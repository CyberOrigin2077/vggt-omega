"""
导出脚本（仅静态高斯）：SplatVoxel 自适应体素 → .splatv。

只导出 SplatVoxelNet 产生的静态高斯，不包含逐帧动态高斯。
可选通过 --semantic_dir 将 SAM3 语义标签传播到高斯，存入 splaTV slot 8。

用法（无语义）:
    python export_static_splatvoxel.py \
        --scene_dir examples/hotel1 \
        --sv_ckpt checkpoints/splatvoxel/checkpoint-best.pth \
        --vggt_ckpt checkpoints/vggt_omega_1b_512.pt \
        --output_dir ./output_static/hotel1 \
        --voxel_grid_res 1200 \
        --num_frames 60

用法（含语义——先运行 infer_semantic.py）:
    python infer_semantic.py --scene_dir examples/hotel1
    python export_static_splatvoxel.py \
        --scene_dir examples/hotel1 \
        ... \
        --semantic_dir examples/hotel1/semantic_masks
"""

import argparse
import glob
import json
import os
import struct
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.gs import depth_to_world_points
from train_splatvoxel import SplatVoxelNet

CACHED_LAYERS = [4, 11, 17, 23]


# ------------------------------------------------------------------
#  splaTV texture 格式
# ------------------------------------------------------------------

def float_to_half(f):
    return int(np.float16(f).view(np.uint16))


def pack_half2x16(x, y):
    hx = float_to_half(x)
    hy = float_to_half(y)
    return (hx | (hy << 16)) & 0xFFFFFFFF


def write_splatv_texture(filepath, positions, scales, colors, opacities, rots,
                         trbf_centers, trbf_scales, semantic_ids=None):
    N = positions.shape[0]
    texwidth = 4096
    texheight = int(np.ceil((4 * N) / texwidth))
    texdata = np.zeros(texwidth * texheight * 4, dtype=np.uint32)
    texdata_f = texdata.view(np.float32)
    texdata_u8 = texdata.view(np.uint8)

    has_sem = semantic_ids is not None and len(semantic_ids) == N

    print(f"Writing {N:,} Gaussians, texture {texwidth}x{texheight} ...")
    for j in range(N):
        texdata_f[16 * j + 0] = -positions[j, 0]
        texdata_f[16 * j + 1] = positions[j, 1]
        texdata_f[16 * j + 2] = positions[j, 2]
        texdata[16 * j + 3] = pack_half2x16(rots[j, 0], rots[j, 1])
        texdata[16 * j + 4] = pack_half2x16(rots[j, 2], rots[j, 3])
        s = scales[j, 0]
        texdata[16 * j + 5] = pack_half2x16(s, s)
        texdata[16 * j + 6] = pack_half2x16(s, 0)
        texdata_u8[4 * (16 * j + 7) + 0] = int(colors[j, 0] * 255)
        texdata_u8[4 * (16 * j + 7) + 1] = int(colors[j, 1] * 255)
        texdata_u8[4 * (16 * j + 7) + 2] = int(colors[j, 2] * 255)
        texdata_u8[4 * (16 * j + 7) + 3] = int(opacities[j, 0] * 255)
        for k in range(8, 15):
            texdata[16 * j + k] = 0
        if has_sem:
            texdata[16 * j + 8] = int(semantic_ids[j])
        texdata[16 * j + 15] = pack_half2x16(
            float(trbf_centers[j]), float(trbf_scales[j]))

    metadata = [{
        "type": "splat",
        "size": int(texdata.nbytes),
        "texwidth": texwidth,
        "texheight": texheight,
        "cameras": [{
            "id": 0, "img_name": "camera_0001",
            "width": 1920, "height": 1080,
            "position": [0.0, 0.0, 3.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "fy": 1000.0, "fx": 1000.0,
        }]
    }]
    json_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", 0x674b))
        f.write(struct.pack("<I", len(json_bytes)))
        f.write(json_bytes)
        f.write(texdata.tobytes())
    print(f"  Done: {os.path.getsize(filepath) / (1024 * 1024):.1f} MB")


# ------------------------------------------------------------------
#  Mask 加载（可选）
# ------------------------------------------------------------------

def load_merged_masks(frame_paths, mask_dirs, target_h, target_w,
                      dilate_pixels, erode_pixels):
    """合并所有 mask 子目录 → 统一动态 mask。任一 mask 标记为动态的像素 → 动态(0)。"""
    masks = []
    for fp in frame_paths:
        frame_id = os.path.splitext(os.path.basename(fp))[0]
        merged = np.ones((target_h, target_w), dtype=np.float32)
        for md in mask_dirs:
            exact = os.path.join(md, f"{frame_id}.png")
            if os.path.exists(exact):
                pm = 1.0 - np.array(Image.open(exact)).astype(np.float32) / 255.0
                pm = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)
                pm = F.interpolate(pm, size=(target_h, target_w), mode="nearest")
                merged = np.minimum(merged, pm.squeeze().numpy())

            pattern = os.path.join(md, f"{frame_id}_*.png")
            for pf in sorted(glob.glob(pattern)):
                pm = 1.0 - np.array(Image.open(pf)).astype(np.float32) / 255.0
                pm = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)
                pm = F.interpolate(pm, size=(target_h, target_w), mode="nearest")
                merged = np.minimum(merged, pm.squeeze().numpy())

        m = torch.from_numpy(merged).unsqueeze(0).unsqueeze(0)
        m = (m.squeeze() > 0.5).float()

        for op, px in [("dilate_dynamic", dilate_pixels), ("erode_dynamic", erode_pixels)]:
            if px > 0:
                k = 2 * px + 1
                if op == "dilate_dynamic":
                    t = (1.0 - m).unsqueeze(0).unsqueeze(0)
                    t = F.max_pool2d(t, kernel_size=k, stride=1, padding=k // 2)
                    m = 1.0 - t.squeeze()
                else:
                    t = m.unsqueeze(0).unsqueeze(0)
                    t = F.max_pool2d(t, kernel_size=k, stride=1, padding=k // 2)
                    m = t.squeeze()
        masks.append(m)
    return torch.stack(masks, dim=0)


# ------------------------------------------------------------------
#  语义 mask 加载
# ------------------------------------------------------------------

def load_semantic_masks(frame_paths, semantic_dir, target_h, target_w):
    """加载 SAM3 生成的 uint16 语义 mask 并缩放到目标分辨率。

    Returns:
        [S, target_h, target_w] int32 张量, 0=背景, 1..N=实例ID
    """
    masks = []
    for fp in frame_paths:
        frame_id = os.path.splitext(os.path.basename(fp))[0]
        sem_path = os.path.join(semantic_dir, f"{frame_id}.png")
        if os.path.exists(sem_path):
            sem = np.array(Image.open(sem_path)).astype(np.int32)
        else:
            sem = np.zeros((target_h, target_w), dtype=np.int32)
        sem_t = torch.from_numpy(sem).unsqueeze(0).unsqueeze(0).float()
        sem_t = F.interpolate(sem_t, size=(target_h, target_w), mode="nearest")
        masks.append(sem_t.squeeze().int())
    return torch.stack(masks, dim=0)


def propagate_semantics_to_gaussians(semantic_masks, static_masks,
                                      inverse_indices, counts, keep):
    """将逐帧 2D 语义标签传播到 3D 高斯。

    利用体素投影阶段的 inverse_indices 映射（像素→体素），
    在每体素内做多数投票，再经 keep 筛选得到每高斯语义 ID。

    Args:
        semantic_masks: [S, H, W] int32, 0=背景
        static_masks:   [S, H, W] float, 1=static
        inverse_indices:[N_static] long, 每个静态像素 → 体素 ID
        counts:         [M] long, 每个体素包含的像素数
        keep:           [M] bool, 哪些体素产出了高斯

    Returns:
        [sum(keep)] int32, 每高斯语义 ID
    """
    device = inverse_indices.device
    S = semantic_masks.shape[0]
    M = counts.shape[0]

    # 逐帧提取静态像素的语义标签
    sem_labels_list = []
    for s in range(S):
        keep_px = static_masks[s].reshape(-1) > 0.5
        sem_frame = semantic_masks[s].reshape(-1)
        sem_labels_list.append(sem_frame[keep_px])

    sem_labels = torch.cat(sem_labels_list, dim=0).to(device)  # [N_static]
    N_static = sem_labels.shape[0]

    if N_static == 0:
        return torch.zeros(keep.sum(), dtype=torch.int32)

    max_label = max(sem_labels.max().item() + 1, 1)
    total_bins = int(M) * int(max_label)

    # 向量化多数投票：scatter 构建 [M, max_label] 直方图
    flat_idx = inverse_indices.long() * max_label + sem_labels.long()
    sem_flat = torch.zeros(total_bins, dtype=torch.long, device=device)
    sem_flat.scatter_add_(0, flat_idx,
                          torch.ones(N_static, dtype=torch.long, device=device))
    sem_hist = sem_flat.reshape(M, max_label)

    # 每个体素取频次最高的标签
    voxel_sem = sem_hist.argmax(dim=1)  # [M]

    return voxel_sem[keep]


# ------------------------------------------------------------------
#  VGGT-Omega 推理
# ------------------------------------------------------------------

@torch.no_grad()
def vggt_inference(model, images_batch):
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        aggregated_tokens_list, patch_token_start = model.aggregator(images_batch)
    with torch.autocast(device_type="cuda", enabled=False):
        pose_enc = model.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
        depth, depth_conf = model.dense_head(
            aggregated_tokens_list, images=images_batch, patch_token_start=patch_token_start)

    cached_tokens = [aggregated_tokens_list[i] for i in CACHED_LAYERS]
    for i in range(len(aggregated_tokens_list)):
        aggregated_tokens_list[i] = None
    del aggregated_tokens_list

    return {"depth": depth, "depth_conf": depth_conf, "pose_enc": pose_enc,
            "cached_tokens": cached_tokens}


# ====================================================================
#  Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="VGGT-Omega SplatVoxel 静态高斯导出")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景目录，需包含 frames/ 子目录")
    parser.add_argument("--vggt_ckpt", type=str, required=True,
                        help="VGGT-Omega 权重路径")
    parser.add_argument("--sv_ckpt", type=str, required=True,
                        help="SplatVoxelNet 权重路径")
    parser.add_argument("--output_dir", type=str, default="output_static")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="VGGT-Omega 输入分辨率（需被 16 整除）")
    parser.add_argument("--voxel_grid_res", type=int, default=1200)
    parser.add_argument("--dilate", type=int, default=5,
                        help="膨胀动态 mask 的像素数，用于收缩静态区域边缘，保证静态高斯干净")
    parser.add_argument("--trbf_center", type=float, default=0.5,
                        help="TRBF 时间中心（静态高斯始终可见）")
    parser.add_argument("--trbf_scale", type=float, default=10.0,
                        help="TRBF 时间尺度（大值=全时域可见）")
    parser.add_argument("--semantic_dir", type=str, default=None,
                        help="语义 mask 目录（含 {frame_id}.png uint16），"
                             "由 infer_semantic.py 生成")
    parser.add_argument("--device_vggt", type=str, default="cuda:0")
    parser.add_argument("--device_splatvoxel", type=str, default="cuda:1")
    args = parser.parse_args()

    if args.image_resolution % 16 != 0:
        raise ValueError("image_resolution 必须能被 16 整除")

    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")

    # 自动检测 mask 子目录（排除 semantic_masks，那是 uint16 语义标签）
    mask_dirs = sorted(
        d for d in glob.glob(os.path.join(scene_dir, "*mask*"))
        if not os.path.basename(d).startswith("semantic")
    )
    if mask_dirs:
        print(f"Detected mask dirs: {[os.path.basename(d) for d in mask_dirs]}")

    device0 = torch.device(args.device_vggt)
    device1 = torch.device(args.device_splatvoxel)
    os.makedirs(args.output_dir, exist_ok=True)

    # ======================== 模型加载 ========================
    print(f"Loading VGGT-Omega backbone on {device0}...")
    backbone = VGGTOmega().eval().to(device0)
    backbone.load_state_dict(torch.load(args.vggt_ckpt, map_location="cpu", weights_only=True))
    for p in backbone.parameters():
        p.requires_grad = False

    print(f"Loading SplatVoxel (static) on {device1}...")
    with torch.cuda.device(device1):
        sv_net = SplatVoxelNet(dpt_out_dim=48, voxel_grid_res=args.voxel_grid_res)
    sv_net = sv_net.to(device1)
    sv_ckpt = torch.load(args.sv_ckpt, map_location=device1, weights_only=True)
    sv_net.load_state_dict(sv_ckpt["model"], strict=False)
    sv_net.eval()

    # ======================== 数据加载 ========================
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if args.num_frames is not None:
        frame_paths = frame_paths[:args.num_frames]
    S = len(frame_paths)
    print(f"Processing {S} frames from {frames_dir}")

    # ======================== VGGT 推理 ========================
    torch.cuda.synchronize()
    t_feedforward_start = time.time()
    print("Running VGGT-Omega inference...")
    images = load_and_preprocess_images(frame_paths, image_resolution=args.image_resolution)
    S = images.shape[0]
    H_out, W_out = images.shape[-2], images.shape[-1]
    print(f"  VGGT input: {H_out}x{W_out}")

    images_batch = images.unsqueeze(0).to(device0)
    out = vggt_inference(backbone, images_batch)
    extrinsics, intrinsics = encoding_to_camera(out["pose_enc"], (H_out, W_out))

    cached_tokens = [t.to(device1) for t in out["cached_tokens"]]
    images_t = images_batch.float().to(device1)
    depth_t = out["depth"].squeeze(0).squeeze(-1).float().to(device1)
    intri_t = intrinsics.squeeze(0).float().to(device1)
    extri_t = extrinsics.squeeze(0).float().to(device1)

    # ======================== Mask（可选） ========================
    if mask_dirs:
        static_masks = load_merged_masks(
            frame_paths, mask_dirs, H_out, W_out,
            dilate_pixels=args.dilate, erode_pixels=0,
        ).to(device1)
    else:
        static_masks = torch.ones(S, H_out, W_out, device=device1)

    # ======================== 语义 mask 加载（可选） ========================
    semantic_masks = None
    if args.semantic_dir and os.path.isdir(args.semantic_dir):
        print(f"Loading semantic masks from {args.semantic_dir} ...")
        semantic_masks = load_semantic_masks(
            frame_paths, args.semantic_dir, H_out, W_out,
        ).to(device1)

    # ======================== SplatVoxel 推理 ========================
    print(f"Running SplatVoxel (voxel_grid_res={args.voxel_grid_res})...")
    need_indices = semantic_masks is not None
    with torch.no_grad(), torch.cuda.device(device1):
        sv_out = sv_net(
            cached_tokens, images_t, static_masks, depth_t, intri_t, extri_t,
            return_indices=need_indices,
        )
    if need_indices:
        s_xyz, s_op, s_rgb, s_sc, inverse_indices, counts, keep = sv_out
    else:
        s_xyz, s_op, s_rgb, s_sc = sv_out
    N = s_xyz.shape[0]
    torch.cuda.synchronize()
    t_feedforward_end = time.time()
    t_feedforward = t_feedforward_end - t_feedforward_start
    print(f"  -> {N:,} Gaussians")
    print(f"  [Timing] Feed-forward (VGGT + SplatVoxel): {t_feedforward:.2f}s")
    if N < 100:
        print("ERROR: Too few Gaussians generated")
        return

    # ======================== 语义传播（可选） ========================
    gaussian_sem_np = None
    if semantic_masks is not None and N > 0:
        print("Propagating semantics to Gaussians ...")
        gaussian_sem = propagate_semantics_to_gaussians(
            semantic_masks, static_masks,
            inverse_indices, counts, keep,
        )
        gaussian_sem_np = gaussian_sem.cpu().numpy()
        # 统计语义分布
        unique_sem, _ = np.unique(gaussian_sem_np, return_counts=True)
        bg_ratio = (gaussian_sem_np == 0).mean() * 100
        print(f"  {len(unique_sem)} unique labels (incl. bg), "
              f"{bg_ratio:.1f}% background")

    # ======================== 组装 splaTV ========================
    print("Building .splatv ...")
    xyz_np = s_xyz.cpu().numpy()
    sc_np = s_sc.cpu().numpy()
    rgb_np = s_rgb.cpu().numpy()
    op_np = s_op.cpu().numpy()

    center = xyz_np.mean(axis=0)
    xyz_np = xyz_np - center

    trbf_c = np.full(N, args.trbf_center, dtype=np.float32)
    trbf_s = np.full(N, args.trbf_scale, dtype=np.float32)

    rgb_np = np.clip(rgb_np, 0, 1).astype(np.float32)
    op_np = np.clip(op_np, 0, 1).astype(np.float32)
    rots = np.zeros((N, 4), dtype=np.float32)
    rots[:, 3] = 1.0

    print(f"  XYZ: x=[{xyz_np[:,0].min():.1f},{xyz_np[:,0].max():.1f}] "
          f"y=[{xyz_np[:,1].min():.1f},{xyz_np[:,1].max():.1f}] "
          f"z=[{xyz_np[:,2].min():.1f},{xyz_np[:,2].max():.1f}]")
    print(f"  Scale: [{sc_np.min():.4f}, {sc_np.max():.4f}]")

    scene_name = os.path.basename(os.path.abspath(scene_dir))
    output_path = os.path.join(args.output_dir, f"{scene_name}_static.splatv")

    t_save_start = time.time()
    write_splatv_texture(output_path, xyz_np, sc_np, rgb_np, op_np,
                         rots, trbf_c, trbf_s,
                         semantic_ids=gaussian_sem_np)
    t_save = time.time() - t_save_start

    print(f"\nDone: {output_path} ({N:,} Gaussians)")
    print(f"[Timing] Save .splatv: {t_save:.2f}s")
    print(f"[Timing] Total (feed-forward + save): {t_feedforward + t_save:.2f}s")


if __name__ == "__main__":
    main()
