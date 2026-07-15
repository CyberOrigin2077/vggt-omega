"""
导出脚本（factory 场景适配）：SplatVoxel 静态高斯 + GS_DPTHead 动态高斯 → .splatv。

基于 VGGT-Omega backbone + SplatVoxelNet (自适应体素) + GS_DPTHead。

适配 factory 场景的输入格式：
  - frames/        : 原始图片 (PNG)
  - obj_mask/      : 动态物体 mask (PNG, 与帧同名)
  - person_mask/   : 动态人物 mask (PNG, 命名: 帧号_objXX.png, 同一帧可能有多个)

合并逻辑：静态 = ~(obj_mask ∪ person_mask)。

用法:
    python export_splatvoxel.py \
        --scene_dir examples/factory0 \
        --sv_ckpt checkpoints/splatvoxel/checkpoint-best.pth \
        --gs_ckpt checkpoints/gs_head/checkpoint-best.pth \
        --vggt_ckpt checkpoints/vggt_omega_1b_512.pt \
        --output_dir ./output_factory0 \
        --voxel_grid_res 1200 \
        --num_frames 60
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.gs import GS_DPTHead, depth_to_world_points
from train_splatvoxel import SplatVoxelNet

CACHED_LAYERS = [4, 11, 17, 23]
PATCH_TOKEN_START = 17


# ------------------------------------------------------------------
#  splaTV texture 格式（与 da3 版本完全一致）
# ------------------------------------------------------------------

def float_to_half(f):
    return int(np.float16(f).view(np.uint16))


def pack_half2x16(x, y):
    hx = float_to_half(x)
    hy = float_to_half(y)
    return (hx | (hy << 16)) & 0xFFFFFFFF


def write_splatv_texture(filepath, positions, scales, colors, opacities, rots,
                         trbf_centers, trbf_scales):
    N = positions.shape[0]
    texwidth = 4096
    texheight = int(np.ceil((4 * N) / texwidth))
    texdata = np.zeros(texwidth * texheight * 4, dtype=np.uint32)
    texdata_f = texdata.view(np.float32)
    texdata_u8 = texdata.view(np.uint8)

    print(f"Writing {N:,} Gaussians, texture {texwidth}×{texheight} ...")
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
#  Mask 合并（与 da3 版本一致）
# ------------------------------------------------------------------

def load_merged_masks(frame_paths, mask_dirs, target_h, target_w,
                      dilate_pixels, erode_pixels):
    """
    合并所有 mask 子目录 → 统一动态 mask。
    任一 mask 中标记为动态的像素 → 最终为动态（0=动态, 1=静态）。
    支持 {frame_id}.png 和 {frame_id}_objXX.png 两种命名。
    """
    masks = []
    for fp in frame_paths:
        frame_id = os.path.splitext(os.path.basename(fp))[0]

        merged = np.ones((target_h, target_w), dtype=np.float32)
        for md in mask_dirs:
            # 精确匹配: frame_id.png
            # 工厂 mask 白色=动态, 需取反 (1-mask) → 静态=1, 动态=0
            exact = os.path.join(md, f"{frame_id}.png")
            if os.path.exists(exact):
                pm = 1.0 - np.array(Image.open(exact)).astype(np.float32) / 255.0
                pm = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)
                pm = F.interpolate(pm, size=(target_h, target_w), mode="nearest")
                merged = np.minimum(merged, pm.squeeze().numpy())

            # 前缀匹配: frame_id_*.png (如 person_mask)
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
#  VGGT-Omega 推理包装
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
    parser = argparse.ArgumentParser(description="VGGT-Omega SplatVoxel 4DGS 导出")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景目录，需包含 frames/ 及 mask 子目录")
    parser.add_argument("--vggt_ckpt", type=str, required=True,
                        help="VGGT-Omega 权重路径")
    parser.add_argument("--sv_ckpt", type=str, required=True,
                        help="SplatVoxelNet 权重路径")
    parser.add_argument("--gs_ckpt", type=str, required=True,
                        help="GS_DPTHead 权重路径")
    parser.add_argument("--output_dir", type=str, default="output_export")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="VGGT-Omega 输入分辨率（需被 16 整除）")
    parser.add_argument("--voxel_grid_res", type=int, default=1200)
    parser.add_argument("--mask_dilate", type=int, default=6)
    parser.add_argument("--mask_erode", type=int, default=2)
    parser.add_argument("--static_t_scale", type=float, default=10.0)
    parser.add_argument("--temporal_sigma", type=float, default=0.6)
    parser.add_argument("--dynamic_scale_mult", type=float, default=4.0)
    parser.add_argument("--dynamic_mask_shrink", type=int, default=5,
                        help="收缩动态 mask 像素数，去除边缘漂浮高斯")
    parser.add_argument("--frames_chunk_size", type=int, default=8)
    parser.add_argument("--split_save", action="store_true", default=False,
                        help="单独保存静态和动态 .splatv 文件")
    parser.add_argument("--device_vggt", type=str, default="cuda:0")
    parser.add_argument("--device_splatvoxel", type=str, default="cuda:1")
    args = parser.parse_args()

    if args.image_resolution % 16 != 0:
        raise ValueError("image_resolution 必须能被 16 整除")

    # 自动检测子目录
    scene_dir = args.scene_dir
    frames_dir = os.path.join(scene_dir, "frames")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"frames dir not found: {frames_dir}")

    # 扫描 mask 子目录: *mask* 匹配 (如 obj_mask, person_mask)
    mask_dirs = sorted(glob.glob(os.path.join(scene_dir, "*mask*")))
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

    print(f"Loading GS head (dynamic) on {device1}...")
    gs_head = GS_DPTHead(
        dim_in=2048, patch_size=16, output_dim=8, features=256,
        out_channels=[256, 512, 1024, 1024],
        intermediate_layer_idx=[0, 1, 2, 3],
        pos_embed=True, down_ratio=1,
    ).to(device1)
    gs_ckpt = torch.load(args.gs_ckpt, map_location=device1, weights_only=True)
    gs_state = gs_ckpt["model"] if "model" in gs_ckpt else gs_ckpt
    gs_head.load_state_dict(gs_state, strict=False)
    gs_head.eval()
    for p in gs_head.parameters():
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
    print("Running VGGT-Omega inference...")
    images = load_and_preprocess_images(frame_paths, image_resolution=args.image_resolution)
    S = images.shape[0]
    H_out, W_out = images.shape[-2], images.shape[-1]
    print(f"  VGGT input: {H_out}×{W_out}")

    images_batch = images.unsqueeze(0).to(device0)
    out = vggt_inference(backbone, images_batch)
    extrinsics, intrinsics = encoding_to_camera(out["pose_enc"], (H_out, W_out))

    # ---- 中间特征 → GPU 1 ----
    cached_tokens = [t.to(device1) for t in out["cached_tokens"]]

    # ---- images & camera params → GPU 1 ----
    images_t = images_batch.float().to(device1)
    depth_t = out["depth"].squeeze(0).squeeze(-1).float().to(device1)
    intri_t = intrinsics.squeeze(0).float().to(device1)
    extri_t = extrinsics.squeeze(0).float().to(device1)

    # ---- 合并所有 mask 目录 ----
    static_masks = load_merged_masks(
        frame_paths, mask_dirs, H_out, W_out,
        args.mask_dilate, args.mask_erode,
    ).to(device1)

    # ================================================================
    #  Phase 1: SplatVoxel → 静态高斯
    # ================================================================
    print(f"Phase 1: Static Gaussians (voxel_grid_res={args.voxel_grid_res})...")
    with torch.no_grad(), torch.cuda.device(device1):
        s_xyz, s_op, s_rgb, s_sc = sv_net(
            cached_tokens, images_t, static_masks, depth_t, intri_t, extri_t)
    Ns = s_xyz.shape[0]
    print(f"  -> {Ns:,} Gaussians")
    if Ns < 100:
        print("ERROR: Too few")
        return

    pts3d = depth_to_world_points(depth_t, intri_t, extri_t)

    # ================================================================
    #  Phase 2: 逐帧动态高斯
    # ================================================================
    print(f"Phase 2: Dynamic Gaussians ({S} frames)...")
    all_d_xyz, all_d_op, all_d_rgb, all_d_sc, all_d_t = [], [], [], [], []
    for i in range(S):
        img_i = images_t[:, i:i+1]  # [1, 1, 3, H, W]
        tokens_list = [t[:, i:i+1, PATCH_TOKEN_START:] for t in cached_tokens]
        pts_i = pts3d[i].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 3]

        with torch.no_grad(), torch.cuda.device(device1):
            gs_all = gs_head(tokens_list, img_i, patch_start_idx=0,
                             frames_num=S, frames_chunk_size=args.frames_chunk_size,
                             pts=pts_i)

        d_mask = (static_masks[i].reshape(-1) <= 0.5)
        # 收缩动态 mask 去除边缘漂浮高斯
        if args.dynamic_mask_shrink > 0:
            k = 2 * args.dynamic_mask_shrink + 1
            static_2d = (~d_mask).float().reshape(H_out, W_out).unsqueeze(0).unsqueeze(0)
            static_2d = F.max_pool2d(static_2d, kernel_size=k, stride=1, padding=k // 2)
            d_mask = (static_2d.squeeze().reshape(-1) <= 0.5)
        if d_mask.any():
            all_d_xyz.append(gs_all.xyz.reshape(-1, 3)[d_mask].cpu())
            all_d_op.append(gs_all.opacity.reshape(-1, 1)[d_mask].cpu())
            all_d_rgb.append(gs_all.rgb.reshape(-1, 3)[d_mask].cpu())
            all_d_sc.append(gs_all.scale_xyz.reshape(-1, 1)[d_mask].cpu())
            all_d_t.append(torch.full((d_mask.sum().item(),), i / max(S - 1, 1)))
        if i % 10 == 0:
            print(f"  Frame {i}/{S}: dynamic={d_mask.sum().item():,}")

    # ================================================================
    #  Phase 3: 组装 splaTV texture
    # ================================================================
    print("Phase 3: Building .splatv ...")
    s_xyz_np = s_xyz.cpu().numpy()
    s_op_np = s_op.cpu().numpy()
    s_rgb_np = s_rgb.cpu().numpy()
    s_sc_np = s_sc.cpu().numpy()

    center = s_xyz_np.mean(axis=0)
    s_xyz_np = s_xyz_np - center

    s_trbf_c = np.full(Ns, 0.5, dtype=np.float32)
    s_trbf_s = np.full(Ns, args.static_t_scale, dtype=np.float32)

    if all_d_xyz:
        d_xyz_np = torch.cat(all_d_xyz, dim=0).numpy() - center
        d_op_np = torch.cat(all_d_op, dim=0).numpy()
        d_rgb_np = torch.cat(all_d_rgb, dim=0).numpy()
        d_sc_np = torch.cat(all_d_sc, dim=0).numpy() * args.dynamic_scale_mult
        d_t = torch.cat(all_d_t, dim=0).numpy()
        Nd = d_xyz_np.shape[0]
        dt = 1.0 / max(S - 1, 1)
        d_trbf_c = d_t.astype(np.float32)
        d_trbf_s = np.full(Nd, dt * args.temporal_sigma, dtype=np.float32)
    else:
        d_xyz_np = np.zeros((0, 3), dtype=np.float32)
        d_op_np = np.zeros((0, 1), dtype=np.float32)
        d_rgb_np = np.zeros((0, 3), dtype=np.float32)
        d_sc_np = np.zeros((0, 1), dtype=np.float32)
        d_trbf_c = np.zeros(0, dtype=np.float32)
        d_trbf_s = np.zeros(0, dtype=np.float32)
        Nd = 0

    all_xyz = np.concatenate([s_xyz_np, d_xyz_np], axis=0).astype(np.float32)
    all_sc = np.concatenate([s_sc_np, d_sc_np], axis=0).astype(np.float32)
    all_rgb = np.clip(np.concatenate([s_rgb_np, d_rgb_np], axis=0), 0, 1).astype(np.float32)
    all_op = np.clip(np.concatenate([s_op_np, d_op_np], axis=0), 0, 1).astype(np.float32)
    all_trbf_c = np.concatenate([s_trbf_c, d_trbf_c], axis=0).astype(np.float32)
    all_trbf_s = np.concatenate([s_trbf_s, d_trbf_s], axis=0).astype(np.float32)

    N_total = Ns + Nd
    all_rots = np.zeros((N_total, 4), dtype=np.float32)
    all_rots[:, 3] = 1.0

    print(f"  XYZ: x=[{all_xyz[:,0].min():.1f},{all_xyz[:,0].max():.1f}] "
          f"y=[{all_xyz[:,1].min():.1f},{all_xyz[:,1].max():.1f}] "
          f"z=[{all_xyz[:,2].min():.1f},{all_xyz[:,2].max():.1f}]")
    print(f"  Scale: [{all_sc.min():.4f}, {all_sc.max():.4f}]")

    scene_name = os.path.basename(os.path.abspath(scene_dir))

    # 合并文件
    output_path = os.path.join(args.output_dir, f"{scene_name}.splatv")
    write_splatv_texture(output_path, all_xyz, all_sc, all_rgb, all_op,
                         all_rots, all_trbf_c, all_trbf_s)
    print(f"  Combined: {output_path}")

    # 单独保存静态/动态
    if args.split_save:
        static_path = os.path.join(args.output_dir, f"{scene_name}_static.splatv")
        write_splatv_texture(static_path, s_xyz_np, s_sc_np, s_rgb_np, s_op_np,
                             all_rots[:Ns], s_trbf_c, s_trbf_s)
        print(f"  Static only: {static_path}")

        if Nd > 0:
            dyn_path = os.path.join(args.output_dir, f"{scene_name}_dynamic.splatv")
            dyn_rots = np.zeros((Nd, 4), dtype=np.float32)
            dyn_rots[:, 3] = 1.0
            write_splatv_texture(dyn_path, d_xyz_np, d_sc_np, d_rgb_np, d_op_np,
                                 dyn_rots, d_trbf_c, d_trbf_s)
            print(f"  Dynamic only: {dyn_path}")

    print(f"\nDone: {N_total:,} Gaussians (static={Ns:,}, dynamic={Nd:,})")


if __name__ == "__main__":
    main()
