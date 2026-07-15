"""
VGGT-Omega 推理 → 静态点云 + SAM3 语义投影 → 带实例标签的 PLY。

在反投影阶段，每个 3D 点源自某帧某个像素，直接查该像素的 SAM3 mask
即可得到全局一致实例 ID，O(1) 无需额外投影。

用法:
    # Step 1: 先生成 SAM3 mask（仅首次需要）
    python splatv/infer_semantic.py --scene_dir examples/room0

    # Step 2: 生成带语义点云
    python splatv/export_semantic_pcd.py \
        --scene_dir examples/room0 \
        --vggt_ckpt checkpoints/vggt_omega_1b_512.pt \
        --semantic_dir examples/room0/semantic_masks \
        --output_dir output_semantic_pcd/room0 \
        --max_points 500000
"""

import argparse, glob, os, sys, time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


# ═══════════════════════════ 深度边缘检测 ═══════════════════════════

def depth_edge(depth, rtol=0.03, kernel_size=3):
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y: y + depth.shape[-2], x: x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


# ═══════════════════════════ VGGT 推理 ═══════════════════════════

@torch.no_grad()
def vggt_inference(model, images_batch):
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        aggregated_tokens_list, patch_token_start = model.aggregator(images_batch)
    with torch.autocast(device_type="cuda", enabled=False):
        pose_enc = model.camera_head(
            aggregated_tokens_list, patch_token_start=patch_token_start)
        depth, depth_conf = model.dense_head(
            aggregated_tokens_list, images=images_batch,
            patch_token_start=patch_token_start)
    return {"depth": depth, "depth_conf": depth_conf, "pose_enc": pose_enc}


def depth_to_world_points(depth, intrinsics, extrinsics):
    if isinstance(depth, np.ndarray): depth = torch.from_numpy(depth)
    if isinstance(intrinsics, np.ndarray): intrinsics = torch.from_numpy(intrinsics)
    if isinstance(extrinsics, np.ndarray): extrinsics = torch.from_numpy(extrinsics)
    depth, intrinsics, extrinsics = depth.float(), intrinsics.float(), extrinsics.float()
    device = depth.device
    N, H, W = depth.shape
    pts_all = []
    for i in range(N):
        fx, fy = intrinsics[i, 0, 0], intrinsics[i, 1, 1]
        cx, cy = intrinsics[i, 0, 2], intrinsics[i, 1, 2]
        d = depth[i]
        u = torch.arange(W, dtype=torch.float32, device=device)
        v = torch.arange(H, dtype=torch.float32, device=device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
        x_cam = (u_grid - cx) * d / fx
        y_cam = (v_grid - cy) * d / fy
        pts_cam = torch.stack([x_cam, y_cam, d], dim=-1).reshape(-1, 3)
        w2c = torch.eye(4, device=device)
        w2c[:3, :] = extrinsics[i][:3, :]
        c2w = torch.linalg.inv(w2c)
        ones = torch.ones(pts_cam.shape[0], 1, device=device)
        pts_w = (c2w @ torch.cat([pts_cam, ones], dim=-1).T).T[:, :3]
        pts_all.append(pts_w.reshape(H, W, 3))
    return torch.stack(pts_all, dim=0)


# ═══════════════════════════ 辅助：加载 mask ═══════════════════════════

def load_masks(frame_paths, mask_dirs, target_h, target_w, dilate_pixels=0):
    """合并动态 mask 目录 → 静态 mask。"""
    S = len(frame_paths)
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
            for pf in sorted(glob.glob(os.path.join(md, f"{frame_id}_*.png"))):
                pm = 1.0 - np.array(Image.open(pf)).astype(np.float32) / 255.0
                pm = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)
                pm = F.interpolate(pm, size=(target_h, target_w), mode="nearest")
                merged = np.minimum(merged, pm.squeeze().numpy())
        m = torch.from_numpy(merged).unsqueeze(0).unsqueeze(0)
        m = (m.squeeze() > 0.5).float()
        if dilate_pixels > 0:
            k = 2 * dilate_pixels + 1
            t = (1.0 - m).unsqueeze(0).unsqueeze(0)
            t = F.max_pool2d(t, kernel_size=k, stride=1, padding=k // 2)
            m = 1.0 - t.squeeze()
        masks.append(m)
    return torch.stack(masks, dim=0)


def load_semantic_masks(frame_paths, semantic_dir, target_h, target_w):
    """加载 SAM3 uint16 语义 mask，缩放到 VGGT 输出分辨率。

    Returns:
        [S, target_h, target_w] int32 张量, 0=背景, >=1=实例ID
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


def id_to_color(obj_id):
    """golden-ratio 哈希 → RGB uint8。"""
    if obj_id <= 0:
        return (60, 60, 70)
    h = (int(obj_id) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    r = max(int(h & 0xFF), 40)
    g = max(int((h >> 8) & 0xFF), 40)
    b = max(int((h >> 16) & 0xFF), 40)
    return (r, g, b)


def _limit_points(vertices, colors, sem_ids, max_points):
    """等距降采样，保持空间均匀。"""
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors, sem_ids
    idx = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[idx], colors[idx], sem_ids[idx]


# ═══════════════════════════ Main ═══════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VGGT-Omega 静态点云 + SAM3 语义 → 带实例标签 PLY")
    parser.add_argument("--scene_dir", required=True,
                        help="场景目录，需包含 frames/")
    parser.add_argument("--vggt_ckpt", required=True,
                        help="VGGT-Omega 权重路径")
    parser.add_argument("--semantic_dir", default=None,
                        help="SAM3 语义 mask 目录（含 {frame_id}.png uint16）"
                             "，默认 {scene_dir}/semantic_masks")
    parser.add_argument("--output_dir", default="output_semantic_pcd")
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--conf_thres", type=float, default=20.0,
                        help="深度置信度百分位阈值，20=保留 top 80%%")
    parser.add_argument("--filter_depth_edges", action="store_true", default=True)
    parser.add_argument("--depth_edge_rtol", type=float, default=0.03)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    frames_dir = os.path.join(args.scene_dir, "frames")
    if not os.path.isdir(frames_dir):
        sys.exit(f"frames dir not found: {frames_dir}")

    # 语义目录
    semantic_dir = args.semantic_dir
    if semantic_dir is None:
        semantic_dir = os.path.join(args.scene_dir, "semantic_masks")
    has_semantic = os.path.isdir(semantic_dir)
    if has_semantic:
        print(f"语义 mask: {semantic_dir}")
    else:
        print("WARNING: 未找到语义 mask 目录，将只输出坐标+颜色")

    # 动态 mask（可选）
    mask_dirs = sorted(
        d for d in glob.glob(os.path.join(args.scene_dir, "*mask*"))
        if not os.path.basename(d).startswith("semantic"))
    if mask_dirs:
        print(f"检测到 mask: {[os.path.basename(d) for d in mask_dirs]}")

    os.makedirs(args.output_dir, exist_ok=True)
    scene_name = os.path.basename(os.path.abspath(args.scene_dir))

    # ---- 帧 ----
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if args.num_frames:
        frame_paths = frame_paths[:args.num_frames]
    S = len(frame_paths)
    print(f"处理 {S} 帧")

    # ---- VGGT ----
    print(f"加载 VGGT-Omega → {device} ...")
    model = VGGTOmega().eval().to(device)
    sd = torch.load(args.vggt_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad = False

    # ---- VGGT 推理 ----
    t0 = time.time()
    print("VGGT 推理 ...")
    images = load_and_preprocess_images(frame_paths, image_resolution=args.image_resolution)
    S = images.shape[0]
    H_out, W_out = images.shape[-2:]
    print(f"  输入分辨率: {H_out}x{W_out}")

    images_batch = images.unsqueeze(0).to(device)
    out = vggt_inference(model, images_batch)
    extrinsics, intrinsics = encoding_to_camera(out["pose_enc"], (H_out, W_out))

    depth = out["depth"].squeeze(0).squeeze(-1).float().cpu()
    depth_conf = out["depth_conf"].squeeze(0).squeeze(-1).float().cpu()
    intri = intrinsics.squeeze(0).float().cpu()
    extri = extrinsics.squeeze(0).float().cpu()
    print(f"  VGGT 推理: {time.time() - t0:.1f}s")

    # ---- 动态 mask ----
    if mask_dirs:
        static_masks = load_masks(frame_paths, mask_dirs, H_out, W_out,
                                  dilate_pixels=args.dilate)
    else:
        static_masks = torch.ones(S, H_out, W_out)

    # ---- 语义 mask ----
    if has_semantic:
        print(f"加载语义 mask ...")
        semantic_masks = load_semantic_masks(
            frame_paths, semantic_dir, H_out, W_out)
        n_instances = len(set(semantic_masks[semantic_masks > 0].tolist()))
        print(f"  {n_instances} 个全局唯一实例")
    else:
        semantic_masks = torch.zeros(S, H_out, W_out, dtype=torch.int32)

    # ---- 深度过滤 ----
    if args.filter_depth_edges:
        depth_np = depth.numpy()
        edge_mask = depth_edge(depth_np, rtol=args.depth_edge_rtol)
        depth_conf = depth_conf * (1.0 - torch.from_numpy(edge_mask).float())
        n_edge = edge_mask.sum()
        print(f"  深度边缘像素: {n_edge:,} ({n_edge / depth_np.size * 100:.1f}%)")

    if args.conf_thres > 0:
        valid_conf = depth_conf.reshape(-1)
        valid_conf = valid_conf[torch.isfinite(valid_conf) & (valid_conf > 1e-5)]
        threshold = float(np.percentile(valid_conf.numpy(), args.conf_thres)) if valid_conf.numel() > 0 else 1.0
    else:
        threshold = 0.0
    print(f"  置信度阈值: perc_{args.conf_thres} = {threshold:.2f}")

    # ---- 筛选 + 反投影 + 语义查表 ----
    print("筛选 & 反投影 & 语义查表 ...")
    all_pts, all_colors, all_sem = [], [], []
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()

    for s in range(S):
        d = depth[s]
        valid_conf = (torch.isfinite(depth_conf[s]) & (depth_conf[s] > 1e-5)
                      & (depth_conf[s] >= threshold) & (static_masks[s] > 0.5))
        if not valid_conf.any():
            continue

        world_pts = depth_to_world_points(
            d.unsqueeze(0), intri[s].unsqueeze(0), extri[s].unsqueeze(0))
        world_pts = world_pts.squeeze(0)

        pts_flat = world_pts.reshape(-1, 3)
        valid_xyz = torch.isfinite(pts_flat).all(dim=1).reshape(H_out, W_out)
        valid = valid_conf & valid_xyz

        # stride 降采样
        pts = world_pts[::args.stride, ::args.stride]
        colors = images_np[s][::args.stride, ::args.stride]
        sem_frame = semantic_masks[s][::args.stride, ::args.stride]
        valid_s = valid[::args.stride, ::args.stride]

        # 查表：每个有效像素的 SAM3 实例 ID
        sem_ids = sem_frame[valid_s].numpy()
        pts_np = pts[valid_s].cpu().numpy()
        colors_np = colors[valid_s.cpu().numpy()]

        all_pts.append(pts_np)
        all_colors.append(colors_np)
        all_sem.append(sem_ids)

    pts_all = np.concatenate(all_pts, axis=0)
    colors_all = np.concatenate(all_colors, axis=0)
    sem_all = np.concatenate(all_sem, axis=0)
    print(f"  总点数: {len(pts_all):,}")

    # ---- 等距降采样 ----
    if args.max_points > 0 and len(pts_all) > args.max_points:
        pts_all, colors_all, sem_all = _limit_points(
            pts_all, colors_all, sem_all, args.max_points)
        print(f"  等距降采样 → {len(pts_all):,}")

    # 去中心
    center = pts_all.mean(axis=0)
    pts_all = pts_all - center
    print(f"  范围: x[{pts_all[:, 0].min():.1f}, {pts_all[:, 0].max():.1f}] "
          f"y[{pts_all[:, 1].min():.1f}, {pts_all[:, 1].max():.1f}] "
          f"z[{pts_all[:, 2].min():.1f}, {pts_all[:, 2].max():.1f}]")

    # ---- 统计 ----
    n_inst = len(set(sem_all[sem_all > 0]))
    n_bg = (sem_all == 0).sum()
    print(f"  有效实例: {n_inst}, 背景点: {n_bg:,} ({n_bg / len(sem_all) * 100:.1f}%)")

    # ---- 计算实例着色 ----
    inst_rgb = np.zeros((len(sem_all), 3), dtype=np.uint8)
    for i in range(3):
        inst_rgb[:, i] = [id_to_color(s)[i] for s in sem_all]
    # 向量化版本更快
    unique_ids = np.unique(sem_all)
    color_lut = {uid: id_to_color(uid) for uid in unique_ids}
    inst_r = np.array([color_lut[s][0] for s in sem_all], dtype=np.uint8)
    inst_g = np.array([color_lut[s][1] for s in sem_all], dtype=np.uint8)
    inst_b = np.array([color_lut[s][2] for s in sem_all], dtype=np.uint8)

    # ---- PLY ----
    colors_u8 = np.clip(colors_all * 255, 0, 255).astype(np.uint8)
    N_out = len(pts_all)
    vertex = np.zeros(N_out, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("instance_id", "i4"),
        ("inst_red", "u1"), ("inst_green", "u1"), ("inst_blue", "u1"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = pts_all[:, 0], pts_all[:, 1], pts_all[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = (
        colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2])
    vertex["instance_id"] = sem_all.astype(np.int32)
    vertex["inst_red"], vertex["inst_green"], vertex["inst_blue"] = inst_r, inst_g, inst_b

    ply_path = os.path.join(args.output_dir, f"{scene_name}_semantic.ply")
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(ply_path)
    t_total = time.time() - t0
    print(f"\n完成: {ply_path} ({N_out:,} 点, {n_inst} 个实例)")
    print(f"总耗时: {t_total:.1f}s")


if __name__ == "__main__":
    main()
