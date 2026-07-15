#!/usr/bin/env python3
"""
轻量场景点云 PLY 导出 — 复用 _vggt_work 缓存, 不做语义标注。

坐标保持 VGGT 原始世界坐标 (不中心化), 确保和 FROSS 3D 高斯对齐。

用法:
    python splatv/export_scene_ply.py \
        --scene_dir examples/office2 \
        --output output_semantic_pcd/office2/office2_semantic.ply \
        --max_points 500000
"""

import argparse, os, sys, glob, time

import numpy as np
import torch
from PIL import Image

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_vggt_cache(scene_dir):
    """加载 _vggt_work 缓存: 深度, 相机, 图像。"""
    work_dir = os.path.join(scene_dir, "_vggt_work")
    cameras_path = os.path.join(work_dir, "cameras.npz")
    images_path = os.path.join(work_dir, "vggt_images.npz")
    depth_dir = os.path.join(work_dir, "depth")
    frame_list_path = os.path.join(work_dir, "frame_list.txt")

    if not os.path.exists(cameras_path):
        raise FileNotFoundError(f"未找到 VGGT 缓存: {cameras_path}\n"
                                f"请先运行 vggt_preprocess.py")

    cdata = np.load(cameras_path)
    intrinsics = cdata["intrinsics"]    # [S, 3, 3]
    extrinsics_3x4 = cdata["extrinsics"]  # [S, 3, 4]
    H, W = int(cdata["resolution"][0]), int(cdata["resolution"][1])

    images = np.load(images_path)["images"]  # [S, H, W, 3] float [0,1]

    with open(frame_list_path) as f:
        frame_ids = [line.strip() for line in f]

    depth = np.zeros((len(frame_ids), H, W), dtype=np.float32)
    for s, fid in enumerate(frame_ids):
        dp = os.path.join(depth_dir, f"{fid}.npy")
        depth[s] = np.load(dp)

    return depth, images, intrinsics, extrinsics_3x4, (H, W), frame_ids


def unproject_to_world(depth, intrinsics, extrinsics_3x4, H, W):
    """
    反投影深度图到世界坐标 (VGGT 约定)。

    VGGT 外参 [R|t] 是世界→相机: P_cam = R·P_world + t
    逆推: P_world = R^T·(P_cam - t)
    """
    S = depth.shape[0]
    device = torch.device("cpu")
    all_pts = []

    for s in range(S):
        d = torch.from_numpy(depth[s])
        fx = float(intrinsics[s, 0, 0]); fy = float(intrinsics[s, 1, 1])
        cx = float(intrinsics[s, 0, 2]); cy = float(intrinsics[s, 1, 2])
        R = torch.from_numpy(extrinsics_3x4[s, :3, :3])
        t = torch.from_numpy(extrinsics_3x4[s, :3, 3])

        u = torch.arange(W, dtype=torch.float32)
        v = torch.arange(H, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

        x_cam = (u_grid - cx) * d / fx
        y_cam = (v_grid - cy) * d / fy
        pts_cam = torch.stack([x_cam, y_cam, d], dim=-1)  # [H, W, 3]

        # VGGT 约定: P_world = R^T·(P_cam - t) = (pts_cam - t) @ R
        pts_world = (pts_cam.reshape(-1, 3) - t) @ R
        all_pts.append(pts_world.reshape(H, W, 3).numpy())

    return np.stack(all_pts, axis=0)  # [S, H, W, 3]


def main():
    parser = argparse.ArgumentParser(description="场景点云 PLY (复用 VGGT 缓存)")
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf_percentile", type=float, default=20.0,
                        help="深度置信度百分位阈值, 20=保留 top 80%%")
    parser.add_argument("--mask_dir", default=None,
                        help="动态物体 mask 目录, 用于滤除动态点")
    args = parser.parse_args()

    depth, images, intrinsics, extrinsics, (H, W), frame_ids = load_vggt_cache(args.scene_dir)
    S = len(frame_ids)
    print(f"加载 VGGT 缓存: {S} 帧, {H}x{W}")

    # 置信度过滤
    conf_files = sorted(glob.glob(
        os.path.join(args.scene_dir, "_vggt_work", "depth", "*_conf.npy")))
    if conf_files:
        all_conf = []
        for cf in conf_files:
            all_conf.append(np.load(cf).ravel())
        all_conf = np.concatenate(all_conf)
        all_conf = all_conf[np.isfinite(all_conf) & (all_conf > 1e-5)]
        threshold = float(np.percentile(all_conf, args.conf_percentile))
        print(f"深度置信度阈值: perc_{args.conf_percentile}={threshold:.2f}")
    else:
        threshold = 0.0
        print("无置信度文件, 使用全部深度")

    # 反投影
    print("反投影到世界坐标 ...")
    t0 = time.time()
    world_pts = unproject_to_world(depth, intrinsics, extrinsics, H, W)
    dt = time.time() - t0
    print(f"  完成 ({dt:.1f}s)")

    # 加载动态 mask
    mask_static = None
    if args.mask_dir and os.path.isdir(args.mask_dir):
        print("加载动态 mask ...")
        mask_static = []
        for fid in frame_ids:
            mp = os.path.join(args.mask_dir, f"{fid}.png")
            if os.path.exists(mp):
                m = np.array(Image.open(mp))
                # mask: 255=动态, 0=静态 → 反转为 1=静态, 0=动态
                if m.ndim == 3:
                    m = m[:, :, 0]
                m = (m < 128).astype(np.float32)
                # 缩放到 VGGT 深度分辨率
                if m.shape[:2] != (H, W):
                    m = np.array(Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(np.float32)
            else:
                m = np.ones((H, W), dtype=np.float32)
            mask_static.append(m)
        print(f"  加载 {len(mask_static)} 帧 mask")

    # 收集有效点
    all_xyz = []
    all_rgb = []
    for s in range(S):
        pts = world_pts[s, ::args.stride, ::args.stride]
        img = images[s, ::args.stride, ::args.stride]  # [0,1]
        dep = depth[s, ::args.stride, ::args.stride]

        valid = (np.isfinite(pts).all(axis=-1) &
                 (dep > 1e-6))

        # 动态 mask 滤除
        if mask_static is not None:
            ms = mask_static[s][::args.stride, ::args.stride]
            valid = valid & (ms > 0.5)

        # 深度置信度过滤
        if conf_files and threshold > 0:
            conf = np.load(conf_files[s])[::args.stride, ::args.stride]
            valid &= (conf > threshold)

        all_xyz.append(pts[valid])
        rgb = np.clip(img[valid] * 255, 0, 255).astype(np.uint8)
        all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    print(f"  有效点: {len(xyz):,}")

    # 降采样 (随机采样避免条纹)
    if args.max_points > 0 and len(xyz) > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xyz), args.max_points, replace=False)
        xyz = xyz[idx]; rgb = rgb[idx]
        print(f"  随机降采样 → {len(xyz):,}")

    # 范围信息
    print(f"  范围: x[{xyz[:, 0].min():.1f}, {xyz[:, 0].max():.1f}] "
          f"y[{xyz[:, 1].min():.1f}, {xyz[:, 1].max():.1f}] "
          f"z[{xyz[:, 2].min():.1f}, {xyz[:, 2].max():.1f}]")

    # 写 PLY
    from plyfile import PlyData, PlyElement
    N = len(xyz)
    vertex = np.zeros(N, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(args.output)
    print(f"\n完成: {args.output} ({N:,} 点)")


if __name__ == "__main__":
    main()
