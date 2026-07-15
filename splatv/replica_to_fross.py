"""
Replica RGB-D 数据集 → FROSS 格式。

用法 (任意 python 环境):
    python splatv/replica_to_fross.py --replica_dir Datasets/Replica --scene room0 \
        --output_dir Datasets/replica_fross --frame_step 50
"""

import argparse, json, os, sys
import numpy as np
from PIL import Image


def read_trajectory(traj_path):
    """读取 Replica traj.txt → [S, 4, 4] c2w 矩阵。"""
    poses = []
    with open(traj_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            pose = np.array(vals).reshape(4, 4)
            poses.append(pose)
    return np.stack(poses, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Replica → FROSS")
    parser.add_argument("--replica_dir", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_dir", default="Datasets/replica_fross")
    parser.add_argument("--frame_step", type=int, default=50, help="每隔N帧取1帧")
    args = parser.parse_args()

    scene_dir = os.path.join(args.replica_dir, args.scene)
    results_dir = os.path.join(scene_dir, "results")
    traj_path = os.path.join(scene_dir, "traj.txt")

    if not os.path.isdir(results_dir):
        sys.exit(f"未找到 {results_dir}")

    # 相机内参
    with open(os.path.join(args.replica_dir, "cam_params.json")) as f:
        cam = json.load(f)["camera"]
    W, H = cam["w"], cam["h"]  # 1200, 680
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]
    depth_scale = cam["scale"]  # 6553.5, 16-bit → meters

    # 帧列表
    rgb_files = sorted(f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg"))
    depth_files = sorted(f for f in os.listdir(results_dir) if f.startswith("depth") and f.endswith(".png"))

    # 降采样
    rgb_files = rgb_files[::args.frame_step]
    depth_files = [d for d in depth_files if
                   d.replace("depth", "").replace(".png", "") in
                   [r.replace("frame", "").replace(".jpg", "") for r in rgb_files]]
    S = len(rgb_files)
    print(f"{args.scene}: {S} 帧 (step={args.frame_step})")

    # 位姿
    if os.path.exists(traj_path):
        all_poses = read_trajectory(traj_path)
        poses = all_poses[::args.frame_step]  # [S, 4, 4] c2w
    else:
        poses = np.tile(np.eye(4)[None], (S, 1, 1))

    # FROSS 目录
    seq_dir = os.path.join(args.output_dir, "data", args.scene, "sequence")
    os.makedirs(seq_dir, exist_ok=True)

    # _info.txt (depth_shift: 16-bit → meters)
    depth_shift = int(depth_scale)
    with open(os.path.join(seq_dir, "_info.txt"), "w") as f:
        f.write(f"m_versionNumber = 1\n")
        f.write(f"m_colorWidth = {W}\n")
        f.write(f"m_colorHeight = {H}\n")
        f.write(f"m_depthWidth = {W}\n")
        f.write(f"m_depthHeight = {H}\n")
        f.write(f"m_depthShift = {depth_shift}\n")
        f.write(f"m_calibrationColorIntrinsic = {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0\n")
        f.write(f"m_calibrationDepthIntrinsic = {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0\n")
    print(f"  _info.txt: {W}x{H}, shift={depth_shift}")

    # 逐帧保存
    for s in range(S):
        # RGB: 直接复制
        rgb_src = os.path.join(results_dir, rgb_files[s])
        rgb_dst = os.path.join(seq_dir, f"{args.scene}-{s:06d}.color.jpg")
        Image.open(rgb_src).save(rgb_dst)

        # Depth: 16-bit PNG → 16-bit PNG (保留原始值)
        depth_src = os.path.join(results_dir, depth_files[s])
        depth_dst = os.path.join(seq_dir, f"{args.scene}-{s:06d}.depth.pgm")
        Image.open(depth_src).save(depth_dst)

        # Pose: c2w → w2c (FROSS 期望 w2c)
        c2w = poses[s]
        w2c = np.linalg.inv(c2w)
        pose_dst = os.path.join(seq_dir, f"{args.scene}-{s:06d}.pose.txt")
        with open(pose_dst, "w") as f:
            for row in range(4):
                f.write(f"{w2c[row, 0]:.6f} {w2c[row, 1]:.6f} "
                        f"{w2c[row, 2]:.6f} {w2c[row, 3]:.6f}\n")

    # 生成 scan_list + class mapping
    ssg_dir = os.path.join(args.output_dir, "ReplicaSSG")
    os.makedirs(ssg_dir, exist_ok=True)
    with open(os.path.join(ssg_dir, "test_scans.txt"), "w") as f:
        f.write(f"{args.scene}\n")

    print(f"\n运行 FROSS:")
    print(f"  bash run_fross_replica.sh {args.scene} vg")


if __name__ == "__main__":
    main()
