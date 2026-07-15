#!/usr/bin/env python3
"""
关键帧筛选 — 支持三种数据源。

  Replica:   位姿 FPS (farthest point sampling on camera centers + 视角增强)
  VLM 视频:  按运动状态分段自适应抽帧 (navigation 90% + manipulation 10%)
  Manifest:  按预选帧号列表从视频直接抽帧

用法:
    # Replica
    python splatv/select_keyframes.py \
        --pose_file Datasets/Replica/office0/traj.txt \
        --total_frames 2000 \
        --output_dir examples/office0/frames \
        --replica_dir Datasets/Replica/office0

    # VLM 视频 (鱼眼自动去畸变)
    python splatv/select_keyframes.py \
        --vlm_json data/scene/vlm.json \
        --video_file data/scene/cam.mp4 \
        --output_dir examples/scene/frames \
        --cam_yaml data/scene/cam.yaml

    # Manifest 预选帧号 (鱼眼自动去畸变)
    python splatv/select_keyframes.py \
        --manifest data/scene/manifest.txt \
        --video_file data/scene/cam.mp4 \
        --output_dir examples/scene/frames \
        --cam_yaml data/scene/cam.yaml
"""

import argparse, os, sys
import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# Replica 模式: 位姿 FPS
# ═══════════════════════════════════════════════════════════════

def load_poses(pose_file):
    """加载 traj.txt: 每行 16 个 float → 4×4 camera-to-world 矩阵。"""
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return np.stack(poses, axis=0)


def extract_camera_info(poses):
    """提取相机中心和 look-at 方向。"""
    centers = poses[:, :3, 3]
    look_at = poses[:, :3, :3] @ np.array([0, 0, 1], dtype=np.float64)
    look_at = look_at / (np.linalg.norm(look_at, axis=1, keepdims=True) + 1e-10)
    return centers, look_at


def farthest_point_sampling(centers, max_frames=300,
                             coverage=0.15, min_distance=0.05):
    """相机中心 farthest point sampling — 保证空间覆盖。"""
    N = len(centers)
    max_dist = np.sqrt(((centers - centers.mean(axis=0)) ** 2).sum(axis=1)).max()
    scene_radius = float(max_dist)
    stop_dist = scene_radius * coverage

    print(f"  场景半径: {scene_radius:.2f}m  stop_dist: {stop_dist:.3f}m "
          f"(半径×{coverage})")

    if N <= 5000:
        dists = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(axis=2))
    else:
        dists = None

    selected = [0]
    min_dists = np.full(N, np.inf)

    for _ in range(1, min(N, max_frames)):
        last = selected[-1]
        if dists is not None:
            min_dists = np.minimum(min_dists, dists[last])
        else:
            d = np.sqrt(((centers - centers[last]) ** 2).sum(axis=1))
            min_dists = np.minimum(min_dists, d)

        candidates = min_dists.copy()
        candidates[selected] = -1
        candidates[min_dists < min_distance] = -1
        if candidates.max() < 0:
            break

        next_idx = int(np.argmax(candidates))
        if min_dists[next_idx] < stop_dist:
            break
        selected.append(next_idx)

    selected.sort()
    return selected


def augment_by_angle(selected, look_at, min_angle_deg=45, min_frame_gap=10):
    """视角增强: 补入大角度变化的帧。"""
    augmented = set(selected)
    for i in range(len(selected) - 1):
        si, ei = selected[i], selected[i + 1]
        if ei - si < min_frame_gap:
            continue
        mid_start = si + min_frame_gap // 2
        mid_end = ei - min_frame_gap // 2
        if mid_end <= mid_start:
            continue
        best_idx, best_angle = -1, 0.0
        for j in range(mid_start, min(mid_end + 1, len(look_at))):
            angle = np.arccos(np.clip(np.abs(np.dot(look_at[si], look_at[j])), 0, 1))
            if angle > best_angle:
                best_angle, best_idx = angle, j
        if np.rad2deg(best_angle) > min_angle_deg:
            augmented.add(best_idx)
    return sorted(augmented)


def min_distance_to_set(centers, set_centers):
    d = np.sqrt(((centers[:, None, :] - set_centers[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).max())


# ═══════════════════════════════════════════════════════════════
# VLM 模式: 分段自适应抽帧
# ═══════════════════════════════════════════════════════════════

def vlm_adaptive_sample(vlm_json, video_file, output_dir, target_frames=100,
                        nav_ratio=0.9):
    """VLM 分段自适应抽帧: navigation 占 nav_ratio, manipulation 占其余。"""
    import json, cv2

    with open(vlm_json) as f:
        data = json.load(f)

    def to_seconds(t):
        m, s = t.split(":")
        return int(m) * 60 + int(s)

    fps = 30.0
    segments = data["segments"]
    nav_sec = sum(to_seconds(s["end_time"]) - to_seconds(s["start_time"])
                  for s in segments if s["type"] == "navigation")
    manip_sec = sum(to_seconds(s["end_time"]) - to_seconds(s["start_time"])
                    for s in segments if s["type"] != "navigation")

    nav_frames = max(int(target_frames * nav_ratio), 1)
    manip_frames = target_frames - nav_frames
    nav_interval = max(1.0 / fps, nav_sec / nav_frames)
    manip_interval = max(1.0 / fps, manip_sec / manip_frames)

    print(f"  VLM 分段: {len(segments)} 段")
    print(f"  navigation:  {nav_sec:.0f}s → {nav_frames} 帧 "
          f"(每 {nav_interval:.1f}s, 占{nav_ratio*100:.0f}%)")
    print(f"  manipulation: {manip_sec:.0f}s → {manip_frames} 帧 "
          f"(每 {manip_interval:.1f}s)")

    # 收集每段抽帧时间点
    frame_times = []
    for seg in segments:
        start_sec = to_seconds(seg["start_time"])
        end_sec = to_seconds(seg["end_time"])
        interval = nav_interval if seg["type"] == "navigation" else manip_interval
        t = start_sec
        while t < end_sec:
            frame_times.append(t)
            t += interval
        if not any(start_sec <= ft < end_sec for ft in frame_times):
            frame_times.append((start_sec + end_sec) / 2)

    frame_times.sort()

    # 从 mp4 逐帧抽取
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        sys.exit(f"无法打开视频: {video_file}")

    idx = 0
    for ft in frame_times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ft * fps))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"{idx:05d}.png"), frame)
            idx += 1
    cap.release()
    print(f"  实际抽取: {idx} 帧 → {output_dir}")
    return idx


# ═══════════════════════════════════════════════════════════════
# Manifest 模式: 按预选帧号列表抽帧
# ═══════════════════════════════════════════════════════════════

def manifest_sample(manifest_file, video_file, output_dir):
    """按 manifest.txt 中预选的帧号列表从视频抽帧。

    manifest.txt 格式:
        keyframes:
          0000: frame=000225, time=7.50s
          0001: frame=000360, time=12.00s
          ...
    """
    import re, cv2

    os.makedirs(output_dir, exist_ok=True)

    # 解析帧号列表 (格式: "NNNN: frame=FFFFFF, time=XX.XXs")
    frames = []
    with open(manifest_file) as f:
        in_keyframes = False
        for line in f:
            line = line.strip()
            if line.startswith("keyframes:"):
                in_keyframes = True
                continue
            if in_keyframes:
                m = re.match(r"^\d{4}:\s*frame=(\d{6})", line)
                if m:
                    frames.append(int(m.group(1)))
                elif line == "" or line.startswith("shots:"):
                    break

    if not frames:
        sys.exit(f"manifest 中未找到 keyframe 条目: {manifest_file}")

    print(f"  manifest 预选: {len(frames)} 帧")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        sys.exit(f"无法打开视频: {video_file}")

    idx = 0
    for fn in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"{idx:05d}.png"), frame)
            idx += 1
    cap.release()
    print(f"  实际抽取: {idx} 帧 → {output_dir}")
    return idx


# ═══════════════════════════════════════════════════════════════
# 鱼眼去畸变 (所有模式共用)
# ═══════════════════════════════════════════════════════════════

def undistort_frames(frame_dir, cam_yaml):
    """对帧目录做鱼眼→pinhole 去畸变, 原地替换。
    cam_yaml: 相机内参 yaml 文件路径 (支持 OPENCV_FISHEYE 模型)"""
    import subprocess
    script = _find_undistort_script()
    if script is None:
        print("  [WARN] 未找到去畸变脚本, 跳过")
        return
    print(f"  鱼眼去畸变: {frame_dir}")
    cmd = [
        sys.executable, script,
        "--cam-dir", frame_dir,
        "--out-dir", frame_dir,
        "--num-frames", "0",
        "--out-size", "1024",
    ]
    if cam_yaml and os.path.exists(cam_yaml):
        cmd += ["--calib", cam_yaml]
    else:
        print("  [WARN] 无相机内参文件, 回退到去畸变脚本内置 cam0 默认值")
        cmd += ["--calib", ""]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] 去畸变失败:\n{r.stderr[-500:]}")
    else:
        n = len([f for f in os.listdir(frame_dir) if f.endswith(".png")])
        print(f"  去畸变完成: {n} 帧")


def _find_undistort_script():
    candidates = [
        os.path.join(os.path.dirname(_project_root), "dynamic_mask", "undistort_center.py"),
        os.path.join(_project_root, "..", "dynamic_mask", "undistort_center.py"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return None

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="关键帧筛选 (Replica / VLM / Manifest)")
    # Replica
    parser.add_argument("--pose_file", default=None, help="traj.txt")
    parser.add_argument("--total_frames", type=int, default=None)
    parser.add_argument("--replica_dir", default=None, help="Replica 场景目录")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--coverage", type=float, default=0.6)
    parser.add_argument("--min_distance", type=float, default=0.05)
    # VLM
    parser.add_argument("--vlm_json", default=None, help="VLM 分段 JSON")
    parser.add_argument("--video_file", default=None, help="mp4 视频")
    parser.add_argument("--nav_ratio", type=float, default=0.9,
                        help="navigation 占比 (默认 0.9)")
    # Manifest
    parser.add_argument("--manifest", default=None,
                        help="预选关键帧列表 manifest.txt (含 keyframes 帧号)")
    # 通用
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_frames", type=int, default=100)
    parser.add_argument("--cam_yaml", default=None,
                        help="鱼眼相机内参 yaml, 如 data/<scene>/cam.yaml (不存在则跳过去畸变)")
    args = parser.parse_args()

    # ---- Manifest 预选帧号模式 ----
    if args.manifest and args.video_file:
        if not os.path.isfile(args.manifest):
            sys.exit(f"未找到: {args.manifest}")
        if not os.path.isfile(args.video_file):
            sys.exit(f"未找到: {args.video_file}")

        manifest_sample(args.manifest, args.video_file, args.output_dir)

        if args.cam_yaml:
            undistort_frames(args.output_dir, args.cam_yaml)
        return

    # ---- VLM 视频模式 ----
    if args.vlm_json and args.video_file:
        if not os.path.isfile(args.vlm_json):
            sys.exit(f"未找到: {args.vlm_json}")
        if not os.path.isfile(args.video_file):
            sys.exit(f"未找到: {args.video_file}")

        vlm_adaptive_sample(args.vlm_json, args.video_file,
                            args.output_dir, args.target_frames, args.nav_ratio)

        if args.cam_yaml:
            undistort_frames(args.output_dir, args.cam_yaml)
        return

    # ---- Replica 位姿模式 ----
    if not args.pose_file or not args.replica_dir:
        sys.exit("需要 --vlm_json+--video_file (VLM) 或 --pose_file+--replica_dir (Replica)")

    results_dir = os.path.join(args.replica_dir, "results")
    if not os.path.isdir(results_dir):
        sys.exit(f"未找到 {results_dir}")

    poses = load_poses(args.pose_file)
    N = min(len(poses), args.total_frames or len(poses))
    poses = poses[:N]
    centers, look_at = extract_camera_info(poses)

    valid_pose_mask = np.linalg.norm(centers, axis=1) > 0.01
    n_valid_poses = int(valid_pose_mask.sum())
    scene_diam = float(np.sqrt(((centers[valid_pose_mask] - centers[valid_pose_mask].mean(axis=0)) ** 2).sum(axis=1)).max()) * 2 if n_valid_poses > 0 else 0
    print(f"全部帧: {N}  有效位姿: {n_valid_poses}  场景直径: {scene_diam:.2f}m")

    original_indices = np.arange(N)

    if n_valid_poses < N * 0.5:
        step = max(1, N // args.target_frames)
        sel = list(range(0, N, step))
        print(f"  有效位姿不足 ({n_valid_poses}/{N}), 退化均匀采样 step={step} → {len(sel)} 帧")
    else:
        MAX_POOL, POOL_SIZE = 5000, 2000
        if N > MAX_POOL:
            step = max(1, N // POOL_SIZE)
            pool_idx = np.arange(0, N, step)
            poses, centers, look_at = poses[pool_idx], centers[pool_idx], look_at[pool_idx]
            original_indices = original_indices[pool_idx]
            print(f"  粗采样: {N} → {len(pool_idx)} 帧 (step={step})")

        print(f"参数: max_frames={args.max_frames} coverage={args.coverage} "
              f"min_dist={args.min_distance}m")

        pool_sel = farthest_point_sampling(centers, args.max_frames, args.coverage, args.min_distance)
        print(f"  位置FPS: {len(pool_sel)} 帧")
        pool_sel = augment_by_angle(pool_sel, look_at, min_angle_deg=40)
        print(f"  视角增强: {len(pool_sel)} 帧")
        sel = original_indices[pool_sel].tolist()

    os.makedirs(args.output_dir, exist_ok=True)
    # 清理旧帧, 避免上一次采样残留 (帧数变少时会留下 stale 帧)
    import glob as _glob
    for _old in _glob.glob(os.path.join(args.output_dir, "[0-9]" * 5 + ".png")):
        os.remove(_old)
    idx = 0
    manifest = []  # (local_stem, original_frame_idx)
    for fn in sel:
        src = os.path.join(results_dir, f"frame{fn:06d}.jpg")
        dst = os.path.join(args.output_dir, f"{idx:05d}.png")
        if os.path.exists(src):
            Image.open(src).save(dst)
            manifest.append((f"{idx:05d}", int(fn)))
            idx += 1
        else:
            print(f"WARNING: 缺失 {src}")

    # 保存 局部帧号 -> 原始 Replica 帧号 映射 (供 GT 深度/位姿回溯)
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(args.output_dir)),
                                 "keyframe_indices.txt")
    with open(manifest_path, "w") as mf:
        mf.write(chr(10).join(f"{stem} frame{ofn:06d}" for stem, ofn in manifest) + chr(10))
    print(f"帧号映射: {manifest_path} ({len(manifest)} 条)")

    gaps = np.diff(np.array(sel))
    print(f"\n入选: {idx} 帧 / {N} 总帧 ({idx/N*100:.1f}%)")
    print(f"帧间隔: min={gaps.min()} median={int(np.median(gaps))} max={gaps.max()}")
    print(f"输出: {args.output_dir}")


if __name__ == "__main__":
    main()
