"""
Replica RGB-D 场景 → examples 格式 (只取 RGB 帧, VGGT 会自己算深度)。

用法:
    python splatv/replica_to_examples.py \
        --replica_dir Datasets/Replica \
        --scene office2 \
        --start 0 --end 600 --step 10 \
        --output examples/office2
"""

import argparse, os, sys
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Replica → examples 格式")
    parser.add_argument("--replica_dir", default="Datasets/Replica")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--start", type=int, default=0, help="起始帧")
    parser.add_argument("--end", type=int, default=600, help="结束帧 (不含)")
    parser.add_argument("--step", type=int, default=10, help="采样间隔")
    parser.add_argument("--output", default=None, help="输出目录, 默认 examples/<scene>")
    args = parser.parse_args()

    results_dir = os.path.join(args.replica_dir, args.scene, "results")
    if not os.path.isdir(results_dir):
        sys.exit(f"未找到 {results_dir}")

    if args.output is None:
        args.output = os.path.join("examples", args.scene)

    frames_dir = os.path.join(args.output, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    idx = 0
    for frame_num in range(args.start, args.end, args.step):
        src = os.path.join(results_dir, f"frame{frame_num:06d}.jpg")
        if not os.path.exists(src):
            print(f"WARNING: 跳过缺失帧 frame{frame_num:06d}.jpg")
            continue
        dst = os.path.join(frames_dir, f"{idx:05d}.png")
        Image.open(src).save(dst)
        idx += 1

    print(f"完成: {idx} 帧 → {frames_dir}/")


if __name__ == "__main__":
    main()
