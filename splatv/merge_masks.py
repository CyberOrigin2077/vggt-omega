"""
将 instance_masks 目录下多实例 mask 合并为每帧单张总 mask。

用法:
    python merge_instance_masks.py \
        --mask_dir examples/factory2/instance_masks \
        --output_dir examples/factory2/obj_mask
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="合并逐实例 mask 为逐帧总 mask")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="输入目录，含 {frame_id}_obj{id}.png 文件")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，产出一帧一张 {frame_id}.png")
    args = parser.parse_args()

    if not os.path.isdir(args.mask_dir):
        sys.exit(f"mask_dir not found: {args.mask_dir}")

    # 按 frame_id 分组
    groups = defaultdict(list)
    pattern = re.compile(r"^(\d+)_obj\d+\.png$")
    for fname in sorted(os.listdir(args.mask_dir)):
        m = pattern.match(fname)
        if m:
            groups[m.group(1)].append(fname)

    if not groups:
        sys.exit("No files matching pattern '{frame_id}_obj{id}.png' found")

    os.makedirs(args.output_dir, exist_ok=True)

    for frame_id, fnames in sorted(groups.items()):
        base = Image.open(os.path.join(args.mask_dir, fnames[0]))
        w, h = base.size
        merged = np.zeros((h, w), dtype=np.uint8)

        for fname in fnames:
            img = np.array(Image.open(os.path.join(args.mask_dir, fname)))
            if img.ndim == 3:
                img = img[:, :, 0]  # 灰度或 RGBA 转单通道
            merged = np.maximum(merged, img)

        out_path = os.path.join(args.output_dir, f"{frame_id}.png")
        Image.fromarray(merged).save(out_path)

    print(f"Merged {len(groups)} frames (from {sum(len(v) for v in groups.values())} masks) → {args.output_dir}")
    print(f"Frames with >1 instance: {sum(1 for v in groups.values() if len(v) > 1)}")


if __name__ == "__main__":
    main()
