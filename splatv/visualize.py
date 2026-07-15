"""
将语义 mask 可视化为彩色图片，方便检查跨帧实例 ID 一致性。

用法:
    python visualize_semantic.py \
        --semantic_dir examples/room0/semantic_masks \
        --frames_dir examples/room0/frames \
        --output_dir examples/room0/semantic_vis
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def id_to_color(obj_id, bg_color=(60, 60, 70)):
    """golden-ratio 哈希 → RGB，同一 ID 始终同一颜色。"""
    if obj_id == 0:
        return bg_color
    h = (int(obj_id) * 2654435761) & 0xFFFFFFFF
    r = (h & 0xFF)
    g = ((h >> 8) & 0xFF)
    b = ((h >> 16) & 0xFF)
    # 调亮，避免太暗
    r = max(r, 40)
    g = max(g, 40)
    b = max(b, 40)
    return (r, g, b)


def main():
    parser = argparse.ArgumentParser(description="语义 mask 可视化")
    parser.add_argument("--semantic_dir", type=str, required=True,
                        help="语义 mask 目录（uint16 PNG）")
    parser.add_argument("--frames_dir", type=str, default=None,
                        help="原始帧目录，用于叠加显示（可选）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认: {semantic_dir}/../semantic_vis）")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="叠加透明度（0=纯原图, 1=纯语义色）")
    args = parser.parse_args()

    if not os.path.isdir(args.semantic_dir):
        sys.exit(f"semantic_dir not found: {args.semantic_dir}")

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.semantic_dir), "semantic_vis"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # 收集 mask 文件
    mask_files = sorted(
        f for f in os.listdir(args.semantic_dir) if f.endswith(".png")
    )
    if not mask_files:
        sys.exit("No mask files found")

    # 统计全局 ID
    all_ids = set()
    for fname in mask_files:
        sem = np.array(Image.open(os.path.join(args.semantic_dir, fname)))
        all_ids.update(np.unique(sem).tolist())
    all_ids.discard(0)
    print(f"{len(mask_files)} 帧, {len(all_ids)} 个全局实例")

    # 逐帧着色
    for fname in mask_files:
        frame_id = os.path.splitext(fname)[0]
        sem_path = os.path.join(args.semantic_dir, fname)
        sem = np.array(Image.open(sem_path)).astype(np.int32)
        H, W = sem.shape

        # 构建彩色语义图
        color = np.zeros((H, W, 3), dtype=np.uint8)
        ids_in_frame = set(np.unique(sem)) - {0}
        for obj_id in ids_in_frame:
            r, g, b = id_to_color(obj_id)
            mask = sem == obj_id
            color[mask] = (r, g, b)
        # 背景色
        bg = id_to_color(0)
        color[sem == 0] = bg

        if args.frames_dir and os.path.isdir(args.frames_dir):
            # 叠加原图
            frame_path = os.path.join(args.frames_dir, f"{frame_id}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(args.frames_dir, f"{frame_id}.jpg")
            if os.path.exists(frame_path):
                orig = np.array(Image.open(frame_path).convert("RGB").resize((W, H)))
                blended = (color * args.alpha + orig * (1 - args.alpha)).astype(np.uint8)
                Image.fromarray(blended).save(
                    os.path.join(args.output_dir, f"{frame_id}.png")
                )
                continue

        Image.fromarray(color).save(
            os.path.join(args.output_dir, f"{frame_id}.png")
        )

    print(f"完成: {args.output_dir}")


if __name__ == "__main__":
    main()
