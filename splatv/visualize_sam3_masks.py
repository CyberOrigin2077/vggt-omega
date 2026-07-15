#!/usr/bin/env python3
"""
SAM3 掩码 2D 可视化 — 在原始帧上绘制 SAM3 精确分割掩码 + 类别标签。

输入: SAM3 提升后的逐帧数据 (per_frame_sam3/)
输出: 每帧融合图 (半透明掩码 + 标签 + 关系线)

用法 (任意有 numpy + PIL 的环境):
    python splatv/visualize_sam3_masks.py \
        --per_frame_dir Datasets/office2/per_frame_sam3 \
        --class_json Datasets/office2/ReplicaSSG/replica_to_visual_genome.json \
        --output output_fross/office2/vis_sam3 \
        --num_frames 5
"""

import argparse, os, sys, glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def id_to_rgb(uid):
    """golden-ratio 哈希 → (R, G, B)"""
    h = (int(uid) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    return (
        max(int(h & 0xFF), 60),
        max(int((h >> 8) & 0xFF), 60),
        max(int((h >> 16) & 0xFF), 60),
    )


def draw_masks_on_image(image, masks, classes, scores, class_names,
                         bboxes=None, rels=None, rel_classes=None,
                         rel_names=None):
    """
    在图像上用半透明 RGBA 绘制 SAM3 掩码 + 标签。

    Args:
        image: [H, W, 3] uint8 numpy array
        masks: [K, H, W] bool
        classes: [K] int
        scores: [K] float
        class_names: list of str
        bboxes: [K, 4] int (cx,cy,w,h) or None
        rels: [E, 2] int or None
        rel_classes: [E] int or None
        rel_names: list of str or None

    Returns:
        PIL Image (RGBA, 已合成)
    """
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = font_sm = ImageFont.load_default()

    H, W = image.shape[:2]
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    K = len(masks)
    alpha_fill = 90   # 掩码填充透明度
    alpha_edge = 180  # 边缘透明度

    # 绘制掩码
    for i in range(K):
        mask_i = masks[i]
        if not mask_i.any():
            continue

        r, g, b = id_to_rgb(i)
        cls_id = int(classes[i])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
        label = f"{i}:{cls_name} ({scores[i]:.2f})"

        # 半透明填充
        ys, xs = np.where(mask_i)
        if len(ys) == 0:
            continue

        # 只绘制边缘 + 少数采样点, 避免逐像素 PIL 操作太慢
        # 方法: 用轮廓线 + 标签代替全填充
        edge_mask = np.zeros_like(mask_i, dtype=bool)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                shifted = np.roll(mask_i, (dy, dx), axis=(0, 1))
                edge_mask |= (mask_i & ~shifted)
        # 限制边缘点数量
        ey, ex = np.where(edge_mask)
        max_edge = 3000
        if len(ey) > max_edge:
            step = max(1, len(ey) // max_edge)
            ey, ex = ey[::step], ex[::step]

        for y, x in zip(ey, ex):
            draw.point((x, y), fill=(r, g, b, alpha_fill))

        # bbox 框 (用于定位)
        if bboxes is not None and i < len(bboxes):
            cx_b, cy_b, bw, bh = bboxes[i]
            x1 = max(0, cx_b - bw // 2)
            y1 = max(0, cy_b - bh // 2)
            x2 = min(W - 1, cx_b + bw // 2)
            y2 = min(H - 1, cy_b + bh // 2)
            draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, alpha_edge), width=1)

        # 标签
        tw = max(len(label) * 7, 50)
        y_label = max(0, int(ys.min()) - 18)
        draw.rectangle(
            [max(0, xs.min()), y_label,
             min(W, xs.min() + tw), y_label + 16],
            fill=(r, g, b, 200))
        draw.text((xs.min() + 2, y_label), label,
                  fill=(255, 255, 255, 255), font=font_sm)

    # 绘制关系线
    if rels is not None and len(rels) > 0 and bboxes is not None:
        for idx, (s, o) in enumerate(rels):
            if s >= K or o >= K:
                continue
            if not masks[s].any() or not masks[o].any():
                continue
            ys_s, xs_s = np.where(masks[s])
            ys_o, xs_o = np.where(masks[o])
            sx, sy = int(xs_s.mean()), int(ys_s.mean())
            ox, oy = int(xs_o.mean()), int(ys_o.mean())
            rel_name = ""
            if rel_classes is not None and rel_names is not None:
                rc = int(rel_classes[idx]) if idx < len(rel_classes) else 0
                rel_name = rel_names[rc] if rc < len(rel_names) else f"rel{rc}"
            draw.line([(sx, sy), (ox, oy)], fill=(255, 200, 50, 180), width=2)
            mx, my = (sx + ox) // 2, (sy + oy) // 2
            draw.ellipse([mx - 3, my - 3, mx + 3, my + 3], fill=(255, 200, 50, 200))
            draw.text((mx + 5, my - 7), rel_name, fill=(255, 200, 50, 255), font=font_sm)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    return result


def main():
    parser = argparse.ArgumentParser(description="SAM3 掩码 2D 可视化")
    parser.add_argument("--per_frame_dir", required=True)
    parser.add_argument("--class_json", required=True)
    parser.add_argument("--output", default="output_sam3_vis")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载 class mapping
    import json
    with open(args.class_json) as f:
        cmap = json.load(f)
    class_names = cmap.get("VisualGenome_list", [])
    rel_names = cmap.get("VisualGenome_rel", [])

    # 找场景目录
    if args.scene:
        scene_dirs = [os.path.join(args.per_frame_dir, d)
                      for d in [args.scene]
                      if os.path.isdir(os.path.join(args.per_frame_dir, d))]
    else:
        scene_dirs = sorted([
            os.path.join(args.per_frame_dir, d)
            for d in os.listdir(args.per_frame_dir)
            if os.path.isdir(os.path.join(args.per_frame_dir, d))
        ])

    if not scene_dirs:
        print(f"未找到场景目录: {args.per_frame_dir}")
        sys.exit(1)

    total = 0
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        frame_files = sorted(glob.glob(os.path.join(scene_dir, "frame_*.npz")))
        if not frame_files:
            continue

        print(f"[{scene_name}] {len(frame_files)} 帧, 取 {args.num_frames} 帧")

        # 均匀采样帧
        indices = np.linspace(0, len(frame_files) - 1, min(args.num_frames, len(frame_files)),
                              dtype=int)

        for idx in indices:
            fp = frame_files[idx]
            data = np.load(fp, allow_pickle=True)

            if "sam_masks" not in data:
                print(f"  跳过 {os.path.basename(fp)}: 无 SAM3 掩码")
                continue

            image = data["image"]
            sam_masks = data["sam_masks"]
            sam_scores = data["sam_scores"]
            classes = data["classes"]
            bboxes = data["bboxes"]
            rels = data["rels"] if "rels" in data else None
            rel_classes = data["rel_classes"] if "rel_classes" in data else None

            if image.ndim != 3 or image.shape[2] != 3:
                continue

            # 只保留有效的掩码
            valid_k = [k for k in range(len(sam_masks))
                       if k < len(sam_masks) and sam_masks[k].any()]
            if not valid_k:
                continue

            masks_v = sam_masks[valid_k]
            scores_v = sam_scores[valid_k]
            classes_v = classes[valid_k]
            bboxes_v = bboxes[valid_k] if len(bboxes) > 0 else None

            result = draw_masks_on_image(
                image, masks_v, classes_v, scores_v, class_names,
                bboxes=bboxes_v, rels=rels, rel_classes=rel_classes,
                rel_names=rel_names)

            out_path = os.path.join(
                args.output,
                f"{scene_name}_frame{os.path.basename(fp).replace('.npz', '')}.png")
            result.save(out_path)
            total += 1

            n_obj = len(valid_k)
            cls_list = [class_names[int(classes_v[i])]
                        if int(classes_v[i]) < len(class_names) else "?"
                        for i in range(min(n_obj, 5))]
            print(f"  → {os.path.basename(out_path)} ({n_obj} 物体: {', '.join(cls_list)}...)")

    print(f"\n完成: {total} 张可视化图像 → {args.output}/")


if __name__ == "__main__":
    main()
