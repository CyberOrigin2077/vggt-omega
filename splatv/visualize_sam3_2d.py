"""
SAM3 2D 检测可视化：在原图上叠加半透明彩色实例 mask。

用法 (sam3 环境):
    python splatv/visualize_sam3_2d.py \
        --scene_dir examples/lab0 \
        --output output_fross/lab0/vis_sam3_2d \
        --num_frames 5
"""

import argparse, os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def id_to_rgba(uid, alpha=100):
    if uid <= 0: return (60, 60, 70, 0)
    h = (int(uid) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    return (max(int(h & 0xFF), 60),
            max(int((h >> 8) & 0xFF), 60),
            max(int((h >> 16) & 0xFF), 60),
            alpha)


def main():
    parser = argparse.ArgumentParser(description="SAM3 2D mask 可视化")
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output", default="output_vis")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--alpha", type=int, default=80, help="mask 透明度 0-255")
    args = parser.parse_args()

    scene_name = os.path.basename(os.path.abspath(args.scene_dir))
    work_dir = os.path.join(args.scene_dir, "_vggt_work")
    sem_dir = os.path.join(work_dir, "semantic_masks")
    frames_dir = os.path.join(args.scene_dir, "frames")

    if not os.path.isdir(sem_dir):
        sys.exit(f"未找到 {sem_dir}。先跑 run_semantic_pcd.sh")

    # 读帧列表
    with open(os.path.join(work_dir, "frame_list.txt")) as f:
        frame_ids = [l.strip() for l in f if l.strip()]
    frame_ids = frame_ids[:args.num_frames]

    os.makedirs(args.output, exist_ok=True)

    for fid in frame_ids:
        # 原图
        fp = os.path.join(frames_dir, f"{fid}.png")
        if not os.path.exists(fp):
            fp = os.path.join(frames_dir, f"{fid}.jpg")
        if not os.path.exists(fp):
            continue

        orig = Image.open(fp).convert("RGBA")
        ow, oh = orig.size

        # SAM3 mask (VGGT 分辨率)
        sem_path = os.path.join(sem_dir, f"{fid}.npy")
        if not os.path.exists(sem_path):
            continue
        sem = np.load(sem_path)  # [H_vggt, W_vggt]
        H_s, W_s = sem.shape

        # 缩放 mask 到原图分辨率
        sem_img = Image.fromarray(sem.astype(np.int32))
        sem_img = sem_img.resize((ow, oh), Image.NEAREST)
        sem_np = np.array(sem_img)

        # 构建半透明叠加层
        overlay = np.zeros((oh, ow, 4), dtype=np.uint8)
        for iid in np.unique(sem_np):
            if iid <= 0: continue
            r, g, b, a = id_to_rgba(iid, args.alpha)
            overlay[sem_np == iid] = [r, g, b, a]

        overlay_img = Image.fromarray(overlay, "RGBA")

        # 合成: 原图 + 半透明 mask
        result = Image.alpha_composite(orig, overlay_img)

        # 画边框 (只画面积 > min_area 的实例)
        draw = ImageDraw.Draw(result)
        min_area_px = 500  # 原图分辨率下的最小面积
        for iid in np.unique(sem_np):
            if iid <= 0: continue
            ys, xs = np.where(sem_np == iid)
            if len(ys) < min_area_px: continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            r, g, b, _ = id_to_rgba(iid, 255)
            draw.rectangle([x1-1, y1-1, x2+1, y2+1], outline=(r, g, b), width=2)

        out_path = os.path.join(args.output, f"{fid}.png")
        result.convert("RGB").save(out_path)

    n_inst = set()
    for fid in frame_ids:
        sp = os.path.join(sem_dir, f"{fid}.npy")
        if os.path.exists(sp):
            sem = np.load(sp)
            n_inst.update(sem[sem > 0].tolist())
    print(f"{len(frame_ids)} 帧, {len(n_inst)} 个全局实例 → {args.output}/")


if __name__ == "__main__":
    main()
