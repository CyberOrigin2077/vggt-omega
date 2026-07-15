"""
用 SAM3 对场景帧做逐帧实例分割，再用跨帧 IoU 匹配关联全局一致实例 ID。

流程:
  1. SAM3 逐帧检测 → 每帧 N 个实例 mask
  2. 相邻帧 IoU 计算 → 匈牙利匹配 → 传播全局 ID
  3. 输出 {output_dir}/{frame_id}.png  uint16 PNG

用法:
    python infer_semantic.py \
        --scene_dir examples/room0 \
        --output_dir examples/room0/semantic_masks
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
try:
    from scipy.optimize import linear_sum_assignment
    def _match_fn(cost):
        return linear_sum_assignment(cost)
except ImportError:
    def _match_fn(cost):
        """贪婪匹配（scipy 不可用时的回退方案）。"""
        N_p, N_c = cost.shape
        row_ind, col_ind = [], []
        used_col = set()
        order = np.argsort(cost.ravel())
        for idx in order:
            r, c = idx // N_c, idx % N_c
            if r not in set(row_ind) and c not in used_col:
                row_ind.append(r)
                col_ind.append(c)
                used_col.add(c)
                if len(row_ind) == min(N_p, N_c):
                    break
        return np.array(row_ind), np.array(col_ind)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sam3_root = os.path.join(_project_root, "third_party", "sam3")
if _sam3_root not in sys.path:
    sys.path.insert(0, _sam3_root)

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------------------------------------------------------------------------
#  跨帧实例匹配
# ---------------------------------------------------------------------------

def match_instances_across_frames(all_masks, all_scores, iou_thresh=0.05):
    """邻帧 IoU + 匈牙利匹配，传播全局实例 ID。

    Args:
        all_masks:  list of [N_i, H, W] bool tensors (每帧的实例 mask)
        all_scores: list of [N_i] float tensors
        iou_thresh: IoU 低于此值的匹配对视为不同实例

    Returns:
        remapped_maps: list of [H, W] int32 tensors (全局一致 ID)
    """
    S = len(all_masks)
    if S == 0:
        return []

    # 第一帧：建立初始全局 ID（从 1 开始）
    N0 = all_masks[0].shape[0]
    global_id_next = N0 + 1
    frame_mappings = [{i: i + 1 for i in range(N0)}]  # local_id → global_id

    for s in range(1, S):
        prev_masks = all_masks[s - 1]  # [N_p, H, W]
        curr_masks = all_masks[s]      # [N_c, H, W]
        N_p, N_c = int(prev_masks.shape[0]), int(curr_masks.shape[0])

        mapping = {}  # curr local_id → global_id

        if N_p > 0 and N_c > 0:
            # 在较小分辨率上计算 IoU（加速）
            scale_h = min(128, prev_masks.shape[1])
            scale_w = min(128, prev_masks.shape[2])
            prev_small = torch.nn.functional.interpolate(
                prev_masks.unsqueeze(1).float(),
                size=(scale_h, scale_w), mode="bilinear",
            ).squeeze(1) > 0.5  # [N_p, h, w]
            curr_small = torch.nn.functional.interpolate(
                curr_masks.unsqueeze(1).float(),
                size=(scale_h, scale_w), mode="bilinear",
            ).squeeze(1) > 0.5  # [N_c, h, w]

            prev_flat = prev_small.reshape(N_p, -1).float()
            curr_flat = curr_small.reshape(N_c, -1).float()
            inter = prev_flat @ curr_flat.T  # [N_p, N_c]
            p_area = prev_flat.sum(dim=1, keepdim=True).clamp(min=1)  # [N_p, 1]
            c_area = curr_flat.sum(dim=1).unsqueeze(0).clamp(min=1)   # [1, N_c]
            iou = inter / (p_area + c_area - inter + 1e-8)  # [N_p, N_c]

            cost = 1.0 - iou.numpy()
            row_ind, col_ind = _match_fn(cost)

            matched_curr = set()
            for pi, ci in zip(row_ind, col_ind):
                if iou[pi, ci] > iou_thresh:
                    prev_local = pi
                    curr_local = ci
                    global_id = frame_mappings[s - 1].get(prev_local, None)
                    if global_id is not None:
                        mapping[curr_local] = global_id
                        matched_curr.add(curr_local)

        # 未匹配的当前帧实例 → 新全局 ID
        for local_id in range(N_c):
            if local_id not in mapping:
                mapping[local_id] = global_id_next
                global_id_next += 1

        frame_mappings.append(mapping)

    # 构建重映射后的语义图
    remapped_maps = []
    for s in range(S):
        N_i = all_masks[s].shape[0]
        H, W = all_masks[s].shape[-2:]
        sem = torch.zeros(H, W, dtype=torch.int32)
        mapping = frame_mappings[s]
        for local_id in range(N_i):
            global_id = mapping[local_id]
            sem = torch.where(all_masks[s][local_id].bool(), global_id, sem)
        remapped_maps.append(sem)

    return remapped_maps


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 语义分割（跨帧匹配）")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景目录，需包含 frames/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认: {scene_dir}/semantic_masks）")
    parser.add_argument("--prompt", type=str, default="object",
                        help="文本提示词")
    parser.add_argument("--confidence_threshold", type=float, default=0.1,
                        help="SAM3 检测置信度阈值")
    parser.add_argument("--iou_thresh", type=float, default=0.05,
                        help="跨帧匹配 IoU 阈值")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--sam3_ckpt", type=str,
                        default=os.path.join(_sam3_root, "checkpoints", "sam3.pt"))
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已有 mask 文件")
    args = parser.parse_args()

    # ---- 路径 ----
    frames_dir = os.path.join(args.scene_dir, "frames")
    if not os.path.isdir(frames_dir):
        sys.exit(f"frames dir not found: {frames_dir}")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.scene_dir, "semantic_masks")
    os.makedirs(args.output_dir, exist_ok=True)

    frame_paths = []
    for ext in (".png", ".jpg", ".jpeg"):
        frame_paths.extend(
            os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
            if f.lower().endswith(ext)
        )
    frame_paths = sorted(set(frame_paths))
    if args.frame_limit:
        frame_paths = frame_paths[:args.frame_limit]
    S = len(frame_paths)
    if S == 0:
        sys.exit("No frames found")
    frame_ids = [os.path.splitext(os.path.basename(fp))[0] for fp in frame_paths]
    print(f"处理 {S} 帧, prompt='{args.prompt}', thresh={args.confidence_threshold}, "
          f"iou_thresh={args.iou_thresh}")

    # ---- 加载模型 ----
    print(f"加载 SAM3 到 {args.device} ...")
    device = torch.device(args.device)
    model = build_sam3_image_model(
        device=device, eval_mode=True,
        enable_inst_interactivity=False,
        checkpoint_path=args.sam3_ckpt, load_from_HF=False,
    )
    processor = Sam3Processor(
        model, resolution=1008, device=device,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"  显存: {torch.cuda.max_memory_allocated(device) / 1024**3:.1f} GB")

    # ---- 阶段 1: 逐帧检测 ----
    torch.cuda.reset_peak_memory_stats(device)
    prompt = args.prompt.strip()
    all_masks = []
    all_scores = []

    t0 = time.time()
    for idx, fp in enumerate(frame_paths):
        pil_img = Image.open(fp).convert("RGB")
        state = processor.set_image(pil_img)
        state = processor.set_text_prompt(prompt, state)

        masks = state["masks"].squeeze(1).cpu()   # [N, H, W] bool
        scores = state["scores"].cpu()             # [N]

        all_masks.append(masks)
        all_scores.append(scores)

        if idx % 15 == 0 or idx == S - 1:
            print(f"  检测 [{idx+1}/{S}] — 共 {masks.shape[0]} 实例")
    t_detect = time.time() - t0

    # ---- 阶段 2: 跨帧匹配 ----
    print(f"\n跨帧匹配 (IoU > {args.iou_thresh}) ...")
    t_m = time.time()
    remapped_maps = match_instances_across_frames(
        all_masks, all_scores, iou_thresh=args.iou_thresh,
    )
    t_match = time.time() - t_m

    # 统计全局 ID
    all_ids = set()
    for sem in remapped_maps:
        all_ids.update(sem.unique().tolist())
    all_ids.discard(0)
    print(f"  全局唯一实例 ID: {len(all_ids)}")

    # ---- 阶段 3: 保存 ----
    print("\n保存...")
    saved = 0
    for s in range(S):
        out_path = os.path.join(args.output_dir, f"{frame_ids[s]}.png")
        if args.skip_existing and os.path.exists(out_path):
            continue
        sem = remapped_maps[s]
        img = Image.fromarray(sem.numpy().astype(np.uint16))
        img.save(out_path)
        saved += 1

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"\n完成: {args.output_dir}")
    print(f"  帧数: {saved}, 全局唯一实例: {len(all_ids)}")
    print(f"  检测: {t_detect:.1f}s | 匹配: {t_match:.1f}s")
    print(f"  峰值显存: {peak_gb:.1f} GB")


if __name__ == "__main__":
    main()
