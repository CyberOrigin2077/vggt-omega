#!/usr/bin/env python3
"""
DeWorldSG SAM3 掩码提取 + 深度感知 3D 高斯提升。

用法 (SAM3 环境):
    /data/users/xjl/envs/sam3/bin/python splatv/sam3_lift.py \
        --per_frame_dir Datasets/office2/per_frame \
        --output_dir Datasets/office2/per_frame_sam3
"""

import argparse, os, sys, time, glob

import numpy as np
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from splatv.depth_aware_gaussian import (
    dual_domain_refinement,
    estimate_3d_gaussian,
)


def build_sam3_model(device="cuda:0", checkpoint_path=None):
    """加载 SAM3 图像模型, 返回 (model, processor)。"""
    import torch as _torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # SAM3 全局 autocast 与某些 CUDA 内核不兼容, 需关闭
    _torch.set_autocast_enabled(False)

    print(f"加载 SAM3 图像模型 → {device} ...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=True,
        enable_inst_interactivity=True,
        load_from_HF=False,
    )
    # 强制参数迁移到 GPU (build_sam3_image_model 的 cuda() 有时不完整)
    model = model.to(device)
    for param in model.parameters():
        if not param.is_cuda:
            param.data = param.data.to(device)

    processor = Sam3Processor(model, device=device)
    return model, processor


def extract_sam3_masks(model, processor, image, boxes_xyxy):
    """
    用 SAM3 box prompt 批量提取实例掩码。

    Args:
        model: Sam3Image (含 inst_interactive_predictor)
        processor: Sam3Processor
        image: [H, W, 3] uint8 numpy array
        boxes_xyxy: [K, 4] (x1, y1, x2, y2) int

    Returns:
        masks: [K, H, W] bool
        scores: [K] float
    """
    from PIL import Image as PILImage

    K = len(boxes_xyxy)
    H, W = image.shape[:2]

    if K == 0:
        return np.zeros((0, H, W), dtype=bool), np.zeros(0, dtype=np.float32)

    # Sam3Processor 对 PIL image 维度解析正确，numpy 会有 bug
    pil_image = PILImage.fromarray(image)

    state = processor.set_image(pil_image)

    all_masks = []
    all_scores = []

    for box in boxes_xyxy:
        masks, scores, logits = model.predict_inst(
            state,
            box=box.astype(np.float32),
            multimask_output=False,
        )
        # masks[0] is float32 [H, W], threshold at 0.5 → bool
        all_masks.append(masks[0] > 0.5)
        all_scores.append(float(scores[0]))

    masks = np.stack(all_masks, axis=0).astype(bool)
    scores = np.array(all_scores, dtype=np.float32)

    processor.reset_all_prompts(state)
    del state
    torch.cuda.empty_cache()

    return masks, scores


def resolve_ambiguous_pixels(masks, min_visible=10):
    """
    处理 SAM3 掩码之间的重叠像素。

    如果像素被 >1 个掩码覆盖 → 歧义像素 → 从所有相关掩码中丢弃。
    """
    if len(masks) == 0:
        return masks

    K, H, W = masks.shape
    clean_masks = masks.copy()

    coverage = masks.sum(axis=0)
    ambiguous = coverage > 1
    if ambiguous.any():
        for k in range(K):
            clean_masks[k][ambiguous] = False

    for k in range(K):
        if clean_masks[k].sum() < min_visible:
            clean_masks[k] = False

    return clean_masks


def process_frame(frame_data, model, processor, dr_params=None):
    """
    处理一帧: SAM3掩码 + DR + 3D高斯。

    Returns:
        dict with means_3d, covs_3d, pcds_list, valid_mask, obs_counts,
             classes, class_probs, bboxes, rels, rel_classes
    """
    if dr_params is None:
        dr_params = {}

    classes = frame_data["classes"]
    bboxes = frame_data["bboxes"]
    class_probs = frame_data["class_probs"]
    image = frame_data["image"]
    depth = frame_data["depth"]
    fx, fy = float(frame_data["fx"]), float(frame_data["fy"])
    cx, cy = float(frame_data["cx"]), float(frame_data["cy"])
    R = frame_data["camera_rot"]
    t = frame_data["camera_trans"]

    K = len(classes)
    H, W = depth.shape

    # bbox (cx,cy,w,h) → (x1,y1,x2,y2)
    boxes_xyxy = np.zeros((K, 4), dtype=np.float32)
    for k in range(K):
        cx_b, cy_b, bw, bh = bboxes[k]
        boxes_xyxy[k, 0] = max(0, cx_b - bw // 2)
        boxes_xyxy[k, 1] = max(0, cy_b - bh // 2)
        boxes_xyxy[k, 2] = min(W - 1, cx_b + bw // 2)
        boxes_xyxy[k, 3] = min(H - 1, cy_b + bh // 2)

    # SAM3 提取掩码
    sam_masks, sam_scores = extract_sam3_masks(model, processor, image, boxes_xyxy)
    sam_masks = resolve_ambiguous_pixels(sam_masks)

    # ---- 动态物体过滤 ----
    # 双重检查: (1) SAM3 mask 与动态区域重叠 > 30%  (2) bbox 中心落在动态区域
    dyn_mask = frame_data.get("dynamic_mask", None)
    is_dynamic = np.zeros(K, dtype=bool)
    if dyn_mask is not None and dyn_mask.ndim == 2:
        dynamic_region = (dyn_mask < 0.5)
        if dynamic_region.any():
            for k in range(K):
                # 检查1: SAM3 mask 重叠率
                if k < len(sam_masks) and sam_masks[k].any():
                    inst_area = sam_masks[k].sum()
                    overlap = (sam_masks[k] & dynamic_region).sum()
                    if overlap / max(inst_area, 1) > 0.3:
                        is_dynamic[k] = True
                        continue
                # 检查2: bbox 中心点
                cx_b, cy_b, _, _ = bboxes[k]
                cx_c = int(np.clip(cx_b, 0, dynamic_region.shape[1] - 1))
                cy_c = int(np.clip(cy_b, 0, dynamic_region.shape[0] - 1))
                if dynamic_region[cy_c, cx_c]:
                    is_dynamic[k] = True

    # DR + 3D 高斯
    means_3d = np.zeros((K, 3), dtype=np.float32)
    covs_3d = np.tile(np.eye(3, dtype=np.float32) * 0.01, (K, 1, 1))
    pcds_list = [np.zeros((0, 3), dtype=np.float32)] * K
    valid_mask = np.zeros(K, dtype=bool)
    obs_counts = np.zeros(K, dtype=np.int32)

    for k in range(K):
        if is_dynamic[k]:
            continue  # 跳过动态物体

        if k < len(sam_masks):
            init_mask = sam_masks[k] & (depth > 1e-6)
        else:
            # 回退到 bbox 掩码
            cx_b, cy_b, bw, bh = bboxes[k]
            x1 = max(0, int(cx_b - bw // 2))
            y1 = max(0, int(cy_b - bh // 2))
            x2 = min(W, int(cx_b + bw // 2))
            y2 = min(H, int(cy_b + bh // 2))
            init_mask = np.zeros((H, W), dtype=bool)
            if x2 > x1 and y2 > y1:
                init_mask[y1:y2, x1:x2] = True
            init_mask = init_mask & (depth > 1e-6)

        if init_mask.sum() < 10:
            continue

        refined_mask, ok = dual_domain_refinement(
            depth, init_mask, min_valid_pixels=10, **dr_params)
        if not ok:
            refined_mask = init_mask

        mean_3d, cov_3d, pts_3d, success = estimate_3d_gaussian(
            depth, refined_mask, fx, fy, cx, cy, R, t, min_points=10)

        if success:
            means_3d[k] = mean_3d
            covs_3d[k] = cov_3d
            pcds_list[k] = pts_3d
            valid_mask[k] = True
            obs_counts[k] = len(pts_3d)

    # 过滤涉及动态物体的关系 (dynamic ↔ static 无意义)
    rels_raw = frame_data["rels"]
    rel_classes_raw = frame_data["rel_classes"]
    if len(rels_raw) > 0 and is_dynamic.any():
        valid_rel = ~(is_dynamic[rels_raw[:, 0]] | is_dynamic[rels_raw[:, 1]])
        rels_raw = rels_raw[valid_rel]
        rel_classes_raw = rel_classes_raw[valid_rel]

    # 将点云列表扁平化为 offset 格式 (兼容 numpy 1.x)
    all_pts = []
    offsets = [0]
    for pcd in pcds_list:
        pts = pcd if isinstance(pcd, np.ndarray) else np.array(pcd)
        if len(pts) == 0 or pts.ndim != 2 or pts.shape[1] != 3:
            pts = np.zeros((0, 3), dtype=np.float32)
        all_pts.append(pts.astype(np.float32))
        offsets.append(offsets[-1] + len(pts))
    pcds_flat = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float32)
    pcds_offsets = np.array(offsets, dtype=np.int32)

    return {
        "means_3d": means_3d,
        "covs_3d": covs_3d,
        "pcds_flat": pcds_flat,
        "pcds_offsets": pcds_offsets,
        "valid_mask": valid_mask,
        "obs_counts": obs_counts,
        "classes": classes,
        "class_probs": class_probs,
        "bboxes": bboxes,
        "rels": rels_raw,
        "rel_classes": rel_classes_raw,
        # SAM3 原始掩码
        "sam_masks": sam_masks,
        "sam_scores": sam_scores,
        "image": frame_data["image"],
        # 相机参数 (透传)
        "camera_rot": frame_data["camera_rot"],
        "camera_trans": frame_data["camera_trans"],
        "fx": frame_data["fx"],
        "fy": frame_data["fy"],
        "cx": frame_data["cx"],
        "cy": frame_data["cy"],
    }


def main():
    parser = argparse.ArgumentParser(description="DeWorldSG SAM3 掩码提升")
    parser.add_argument("--per_frame_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--gamma_base", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.02)
    args = parser.parse_args()

    dr_params = dict(tau=args.tau, eps=args.eps,
                     gamma_base=args.gamma_base,
                     alpha=args.alpha, beta=args.beta)

    per_frame_root = args.per_frame_dir
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted([
            d for d in os.listdir(per_frame_root)
            if os.path.isdir(os.path.join(per_frame_root, d))
        ])

    if not scenes:
        print(f"错误: 未在 {per_frame_root} 中找到场景目录")
        sys.exit(1)

    print(f"场景列表: {scenes}")
    print(f"DR 参数: {dr_params}")

    # 自动查找 checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = [
            os.path.join(_project_root, "third_party", "FROSS", "..", "..",
                         "dynamic_mask", "sam3", "checkpoints", "sam3.pt"),
            os.path.join(os.path.dirname(_project_root), "dynamic_mask",
                         "sam3", "checkpoints", "sam3.pt"),
            "/data/users/xjl/beifen/dynamic_mask/sam3/checkpoints/sam3.pt",
        ]
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                checkpoint = c
                break
        if checkpoint:
            print(f"使用本地 checkpoint: {checkpoint}")
        else:
            print("未找到本地 checkpoint, 尝试从 HuggingFace 下载...")

    model, processor = build_sam3_model(args.device, checkpoint_path=checkpoint)
    print("SAM3 加载完成\n")

    total_frames = 0
    total_objects = 0
    t_start = time.time()

    for scene in scenes:
        scene_dir = os.path.join(per_frame_root, scene)
        out_dir = os.path.join(args.output_dir, scene)
        os.makedirs(out_dir, exist_ok=True)

        frame_files = sorted(glob.glob(os.path.join(scene_dir, "frame_*.npz")))
        if not frame_files:
            print(f"  [{scene}] 未找到帧文件, 跳过")
            continue

        print(f"\n[{scene}] {len(frame_files)} 帧")

        for fi, fp in enumerate(frame_files):
            frame_data = np.load(fp, allow_pickle=True)

            if len(frame_data["classes"]) == 0:
                continue

            img_data = frame_data["image"]
            if img_data.ndim != 3 or img_data.shape[2] != 3:
                continue

            result = process_frame(frame_data, model, processor, dr_params)

            out_file = os.path.join(out_dir, os.path.basename(fp))
            np.savez_compressed(out_file, **result)
            total_frames += 1
            total_objects += int(result["valid_mask"].sum())

            if (fi + 1) % 10 == 0:
                n_valid = int(result["valid_mask"].sum())
                print(f"  [{scene}] {fi+1}/{len(frame_files)} "
                      f"({n_valid}/{len(result['classes'])} 有效物体)")

    elapsed = time.time() - t_start
    print(f"\n完成: {total_frames} 帧, {total_objects} 个有效物体")
    print(f"耗时: {elapsed:.1f}s ({elapsed/max(total_frames,1):.2f}s/帧)")
    print(f"输出: {args.output_dir}/")


if __name__ == "__main__":
    main()
