"""
FROSS 2D 检测可视化：在原始帧上画出物体框 + 关系连线。

用法 (fross 环境):
    python splatv/visualize_fross_2d.py \
        --scene_dir examples/lab0 \
        --artifact_path third_party/FROSS/weights/RT-DETR-EGTR/VG/.../version_0/ \
        --class_json Datasets/lab0/ReplicaSSG/replica_to_visual_genome.json \
        --output output_fross/lab0/vis_2d \
        --num_frames 5
"""

import argparse, os, sys, glob as _glob

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "third_party", "FROSS", "EGTR"))

import json, time
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from model.rtdetr.feature_extractor import RtDetrFeatureExtractor
from model.rtdetr.rtdetr import RtDetrConfig
from model.rtdetr_egtr import RtDetrForSceneGraphGeneration
from lib.pytorch_misc import argsort_desc


def id_to_rgb(uid, alpha=1.0):
    h = (int(uid) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    return (max(int(h & 0xFF), 60),
            max(int((h >> 8) & 0xFF), 60),
            max(int((h >> 16) & 0xFF), 60))


def draw_scene_graph(image, classes, bboxes, scores, rels, rel_classes,
                     class_names, rel_names, filepath, alpha=70):
    """在 PIL 图像上画半透明物体框 + 关系箭头 (RGBA 分层合成)"""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = font_sm = ImageFont.load_default()

    # 原图转 RGBA
    base = image.convert("RGBA")
    ov_w, ov_h = base.size
    overlay = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    N = len(classes)

    # 画半透明填充 + 边框
    for i in range(N):
        bbox = bboxes[i]
        x1 = max(0, int(bbox[0] - bbox[2] // 2))
        y1 = max(0, int(bbox[1] - bbox[3] // 2))
        x2 = min(ov_w - 1, int(bbox[0] + bbox[2] // 2))
        y2 = min(ov_h - 1, int(bbox[1] + bbox[3] // 2))

        r, g, b = id_to_rgb(i)
        cls_id = int(classes[i])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
        label = f"{i}:{cls_name} ({scores[i]:.2f})"

        # 半透明填充
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, alpha), outline=(r, g, b, 200), width=2)

        # 标签: 不透明背景
        tw = max(len(label) * 8, 40)
        draw.rectangle([x1, y1 - 18, x1 + tw, y1], fill=(r, g, b, 220))
        draw.text((x1 + 2, y1 - 16), label, fill=(255, 255, 255, 255), font=font_sm)

    # 关系线 + 标签
    if rels is not None and len(rels) > 0:
        for idx, (s, o) in enumerate(rels):
            if s >= N or o >= N: continue
            rcls = rel_classes[idx]
            rel_name = rel_names[rcls] if rcls < len(rel_names) else f"rel{rcls}"
            sx, sy = int(bboxes[s][0]), int(bboxes[s][1])
            ox, oy = int(bboxes[o][0]), int(bboxes[o][1])
            # 箭头线
            draw.line([(sx, sy), (ox, oy)], fill=(255, 200, 50, 180), width=2)
            # 中点
            mx, my = (sx + ox) // 2, (sy + oy) // 2
            draw.ellipse([mx - 3, my - 3, mx + 3, my + 3], fill=(255, 200, 50, 200))
            draw.text((mx + 5, my - 7), rel_name, fill=(255, 200, 50, 255), font=font_sm)

    # 合成
    result = Image.alpha_composite(base, overlay)
    result.convert("RGB").save(filepath)
    print(f"  → {filepath}")


def main():
    parser = argparse.ArgumentParser(description="FROSS 2D 检测可视化")
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--artifact_path", required=True)
    parser.add_argument("--class_json", required=True)
    parser.add_argument("--output", default="output_fross")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--obj_thresh", type=float, default=0.5)
    parser.add_argument("--rel_topk", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载 class mapping
    with open(args.class_json) as f:
        cmap = json.load(f)
    class_names = cmap.get("VisualGenome_list", [])
    rel_names = cmap.get("VisualGenome_rel", [])

    # 加载模型
    device = torch.device(args.device)
    print(f"加载 RT-DETR-EGTR → {device}")
    feature_extractor = RtDetrFeatureExtractor.from_pretrained(
        "SenseTime/deformable-detr", size=800, max_size=1333)
    config = RtDetrConfig.from_pretrained(args.artifact_path)
    config.logit_adjustment = False
    config.deploy = True
    config.obj_det_engine_path = os.path.join(args.artifact_path, "rt-detr.engine")
    config.egtr_head_engine_path = os.path.join(args.artifact_path, "egtr-head.engine")
    model = RtDetrForSceneGraphGeneration(config).to(device).eval()

    # warmup (跳过——避免显存不足)
    print("跳过预热 (显存限制)")

    # 读帧
    frames_dir = os.path.join(args.scene_dir, "frames")
    frame_paths = sorted(_glob.glob(os.path.join(frames_dir, "*.png")) +
                         _glob.glob(os.path.join(frames_dir, "*.jpg")))
    frame_paths = frame_paths[:args.num_frames]
    print(f"处理 {len(frame_paths)} 帧")

    for idx, fp in enumerate(frame_paths):
        print(f"\n[{idx+1}/{len(frame_paths)}] {os.path.basename(fp)}")

        # 加载原图
        pil_img = Image.open(fp).convert("RGB")
        ori_w, ori_h = pil_img.size

        # 预处理 (和 SG_Predictor 一致)
        img_t = torch.tensor(np.array(pil_img)).permute(2, 0, 1).to(device)
        img_t = torchvision.transforms.functional.resize(img_t, (640, 640))
        img_t = torchvision.transforms.functional.normalize(
            img_t / 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_t = img_t.unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            # 物体检测
            obj_out = model.forward_obj_det(img_t)
            pred_logits = obj_out["logits"][0]  # [N, 151]
            obj_scores, pred_classes = torch.max(pred_logits.softmax(-1), -1)
            keep = obj_scores > args.obj_thresh
            pred_class_np = pred_classes[keep].cpu().numpy()
            pred_logits_np = pred_logits[keep].softmax(-1).cpu().numpy()
            pred_boxes = obj_out["pred_boxes"][0][keep]
            pred_boxes[:, ::2] = pred_boxes[:, ::2] * ori_w
            pred_boxes[:, 1::2] = pred_boxes[:, 1::2] * ori_h
            pred_boxes_np = pred_boxes.cpu().numpy().astype(int)
            obj_scores_np = obj_scores

            # 关系提取
            if keep.sum() > 1:
                rel_out = model(obj_det_output=obj_out)
                keep_scores = obj_scores[keep]
                sub_ob_scores = torch.outer(keep_scores, keep_scores)
                sub_ob_scores[torch.arange(len(keep_scores)), torch.arange(len(keep_scores))] = 0
                pred_rel = torch.clamp(rel_out["pred_rel"][0][keep][:, keep], 0, 1)
                if "pred_connectivity" in rel_out:
                    pred_conn = torch.clamp(rel_out["pred_connectivity"][0][keep][:, keep], 0, 1)
                    pred_rel = torch.mul(pred_rel, pred_conn)
                triplet = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)
                rel_inds = argsort_desc(triplet.cpu().clone().numpy())[:args.rel_topk, :]
                rel_classes_np = torch.argmax(pred_rel[rel_inds[:, 0], rel_inds[:, 1]], -1).cpu().numpy()
                rels_np = rel_inds
            else:
                rels_np = np.zeros((0, 2), dtype=int)
                rel_classes_np = np.zeros(0, dtype=int)

        dt = time.time() - t0

        N_obj = len(pred_class_np)
        print(f"  物体={N_obj}, 关系={len(rels_np)}, 耗时={dt*1000:.0f}ms")
        for i in range(min(N_obj, 5)):
            cls_n = class_names[int(pred_class_np[i])] if int(pred_class_np[i]) < len(class_names) else "?"
            print(f"    [{i}] {cls_n} score={obj_scores_np[keep][i]:.2f} "
                  f"bbox=({pred_boxes_np[i][0]},{pred_boxes_np[i][1]},{pred_boxes_np[i][2]},{pred_boxes_np[i][3]})")

        # 画图
        out_path = os.path.join(args.output, os.path.basename(fp))
        draw_scene_graph(pil_img.copy(), pred_class_np, pred_boxes_np,
                         obj_scores_np[keep].cpu().numpy(), rels_np, rel_classes_np,
                         class_names, rel_names, out_path)

    print(f"\n完成 → {args.output}/")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
