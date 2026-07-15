"""
用文本查询 SceneSplat 输出的语言特征，生成逐高斯语义标签。

用法:
    python query_scenesplat.py \
        --features_dir output/room0 \
        --output semantic_labels.npy \
        --prompts "wall,floor,ceiling,chair,table,sofa,bed,cabinet,door,window"
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


def encode_texts(texts, model, processor, device):
    """CLIP 文本编码 → L2 归一化。"""
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return F.normalize(text_features, dim=-1)


def main():
    parser = argparse.ArgumentParser(description="SceneSplat 特征文本查询")
    parser.add_argument("--features_dir", type=str, required=True,
                        help="lang_inference 输出目录（含 *_feat.pt）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出语义标签 .npy 路径")
    parser.add_argument("--prompts", type=str, required=True,
                        help="逗号分隔的文本类别，如 'chair,table,wall'")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="余弦相似度阈值（低于此值归为背景 0）")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- 加载特征 ----
    feat_files = sorted(f for f in os.listdir(args.features_dir) if f.endswith("_feat.pt"))
    if not feat_files:
        raise FileNotFoundError(f"未在 {args.features_dir} 找到 *_feat.pt")
    feat_path = os.path.join(args.features_dir, feat_files[0])
    print(f"加载特征: {feat_path}")
    features = torch.load(feat_path, map_location="cpu", weights_only=True)
    if isinstance(features, dict):
        features = features["feat"]
    features = features.float().to(device)  # [N, 768]
    N = features.shape[0]
    print(f"  {N:,} 高斯, 特征维度 {features.shape[1]}")

    # ---- L2 归一化特征 ----
    features = F.normalize(features, dim=-1)

    # ---- 文本编码 ----
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print(f"文本类别: {prompts}")

    # 使用较小的 CLIP 模型
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    text_embeds = encode_texts(prompts, clip_model, clip_processor, device)  # [C, D]
    C = text_embeds.shape[0]

    # ---- 余弦相似度 ----
    # 如果特征维度 != CLIP 维度，需要投影
    if features.shape[1] != text_embeds.shape[1]:
        print(f"  特征维度 {features.shape[1]} != CLIP 维度 {text_embeds.shape[1]}, "
              f"使用 PCA 对齐")
        # 简单方案: 截断或零填充
        min_dim = min(features.shape[1], text_embeds.shape[1])
        features = features[:, :min_dim]
        text_embeds = text_embeds[:, :min_dim]
        features = F.normalize(features, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

    sim = features @ text_embeds.T  # [N, C]

    # ---- 分配标签 ----
    max_sim, labels = sim.max(dim=1)  # [N]
    labels = labels + 1  # class 1..C
    labels[max_sim < args.threshold] = 0  # 低于阈值 → 背景

    labels_np = labels.cpu().numpy().astype(np.int32)

    # 统计
    for i, p in enumerate(prompts):
        count = (labels_np == i + 1).sum()
        print(f"  {p}: {count:,} ({count / N * 100:.1f}%)")
    bg_count = (labels_np == 0).sum()
    print(f"  背景: {bg_count:,} ({bg_count / N * 100:.1f}%)")

    np.save(args.output, labels_np)
    print(f"\n已保存: {args.output}")


if __name__ == "__main__":
    main()
