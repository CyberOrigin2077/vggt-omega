#!/bin/bash
# 从项目根目录运行: bash splatv/test.sh
cd "$(dirname "$0")/.."

# MinkowskiEngine 环境变量 (vggt_omega 环境)
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=${CONDA_PREFIX}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 方案 A (SAM3 2D→3D) — 在 sam3 环境中运行
# ============================================================
# conda activate sam3

# A1: SAM3 逐帧实例分割 + 跨帧匹配
# CUDA_VISIBLE_DEVICES=6 python splatv/infer_semantic.py \
#     --scene_dir examples/factory1 \
#     --confidence_threshold 0.1

# A2: 语义 mask 可视化
# python splatv/visualize.py \
#     --semantic_dir examples/factory1/semantic_masks \
#     --frames_dir examples/factory1/frames \
#     --alpha 0.5

# A3: 静态高斯导出 + 语义投影
# CUDA_VISIBLE_DEVICES=6,7 python splatv/export_static.py \
#     --scene_dir examples/factory1 \
#     --vggt_ckpt checkpoints/vggt_omega_1b_512.pt \
#     --sv_ckpt checkpoints/splatvoxel/checkpoint-best.pth \
#     --output_dir ./output_static/factory1 \
#     --voxel_grid_res 1200 \
#     --semantic_dir examples/factory1/semantic_masks


# ============================================================
# 方案 B (SceneSplat 3D 原生) — 在 scene_splat 环境中运行
# ============================================================
# conda activate scene_splat

# B1-B5 一键: 实例分割 (OverKMeans, 自动确定聚类数)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=third_party/SceneSplat python splatv/run_scenesplat.py \
    --input output_static/room0/room0_static.splatv \
    --output output_static/room0/room0_scenesplat.splatv

# 调参: --over_k 过分割粒度, --merge_thresh 合并激进程度
# CUDA_VISIBLE_DEVICES=6 PYTHONPATH=third_party/SceneSplat python splatv/run_scenesplat.py \
#     --input ... --output ... --over_k 500 --merge_thresh 0.8

# B1-B5 一键: CLIP 语义模式
# CUDA_VISIBLE_DEVICES=6 PYTHONPATH=third_party/SceneSplat python splatv/run_scenesplat.py \
#     --input output_static/room0/room0_static.splatv \
#     --output output_static/room0/room0_scenesplat_clip.splatv \
#     --use_clip --threshold 0.04 \
#     --prompts "wall,floor,ceiling,chair,table,sofa,bed,cabinet,door,window"

# (分步运行——参考方案 B 的步骤注释在下方)


# ============================================================
# 无语义版本 (vggt_omega 环境)
# ============================================================
# CUDA_VISIBLE_DEVICES=6,7 python splatv/export_static.py \
#     --scene_dir examples/hotel0 \
#     --vggt_ckpt checkpoints/vggt_omega_1b_512.pt \
#     --sv_ckpt checkpoints/splatvoxel/checkpoint-best.pth \
#     --output_dir ./output_static/hotel0 \
#     --voxel_grid_res 1200


# ============================================================
# 方案 B 分步 (调试用)
# ============================================================
# B1: splaTV → PLY
# python splatv/convert_to_ply.py --input ... --output ...

# B2: PLY → NPY (scene_splat 环境)
# python -m scripts.preprocess_gs --input ... --output ...

# B3: NPY → 语言特征 (scene_splat 环境)
# python -m tools.lang_inference --config ... --checkpoint ... --input-root ... --output-dir ...

# B4: 特征 → 语义 (scene_splat 环境)
# python splatv/query_scenesplat.py --features_dir ... --output ... --prompts "..."

# B5: 语义写回 splaTV
# python splatv/patch_semantic.py --input ... --labels ... --output ...
