#!/usr/bin/env python3
"""
动态物体 mask 生成 — 集成 SAM3 人掩码 + InterFormer 手持物掩码。

两条流水线独立运行 (各自环境不同), 输出合并为统一的二值动态 mask:
  255 = 动态像素, 0 = 静态背景

用法:
    python splatv/generate_dynamic_mask.py \
        --frame_dir examples/scene/frames \
        --output_dir examples/scene/mask \
        --sam3_root /path/to/sam3 \
        --mask_module_dir third_party/mask_module
"""

import argparse, os, subprocess, sys, glob, shutil

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(cmd, desc, env=None):
    print(f"\n  [{desc}]")
    print(f"  {' '.join(cmd)}")
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=merged_env,
                       capture_output=False)
    if r.returncode != 0:
        print(f"  [ERROR] {desc} 失败 (exit {r.returncode})")
    return r.returncode == 0


def merge_masks(person_dir, object_dir, output_dir, dilate=5):
    """
    合并 person mask + object mask → 统一动态 mask + 膨胀。

    person mask:  combined_masks/<stem>.png  像素值 = 实例ID (>0 = 人)
    object mask:  mask/<stem>.png            0/255 (>0 = 手持物)
    dilate: 合并后膨胀像素数 (默认 5)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 从 person mask 目录获取帧列表
    person_mask_dir = os.path.join(person_dir, "combined_masks") if person_dir else None
    if not os.path.isdir(person_mask_dir):
        print(f"  [WARN] 无 person mask 目录: {person_mask_dir}")
        person_mask_dir = None

    object_mask_dir = os.path.join(object_dir, "mask") if object_dir else None
    if object_mask_dir and not os.path.isdir(object_mask_dir):
        print(f"  [WARN] 无 object mask 目录: {object_mask_dir}")
        object_mask_dir = None

    # 收集所有帧的文件名
    stems = set()
    for md in [person_mask_dir, object_mask_dir]:
        if md:
            for f in os.listdir(md):
                if f.endswith(".png"):
                    stems.add(os.path.splitext(f)[0])

    if not stems:
        print("  [ERROR] 未找到任何 mask 文件")
        return 0

    for stem in sorted(stems):
        merged = np.zeros((1, 1), dtype=np.uint8)

        # 加载 person mask
        if person_mask_dir:
            pp = os.path.join(person_mask_dir, f"{stem}.png")
            if os.path.exists(pp):
                pm = np.array(Image.open(pp))
                if merged.shape == (1, 1):
                    merged = np.zeros(pm.shape[:2], dtype=np.uint8)
                merged[pm > 0] = 255

        # 加载 object mask
        if object_mask_dir:
            op = os.path.join(object_mask_dir, f"{stem}.png")
            if os.path.exists(op):
                om = np.array(Image.open(op))
                if merged.shape == (1, 1):
                    merged = np.zeros(om.shape[:2], dtype=np.uint8)
                # 统一尺寸
                if om.shape[:2] != merged.shape[:2]:
                    om = np.array(Image.fromarray(om).resize(
                        (merged.shape[1], merged.shape[0]), Image.NEAREST))
                merged[om > 0] = 255

        # 膨胀 (消除边缘锯齿, 确保动态物体完整滤除)
        if dilate > 0:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
            merged = cv2.dilate(merged, kernel)

        out_path = os.path.join(output_dir, f"{stem}.png")
        Image.fromarray(merged).save(out_path)

    print(f"  合并完成: {len(stems)} 帧 → {output_dir}")
    return len(stems)


def main():
    parser = argparse.ArgumentParser(description="动态物体 mask 生成")
    parser.add_argument("--frame_dir", required=True, help="帧目录")
    parser.add_argument("--output_dir", required=True, help="输出 mask 目录")
    parser.add_argument("--sam3_root", default="/data/users/yzr/code/sam3",
                        help="SAM3 仓库路径")
    parser.add_argument("--sam3_checkpoint", default=None)
    parser.add_argument("--mask_module_dir",
                        default=os.path.join(PROJECT_ROOT, "third_party", "mask_module"))
    # 环境
    parser.add_argument("--sam3_python",
                        default="/data/users/yzr/envs/lh3dod-sam3/bin/python")
    parser.add_argument("--mmseg_python",
                        default="/data/users/yzr/miniconda3/envs/mmseg/bin/python")
    parser.add_argument("--skip_person", action="store_true")
    parser.add_argument("--skip_object", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    mask_module = os.path.abspath(args.mask_module_dir)
    frame_dir = os.path.abspath(args.frame_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(frame_dir):
        sys.exit(f"帧目录不存在: {frame_dir}")
    if not os.path.isdir(mask_module):
        sys.exit(f"mask_module 不存在: {mask_module}")

    n_frames = len([f for f in os.listdir(frame_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"输入: {frame_dir} ({n_frames} 帧)")
    print(f"输出: {output_dir}")
    print(f"mask_module: {mask_module}")

    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    person_tmp = os.path.join(tmp_dir, "person")
    object_tmp = os.path.join(tmp_dir, "object")

    # ---- Step 1: person mask (SAM3) ----
    person_ok = False
    if not args.skip_person and os.path.isfile(args.sam3_python):
        person_script = os.path.join(mask_module, "person_mask", "mask_person_sam3.py")
        if not os.path.exists(person_script):
            print(f"  [WARN] person mask 脚本不存在: {person_script}")
        else:
            # SAM3 内部有多 GPU 设备 bug, 用 CUDA_VISIBLE_DEVICES 隔离
            gpu_id = args.device.replace("cuda:", "")
            person_cmd = [
                args.sam3_python, person_script,
                "--img-dir", frame_dir,
                "--out-dir", person_tmp,
                "--sam3-root", args.sam3_root,
                "--limit", "0",
                "--prompt", "person",
                "--device", "cuda:0",       # CUDA_VISIBLE_DEVICES 隔离后总是 0
            ]
            if args.sam3_checkpoint:
                person_cmd += ["--checkpoint", args.sam3_checkpoint]
            person_ok = run_step(person_cmd, "SAM3 person mask",
                                 env={"CUDA_VISIBLE_DEVICES": gpu_id,
                                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                      "TMPDIR": "/data/users/xjl/tmp"})

    # ---- Step 2: object mask (InterFormer) ----
    object_ok = False
    if not args.skip_object:
        inter_dir = os.path.join(mask_module, "InterFormer")
        object_script = os.path.join(inter_dir, "tools", "infer_object_mask.py")
        if not os.path.exists(object_script):
            print(f"  [WARN] object mask 脚本不存在: {object_script}")
        else:
            object_cmd = [
                args.mmseg_python, object_script,
                "--img-dir", frame_dir,
                "--out-dir", object_tmp,
                "--config", os.path.join(inter_dir, "work_dirs",
                                         "ktvmopping_cam01_first10",
                                         "config_interformer.py"),
                "--checkpoint", os.path.join(inter_dir, "checkpoints",
                                             "interformer_checkpoint.pth"),
                "--limit", "0",
                "--device", args.device,
            ]
            object_ok = run_step(object_cmd, "InterFormer object mask",
                                 env={"PYTHONPATH": inter_dir})

    # ---- Step 3: merge ----
    if person_ok or object_ok:
        p_dir = person_tmp if person_ok else None
        o_dir = object_tmp if object_ok else None
        n = merge_masks(p_dir, o_dir, output_dir, dilate=5)
        print(f"\n完成: {n} 帧动态 mask → {output_dir}")
    else:
        print("\n[ERROR] 两条 mask 流程均失败")

    # 清理临时文件
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
