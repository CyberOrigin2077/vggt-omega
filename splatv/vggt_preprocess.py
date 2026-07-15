"""
VGGT-Omega 预处理：对场景所有帧推理，保存深度图 + 相机参数到磁盘。

输出目录: <scene>/_vggt_work/
  depth/{frame_id}.npy     — 每帧深度图 float32 [H, W]
  cameras.npz              — intr [S,3,3], extr [S,4,4], resolution [H,W]
  frame_list.txt           — 帧文件名列表（保持顺序）

用法 (vggt-omega 环境):
    python splatv/vggt_preprocess.py \
        --scene_dir examples/room0 \
        --vggt_ckpt checkpoints/vggt_omega_1b_512.pt
"""

import argparse, glob, os, sys, time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


@torch.no_grad()
def vggt_inference(model, images_batch):
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        aggregated_tokens_list, patch_token_start = model.aggregator(images_batch)
    with torch.autocast(device_type="cuda", enabled=False):
        pose_enc = model.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
        depth, depth_conf = model.dense_head(
            aggregated_tokens_list, images=images_batch, patch_token_start=patch_token_start)
    return {"depth": depth, "depth_conf": depth_conf, "pose_enc": pose_enc}


def main():
    parser = argparse.ArgumentParser(description="VGGT 预处理：保存深度+相机")
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--vggt_ckpt", required=True)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    frames_dir = os.path.join(args.scene_dir, "frames")
    if not os.path.isdir(frames_dir):
        sys.exit(f"frames dir not found: {frames_dir}")

    work_dir = os.path.join(args.scene_dir, "_vggt_work")
    depth_dir = os.path.join(work_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    # ---- 帧列表 ----
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if args.num_frames:
        frame_paths = frame_paths[:args.num_frames]
    S = len(frame_paths)
    frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in frame_paths]

    # 保存帧列表
    with open(os.path.join(work_dir, "frame_list.txt"), "w") as f:
        for fid in frame_ids:
            f.write(f"{fid}\n")

    print(f"场景: {args.scene_dir}, {S} 帧")

    # ---- VGGT 推理 ----
    print(f"加载 VGGT-Omega → {device} ...")
    model = VGGTOmega().eval().to(device)
    sd = torch.load(args.vggt_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad = False

    t0 = time.time()
    print("VGGT 推理 ...")
    images = load_and_preprocess_images(frame_paths, image_resolution=args.image_resolution)
    S = images.shape[0]
    H_out, W_out = images.shape[-2:]
    print(f"  分辨率: {H_out}x{W_out}")

    images_batch = images.unsqueeze(0).to(device)
    out = vggt_inference(model, images_batch)
    extrinsics, intrinsics = encoding_to_camera(out["pose_enc"], (H_out, W_out))

    depth = out["depth"].squeeze(0).squeeze(-1).float().cpu()       # [S, H, W]
    depth_conf = out["depth_conf"].squeeze(0).squeeze(-1).float().cpu()
    intri = intrinsics.squeeze(0).float().cpu()                      # [S, 3, 3]
    extri = extrinsics.squeeze(0).float().cpu()                      # [S, 4, 4]
    print(f"  VGGT 推理: {time.time() - t0:.1f}s")

    # ---- 逐帧保存 ----
    print(f"保存深度/置信度 → {depth_dir}/ ...")
    for s in range(S):
        np.save(os.path.join(depth_dir, f"{frame_ids[s]}.npy"), depth[s].numpy())
        np.save(os.path.join(depth_dir, f"{frame_ids[s]}_conf.npy"), depth_conf[s].numpy())

    # ---- 保存相机参数 ----
    cameras_path = os.path.join(work_dir, "cameras.npz")
    np.savez(cameras_path,
             intrinsics=intri.numpy(),
             extrinsics=extri.numpy(),
             resolution=np.array([H_out, W_out], dtype=np.int32))
    print(f"保存相机 → {cameras_path}")

    # ---- 保存原始 VGGT 输入图像 (供 SAM3 端加载) ----
    # 保存为 npz 方便跨环境读取
    images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # [S, H, W, 3]
    images_path = os.path.join(work_dir, "vggt_images.npz")
    np.savez(images_path, images=images_np)
    print(f"保存 VGGT 图像 → {images_path}")

    del model, images_batch, out
    torch.cuda.empty_cache()
    print(f"\n完成: {work_dir}/ ({S} 帧, 总耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
