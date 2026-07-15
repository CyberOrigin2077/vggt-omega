"""
VGGT-omega 输出 → FROSS 3RScan 格式转换。

用法 (vggt-omega 环境):
    python splatv/export_fross_format.py --scene_dir examples/lab0 --output_dir Datasets/lab0
"""

import argparse, os, sys, struct

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from PIL import Image

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
    parser = argparse.ArgumentParser(description="VGGT → FROSS 3RScan 格式")
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--vggt_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--label_categories", choices=["scannet", "replica"],
                        default="replica", help="replica=VG权重, scannet=3RScan权重")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    frames_dir = os.path.join(args.scene_dir, "frames")
    if not os.path.isdir(frames_dir):
        sys.exit(f"frames dir not found: {frames_dir}")

    scene_name = os.path.basename(os.path.abspath(args.scene_dir))
    is_scannet = args.label_categories == "scannet"

    # FROSS 目录结构: output_dir/data/<scene>/sequence/
    seq_dir = os.path.join(args.output_dir, "data", scene_name, "sequence")
    os.makedirs(seq_dir, exist_ok=True)

    # ---- 帧列表 ----
    frame_paths = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
         if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    S = len(frame_paths)
    frame_ids = [os.path.splitext(os.path.basename(p))[0] for p in frame_paths]
    print(f"{scene_name}: {S} 帧")

    # ---- 优先复用 _vggt_work/ 缓存 (避免重复 VGGT 推理) ----
    vggt_work_dir = os.path.join(args.scene_dir, "_vggt_work")
    vggt_cameras_path = os.path.join(vggt_work_dir, "cameras.npz")
    vggt_images_path = os.path.join(vggt_work_dir, "vggt_images.npz")
    vggt_depth_dir = os.path.join(vggt_work_dir, "depth")

    if os.path.exists(vggt_cameras_path) and os.path.exists(vggt_images_path):
        print(f"复用 _vggt_work 缓存: {vggt_work_dir}")
        cdata = np.load(vggt_cameras_path)
        intri = torch.from_numpy(cdata["intrinsics"]).float()   # [S, 3, 3]
        extri_4x4 = torch.zeros(intri.shape[0], 4, 4)
        extri_4x4[:, :3, :] = torch.from_numpy(cdata["extrinsics"]).float()
        extri_4x4[:, 3, 3] = 1.0
        extri = extri_4x4
        H, W = int(cdata["resolution"][0]), int(cdata["resolution"][1])
        S = extri.shape[0]

        # 深度从逐个文件加载
        depth = torch.zeros(S, H, W)
        for s, fid in enumerate(frame_ids):
            dp = os.path.join(vggt_depth_dir, f"{fid}.npy")
            depth[s] = torch.from_numpy(np.load(dp))

        # VGGT 图像已是 [0,1] (ToTensor 转换)
        images_u8 = np.load(vggt_images_path)["images"]  # [S, H, W, 3]
        images_u8 = np.clip(images_u8 * 255, 0, 255).astype(np.uint8)
        print(f"  {S} 帧, {H}x{W}")
    else:
        print(f"未找到缓存, 运行 VGGT 推理 ...")
        model = VGGTOmega().eval().to(device)
        sd = torch.load(args.vggt_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        for p in model.parameters():
            p.requires_grad = False

        images = load_and_preprocess_images(frame_paths, image_resolution=args.image_resolution)
        S = images.shape[0]
        H, W = images.shape[-2:]
        print(f"  VGGT 分辨率: {H}x{W}")

        images_batch = images.unsqueeze(0).to(device)
        out = vggt_inference(model, images_batch)
        extrinsics, intrinsics = encoding_to_camera(out["pose_enc"], (H, W))

        depth = out["depth"].squeeze(0).squeeze(-1).float().cpu()
        intri = intrinsics.squeeze(0).float().cpu()
        extri_raw = extrinsics.squeeze(0).float().cpu()
        if extri_raw.shape[1] == 3:
            extri = torch.zeros(S, 4, 4)
            extri[:, :3, :] = extri_raw
            extri[:, 3, 3] = 1.0
        else:
            extri = extri_raw

        images_u8 = np.clip(images.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)

    # ---- 取第一帧的相机内参 (VGGT 所有帧共享) ----
    K = intri[0]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # ---- 写 _info.txt ----
    # depthShift: mm→m 都用 1000 (16-bit 存储)
    depth_shift = 1000
    info_path = os.path.join(seq_dir, "_info.txt")
    with open(info_path, "w") as f:
        f.write(f"m_versionNumber = 1\n")
        f.write(f"m_colorWidth = {W}\n")
        f.write(f"m_colorHeight = {H}\n")
        f.write(f"m_depthWidth = {W}\n")
        f.write(f"m_depthHeight = {H}\n")
        f.write(f"m_depthShift = {depth_shift}\n")
        # FROSS 按 3x4 投影矩阵格式读取: [0]=fx [2]=cx [5]=fy [6]=cy
        # 格式: [[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0]] 展平
        f.write(f"m_calibrationColorIntrinsic = {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0\n")
        f.write(f"m_calibrationDepthIntrinsic = {fx} 0 {cx} 0 0 {fy} {cy} 0 0 0 1 0\n")
    print(f"  _info.txt: {W}x{H}, fx={fx:.1f} fy={fy:.1f}, shift={depth_shift}")

    # ---- 逐帧保存 ----
    # FROSS 固定用 .color.jpg (scannet 和 replica 都一样, sg_loader.py:25)
    extension = ".color.jpg"
    depth_ext = ".rendered.depth.png" if is_scannet else ".depth.pgm"
    print(f"  格式: {extension}, 深度: {depth_ext}")

    for s in range(S):
        # Color
        color_name = f"{scene_name}-{s:06d}{extension}"
        color_path = os.path.join(seq_dir, color_name)
        Image.fromarray(images_u8[s]).save(color_path)

        # Depth: 16-bit, mm 单位
        depth_name = f"{scene_name}-{s:06d}{depth_ext}"
        depth_path = os.path.join(seq_dir, depth_name)
        d_mm = np.clip(depth[s].numpy() * 1000, 0, 65535).astype(np.uint16)
        Image.fromarray(d_mm).save(depth_path)

        # Pose: 4x4 extrinsic matrix
        pose_name = f"{scene_name}-{s:06d}.pose.txt"
        pose_path = os.path.join(seq_dir, pose_name)
        e = extri[s].numpy()
        with open(pose_path, "w") as f:
            for row in range(4):
                f.write(f"{e[row, 0]:.6f} {e[row, 1]:.6f} {e[row, 2]:.6f} {e[row, 3]:.6f}\n")

    # ---- 写 scan 列表文件 ----
    ssg_dir = os.path.join(args.output_dir, "3DSSG_subset" if is_scannet else "ReplicaSSG")
    os.makedirs(ssg_dir, exist_ok=True)
    scan_list = os.path.join(ssg_dir, "test_scans.txt")
    with open(scan_list, "w") as f:
        f.write(f"{scene_name}\n")

    # ---- 空 2DSG 标注（让 FROSS 用 RT-DETR-EGTR 预测，不用 GT） ----
    dsg_dir = os.path.join(args.output_dir, "2DSG20" if is_scannet else "2DSG")
    os.makedirs(dsg_dir, exist_ok=True)

    print(f"\n完成: {seq_dir}/ ({S} 帧)")
    print(f"\n运行 FROSS ({args.label_categories} 模式):")
    print(f"  conda activate fross")
    print(f"  cd third_party/FROSS/Merging")
    print(f"  python main.py \\")
    print(f"    --dataset_path ../../{args.output_dir} \\")
    print(f"    --label_categories {args.label_categories} \\")
    print(f"    --artifact_path ../../weights/RT-DETR-EGTR/VG/<version_dir>/ \\")
    print(f"    --split test")


if __name__ == "__main__":
    main()
