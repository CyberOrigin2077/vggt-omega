"""
SAM3 逐帧分割 + VGGT 3D 投影跨帧匹配 → 带实例标签的语义点云 PLY。

核心改进：用 VGGT 的深度+相机做 3D 投影匹配，替代纯 2D IoU。
  - 帧A 每个 mask 的像素 → unproject 到 3D → project 到帧B
  - 查帧B 中被投影点覆盖最多的 mask → 匹配
  - 即使 2D 上完全不重叠的同一物体，3D 投影后也能正确关联

前提：先运行 vggt_preprocess.py 生成 _vggt_work/

用法 (sam3 环境):
    python splatv/sam3_3d_pcd.py \
        --scene_dir examples/room0 \
        --output_dir output_semantic_pcd/room0 \
        --max_points 500000
"""

import argparse, glob, os, sys, time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

# SAM3
_sam3_root = os.path.join(_project_root, "third_party", "sam3")
if _sam3_root not in sys.path:
    sys.path.insert(0, _sam3_root)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ═══════════════════════════ 匹配算法 ═══════════════════════════

def _match_fn(cost):
    """匈牙利匹配，scipy 不可用时回退贪婪。"""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:
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


def unproject_frame(depth, intrinsic, extrinsic):
    """深度图 → 世界坐标 [H, W, 3]。
    extrinsic 格式: VGGT 4x4 (w2c 的前3行在 [:3, :])
    """
    if isinstance(depth, np.ndarray): depth = torch.from_numpy(depth)
    if isinstance(intrinsic, np.ndarray): intrinsic = torch.from_numpy(intrinsic)
    if isinstance(extrinsic, np.ndarray): extrinsic = torch.from_numpy(extrinsic)
    depth, intrinsic, extrinsic = depth.float(), intrinsic.float(), extrinsic.float()
    device = depth.device
    H, W = depth.shape

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = torch.arange(W, dtype=torch.float32, device=device)
    v = torch.arange(H, dtype=torch.float32, device=device)
    v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")

    x_cam = (u_grid - cx) * depth / fx
    y_cam = (v_grid - cy) * depth / fy
    pts_cam = torch.stack([x_cam, y_cam, depth], dim=-1).reshape(-1, 3)  # [H*W, 3]

    # w2c → c2w
    w2c = torch.eye(4, device=device)
    w2c[:3, :] = extrinsic[:3, :]
    c2w = torch.linalg.inv(w2c)
    ones = torch.ones(pts_cam.shape[0], 1, device=device)
    pts_w = (c2w @ torch.cat([pts_cam, ones], dim=-1).T).T[:, :3]

    return pts_w.reshape(H, W, 3)


def project_to_frame(pcd_world, intrinsic, extrinsic, H, W):
    """将世界坐标 3D 点投影到指定帧的图像平面。
    Returns: u [N], v [N], valid [N] bool
    """
    if isinstance(pcd_world, np.ndarray): pcd_world = torch.from_numpy(pcd_world)
    if isinstance(intrinsic, np.ndarray): intrinsic = torch.from_numpy(intrinsic)
    if isinstance(extrinsic, np.ndarray): extrinsic = torch.from_numpy(extrinsic)
    pcd_world = pcd_world.float()
    intrinsic = intrinsic.float()
    extrinsic = extrinsic.float()
    device = pcd_world.device

    # world → cam: P_cam = R @ P_world + t
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pts_cam = (R @ pcd_world.T).T + t  # [N, 3]

    # cam → image
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    u = (pts_cam[:, 0] * fx / pts_cam[:, 2] + cx)
    v = (pts_cam[:, 1] * fy / pts_cam[:, 2] + cy)

    valid = (pts_cam[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u.long(), v.long(), valid


def match_instances_3d(all_masks, depths, intrinsics, extrinsics, window=10):
    """用 3D 投影 + 滑动窗口做跨帧实例匹配。

    每帧与前面 window 帧内的所有帧做 3D 投影匹配，累积投票，
    不再只依赖相邻帧。帧间偶尔漏检或被遮挡不会导致 ID 断裂。

    Args:
        all_masks:  list of [N_s, H, W] bool tensors
        depths:     [S, H, W] float32
        intrinsics: [S, 3, 3] float32
        extrinsics: [S, 4, 4] float32
        window:     前向搜索窗口大小

    Returns:
        frame_mappings: list of dict {local_id: global_id}
    """
    S = len(all_masks)
    device = depths.device

    # 预处理：每帧所有像素的世界坐标 [H, W, 3]
    frame_pcds = []
    for s in range(S):
        pcd = unproject_frame(depths[s], intrinsics[s], extrinsics[s])
        frame_pcds.append(pcd.to(device))

    H, W = depths.shape[1], depths.shape[2]

    # 第一帧
    N0 = all_masks[0].shape[0]
    frame_mappings = [{i: i + 1 for i in range(N0)}] if N0 > 0 else [{}]
    global_id_next = max(frame_mappings[0].values()) + 1 if frame_mappings[0] else 1

    for s in range(1, S):
        curr_masks = all_masks[s]  # [N_c, H, W]
        N_c = curr_masks.shape[0]

        if N_c == 0:
            frame_mappings.append({})
            continue

        # 累积投票：curr_local_id → {global_id: vote_count}
        gid_votes = {cj: {} for cj in range(N_c)}

        # 对窗口内的每一帧 t 做投影匹配
        t_start = max(0, s - window)
        for t in range(t_start, s):
            prev_masks = all_masks[t]
            N_p = prev_masks.shape[0]
            if N_p == 0:
                continue

            # 投影：帧 t 的世界坐标 → 帧 s 的图像平面
            pcd_t = frame_pcds[t].reshape(-1, 3)
            u_proj, v_proj, valid_proj = project_to_frame(
                pcd_t, intrinsics[s], extrinsics[s], H, W)

            valid_idx = torch.arange(H * W, device=device)[valid_proj]
            if len(valid_idx) < 10:
                continue

            u_valid = u_proj[valid_proj]
            v_valid = v_proj[valid_proj]

            # 计算帧 t 每个 mask 对帧 s 每个 mask 的共现
            prev_flat = prev_masks.reshape(N_p, -1).float()
            for pi in range(N_p):
                prev_mask_valid = prev_flat[pi, valid_idx] > 0.5
                if prev_mask_valid.sum() < 5:
                    continue
                nz_u = u_valid[prev_mask_valid]
                nz_v = v_valid[prev_mask_valid]
                for cj in range(N_c):
                    hits = curr_masks[cj, nz_v, nz_u].float().sum().item()
                    if hits > 3:
                        gid = frame_mappings[t].get(pi)
                        if gid is not None:
                            gid_votes[cj][gid] = gid_votes[cj].get(gid, 0) + hits

        # 每个当前mask选票数最高的全局ID
        mapping = {}
        for cj in range(N_c):
            votes = gid_votes[cj]
            if votes:
                best_gid = max(votes, key=votes.get)
                mapping[cj] = best_gid
            else:
                mapping[cj] = global_id_next
                global_id_next += 1

        frame_mappings.append(mapping)

    return frame_mappings


# ═══════════════════════════ 辅助 ═══════════════════════════

def depth_edge(depth, rtol=0.03, kernel_size=3):
    """检测深度图中局部跳变过大的像素（边缘/遮挡边界）。
    来源: VGGT-omega visual_util.py
    """
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y: y + depth.shape[-2], x: x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def id_to_color(obj_id):
    if obj_id <= 0:
        return (60, 60, 70)
    h = (int(obj_id) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    return (max(int(h & 0xFF), 40),
            max(int((h >> 8) & 0xFF), 40),
            max(int((h >> 16) & 0xFF), 40))


def load_static_masks(frame_ids, mask_dir, target_h, target_w, dilate_pixels=5):
    """加载动态 mask，支持两种格式:
      - {frame_id}.png          单文件模式 (obj_mask, mask)
      - {frame_id}_obj*.png     多物体模式 (person_mask)
    动态 mask 为 uint8 PNG: 255=动态, 0=静态。
    返回 [S, H, W] float tensor: 1=静态, 0=动态（已膨胀收缩）。
    """
    S = len(frame_ids)
    masks = []
    for fid in frame_ids:
        # 合并该帧所有 mask 文件: 任一标记为动态 → 动态
        merged = np.ones((target_h, target_w), dtype=np.float32)  # 默认全静态

        # 模式1: {frame_id}.png
        exact = os.path.join(mask_dir, f"{fid}.png")
        if os.path.exists(exact):
            pm = 1.0 - np.array(Image.open(exact)).astype(np.float32) / 255.0
            pm = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)
            pm = F.interpolate(pm, size=(target_h, target_w), mode="nearest")
            merged = np.minimum(merged, pm.squeeze().numpy())

        # 模式2: {frame_id}_*.png (逐物体分开的mask)
        for pf in sorted(glob.glob(os.path.join(mask_dir, f"{fid}_*.png"))):
            pm = 1.0 - np.array(Image.open(pf)).astype(np.float32) / 255.0
            pm = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0)
            pm = F.interpolate(pm, size=(target_h, target_w), mode="nearest")
            merged = np.minimum(merged, pm.squeeze().numpy())

        m = torch.from_numpy(merged)
        m = (m > 0.5).float()

        # 膨胀动态区域 → 收缩静态边缘
        if dilate_pixels > 0:
            k = 2 * dilate_pixels + 1
            t = (1.0 - m).unsqueeze(0).unsqueeze(0)
            t = F.max_pool2d(t, kernel_size=k, stride=1, padding=k // 2)
            m = 1.0 - t.squeeze()
        masks.append(m)
    return torch.stack(masks, dim=0)  # [S, H, W]


def _limit_points(vertices, colors, sem_ids, max_points):
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices, colors, sem_ids
    idx = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
    return vertices[idx], colors[idx], sem_ids[idx]


# ═══════════════════════════ Main ═══════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 + VGGT 3D 匹配 → 语义点云 PLY")
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--output_dir", default=None,
                        help="默认 output_semantic_pcd/<scene_name>")
    parser.add_argument("--sam3_ckpt", default=None,
                        help="SAM3 权重路径，默认 third_party/sam3/checkpoints/sam3.pt")
    parser.add_argument("--prompt", default="object",
                        help="SAM3 文本提示词")
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    parser.add_argument("--max_points", type=int, default=500000)
    parser.add_argument("--conf_thres", type=float, default=20.0,
                        help="深度置信度百分位阈值")
    parser.add_argument("--dilate", type=int, default=5,
                        help="膨胀动态 mask 的像素数，收缩静态区域边缘")
    parser.add_argument("--match_window", type=int, default=10,
                        help="3D匹配窗口大小(帧数)，越大合得越狠 (默认: 10)")
    parser.add_argument("--filter_depth_edges", action="store_true", default=True,
                        help="剔除深度边缘像素 (推荐)")
    parser.add_argument("--depth_edge_rtol", type=float, default=0.03,
                        help="深度边缘检测相对容差")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    scene_name = os.path.basename(os.path.abspath(args.scene_dir))
    work_dir = os.path.join(args.scene_dir, "_vggt_work")

    # ---- 读取 VGGT 预处理结果 ----
    cameras_path = os.path.join(work_dir, "cameras.npz")
    frame_list_path = os.path.join(work_dir, "frame_list.txt")
    images_path = os.path.join(work_dir, "vggt_images.npz")
    depth_dir = os.path.join(work_dir, "depth")

    if not os.path.exists(cameras_path):
        sys.exit(f"未找到 VGGT 预处理结果。请先运行 vggt_preprocess.py。\n"
                 f"期望路径: {cameras_path}")

    print(f"加载 VGGT 预处理结果: {work_dir}")
    cdata = np.load(cameras_path)
    intrinsics = torch.from_numpy(cdata["intrinsics"]).float()    # [S, 3, 3]
    extrinsics = torch.from_numpy(cdata["extrinsics"]).float()    # [S, 4, 4]
    H_vggt, W_vggt = int(cdata["resolution"][0]), int(cdata["resolution"][1])
    S = intrinsics.shape[0]
    print(f"  {S} 帧, VGGT 分辨率 {H_vggt}x{W_vggt}")

    with open(frame_list_path) as f:
        frame_ids = [line.strip() for line in f if line.strip()]
    print(f"  帧 ID: {frame_ids[0]} ... {frame_ids[-1]}")

    # 加载 VGGT 图像
    images_np = np.load(images_path)["images"]  # [S, H, W, 3]
    print(f"  VGGT 图像: {images_np.shape}")

    # 加载深度图 + 置信度
    depths, depth_confs = [], []
    for fid in frame_ids:
        d = np.load(os.path.join(depth_dir, f"{fid}.npy"))
        depths.append(torch.from_numpy(d))
        cf_path = os.path.join(depth_dir, f"{fid}_conf.npy")
        if os.path.exists(cf_path):
            depth_confs.append(torch.from_numpy(np.load(cf_path)))
        else:
            depth_confs.append(torch.ones_like(torch.from_numpy(d)))
    depths = torch.stack(depths, dim=0).float()           # [S, H, W]
    depth_confs = torch.stack(depth_confs, dim=0).float()  # [S, H, W]
    print(f"  深度: {depths.shape}, 置信度: {depth_confs.shape}")

    # ---- 动态 mask（可选）—— 扫描所有含 "mask" 的目录 ----
    def _has_mask_files(mask_dir):
        """检查目录是否包含匹配 frame_ids 的 mask 文件。"""
        fid0 = frame_ids[0]
        # 模式1: {frame_id}.png
        if os.path.exists(os.path.join(mask_dir, f"{fid0}.png")):
            return True
        # 模式2: {frame_id}_*.png
        if glob.glob(os.path.join(mask_dir, f"{fid0}_*.png")):
            return True
        return False

    mask_dirs = sorted(
        d for d in glob.glob(os.path.join(args.scene_dir, "*mask*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith("semantic")
        and _has_mask_files(d)
    )
    static_masks = None
    if mask_dirs:
        print(f"加载动态 mask: {[os.path.basename(d) for d in mask_dirs]}")
        # 对每个 mask 目录加载，然后取交集（任一目录标记为动态→动态）
        merged = torch.ones(S, H_vggt, W_vggt)
        for md in mask_dirs:
            sm = load_static_masks(
                frame_ids, md, H_vggt, W_vggt, dilate_pixels=args.dilate)
            merged = torch.minimum(merged, sm)
        static_masks = merged
        n_dynamic = (static_masks < 0.5).sum().item()
        n_total = static_masks.numel()
        print(f"  动态区域: {n_dynamic:,} px ({n_dynamic / n_total * 100:.1f}%)")
    else:
        print("未检测到含 mask 文件的目录，全部视为静态")

    # ---- 原始帧路径 (SAM3 处理用) ----
    frames_dir = os.path.join(args.scene_dir, "frames")
    frame_paths = []
    for fid in frame_ids:
        for ext in (".png", ".jpg"):
            fp = os.path.join(frames_dir, f"{fid}{ext}")
            if os.path.exists(fp):
                frame_paths.append(fp)
                break
    print(f"  帧路径: {len(frame_paths)} 个")

    # ---- SAM3 ----
    sam3_ckpt = args.sam3_ckpt
    if sam3_ckpt is None:
        sam3_ckpt = os.path.join(_sam3_root, "checkpoints", "sam3.pt")
    print(f"\n加载 SAM3 → {device}")
    model = build_sam3_image_model(
        device=device, eval_mode=True,
        enable_inst_interactivity=False,
        checkpoint_path=sam3_ckpt, load_from_HF=False,
    )
    processor = Sam3Processor(
        model, resolution=1008, device=device,
        confidence_threshold=args.confidence_threshold,
    )

    # ---- 阶段 1: SAM3 逐帧检测 ----
    print(f"\nSAM3 逐帧检测 (prompt='{args.prompt}') ...")
    t0 = time.time()
    all_masks_sam3 = []  # 每帧在原分辨率下的 mask
    all_scores = []

    for idx, fp in enumerate(frame_paths):
        pil_img = Image.open(fp).convert("RGB")
        state = processor.set_image(pil_img)
        state = processor.set_text_prompt(args.prompt, state)
        masks = state["masks"].squeeze(1).cpu()   # [N, H_orig, W_orig] bool
        scores = state["scores"].cpu()
        all_masks_sam3.append(masks)
        all_scores.append(scores)

        if idx % 15 == 0 or idx == S - 1:
            print(f"  检测 [{idx+1}/{S}] — {masks.shape[0]} 实例")
    t_detect = time.time() - t0
    total_instances = sum(m.shape[0] for m in all_masks_sam3)
    print(f"  检测完成: {t_detect:.1f}s, 共 {total_instances} 个局部实例")

    # ---- 阶段 2: 缩放 mask 到 VGGT 分辨率做 3D 匹配 ----
    print(f"\n缩放 mask 到 VGGT 分辨率 ({H_vggt}x{W_vggt}) ...")
    all_masks_vggt = []
    for masks in all_masks_sam3:
        if masks.shape[0] == 0:
            all_masks_vggt.append(torch.zeros(0, H_vggt, W_vggt, dtype=torch.bool))
            continue
        m = masks.unsqueeze(1).float()  # [N, 1, H_orig, W_orig]
        m = F.interpolate(m, size=(H_vggt, W_vggt), mode="nearest")
        all_masks_vggt.append(m.squeeze(1) > 0.5)  # [N, H_vggt, W_vggt]

    # ---- 阶段 3: 3D 投影跨帧匹配 ----
    print("3D 投影跨帧匹配 ...")
    t_m = time.time()
    depths_dev = depths.to(device)
    intrinsics_dev = intrinsics.to(device)
    extrinsics_dev = extrinsics.to(device)
    all_masks_dev = [m.to(device) for m in all_masks_vggt]

    frame_mappings = match_instances_3d(
        all_masks_dev, depths_dev, intrinsics_dev, extrinsics_dev,
        window=args.match_window)

    t_match = time.time() - t_m

    # 统计全局 ID
    all_gids = set()
    for fm in frame_mappings:
        all_gids.update(fm.values())
    print(f"  全局唯一实例: {len(all_gids)}, 匹配耗时: {t_match:.1f}s")

    # ---- 阶段 4: 生产语义 remap (逐帧) ----
    print("生成逐帧语义图 ...")
    remapped_maps = []
    for s in range(S):
        sem = torch.zeros(H_vggt, W_vggt, dtype=torch.int32)
        masks = all_masks_vggt[s]  # [N_s, H, W]
        mapping = frame_mappings[s]
        for local_id in range(masks.shape[0]):
            gid = mapping[local_id]
            sem = torch.where(masks[local_id].bool(), gid, sem)
        remapped_maps.append(sem)

    # 保存逐帧语义图 → _vggt_work/semantic_masks/ (供 FROSS 集成)
    sem_save_dir = os.path.join(work_dir, "semantic_masks")
    os.makedirs(sem_save_dir, exist_ok=True)
    for s, fid in enumerate(frame_ids):
        np.save(os.path.join(sem_save_dir, f"{fid}.npy"), remapped_maps[s].numpy())
    print(f"  逐帧语义图已保存: {sem_save_dir}/ ({S} 帧)")

    # ---- 阶段 5: 筛选 & 反投影 → 点云 ----
    print("反投影生成语义点云 ...")
    t_proj = time.time()

    # 深度置信度 + 深度边缘剔除 (与 VGGT 官方 visual_util.py 一致)
    depth_conf = depth_confs.clone()
    if args.filter_depth_edges:
        depth_np = depths.numpy()
        edge_mask = depth_edge(depth_np, rtol=args.depth_edge_rtol)
        depth_conf = depth_conf * (1.0 - torch.from_numpy(edge_mask).float())
        n_edge = edge_mask.sum()
        print(f"  深度边缘像素: {n_edge:,} ({n_edge / edge_mask.size * 100:.1f}%)")

    if args.conf_thres > 0:
        valid_conf = depth_conf.reshape(-1)
        valid_conf = valid_conf[torch.isfinite(valid_conf)]
        threshold = float(np.percentile(valid_conf.numpy(), args.conf_thres)) if valid_conf.numel() > 0 else 1.0
    else:
        threshold = 0.0
    print(f"  置信度阈值: perc_{args.conf_thres} = {threshold:.2f}")

    all_pts, all_colors, all_sem = [], [], []

    for s in range(S):
        d = depths[s]
        valid_c = (torch.isfinite(d) & (d > 1e-5) & (depth_conf[s] >= threshold))
        if static_masks is not None:
            valid_c = valid_c & (static_masks[s] > 0.5)
        if not valid_c.any():
            continue

        world_pts = unproject_frame(d, intrinsics[s], extrinsics[s])
        pts_flat = world_pts.reshape(-1, 3)
        valid_xyz = torch.isfinite(pts_flat).all(dim=1).reshape(H_vggt, W_vggt)
        valid = valid_c & valid_xyz

        pts = world_pts[valid].cpu().numpy()
        colors = images_np[s][valid.cpu().numpy()]
        sem_ids = remapped_maps[s][valid].numpy()

        if len(pts) > 0:
            all_pts.append(pts)
            all_colors.append(colors)
            all_sem.append(sem_ids)

    pts_all = np.concatenate(all_pts, axis=0)
    colors_all = np.concatenate(all_colors, axis=0)
    sem_all = np.concatenate(all_sem, axis=0)
    print(f"  总点数: {len(pts_all):,}")

    # 等距降采样
    if args.max_points > 0 and len(pts_all) > args.max_points:
        pts_all, colors_all, sem_all = _limit_points(
            pts_all, colors_all, sem_all, args.max_points)
        print(f"  降采样 → {len(pts_all):,}")

    # 保持 VGGT 原始世界坐标系 (与 FROSS 一致, 不做去中心化)

    # ---- 统计 ----
    n_inst = len(set(sem_all[sem_all > 0]))
    n_bg = (sem_all == 0).sum()
    print(f"  有效实例: {n_inst}, 背景: {n_bg:,} ({n_bg / len(sem_all) * 100:.1f}%)")

    # ---- PLY ----
    colors_u8 = np.clip(colors_all * 255, 0, 255).astype(np.uint8)
    N_out = len(pts_all)

    # 实例着色
    unique_ids = np.unique(sem_all)
    color_lut = {uid: id_to_color(uid) for uid in unique_ids}
    inst_r = np.array([color_lut[s][0] for s in sem_all], dtype=np.uint8)
    inst_g = np.array([color_lut[s][1] for s in sem_all], dtype=np.uint8)
    inst_b = np.array([color_lut[s][2] for s in sem_all], dtype=np.uint8)

    vertex = np.zeros(N_out, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("instance_id", "i4"),
        ("inst_red", "u1"), ("inst_green", "u1"), ("inst_blue", "u1"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = pts_all[:, 0], pts_all[:, 1], pts_all[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = (
        colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2])
    vertex["instance_id"] = sem_all.astype(np.int32)
    vertex["inst_red"], vertex["inst_green"], vertex["inst_blue"] = inst_r, inst_g, inst_b

    # 输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join("output_semantic_pcd", scene_name)
    os.makedirs(args.output_dir, exist_ok=True)
    ply_path = os.path.join(args.output_dir, f"{scene_name}_semantic.ply")
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(ply_path)

    t_total = time.time() - t0
    print(f"\n完成: {ply_path} ({N_out:,} 点, {n_inst} 个实例)")
    print(f"耗时: 检测 {t_detect:.1f}s | 匹配 {t_match:.1f}s | "
          f"反投影 {time.time() - t_proj:.1f}s | 总计 {t_total:.1f}s")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
