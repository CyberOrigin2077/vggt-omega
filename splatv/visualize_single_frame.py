#!/usr/bin/env python3
"""
单帧 3D 拓扑图投影到场景点云 — 保持原有点云颜色 + 语义高斯球 + 拓扑关系线。

用法:
    python splatv/visualize_single_frame.py \
        --frame_npz Datasets/office2/per_frame_sam3/office2/frame_000000.npz \
        --scene_ply output_semantic_pcd/office2/office2_semantic.ply \
        --class_json Datasets/office2/ReplicaSSG/replica_to_visual_genome.json \
        --output output_fross/office2/office2_frame000
"""

import argparse, os, sys, json

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def id_to_rgb(uid):
    h = (int(uid) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    return (max(int(h & 0xFF), 50),
            max(int((h >> 8) & 0xFF), 50),
            max(int((h >> 16) & 0xFF), 50))


def sample_ellipsoid(mean, cov, n_pts=500, base_radius=0.03):
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    eigenvals = np.maximum(eigenvals, 1e-8)
    radii = np.sqrt(eigenvals)
    radii = np.maximum(radii, base_radius)

    phi = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(n_pts)
    y = 1.0 - (idx / (n_pts - 1)) * 2.0
    r = np.sqrt(1.0 - y * y)
    theta = phi * idx
    unit_sphere = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)
    return unit_sphere * radii @ eigenvecs.T + mean


def build_tube_edge(start, end, ring_radius=0.015, n_segments=None, n_ring=6):
    """构建两个 3D 点之间的管状连线。"""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.zeros((0, 3), dtype=np.float32)
    direction = direction / length
    n_segments = n_segments or max(int(length / 0.03), 5)

    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, [1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    pts = []
    for seg in range(n_segments + 1):
        t = seg / n_segments
        center = start + t * (end - start)
        for a in range(n_ring):
            angle = 2 * np.pi * a / n_ring
            pt = center + ring_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            pts.append(pt)
    return np.array(pts, dtype=np.float32)


def build_camera_frustum(R, t, fx, fy, cx, cy, H, W, scale=0.20):
    """构建相机锥体 5 点 (4角+光心), 8 条线。"""
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c); center = c2w[:3, 3]
    corners_cam = np.array([
        [(-cx) / fx, (-cy) / fy, 1],
        [(W - cx) / fx, (-cy) / fy, 1],
        [(W - cx) / fx, (H - cy) / fy, 1],
        [(-cx) / fx, (H - cy) / fy, 1],
    ]) * scale
    corners = (c2w @ np.hstack([corners_cam, np.ones((4, 1))]).T).T[:, :3]
    return np.vstack([corners, center.reshape(1, 3)])


def main():
    parser = argparse.ArgumentParser(description="单帧 3D 拓扑图 — 场景点云+高斯球+关系线")
    parser.add_argument("--frame_npz", required=True)
    parser.add_argument("--scene_ply", default=None)
    parser.add_argument("--class_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_scene_pts", type=int, default=300000)
    args = parser.parse_args()

    from plyfile import PlyData, PlyElement

    # ---- 加载帧数据 ----
    data = np.load(args.frame_npz, allow_pickle=True)
    valid_idx = np.where(data["valid_mask"])[0]
    print(f"帧: {len(data['valid_mask'])} 检测, {len(valid_idx)} 有效物体")

    if len(valid_idx) == 0:
        print("无有效物体"); return

    means = data["means_3d"][valid_idx]
    covs = data["covs_3d"][valid_idx]
    classes = data["classes"][valid_idx]
    pcds_flat = data["pcds_flat"]
    pcds_offsets = data["pcds_offsets"]

    fx = float(data["fx"]); fy = float(data["fy"])
    cx = float(data["cx"]); cy = float(data["cy"])
    R = data["camera_rot"]; t = data["camera_trans"]
    H_img = data["image"].shape[0]; W_img = data["image"].shape[1]

    with open(args.class_json) as f:
        class_names = json.load(f).get("VisualGenome_list", [])
    rel_names = json.load(open(args.class_json)).get("VisualGenome_rel", [])

    N = len(valid_idx)

    # ---- 场景点云 (保持原始颜色) ----
    if args.scene_ply and os.path.exists(args.scene_ply):
        scene = PlyData.read(args.scene_ply)
        sv = scene["vertex"]
        N_scene = len(sv)
        step = max(1, N_scene // args.max_scene_pts)
        si = np.arange(0, N_scene, step)
        scene_xyz = np.stack([sv["x"][si], sv["y"][si], sv["z"][si]], axis=-1)
        scene_r = sv["red"][si].astype(np.uint8)
        scene_g = sv["green"][si].astype(np.uint8)
        scene_b = sv["blue"][si].astype(np.uint8)
        n_scene = len(si)
        print(f"场景: {n_scene:,} 点 (原始色)")
    else:
        # 空场景 (只显示高斯球+连线)
        scene_xyz = np.zeros((0, 3), dtype=np.float32)
        scene_r = scene_g = scene_b = np.zeros(0, dtype=np.uint8)
        n_scene = 0
        print(f"场景: 无 (仅显示高斯球+连线)")

    # ---- 物体: 椭球节点 + 物体点云 ----
    all_xyz = [scene_xyz]
    all_r = [scene_r]; all_g = [scene_g]; all_b = [scene_b]
    all_ptype = [np.zeros(n_scene, dtype=np.int32)]
    all_obj_id = [np.full(n_scene, -1, dtype=np.int32)]

    for i in range(N):
        m = means[i]; c = covs[i]
        cls_id = int(classes[i])
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"?{cls_id}"
        rc, gc, bc = id_to_rgb(i)

        # 椭球 (语义颜色)
        ell = sample_ellipsoid(m, c, n_pts=500)
        ne = len(ell)
        all_xyz.append(ell)
        all_r.append(np.full(ne, rc, dtype=np.uint8))
        all_g.append(np.full(ne, gc, dtype=np.uint8))
        all_b.append(np.full(ne, bc, dtype=np.uint8))
        all_ptype.append(np.full(ne, 1, dtype=np.int32))
        all_obj_id.append(np.full(ne, i, dtype=np.int32))

        print(f"  [{i}] {cls_name:15s} @ ({m[0]:+.2f} {m[1]:+.2f} {m[2]:+.2f})  "
              f"σ=({np.sqrt(c[0,0]):.2f},{np.sqrt(c[1,1]):.2f},{np.sqrt(c[2,2]):.2f})")

    # ---- 关系边 (管状连线) ----
    rels = data["rels"]
    rel_classes = data["rel_classes"]
    rel_valid, edge_xyz_parts = [], []

    if len(rels) > 0:
        # 重新映射到 valid_idx
        global_to_local = np.full(len(data["valid_mask"]), -1, dtype=int)
        global_to_local[valid_idx] = np.arange(N)

        for e, (s, o) in enumerate(rels):
            ls = global_to_local[s] if s < len(global_to_local) else -1
            lo = global_to_local[o] if o < len(global_to_local) else -1
            if ls < 0 or lo < 0 or ls == lo:
                continue

            # 渐变颜色 (subject→object)
            sr_c, sg_c, sb_c = id_to_rgb(ls)
            or_c, og_c, ob_c = id_to_rgb(lo)

            tube = build_tube_edge(means[ls], means[lo])
            if len(tube) == 0:
                continue

            n_tube = len(tube)
            # 渐变着色
            t_vals = np.linspace(0, 1, n_tube // 6).repeat(6)[:n_tube]
            tr = (sr_c * (1 - t_vals) + or_c * t_vals).astype(np.uint8)
            tg = (sg_c * (1 - t_vals) + og_c * t_vals).astype(np.uint8)
            tb = (sb_c * (1 - t_vals) + ob_c * t_vals).astype(np.uint8)

            all_xyz.append(tube)
            all_r.append(tr); all_g.append(tg); all_b.append(tb)
            all_ptype.append(np.full(n_tube, 2, dtype=np.int32))
            all_obj_id.append(np.full(n_tube, -1, dtype=np.int32))

            rc = int(rel_classes[e]) if e < len(rel_classes) else 0
            rn = rel_names[rc] if rc < len(rel_names) else f"rel{rc}"
            rel_valid.append(f"  {ls}({class_names[int(classes[ls])]}) → "
                             f"{lo}({class_names[int(classes[lo])]}) : {rn}")

    n_edges = len(rel_valid)
    print(f"关系: {n_edges} 条")
    for rv in rel_valid[:10]:
        print(rv)

    # ---- 写 PLY ----
    total = sum(len(x) for x in all_xyz)
    vertex = np.zeros(total, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("point_type", "i4"), ("object_id", "i4"),
    ])

    off = 0
    for xyz, r, g, b, pt, oid in zip(all_xyz, all_r, all_g, all_b, all_ptype, all_obj_id):
        n = len(xyz)
        if n == 0: continue
        s = slice(off, off + n)
        vertex["x"][s], vertex["y"][s], vertex["z"][s] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vertex["red"][s], vertex["green"][s], vertex["blue"][s] = r[:n], g[:n], b[:n]
        vertex["point_type"][s] = pt[:n]
        vertex["object_id"][s] = oid[:n]
        off += n

    path = f"{args.output}_frame000.ply"
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

    print(f"\n→ {path}  ({off:,} 点)")
    print(f"  point_type: 0=场景(原色) 1=高斯椭球 2=关系连线")
    print(f"  物体={N}  关系={n_edges}  场景={n_scene:,}")


if __name__ == "__main__":
    main()
