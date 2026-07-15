"""
FROSS 输出可视化：pkl → 图结构 PLY + 场景叠加 PLY + 文本报告。

输出:
  - {output}_graph.ply     节点(椭球,按3D高斯缩放) + 边(管状连线)
  - {output}_objects.ply   每物体点云 (不同颜色)
  - {output}_scene.ply     场景点云 + 图叠加 (需 --scene_ply)
  - {output}_report.txt    文本报告

用法 (fross 环境):
    python splatv/visualize_fross.py \
        --pkl output_fross/lab0/predictions_gaussian_*.pkl \
        --class_json Datasets/lab0/ReplicaSSG/replica_to_visual_genome.json \
        --scene_ply output_semantic_pcd/lab0/lab0_semantic.ply \
        --output output_fross/lab0/lab0
"""

import argparse, os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import json, pickle
import numpy as np
from plyfile import PlyData, PlyElement


def id_to_rgb(uid):
    if uid < 0: return (60, 60, 70)
    h = (int(uid) * 2654435761) & 0xFFFFFFFFFFFFFFFF
    return (max(int(h & 0xFF), 40),
            max(int((h >> 8) & 0xFF), 40),
            max(int((h >> 16) & 0xFF), 40))


def sample_ellipsoid(mean, cov, n_pts=300, base_radius=0.04):
    """根据 3D 高斯协方差采样椭球面。

    协方差矩阵的特征值开方 → 各轴半径, 特征向量 → 朝向。
    base_radius 是最小球体半径 (避免物体太小看不见)。
    """
    # 特征分解
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    eigenvals = np.maximum(eigenvals, 1e-8)
    radii = np.sqrt(eigenvals)
    # 保证最小半径
    radii = np.maximum(radii, base_radius)

    # Fibonacci 球面 (单位球)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(n_pts)
    y = 1.0 - (idx / (n_pts - 1)) * 2.0
    r = np.sqrt(1.0 - y * y)
    theta = phi * idx
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    unit_sphere = np.stack([x, y, z], axis=1)  # [N, 3]

    # 变换: unit_sphere @ diag(radii) @ eigenvecs.T + mean
    ellipsoid = unit_sphere * radii  # 缩放
    ellipsoid = ellipsoid @ eigenvecs.T  # 旋转
    ellipsoid = ellipsoid + mean  # 平移
    return ellipsoid


def write_object_ply(pcds, classes, class_names, filepath):
    """每物体不同颜色 → PLY"""
    all_xyz, all_rgb, all_obj_id = [], [], []
    for i, pts in enumerate(pcds):
        if pts is None or len(pts) == 0:
            continue
        r, g, b = id_to_rgb(i)
        all_xyz.append(pts)
        all_rgb.append(np.full((len(pts), 3), [r, g, b], dtype=np.uint8))
        all_obj_id.append(np.full(len(pts), i, dtype=np.int32))

    if not all_xyz:
        print("WARNING: 物体点云为空")
        return

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    obj_id = np.concatenate(all_obj_id, axis=0)

    vertex = np.zeros(len(xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("object_id", "i4"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertex["object_id"] = obj_id
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(filepath)
    print(f"  物体 PLY: {filepath} ({len(xyz):,} 点, {len(pcds)} 物体)")


def write_graph_ply(means, covs, classes, edge_index, edge_cls, filepath):
    """节点=3D高斯椭球, 边=管状连线 → PLY"""
    N = len(means)
    if N == 0:
        print(f"  [WARN] 0 物体, 跳过 graph PLY 写入")
        return
    E = edge_index.shape[1]

    node_xyz_parts, node_rgb_parts, node_id_parts = [], [], []

    for i in range(N):
        # 根据协方差生成椭球
        pts = sample_ellipsoid(means[i], covs[i], n_pts=300)
        r, g, b = id_to_rgb(i)
        node_xyz_parts.append(pts)
        node_rgb_parts.append(np.full((len(pts), 3), [r, g, b], dtype=np.uint8))
        node_id_parts.append(np.full(len(pts), i, dtype=np.int32))

    node_xyz = np.concatenate(node_xyz_parts, axis=0)
    node_rgb = np.concatenate(node_rgb_parts, axis=0)
    node_id = np.concatenate(node_id_parts, axis=0)

    # ---- 边: 管状加粗 ----
    edge_xyz, edge_rgb = [], []
    for e in range(E):
        s, o = edge_index[0, e], edge_index[1, e]
        if s >= N or o >= N: continue
        rel_probs = edge_cls[e]
        top_rel = rel_probs.argmax()
        conf = float(rel_probs[top_rel])

        start = means[s]
        end = means[o]
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length

        sr, sg, sb = id_to_rgb(s)
        er, eg, eb = id_to_rgb(o)

        n_segments = max(int(length / 0.02), 5)
        n_ring = 6
        ring_radius = 0.012

        if abs(direction[0]) < 0.9:
            perp1 = np.cross(direction, [1, 0, 0])
        else:
            perp1 = np.cross(direction, [0, 1, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)

        for seg in range(n_segments + 1):
            t = seg / n_segments
            center = start + t * (end - start)
            for a in range(n_ring):
                angle = 2 * np.pi * a / n_ring
                pt = center + ring_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                edge_xyz.append(pt)
                r = int(sr * (1 - t) + er * t)
                g = int(sg * (1 - t) + eg * t)
                b = int(sb * (1 - t) + eb * t)
                edge_rgb.append([r, g, b])

    if edge_xyz:
        edge_xyz_np = np.array(edge_xyz, dtype=np.float32)
        edge_rgb_np = np.array(edge_rgb, dtype=np.uint8)
    else:
        edge_xyz_np = np.zeros((0, 3), dtype=np.float32)
        edge_rgb_np = np.zeros((0, 3), dtype=np.uint8)

    # 合并
    all_xyz = np.concatenate([node_xyz, edge_xyz_np], axis=0)
    all_rgb = np.concatenate([node_rgb, edge_rgb_np], axis=0)
    is_node = np.concatenate([
        np.ones(len(node_xyz), dtype=np.int32),
        np.zeros(len(edge_xyz_np), dtype=np.int32)
    ])
    all_node_id = np.concatenate([
        node_id,
        np.full(len(edge_xyz_np), -1, dtype=np.int32)
    ])

    vertex = np.zeros(len(all_xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("is_node", "i4"),
        ("node_id", "i4"),
    ])
    vertex["x"], vertex["y"], vertex["z"] = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = all_rgb[:, 0], all_rgb[:, 1], all_rgb[:, 2]
    vertex["is_node"] = is_node
    vertex["node_id"] = all_node_id
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(filepath)
    print(f"  图结构 PLY: {filepath} ({N} 椭球节点, {E} 管状边, {len(all_xyz):,} 点)")


def write_scene_ply(scene_ply_path, graph_ply_path, output_path, max_scene_pts=250000):
    """合并场景点云 + 图结构 → 单一 PLY。
    两者均保持 VGGT 原始世界坐标系，无需额外对齐。
    """
    scene = PlyData.read(scene_ply_path)
    sv = scene["vertex"]
    N = len(sv)
    step = max(1, N // max_scene_pts)
    idx = np.arange(0, N, step)
    s_xyz = np.stack([sv["x"][idx], sv["y"][idx], sv["z"][idx]], axis=-1)
    s_r = sv["red"][idx].astype(np.uint8)
    s_g = sv["green"][idx].astype(np.uint8)
    s_b = sv["blue"][idx].astype(np.uint8)
    n_scene = len(s_xyz)

    graph = PlyData.read(graph_ply_path)
    gv = graph["vertex"]
    n_graph = len(gv)

    N_total = n_scene + n_graph
    vertex = np.zeros(N_total, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("point_type", "i4"),
        ("node_id", "i4"),
    ])
    # 场景
    vertex["x"][:n_scene], vertex["y"][:n_scene], vertex["z"][:n_scene] = (
        s_xyz[:, 0], s_xyz[:, 1], s_xyz[:, 2])
    vertex["red"][:n_scene], vertex["green"][:n_scene], vertex["blue"][:n_scene] = (
        s_r, s_g, s_b)
    vertex["point_type"][:n_scene] = 0
    vertex["node_id"][:n_scene] = -1

    # 图 (VGGT 原始坐标, 与场景一致)
    vertex["x"][n_scene:] = gv["x"]
    vertex["y"][n_scene:] = gv["y"]
    vertex["z"][n_scene:] = gv["z"]
    vertex["red"][n_scene:] = gv["red"]
    vertex["green"][n_scene:] = gv["green"]
    vertex["blue"][n_scene:] = gv["blue"]
    vertex["point_type"][n_scene:] = np.where(gv["is_node"] == 1, 1, 2)
    vertex["node_id"][n_scene:] = gv["node_id"]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(output_path)
    print(f"  场景+图 PLY: {output_path} ({n_scene:,} 场景 + {n_graph:,} 图)")


def write_report(classes, class_names, means, covs, edge_index, edge_cls, rel_names, filepath):
    lines = []
    lines.append("=" * 60)
    lines.append("FROSS 3D Scene Graph 报告")
    lines.append("=" * 60)

    N = len(classes)
    lines.append(f"\n物体 ({N} 个):")
    lines.append(f"{'ID':<4} {'类别':<20} {'中心 (x,y,z)':<30} {'尺寸 (宽×深×高,m)'}")
    lines.append("-" * 75)
    for i in range(N):
        cls_id = classes[i].argmax() if classes[i].ndim == 1 else classes[i]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        m = means[i]
        c = covs[i]
        extent = np.sqrt(np.maximum(np.diag(c), 0)) * 2  # 2-sigma
        lines.append(f"{i:<4} {cls_name:<20} ({m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f})     "
                     f"{extent[0]:.2f}×{extent[1]:.2f}×{extent[2]:.2f}")

    E = edge_index.shape[1]
    lines.append(f"\n关系 ({E} 条):")
    lines.append(f"{'主体':<4} → {'客体':<4} {'关系':<20} {'置信度'}")
    lines.append("-" * 50)
    for e in range(E):
        s, o = edge_index[0, e], edge_index[1, e]
        probs = edge_cls[e]
        top_k = probs.argsort()[-3:][::-1]
        for tk in top_k:
            if probs[tk] < 0.1: continue
            rel_name = rel_names[tk] if tk < len(rel_names) else f"rel_{tk}"
            lines.append(f"{s:<4} → {o:<4} {rel_name:<20} ({probs[tk]:.2f})")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"  报告: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="FROSS 输出可视化")
    parser.add_argument("--pkl", required=True, help="FROSS predictions .pkl")
    parser.add_argument("--class_json", required=True, help="class mapping JSON")
    parser.add_argument("--scene_ply", default=None, help="场景点云 PLY (合并用)")
    parser.add_argument("--output", required=True, help="输出前缀")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        pred = pickle.load(f)
    with open(args.class_json) as f:
        class_map = json.load(f)
    class_names = class_map.get("VisualGenome_list", [])
    rel_names = class_map.get("VisualGenome_rel", [])

    for scan_id, data in pred.items():
        print(f"场景: {scan_id}")

        classes = data["cls"]
        means = data["mean"]
        covs = data["cov"]
        pcds = data["pcd"]
        edge_index = data["edge_index"]
        edge_cls = data["edge_cls"]
        N, E = len(classes), edge_index.shape[1]
        print(f"  {N} 物体, {E} 关系")

        # 1. 物体点云
        write_object_ply(pcds, classes, class_names, f"{args.output}_objects.ply")

        # 2. 图结构 (椭球节点 + 管状边)
        write_graph_ply(means, covs, classes, edge_index, edge_cls,
                        f"{args.output}_graph.ply")

        # 3. 场景+图合并
        graph_ply = f"{args.output}_graph.ply"
        if args.scene_ply and os.path.exists(args.scene_ply) and os.path.exists(graph_ply):
            write_scene_ply(args.scene_ply, graph_ply,
                            f"{args.output}_scene.ply")

        # 4. 报告
        write_report(classes, class_names, means, covs, edge_index, edge_cls,
                     rel_names, f"{args.output}_report.txt")


if __name__ == "__main__":
    main()
