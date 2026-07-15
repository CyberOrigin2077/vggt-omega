"""
DeWorldSG 深度感知 3D 高斯估计核心模块。
参考: DeWorldSG (ECCV 2026) Sec 4.1-4.2, 排除 JEPA/世界模型部分。

包含:
  - 双域深度精炼 (Dual-Domain Depth Refinement, DR)
  - 从点云直接估计 3D 高斯 (替代 FROSS 的单像素+雅可比)
  - 双重合并标准 (Hellinger距离 + 类别概率内积)
  - 几何邻居配对过滤
"""

import numpy as np
from scipy.ndimage import median_filter


# ============================================================
# 双域深度精炼 (Dual-Domain Refinement)
# ============================================================

def spatial_depth_filter(depth, mask, tau=0.05, eps=1e-3):
    """
    空间域过滤: 每个mask内像素与3×3邻域中位数比较, 偏差过大则剔除。

    Eq.(1): |D(p) - median_3x3(p)| < τ·median_3x3(p) + ε

    Args:
        depth: [H, W] float32 深度图 (米)
        mask: [H, W] bool 实例掩码
        tau: 相对偏差容忍度 (论文默认 0.05)
        eps: 绝对偏差容忍度 (论文默认 1e-3)

    Returns:
        spatial_mask: [H, W] bool 空间域过滤后的掩码
    """
    depth_median = median_filter(depth, size=3, mode="nearest")
    deviation = np.abs(depth - depth_median)
    threshold = tau * depth_median + eps

    spatial_mask = mask.copy()
    spatial_mask[mask] = deviation[mask] < threshold[mask]
    return spatial_mask


def distribution_depth_filter(depth_values, gamma_base=0.03, alpha=10.0, beta=0.02):
    """
    深度分布域过滤: 1D聚类, 保留最靠近参考深度的簇。

    Eq.(2): Δ_thr = max(γ_base, α·median(Δz), β·d_ref)

    Args:
        depth_values: [N] 空间域过滤后的深度值 (米)
        gamma_base: 最小分离距离 (米), 论文默认 0.03
        alpha: 局部深度变化缩放系数, 论文默认 10.0
        beta: 全局深度尺度缩放系数, 论文默认 0.02

    Returns:
        inlier_mask: [N] bool, True=保留
    """
    N = len(depth_values)
    if N < 5:
        return np.ones(N, dtype=bool)

    sorted_depth = np.sort(depth_values)
    d_ref = np.median(sorted_depth)

    # 相邻深度差
    adj_diff = np.diff(sorted_depth)
    median_adj = float(np.median(adj_diff)) if len(adj_diff) > 0 else 0.01

    # 动态聚类阈值 Eq.(2)
    delta_thr = max(gamma_base, alpha * median_adj, beta * d_ref)

    # 按阈值分割成簇
    split_points = np.where(adj_diff > delta_thr)[0] + 1
    clusters = np.split(sorted_depth, split_points)

    if len(clusters) == 0:
        return np.ones(N, dtype=bool)

    # 找最靠近参考深度的簇
    cluster_means = np.array([c.mean() for c in clusters])
    best_idx = int(np.argmin(np.abs(cluster_means - d_ref)))

    # 保留该簇范围的深度值
    best_mean = float(cluster_means[best_idx])
    lower = best_mean - delta_thr
    upper = best_mean + delta_thr
    inlier_mask = (depth_values >= lower) & (depth_values <= upper)

    return inlier_mask


def dual_domain_refinement(depth, mask, tau=0.05, eps=1e-3,
                           gamma_base=0.03, alpha=10.0, beta=0.02,
                           min_valid_pixels=10):
    """
    双域深度精炼: 空间域 → 深度分布域, 顺序执行。

    Args:
        depth: [H, W] float32 深度图 (米)
        mask: [H, W] bool 初始实例掩码
        tau, eps: 空间域参数
        gamma_base, alpha, beta: 深度分布域参数
        min_valid_pixels: 最少有效像素 (< 此值视为无效)

    Returns:
        refined_mask: [H, W] bool 精炼后的掩码
        valid: bool 是否有足够有效像素
    """
    # Step 1: 空间域过滤
    spatial_mask = spatial_depth_filter(depth, mask, tau, eps)

    n_spatial = int(np.sum(spatial_mask))
    if n_spatial < min_valid_pixels:
        return spatial_mask, False

    # Step 2: 提取空间域过滤后的深度值
    ys, xs = np.where(spatial_mask)
    depth_vals = depth[ys, xs]

    # Step 3: 深度分布域过滤
    dist_inlier = distribution_depth_filter(depth_vals, gamma_base, alpha, beta)

    final_ys = ys[dist_inlier]
    final_xs = xs[dist_inlier]

    refined_mask = np.zeros_like(mask, dtype=bool)
    refined_mask[final_ys, final_xs] = True

    return refined_mask, int(np.sum(refined_mask)) >= min_valid_pixels


# ============================================================
# 3D 高斯估计 (替代 FROSS 的单像素 + 雅可比传播)
# ============================================================

def unproject_depth_to_3d(depth, mask, fx, fy, cx, cy, R, t):
    """
    将掩码内所有深度像素反投影到世界坐标系 (VGGT 外参约定)。

    VGGT 外参 [R|t] 是世界→相机: P_cam = R·P_world + t
    因此世界坐标: P_world = R^T·(P_cam - t)

    在 numpy row-vector 形式下: world_pts = (camera_pts - t) @ R

    Args:
        depth: [H, W] float32 深度图 (米)
        mask: [H, W] bool
        fx, fy, cx, cy: 相机内参
        R: [3, 3] 旋转矩阵 (world→camera)
        t: [3] 平移向量 (world→camera)

    Returns:
        world_pts: [N, 3] float32 世界坐标点
    """
    ys, xs = np.where(mask)
    N = len(ys)
    if N == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[ys, xs]  # [N]

    # 像素 → 相机坐标
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    camera_pts = np.stack([x_cam, y_cam, z], axis=-1)  # [N, 3]

    # 相机坐标 → 世界坐标 (VGGT 约定)
    # P_world = R^T @ (P_camera - t)  → row: (camera_pts - t) @ R
    world_pts = (camera_pts - t) @ R

    return world_pts.astype(np.float32)


def estimate_3d_gaussian(depth, mask, fx, fy, cx, cy, R, t,
                         eps=1e-4, min_points=10):
    """
    从精炼后的深度掩码直接估计 3D 高斯分布 (样本均值和协方差)。

    Eq.(4):
      μ = mean(P_final)
      Σ = cov(P_final) + εI

    相比 FROSS 的单像素+雅可比:
      - 利用掩码内全部有效像素, 位置更稳定
      - 协方差直接反映物体3D几何形状, 而非2D bbox的均匀分布假设

    Args:
        depth: [H, W] float32 深度图
        mask: [H, W] bool 精炼后的掩码
        fx, fy, cx, cy: 内参
        R: [3,3], t: [3] 外参
        eps: 协方差正则化系数
        min_points: 最少点数

    Returns:
        mean_3d: [3] float32
        cov_3d: [3,3] float32
        points_3d: [N,3] float32 世界坐标点 (用于后续合并时的加权)
        valid: bool
    """
    points_3d = unproject_depth_to_3d(depth, mask, fx, fy, cx, cy, R, t)
    N = len(points_3d)

    if N < min_points:
        return (np.zeros(3, dtype=np.float32),
                np.eye(3, dtype=np.float32) * 0.01,
                points_3d, False)

    mean_3d = np.mean(points_3d, axis=0).astype(np.float32)
    cov_3d = np.cov(points_3d.T).astype(np.float32)
    cov_3d += np.eye(3, dtype=np.float32) * eps

    return mean_3d, cov_3d, points_3d, True


def full_instance_pipeline(depth, bboxes, fx, fy, cx, cy, R, t,
                           dr_params=None, min_valid_pixels=10):
    """
    一帧所有bbox的完整深度感知3D高斯估计管线。

    流程: bbox掩码 → 双域DR → 3D高斯估计

    Args:
        depth: [H, W] float32 深度图 (米)
        bboxes: [K, 4] int (cx, cy, w, h)
        fx, fy, cx, cy: 内参
        R: [3,3], t: [3] 外参
        dr_params: dict or None
        min_valid_pixels: 最少有效像素

    Returns:
        means:     [K, 3] float32
        covs:      [K, 3, 3] float32
        pcds:      list of [N_k, 3] float32
        valid:     [K] bool
    """
    if dr_params is None:
        dr_params = {}

    K = len(bboxes)
    H, W = depth.shape

    means = np.zeros((K, 3), dtype=np.float32)
    covs = np.tile(np.eye(3, dtype=np.float32) * 0.01, (K, 1, 1))
    pcds = [np.zeros((0, 3), dtype=np.float32)] * K
    valid = np.zeros(K, dtype=bool)

    for k in range(K):
        cx_b, cy_b, bw, bh = bboxes[k]

        # bbox → 初始掩码
        x1 = max(0, int(cx_b - bw // 2))
        y1 = max(0, int(cy_b - bh // 2))
        x2 = min(W, int(cx_b + bw // 2))
        y2 = min(H, int(cy_b + bh // 2))

        if x2 <= x1 or y2 <= y1:
            continue

        init_mask = np.zeros((H, W), dtype=bool)
        init_mask[y1:y2, x1:x2] = True
        # 过滤无效深度
        init_mask = init_mask & (depth > 1e-6)

        if init_mask.sum() < min_valid_pixels:
            continue

        # 双域深度精炼
        refined_mask, ok = dual_domain_refinement(
            depth, init_mask, min_valid_pixels=min_valid_pixels, **dr_params)

        if not ok:
            continue

        # 3D高斯估计
        mean_3d, cov_3d, pts_3d, v = estimate_3d_gaussian(
            depth, refined_mask, fx, fy, cx, cy, R, t,
            min_points=min_valid_pixels)

        if v:
            means[k] = mean_3d
            covs[k] = cov_3d
            pcds[k] = pts_3d
            valid[k] = True

    return means, covs, pcds, valid


# ============================================================
# 双重合并标准 (DeWorldSG Sec 4.2)
# ============================================================

def class_prob_distance(p_i, p_j):
    """
    类别概率内积距离 Eq.(5):
    D_c(i,j) = 1 - Σ_k p_i(k)·p_j(k)

    当两个节点的概率质量集中在相同类别时, 距离趋近于0。
    这比 FROSS 原始 hard-label 比较更细粒度。

    Args:
        p_i: [num_classes] 或 [1] (hard label) 类别概率
        p_j: 同上

    Returns:
        float, ∈ [0, 1]
    """
    if p_i.ndim == 0 or p_i.shape[0] == 1:
        # hard label → one-hot → 内积=1 if same else 0
        return 0.0 if int(p_i) == int(p_j) else 1.0
    return float(1.0 - np.dot(p_i, p_j))


def _batched_hellinger_distance(mean1, cov1, mean2, cov2):
    """
    批量计算 Hellinger 距离 (复制自 FROSS utils.py, 保持一致)。

    给定一个高斯与一组高斯的比较。
    """
    if mean2.ndim == 1:
        mean2 = mean2[None, :]
    if cov2.ndim == 2:
        cov2 = cov2[None, :, :]

    mean1 = mean1[None, :, None]      # [1, 3, 1]
    mean2 = mean2[..., None]          # [M, 3, 1]
    cov1 = cov1[None, ...]            # [1, 3, 3]
    mean_diff = mean1 - mean2          # [M, 3, 1]
    cov_mean = (cov1 + cov2) / 2.0
    cov_mean_inv = np.linalg.inv(cov_mean)

    det_cov_mean = np.linalg.det(cov_mean)
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)

    # 避免数值下溢
    safe_sqrt = np.maximum(det_cov1 * det_cov2, 1e-30)
    B_D = (0.125 * mean_diff.transpose(0, 2, 1) @ cov_mean_inv @ mean_diff).flatten() \
        + 0.5 * np.log(np.maximum(det_cov_mean / np.sqrt(safe_sqrt), 1e-12))

    return np.sqrt(np.maximum(1.0 - np.exp(-B_D), 0.0))


def should_merge(mean_i, cov_i, class_i, mean_j, cov_j, class_j,
                 delta_g=0.7, delta_c=0.8):
    """
    双重合并标准 Eq.(6):
      HD(i,j) < δ_g  AND  D_c(i,j) < δ_c

    Args:
        delta_g: Hellinger距离阈值 (论文默认 0.7, 比FROSS的0.85更严格)
        delta_c: 类别距离阈值 (论文默认 0.8)

    Returns:
        (should_merge: bool, hd: float, cd: float)
    """
    hd = float(_batched_hellinger_distance(mean_i, cov_i, mean_j, cov_j)[0])
    cd = class_prob_distance(class_i, class_j)
    return (hd < delta_g and cd < delta_c), hd, cd


def merge_two_gaussians(mu_i, cov_i, w_i, mu_j, cov_j, w_j):
    """
    加权合并两个高斯 Eq.(7):
      μ_k = (w_i·μ_i + w_j·μ_j) / (w_i + w_j)
      Σ_k = (w_i·Σ_i + w_j·Σ_j)/(w_i+w_j)
            + w_i·w_j/(w_i+w_j)² · (μ_i-μ_j)(μ_i-μ_j)ᵀ

    Args:
        w_i, w_j: 观测频次 (点数)

    Returns:
        mu_k: [3], cov_k: [3,3]
    """
    w_total = w_i + w_j
    mu_k = (w_i * mu_i + w_j * mu_j) / w_total
    diff = (mu_i - mu_j).reshape(-1, 1)
    cov_k = (w_i * cov_i + w_j * cov_j) / w_total \
            + (w_i * w_j / (w_total ** 2)) * (diff @ diff.T)
    return mu_k.astype(np.float32), cov_k.astype(np.float32)


# ============================================================
# 几何邻居配对 (DeWorldSG Eq.8, 用于过滤不合理的关系候选)
# ============================================================

def geometric_neighbor_cost(bbox_i, bbox_j, z_i, z_j, img_diag):
    """
    Eq.(8): GeoCost(i,j) = 0.5·d_2D(i,j)/d_diag + |z_i - z_j|

    只对 GeoCost < 0.45 的 pairs 保留关系。

    Args:
        bbox_i, bbox_j: (cx, cy, w, h) int
        z_i, z_j: 物体深度 (米)
        img_diag: 图像对角线长度 sqrt(W²+H²)

    Returns:
        float
    """
    ci = np.array([bbox_i[0], bbox_i[1]], dtype=np.float64)
    cj = np.array([bbox_j[0], bbox_j[1]], dtype=np.float64)
    d_2d = np.linalg.norm(ci - cj)
    d_depth = abs(z_i - z_j)
    return 0.5 * d_2d / img_diag + d_depth


def filter_relation_pairs(bboxes, depth_vals, rels, img_h, img_w,
                          geo_thresh=0.45):
    """
    用几何代价过滤不合理的关系对。

    Args:
        bboxes: [K, 4] (cx, cy, w, h)
        depth_vals: [K] 每个物体的深度值
        rels: [E, 2] (subject, object) 索引
        img_h, img_w: 图像尺寸
        geo_thresh: 代价阈值

    Returns:
        valid_rels: [E] bool
    """
    if len(rels) == 0:
        return np.zeros(0, dtype=bool)

    img_diag = np.sqrt(img_h ** 2 + img_w ** 2)
    valid = np.zeros(len(rels), dtype=bool)

    for e, (s, o) in enumerate(rels):
        if s >= len(bboxes) or o >= len(bboxes):
            continue
        cost = geometric_neighbor_cost(
            bboxes[s], bboxes[o], depth_vals[s], depth_vals[o], img_diag)
        if cost < geo_thresh:
            valid[e] = True

    return valid


# ============================================================
# 时序关系累积 (DeWorldSG Eq.9, 跨帧累加logits)
# ============================================================

class TemporalRelationAccumulator:
    """
    跨帧累积关系预测logits, 软化单帧噪声。

    Eq.(9): P_vis(r|i,j) = Softmax(Σ_t ℓ_t(r|i,j))

    同时按 pair 追踪不确定度 (熵), 供后续融合使用。
    """

    def __init__(self, num_rel_classes, num_obj_classes, max_objects=2000):
        self.num_rel_classes = num_rel_classes
        self.num_obj_classes = num_obj_classes
        # 按 (class_s, class_o) 累积logits
        self._logits = {}     # (cls_s, cls_o) → np.array [num_rel_classes]
        # 按 specific pair (i, j) 累积logits (global idx)
        self._pair_logits = {}  # (global_i, global_j) → np.array [num_rel_classes]
        self._pair_counts = {}  # (global_i, global_j) → int (观测帧数)

    def accumulate(self, global_classes, rels, rel_classes, num_classes):
        """
        累积一帧的关系证据。

        Args:
            global_classes: [K_g] 全局图中的类别 (int)
            rels: [E, 2] (subject, object) 全局索引
            rel_classes: [E] 关系类别
            num_classes: 总类别数
        """
        for idx, (s, o) in enumerate(rels):
            r = int(rel_classes[idx])

            # 按具体 pair 累积
            key = (int(s), int(o))
            if key not in self._pair_logits:
                self._pair_logits[key] = np.zeros(self.num_rel_classes)
            self._pair_logits[key][r] += 1.0
            self._pair_counts[key] = self._pair_counts.get(key, 0) + 1

            # 按类别对累积
            cs = int(global_classes[s]) if s < len(global_classes) else -1
            co = int(global_classes[o]) if o < len(global_classes) else -1
            if cs >= 0 and co >= 0:
                cls_key = (cs, co)
                if cls_key not in self._logits:
                    self._logits[cls_key] = np.zeros(self.num_rel_classes)
                self._logits[cls_key][r] += 1.0

    def get_pair_distribution(self, i, j):
        """获取 pair (i,j) 的累积关系分布 P_vis(r|i,j) (Eq.9)"""
        key = (int(i), int(j))
        if key not in self._pair_logits:
            return None
        logits = self._pair_logits[key]
        if logits.sum() == 0:
            return None
        # softmax
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        return exp_logits / exp_logits.sum()

    def get_pair_uncertainty(self, i, j):
        """计算 pair 的不确定度 (熵), 用于判断是否需要世界模型先验"""
        dist = self.get_pair_distribution(i, j)
        if dist is None:
            return float("inf")
        # 避免 log(0)
        dist_safe = np.clip(dist, 1e-12, 1.0)
        return float(-np.sum(dist_safe * np.log(dist_safe)))

    def get_class_pair_distribution(self, cs, co):
        """按类别对获取累积分布"""
        key = (int(cs), int(co))
        if key not in self._logits:
            return None
        logits = self._logits[key]
        if logits.sum() == 0:
            return None
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        return exp_logits / exp_logits.sum()
