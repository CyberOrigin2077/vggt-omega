"""
SceneSplat 语义管线一键脚本。
输入 .splatv → 输出带语义 ID (slot 8) 的 .splatv。

用法 (scene_splat 环境):
    # 实例分割 (OverKMeans, 自动确定聚类数)
    PYTHONPATH=../third_party/SceneSplat python splatv/run_scenesplat.py \
        --input output_static/room0/room0_static.splatv \
        --output output_static/room0/room0_scenesplat.splatv

    # CLIP 语义模式
    PYTHONPATH=../third_party/SceneSplat python splatv/run_scenesplat.py \
        --input ... --output ... --use_clip \
        --prompts "wall,floor,chair,table"
"""

import argparse, json, os, struct, subprocess, sys, time
import numpy as np
import torch, torch.nn.functional as F

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ss_root = os.path.join(_proj_root, "third_party", "SceneSplat")
C0 = 0.28209479177387814

# ═══════════════════════════ splaTV ↔ numpy ═══════════════════════════

def read_splatv(filepath):
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x674B
        json_len = struct.unpack("<I", f.read(4))[0]
        metadata = json.loads(f.read(json_len))
        td = np.frombuffer(f.read(), dtype=np.uint32).copy()
    tw, th = metadata[0]["texwidth"], metadata[0]["texheight"]
    N = (tw * th) // 4
    tf, tu = td.view(np.float32), td.view(np.uint8)
    x, c = np.zeros((N, 3), dtype=np.float32), np.zeros((N, 3), dtype=np.float32)
    o, s = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)
    for j in range(N):
        x[j, 0] = -tf[16 * j + 0]; x[j, 1] = tf[16 * j + 1]; x[j, 2] = tf[16 * j + 2]
        c[j, 0] = tu[4 * (16 * j + 7) + 0] / 255.0
        c[j, 1] = tu[4 * (16 * j + 7) + 1] / 255.0
        c[j, 2] = tu[4 * (16 * j + 7) + 2] / 255.0
        o[j] = tu[4 * (16 * j + 7) + 3] / 255.0
        raw = int(td[16 * j + 5]) & 0xFFFF
        s[j] = struct.unpack('<e', struct.pack('<H', raw))[0]
    return x, c, o, s, metadata, td


def write_splatv(filepath, td, metadata, labels):
    N = (metadata[0]["texwidth"] * metadata[0]["texheight"]) // 4
    for j in range(N): td[16 * j + 8] = 0
    for j in range(min(N, len(labels))): td[16 * j + 8] = int(labels[j])
    jb = json.dumps(metadata, separators=(',', ':')).encode("utf-8")
    with open(filepath, "wb") as f:
        f.write(struct.pack("<I", 0x674B))
        f.write(struct.pack("<I", len(jb))); f.write(jb)
        f.write(td.tobytes())
    nz = sum(1 for j in range(N) if td[16 * j + 8] > 0)
    print(f"写 {N:,} 高斯, {nz:,} 有语义 → {filepath}")

# ═══════════════════════════ PLY ═══════════════════════════

def export_ply(xyz, rgb, opacity, scale, filepath):
    from plyfile import PlyData, PlyElement
    N = len(xyz)
    lo = np.log(np.clip(opacity, 1e-6, 1 - 1e-6) /
                (1 - np.clip(opacity, 1e-6, 1 - 1e-6))).astype(np.float32)
    ls = np.log(np.clip(scale, 1e-8, None)).astype(np.float32)
    sh = ((rgb - 0.5) / C0).astype(np.float32)
    q = np.zeros((N, 4), dtype=np.float32); q[:, 0] = 1.0
    v = np.zeros(N, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v["f_dc_0"], v["f_dc_1"], v["f_dc_2"] = sh[:, 0], sh[:, 1], sh[:, 2]
    v["opacity"] = lo
    v["scale_0"] = ls
    v["scale_1"] = ls
    v["scale_2"] = ls
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(filepath)
    print(f"  写 PLY: {filepath} ({N:,} 高斯)")

# ═══════════════════════════ SceneSplat 子进程 ═══════════════════════════

def run_preprocess(ply_path, npy_dir):
    print("B2: PLY → NPY ...")
    subprocess.run([sys.executable, "-m", "scripts.preprocess_gs",
                    "--input", os.path.abspath(ply_path),
                    "--output", os.path.abspath(npy_dir)],
                   cwd=_ss_root, check=True)

def run_inference(npy_dir, feat_dir, ckpt):
    print("B3: NPY → 特征 ...")
    subprocess.run([sys.executable, "-m", "tools.lang_inference",
                    "--config", "configs/inference/lang-pretrain-pt-v3m1-3dgs.py",
                    "--checkpoint", os.path.abspath(ckpt),
                    "--input-root", os.path.abspath(npy_dir),
                    "--output-dir", os.path.abspath(feat_dir)],
                   cwd=_ss_root, check=True)

# ═══════════════════════════ B4: 标签 ═══════════════════════════

def _load_features(feat_dir, device):
    fs = sorted(f for f in os.listdir(feat_dir) if f.endswith("_feat.pt"))
    if not fs: raise FileNotFoundError(f"未在 {feat_dir} 找到 *_feat.pt")
    fp = os.path.join(feat_dir, fs[0])
    print(f"B4: 加载特征 {fp}")
    feat = torch.load(fp, map_location="cpu", weights_only=True)
    if isinstance(feat, dict): feat = feat["feat"]
    feat = feat.float().to(device); N, D = feat.shape
    print(f"  {N:,} 高斯, {D}D")
    return F.normalize(feat, dim=-1), N, D


def cluster_overkmeans(feat_dir, device, over_k=300, merge_thresh=0.85, sample=300_000):
    """过分割 + 质心合并 → 自动确定聚类数。"""
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    features, N, D = _load_features(feat_dir, device)
    fn = features.cpu().numpy()
    pdim = min(64, D)
    print(f"  OverKMeans: PCA {D}D→{pdim}D, k={over_k}, merge>{merge_thresh}")

    pca = PCA(n_components=pdim, random_state=42)
    if N > sample:
        pca.fit(fn[np.random.choice(N, sample, replace=False)])
    else:
        pca.fit(fn)
    fp = pca.transform(fn).astype(np.float32)
    del fn, features

    km = MiniBatchKMeans(n_clusters=over_k, random_state=42,
                         batch_size=10000, n_init=3, max_iter=100)
    raw = km.fit_predict(fp)
    cents = normalize(km.cluster_centers_.astype(np.float32))
    sim = np.dot(cents, cents.T); np.fill_diagonal(sim, 0)

    # Union-Find 合并
    K = over_k; parent = list(range(K))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py

    pairs = np.argwhere(sim > merge_thresh)
    for pi, pj in pairs[np.argsort(-sim[pairs[:, 0], pairs[:, 1]])]:
        union(pi, pj)

    r2n = {}; nl = np.zeros(K, dtype=np.int32)
    for i in range(K):
        r = find(i)
        if r not in r2n: r2n[r] = len(r2n)
        nl[i] = r2n[r]
    mapped = nl[raw] + 1

    # 合并小簇
    uniq, cnts = np.unique(mapped, return_counts=True)
    mn = max(100, N // 5000)
    small = uniq[cnts < mn]; big = uniq[cnts >= mn]
    if len(small) > 0 and len(big) > 0:
        bm = normalize(np.vstack([fp[mapped == b].mean(axis=0) for b in big]))
        for sc in small:
            msk = mapped == sc
            if not msk.any(): continue
            sm = normalize(fp[msk].mean(axis=0, keepdims=True))
            mapped[msk] = big[np.dot(sm, bm.T).argmax()]
        uniq, cnts = np.unique(mapped, return_counts=True)

    print(f"  {len(uniq)} 个簇, min={cnts.min():,}, max={cnts.max():,}, "
          f"mean={cnts.mean():.0f}")
    return mapped.astype(np.int32)


def cluster_kmeans(feat_dir, num_clusters, device, sample=300_000):
    """固定 K 的 KMeans。"""
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    features, N, D = _load_features(feat_dir, device)
    fn = features.cpu().numpy(); pdim = min(64, D)
    print(f"  KMeans: PCA {D}D→{pdim}D, k={num_clusters}")
    pca = PCA(n_components=pdim, random_state=42)
    if N > sample:
        pca.fit(fn[np.random.choice(N, sample, replace=False)])
    else:
        pca.fit(fn)
    fp = pca.transform(fn).astype(np.float32); del fn, features
    km = MiniBatchKMeans(n_clusters=num_clusters, random_state=42,
                         batch_size=10000, n_init=3, max_iter=100)
    labels = km.fit_predict(fp) + 1
    uniq, cnts = np.unique(labels, return_counts=True)
    print(f"  {len(uniq)} 聚类, min={cnts.min():,}, max={cnts.max():,}, "
          f"mean={cnts.mean():.0f}")
    return labels.astype(np.int32)


def query_clip(feat_dir, prompts, temp, device):
    """SigLIP2 文本嵌入 → 语义标签。"""
    from transformers import AutoModel, AutoTokenizer
    features, N, D = _load_features(feat_dir, device)
    plist = [p.strip() for p in prompts.split(",") if p.strip()]
    print(f"  CLIP: {len(plist)} 类别")
    tp = [f"this is a {p}" for p in plist]
    m = AutoModel.from_pretrained("google/siglip2-base-patch16-512").to(device)
    tok = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-512")
    m.eval()
    inp = tok(tp, padding="max_length", max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        te = F.normalize(m.get_text_features(**inp).float(), dim=-1)
    if D != te.shape[1]:
        q = min(D, te.shape[1]); features = features[:, :q]; te = te[:, :q]
        features = F.normalize(features, dim=-1); te = F.normalize(te, dim=-1)
    sim = features @ te.T; probs = F.softmax(sim / temp, dim=1)
    mp, lab = probs.max(dim=1); lab = lab + 1; lab[mp < 0.3] = 0
    ln = lab.cpu().numpy().astype(np.int32)
    for i, p in enumerate(plist):
        n = (ln == i + 1).sum()
        if n > 0: print(f"    {p}: {n:,} ({n / N * 100:.1f}%)")
    print(f"  背景: {(ln == 0).sum():,} ({(ln == 0).sum() / N * 100:.1f}%)")
    return ln

# ═══════════════════════════ Main ═══════════════════════════

def main():
    p = argparse.ArgumentParser(description="SceneSplat 语义管线")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--use_clip", action="store_true")
    p.add_argument("--prompts", default="")
    p.add_argument("--threshold", type=float, default=0.04)
    p.add_argument("--cluster_method", default="overkmeans",
                   choices=["overkmeans", "kmeans"])
    p.add_argument("--num_clusters", type=int, default=50)
    p.add_argument("--over_k", type=int, default=300,
                   help="OverKMeans 过分割簇数")
    p.add_argument("--merge_thresh", type=float, default=0.85,
                   help="OverKMeans 合并余弦相似度阈值")
    p.add_argument("--checkpoint",
                   default=os.path.join(_ss_root, "ckpts",
                       "lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth"))
    p.add_argument("--work_dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--keep_work", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.work_dir is None:
        args.work_dir = os.path.splitext(os.path.abspath(args.output))[0] + "_work"
    args.work_dir = os.path.abspath(args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    t0 = time.time()

    # B1
    print("B1: splaTV → PLY ...")
    xyz, rgb, op, sc, meta, td = read_splatv(os.path.abspath(args.input))
    print(f"  读取 {len(xyz):,} 高斯, scale [{sc.min():.4f}, {sc.max():.4f}]")
    plp = os.path.join(args.work_dir, "scene.ply")
    export_ply(xyz, rgb, op, sc, plp)

    # B2
    nd = os.path.join(args.work_dir, "npy")
    run_preprocess(plp, nd)

    # B3
    fd = os.path.join(args.work_dir, "feat")
    run_inference(nd, fd, args.checkpoint)

    # B4
    if args.use_clip:
        if not args.prompts: p.error("--use_clip 需要 --prompts")
        labels = query_clip(fd, args.prompts, args.threshold, device)
    elif args.cluster_method == "kmeans":
        labels = cluster_kmeans(fd, args.num_clusters, device)
    else:
        labels = cluster_overkmeans(fd, device,
                                     over_k=args.over_k,
                                     merge_thresh=args.merge_thresh)

    # B5
    print("B5: 语义写回 splaTV ...")
    write_splatv(os.path.abspath(args.output), td, meta, labels)

    if not args.keep_work:
        import shutil; shutil.rmtree(args.work_dir)
    print(f"\n完成: {args.output} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()
