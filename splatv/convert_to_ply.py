"""
将 splaTV 文件转换为标准 3DGS PLY 格式，供 SceneSplat 预处理。

splaTV → PLY → SceneSplat NPY → 语言特征 → 语义标签
"""

import argparse
import json
import os
import struct

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814  # SH DC 系数


def read_splatv(filepath):
    """读取 splaTV 文件，返回高斯参数。"""
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x674B, f"Bad magic: 0x{magic:04x}"
        json_len = struct.unpack("<I", f.read(4))[0]
        metadata = json.loads(f.read(json_len))
        texdata = np.frombuffer(f.read(), dtype=np.uint32)

    texwidth = metadata[0]["texwidth"]
    texheight = metadata[0]["texheight"]
    N = (texwidth * texheight) // 4

    # 解析每个高斯 (16 个 uint32)
    tex_f = texdata.view(np.float32)
    tex_u8 = texdata.view(np.uint8)

    xyz = np.zeros((N, 3), dtype=np.float32)
    rgb = np.zeros((N, 3), dtype=np.float32)
    opacity = np.zeros(N, dtype=np.float32)
    scale = np.zeros(N, dtype=np.float32)
    semantic = np.zeros(N, dtype=np.int32)

    for j in range(N):
        xyz[j, 0] = -tex_f[16 * j + 0]  # x (取反)
        xyz[j, 1] = tex_f[16 * j + 1]   # y
        xyz[j, 2] = tex_f[16 * j + 2]   # z
        # slot 3-6: rotation, scale (不关心值）
        rgb[j, 0] = tex_u8[4 * (16 * j + 7) + 0] / 255.0
        rgb[j, 1] = tex_u8[4 * (16 * j + 7) + 1] / 255.0
        rgb[j, 2] = tex_u8[4 * (16 * j + 7) + 2] / 255.0
        opacity[j] = tex_u8[4 * (16 * j + 7) + 3] / 255.0
        scale[j] = np.float16(np.uint16(texdata[16 * j + 5] & 0xFFFF)).view(np.float32)
        semantic[j] = int(texdata[16 * j + 8])

    return xyz, rgb, opacity, scale, semantic, metadata


def safe_sigmoid_inv(x, eps=1e-6):
    return np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps)))


def write_ply(filepath, xyz, rgb, opacity, scale):
    """写标准 3DGS PLY 文件。"""
    N = len(xyz)

    # 转换到 PLY 存储格式
    logit_opacity = safe_sigmoid_inv(opacity).astype(np.float32)
    log_scale = np.log(np.clip(scale, 1e-8, None)).astype(np.float32)
    log_scale_3 = np.tile(log_scale[:, None], (1, 3))  # 3 通道等值

    # RGB → SH DC
    sh_dc = ((rgb - 0.5) / C0).astype(np.float32)

    # 单位四元数
    quat = np.zeros((N, 4), dtype=np.float32)
    quat[:, 0] = 1.0  # w=1

    vertex = np.zeros(N, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["f_dc_0"] = sh_dc[:, 0]
    vertex["f_dc_1"] = sh_dc[:, 1]
    vertex["f_dc_2"] = sh_dc[:, 2]
    vertex["opacity"] = logit_opacity
    vertex["scale_0"] = log_scale_3[:, 0]
    vertex["scale_1"] = log_scale_3[:, 1]
    vertex["scale_2"] = log_scale_3[:, 2]
    vertex["rot_0"] = quat[:, 0]
    vertex["rot_1"] = quat[:, 1]
    vertex["rot_2"] = quat[:, 2]
    vertex["rot_3"] = quat[:, 3]

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(filepath)
    print(f"写 {N:,} 高斯 → {filepath}")


def main():
    parser = argparse.ArgumentParser(description="splaTV → 3DGS PLY")
    parser.add_argument("--input", type=str, required=True, help=".splatv 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出 .ply 路径")
    parser.add_argument("--save_semantic", type=str, default=None,
                        help="同时保存语义 ID 到指定 .npy 文件")
    args = parser.parse_args()

    xyz, rgb, opacity, scale, semantic, meta = read_splatv(args.input)
    print(f"读取 {len(xyz):,} 高斯")

    write_ply(args.output, xyz, rgb, opacity, scale)

    if args.save_semantic:
        np.save(args.save_semantic, semantic)
        print(f"语义 ID 已保存: {args.save_semantic}")


if __name__ == "__main__":
    main()
