"""
将语义标签写回 splaTV 文件的 Slot 8。

用法:
    python patch_semantic_to_splatv.py \
        --input output_static/room0/room0_static.splatv \
        --labels labels.npy \
        --output output_static/room0/room0_static_scenesplat.splatv
"""

import argparse
import json
import os
import struct

import numpy as np


def patch_semantic(input_path, labels, output_path):
    with open(input_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x674B, f"Bad magic: 0x{magic:04x}"
        json_len = struct.unpack("<I", f.read(4))[0]
        metadata = json.loads(f.read(json_len))
        texdata = np.frombuffer(f.read(), dtype=np.uint32)

    texwidth = metadata[0]["texwidth"]
    texheight = metadata[0]["texheight"]
    N = (texwidth * texheight) // 4

    # 清除旧语义，写入新语义
    for j in range(N):
        texdata[16 * j + 8] = 0
    for j in range(min(N, len(labels))):
        texdata[16 * j + 8] = int(labels[j])

    json_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", 0x674B))
        f.write(struct.pack("<I", len(json_bytes)))
        f.write(json_bytes)
        f.write(texdata.tobytes())

    nz = (np.array([texdata[16 * j + 8] for j in range(N)]) > 0).sum()
    print(f"写 {N:,} 高斯, {nz:,} 有语义 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="语义标签 → splaTV")
    parser.add_argument("--input", type=str, required=True, help="输入 .splatv")
    parser.add_argument("--labels", type=str, required=True, help="语义标签 .npy (int32)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认: {input_stem}_semantic.splatv）")
    args = parser.parse_args()

    labels = np.load(args.labels)
    print(f"加载 {len(labels):,} 个标签, 唯一值: {np.unique(labels)}")

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_semantic.splatv"

    patch_semantic(args.input, labels, args.output)


if __name__ == "__main__":
    main()
