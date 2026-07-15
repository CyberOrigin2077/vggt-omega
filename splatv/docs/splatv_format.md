# splaTV 语义格式说明

## 文件结构

`.splatv` 文件是自定义二进制格式，包含一个静态高斯场景的完整描述。

```
┌─────────────────────────────────┐
│  Magic:      uint32 (0x674b)   │  4 bytes
│  JSON len:   uint32             │  4 bytes
│  Metadata:   UTF-8 JSON         │  变长
│  Texture:    4096 × H × uint32  │  剩余数据
└─────────────────────────────────┘
```

## 纹理布局

每个高斯占用连续的 **16 个 uint32** 槽位（64 bytes），按行优先排列在 4096 宽的纹理中。

| Slot | 内容 | 编码 |
|------|------|------|
| 0 | position.x | float32，首项取反 |
| 1 | position.y | float32 |
| 2 | position.z | float32 |
| 3 | rotation (x, y) | packed half2x16 |
| 4 | rotation (z, w) | packed half2x16 |
| 5 | scale | packed half2x16 (s, s) |
| 6 | scale + 预留 | packed half2x16 (s, 0) |
| 7 | RGB + opacity | uint8×4 (R, G, B, α) |
| **8** | **语义 ID** | **uint32** |
| 9–14 | 预留（全 0） | uint32 |
| 15 | TRBF (center, scale) | packed half2x16 |

## 语义 ID（Slot 8）— 新增

Slot 8 在上一个版本中恒为 `0`（属于填充区）。当前版本用于存储高斯粒子的语义标签。

- **来源**：SAM3 对每帧做文本提示实例分割，再由 VGGT-Omega 深度图反投影到 3D，通过体素内多数投票传播到高斯
- **取值**：`0` = 背景/未分类，`1..N` = 实例 ID
- **意义**：同一实例 ID 的高斯属于 SAM3 检测到的同一物体
- **跨帧一致性**：同一物体在不同视角下的高斯会得到相同语义 ID（体素投票保证）

## 兼容性

- 无语义的旧 `.splatv` 文件 Slot 8 全为 0，新格式完全向后兼容
- 渲染器如不使用语义信息，忽略 Slot 8 即可，其余字段无变化
- `export_static_splatvoxel.py` 不带 `--semantic_dir` 时行为与旧版一致

## 生成管线

```
帧图像
  ├─ VGGT-Omega → depth + camera + ViT features
  ├─ SAM3        → 2D 实例分割 mask (uint16)
  └─ SplatVoxelNet → 3D 高斯 + 体素映射
                         ↓
           逆投影 + 体素投票 → 每高斯语义 ID
                         ↓
                    .splatv
```

详见 `infer_semantic.py`（步骤1）和 `export_static_splatvoxel.py`（步骤2）。
