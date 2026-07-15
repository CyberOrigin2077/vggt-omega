# 语义信息着色说明

## 数据来源

`.splatv` 文件中每个高斯在 Slot 8 存储一个 `uint32` 语义 ID：

- **来源**：SAM3 逐帧实例分割 → VGGT-Omega 深度图反投影到 3D → 体素多数投票传播到高斯
- **取值**：`0` = 背景/未分类，`1..N` = 实例 ID（同一物体在不同视角下共享相同 ID）
- **兼容性**：旧版 `.splatv` 该位为 `0` 填充，不影响原有渲染

详见 [splatv_semantic_format.md](docs/splatv_semantic_format.md)。

## 着色逻辑

着色器在顶点阶段根据 `u_semantic` uniform 分支：

| 模式 | 颜色来源 | 运动插值 |
|------|---------|---------|
| RGB（默认） | Slot 7 的 RGBA 颜色 | 从 Slot 8–14 解包运动向量 + 旋转插值 |
| 语义 | Slot 8 的语义 ID 哈希着色 | 跳过（`tpos=0`），仅保留 TRBF 时间透明度 |

### 语义颜色映射

```
语义 ID = 0  → 深灰色 vec3(0.3, 0.3, 0.35)

语义 ID > 0  → golden-ratio 哈希:
    h = id × 2654435761
    R = (h & 0xff)       / 255
    G = ((h >> 8) & 0xff) / 255
    B = ((h >> 16) & 0xff) / 255
```

不同实例 ID 会被映射为视觉上容易区分的不同颜色。

## 代码路径

所有改动集中在 [hybrid.js](hybrid.js)：

- **Shader uniform**（第 287 行）：`uniform bool u_semantic;`
- **语义 ID 读取**（第 308 行）：`uint semanticId = motion0.x;` — 直接读取 uint32，不解包
- **条件分支**（第 311–319 行）：`u_semantic` 为 true 时 `tpos/trot` 置零，否则走原有运动解包
- **颜色分支**（第 358–368 行）：`u_semantic` 为 true 时用哈希颜色，否则用原始 RGBA

JS 侧通过 `#semBtn` 按钮切换 `semanticMode` 状态，每帧将 `u_semantic` uniform 传入 GPU。
