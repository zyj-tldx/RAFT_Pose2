# RAFT-Pose 模型架构与推理流程

> 本文描述 **Stage 1（RGB↔RGB 共享 encoder）** 之后的当前代码状态（`newencoder` 分支）。
> 对应文件：[raft_pose.py](raft_pose.py)、[modules/pose_extractor.py](modules/pose_extractor.py)、[modules/depth_projection.py](modules/depth_projection.py)、[modules/pose_update.py](modules/pose_update.py)、[modules/pose_utils.py](modules/pose_utils.py)。

---

## 1. 概述

RAFT-Pose 是一个**迭代式（iterative）的 RGB-D 相对位姿估计网络**：给定两帧（帧 A = RGB，帧 B = RGB + 深度），估计把帧 B 变换到帧 A 坐标系的相对位姿（6DoF）。思路借鉴 RAFT 光流的「相关代价体 + ConvGRU 迭代精修」，把 2D 光流换成由深度驱动的 6DoF 几何投影。

**核心设计（Stage 1）**：
- **一个共享 RGB encoder**（真 Siamese，一套权重）分别编码帧 A、帧 B 的 RGB → 匹配是 **RGB↔RGB 外观相关**（判别性来自 encoder 学到的边缘/纹理）。
- **深度不进 encoder**，只作**几何投影**：决定相关在哪采样、把帧 B 的点反投影到帧 A。
- 迭代分两段：**Coarse**（1/8，方向位姿假设探测 + 置信度聚合）→ **Fine**（1/4，动态特征对齐残差）。

```
帧A RGB ──► image_encoder ──► fmap_A (外观) ─┐
                (共享权重)                     ├─ 相关 = 外观 vs 外观 ✓
帧B RGB ──► image_encoder ──► fmap_B (外观) ─┘
帧B raw depth ──► depth_projector ──► 采样坐标 (几何: 在哪采)
```

---

## 2. 输入与数据格式

模型 `forward(image, depth, intrinsic_rgb, intrinsic_depth, init_pose=None)`：

| 输入 | 形状 | 含义 |
|------|------|------|
| `image` | `(B, 3, H, W)` | 帧 A 的 RGB，归一化到 [0,1] |
| `depth` | `(B, 4, H, W)` | 帧 B，`[raw_depth, R, G, B]`：ch0 原始深度（米），ch1:3 帧 B 的 RGB（归一化 [0,1]，喂 encoder） |
| `intrinsic_rgb` / `intrinsic_depth` | `(B, 3, 3)` | 两相机内参（7Scenes/TartanAir 中相同） |
| `init_pose` | `(B, 7)` 或 None | 初始位姿；None 时用单位位姿 |

**位姿约定**（[dataloader.py](dataloader.py) `_compute_relative_pose`）：`gt_pose = inv(T_img) @ T_depth`，即**帧 B → 帧 A** 的变换。7D 表示 `[qw, qx, qy, qz, tx, ty, tz]`（单位四元数 + 平移米）。

**输出**：`final_pose (B, 7)`；可选 `pose_sequence (B, K+1, 7)`（初始 + 每次迭代后）、`delta_sequence`（每次迭代的 rot_vec / dt）。

---

## 3. 编码器（共享 ResNet18）

[modules/pose_extractor.py](modules/pose_extractor.py) `ResNet18Encoder`，ImageNet 预训练，InstanceNorm。**两帧共用同一个实例**（`self.image_encoder`），真权重共享。

- stem（conv7×7 s2）→ layer1(s1) → layer2(s2) → layer3(s2) → layer4(s1，无下采样精修)；总步长 8。
- **多尺度输出**：
  - `feat_1_8`：layer4 → conv_out（512→256），H/8，供 coarse 相关。
  - `feat_1_4`：layer2（128ch）→ conv_1_4（128→256），H/4，供 fine 对齐。

forward 中（[raft_pose.py](raft_pose.py) `forward`）：
```python
fmap_A_1_8, fmap_A_1_4 = image_encoder(image)            # 帧 A
fmap_B_1_8, fmap_B_1_4 = image_encoder(depth[:, 1:4])    # 帧 B RGB（共享权重）
# 变量名 fmap_rgb_*=帧A、fmap_depth_*=帧B 沿用旧名，二者现在都是 RGB 外观特征
```

---

## 4. 整体推理流程

```
1. 共享 encoder 提特征：fmap_A / fmap_B（各 1/8 + 1/4）
2. context_feat   = context_proj(fmap_B_1_8)              # coarse 用的上下文
3. 静态相关体     = CorrBlock(fmap_A_1_8, fmap_B_1_8)     # 全对余弦相似度，建一次复用
4. current_pose   = identity（或 init_pose）
5. 迭代 num_iterations 次：
     it < coarse_iters  → Coarse 阶段（1/8，方向假设探测）
     it ≥ coarse_iters  → Fine 阶段（1/4，特征对齐残差）
6. 返回 final_pose
```

默认 `num_iterations=6, coarse_iters=3` → **3 次 coarse + 3 次 fine**。两段各自维护一个 ConvGRU hidden state（coarse 在 1/8、fine 在 1/4，分辨率不同故分离）。

---

## 5. Coarse 阶段（1/8，方向假设探测）

[raft_pose.py](raft_pose.py) `_single_iteration` + `generate_directional_samples` + `sample_correlation_with_poses`。**目的**：从（可能很差的）初始位姿，靠「假设-检验」拉到正确盆地。

1. **生成方向位姿假设**：在当前位姿的 SE(3) 切空间采 ~37 个方向（多轴 × 多尺度 + 单位），每个是一个候选 delta。
2. **几何投影**：对每个假设，用 `depth[:,0:1]`（帧 B 原始深度）+ 该假设位姿做 `depth_projector`，把帧 B 的 3D 点投影到帧 A 的 **1/8** 特征图坐标。
3. **相关采样**：在静态 `CorrBlock` 里，于投影坐标的局部邻域（radius=4，4 层金字塔）采样相关 → 每个假设一组相关特征 + 一个置信度。
4. **置信度聚合**：取置信度 top-K 假设，softmax 加权聚合相关特征与方向编码（coarse_to_fine 分支）。
5. **ConvGRU 更新**：`pose_update_net(聚合相关, context_feat, hidden_coarse, direction_encoding)` → 6D delta。
6. **应用 delta**：`_apply_pose_delta`，受 `max_rot_step/max_trans_step`（默认 0.3）裁剪，Rodrigues 转四元数，与当前位姿复合。

> `CorrBlock`（[modules/depth_projection.py](modules/depth_projection.py)）是两帧 1/8 特征的**全对余弦相似度**（L2 归一化后点积，尺度无关），构建一次、迭代中复用。

---

## 6. Fine 阶段（1/4，动态特征对齐残差）

[raft_pose.py](raft_pose.py) `_single_iteration_fine`。**目的**：coarse 收敛到盆地后，在更高分辨率做局部精修。

1. **1/4 投影**：用当前位姿把帧 B 深度（`depth[:,0:1]`）投影到帧 A 的 **1/4** 坐标 `coords_1_4`（B→A）。
2. **Warp 帧 A 特征到帧 B 网格**（关键修正）：`fmap_A_warped = grid_sample(fmap_A_1_4, grid)`，`grid` 由 `coords_1_4` 归一化 → `fmap_A_warped[p_B] = fmap_A[g(p_B)]` = 点 P（帧 B 像素 p_B 的 3D 点）在帧 A 的特征。
3. **残差**：`feat_diff = (fmap_B_1_4 − fmap_A_warped) * valid_mask`。
   - 同一个 3D 点 P 在两帧的 RGB 外观特征之差；**位姿正确时趋于 0**（Stage 1 RGB↔RGB 修复后恢复判别性的关键）。
   - `valid_mask`：投影出界的像素（无对应关系）置零，消除 `grid_sample` zero-padding 的伪残差。
4. **相似度图**：`similarity = cosine(fmap_A_1_4, fmap_A_warped) * valid_mask`（辅助信号）。
5. **投影聚合**：`feat_diff → fine_feat_proj`、`similarity → fine_sim_proj`，拼接后 `fine_corr_proj` → `align_corr`（保持 1/4，不降采样，保精度）。
6. **Fine 更新**：`context_1_4 = context_proj_1_4(fmap_B_1_4)`；`fine_update_net(align_corr, context_1_4, hidden_fine)` → 6D delta（独立 PoseUpdateNet + 独立 1/4 hidden state）。
7. **应用 delta**：`_apply_pose_delta`，受 `max_rot_step_fine/max_trans_step_fine`（默认 0.05，真 fine 步长）裁剪。

> Fine 用**直接残差**而非相关体（避免 1/4 全对相关的高显存）。RGB↔RGB 下残差重新带上位姿信号。

---

## 7. 位姿表示与更新

[modules/pose_utils.py](modules/pose_utils.py) + [raft_pose.py](raft_pose.py) `_apply_pose_delta`。

- **表示**：单位四元数 q（旋转）+ 平移 t（米）。
- **delta**：6D → rot_vec（3）+ dt（3）。rot_vec 经 Rodrigues 转增量四元数 dq。
- **复合**（camera frame，右乘）：`T_new = T_cur · Δ`，即 `q_new = q ⊗ dq`，`t_new = t + R(q)·dt`。
- **步长裁剪**：rot_vec 与 dt 各自 clamp 到 `±max_*_step`，防发散；旋转幅度超限时按角度缩放（可微）。

---

## 8. 训练损失（简）

[train.py](train.py) + [pose_loss.py](pose_loss.py)。
- **Sequence loss**：对每次迭代的位姿都算误差，按 `seq_loss_gamma`（默认 0.8）递减加权——鼓励逐步收敛。
- **位姿误差**：旋转用测地距离（GeodesicRotationLoss）、平移用 L2（TranslationLoss），分别乘 `rot_weight / trans_weight`（默认 100/100）。
- **Delta 监督**（`delta_loss_weight=1.0`）：直接监督每次迭代预测的 delta 接近「当前位姿到 GT 的剩余增量」。
- **Curriculum**：初始位姿加噪声从大到小（`curriculum_start/end/warmup`），逐步逼近真实分布。

---

## 9. 关键设计决策（为什么这么做）

| 决策 | 原因 |
|------|------|
| **共享 RGB encoder（RGB↔RGB 匹配）** | 旧版用独立 depth_encoder 编码深度图，造成「外观 vs 几何」跨模态，局部相关平坦、fine 阶段失效。两帧都用 RGB 外观 → 同模态匹配，判别性恢复。 |
| **深度不进 encoder，只做投影** | 深度的本职是几何（决定相关在哪采），不是外观匹配。解耦后职责清晰：RGB=「匹配什么」，深度=「在哪采」。 |
| **Coarse 用方向假设探测** | 从差初值出发，单点局部相关不够；多假设 + 置信度聚合能全局定位到正确盆地。 |
| **Fine 用 1/4 直接残差** | coarse 收敛后残差小、落在 1/4 局部盆地内；更高分辨率提升平移精度。 |
| **两段独立 hidden state** | coarse(1/8) 与 fine(1/4) 分辨率不同，ConvGRU hidden 不能跨分辨率复用。 |

---

## 10. 文件地图

| 文件 | 作用 |
|------|------|
| [raft_pose.py](raft_pose.py) | 主模型：encoder 调用、coarse/fine 迭代、位姿更新、forward |
| [modules/pose_extractor.py](modules/pose_extractor.py) | `ResNet18Encoder`（多尺度 1/8+1/4）、`BasicEncoder` 等 |
| [modules/depth_projection.py](modules/depth_projection.py) | `DepthProjector`（3D 点投影）、`CorrBlock`（全对相关 + 邻域采样） |
| [modules/pose_update.py](modules/pose_update.py) | `PoseUpdateNet`（corr/context/dir 投影 + ConvGRU + 位姿回归头） |
| [modules/pose_utils.py](modules/pose_utils.py) | 四元数运算、位姿复合、误差计算 |
| [dataloader.py](dataloader.py) | 训练 `SevenScenesDataset`：产 `image`(RGB)、`depth`([raw,R,G,B])、`gt_pose` |
| [evalute_benchmark.py](evalute_benchmark.py) | 评测 `BenchmarkDataset`（同格式）+ benchmark 引擎 |
| [train.py](train.py) | 训练循环（curriculum、sequence loss、delta 监督、freeze encoder） |

---

## 11. 关键配置参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `image_encoder` | resnet18 | 共享 encoder 类型 |
| `num_iterations` | 6 | 总迭代数 |
| `coarse_iters` | 3 | 前 N 次走 coarse；设成 `num_iterations` 即关闭 fine |
| `hidden_dim / context_dim` | 128 / 64 | ConvGRU hidden / context 通道 |
| `corr_levels / corr_radius` | 4 / 4 | 相关金字塔层数 / 邻域半径 |
| `max_rot_step / max_trans_step` | 0.3 / 0.3 | coarse 每步位姿裁剪 |
| `max_rot_step_fine / max_trans_step_fine` | 0.05 / 0.05 | fine 每步裁剪 |
| `rot_weight / trans_weight` | 100 / 100 | loss 中旋转/平移权重 |
| `freeze_encoder_epochs` | 10 | 前 N epoch 冻结 encoder，让更新头先适应 |

---

## 12. 参数量（Stage 1）

- 单共享 ResNet18 encoder ≈ 11M；`pose_update_net`（coarse）+ `fine_update_net`（fine）+ 各 proj 层 ≈ 5M。
- **总 ≈ 16M**（相比旧版双 encoder ~25M 下降，因为去掉独立 depth_encoder、两帧共用一套权重）。
