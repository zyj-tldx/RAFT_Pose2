# RAFT-Pose 模型架构详解

> 基于当前代码库（`newencoder` 分支），包含 Dual ResNet18 + Cross-Attention Matcher。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [Step 0: 数据输入](#step-0-数据输入)
3. [Step 1: 特征编码 (Feature Encoding)](#step-1-特征编码-feature-encoding)
4. [Step 2: 特征匹配 (Feature Matching)](#step-2-特征匹配-feature-matching)
5. [Step 3: 深度投影 (Depth Projection)](#step-3-深度投影-depth-projection)
6. [Step 4: 相关性采样 (Correlation Sampling)](#step-4-相关性采样-correlation-sampling)
7. [Step 5: Top-K 聚合 (Confidence Aggregation)](#step-5-top-k-聚合-confidence-aggregation)
8. [Step 6: 姿态更新网络 (Pose Update Network)](#step-6-姿态更新网络-pose-update-network)
9. [Step 7: 姿态更新 (Pose Update via SE(3))](#step-7-姿态更新-pose-update-via-se3)
10. [Step 8: 迭代优化 (Iterative Refinement)](#step-8-迭代优化-iterative-refinement)
11. [损失函数](#11-损失函数)
12. [参数量统计](#12-参数量统计)

---

## 1. 整体架构概览

```
输入: image (B,3,480,640) + depth (B,2,480,640) + intrinsics
                    │                              │
              ┌─────┴─────┐                  ┌─────┴─────┐
              │ RGB Encoder│                  │Depth Encoder│
              │ (ResNet18) │                  │ (ResNet18) │
              │ ImageNet预训练│                │ 随机初始化    │
              └─────┬─────┘                  └─────┬─────┘
                    │                              │
              fmap_rgb                       fmap_depth
           (B,256,60,80)                  (B,256,60,80)
                    │                              │
                    │    ┌─────────────────────────┘
                    │    │  Context Projection
                    │    │  context_feat (B,64,60,80)
                    │    │
              ┌─────┴────┴────────────────────────────┐
              │     Feature Matching                   │
              │  Cross-Attention Matcher / CorrBlock   │
              └──────────────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Iterative Refinement    │
                    │  (K=12 iterations)       │
                    │                          │
                    │  ┌──────────────────┐    │
                    │  │ 1. 生成37个pose采样│    │
                    │  │ 2. 深度投影       │    │
                    │  │ 3. 相关性采样     │    │
                    │  │ 4. Top-K聚合      │    │
                    │  │ 5. ConvGRU更新    │    │
                    │  │ 6. 预测pose delta │    │
                    │  │ 7. SE(3)更新pose  │    │
                    │  └──────────────────┘    │
                    └────────────┬────────────┘
                                 │
                        final_pose (B, 7)
                   [qw, qx, qy, qz, tx, ty, tz]
```

**核心思想**：借鉴 RAFT 光流估计的迭代优化范式，将 pose 估计问题转化为"在 pose 空间中搜索最优匹配"的问题。每一轮迭代生成多个候选 pose，通过深度投影+特征匹配评估每个候选的质量，然后预测一个小的 pose 修正量。

---

## Step 0: 数据输入

**代码位置**: `dataloader.py` → `SevenScenesDataset.__getitem__()` (line ~195)

### 输入数据

| 数据 | 形状 | 说明 |
|------|------|------|
| `image` | `(B, 3, H, W)` | RGB 图像，归一化到 [0, 1] |
| `depth` | `(B, 2, H, W)` | 双通道深度表示 |
| `intrinsic_rgb` | `(B, 3, 3)` | RGB 相机内参矩阵 |
| `intrinsic_depth` | `(B, 3, 3)` | 深度相机内参矩阵 |
| `gt_pose` | `(B, 7)` | GT 相对位姿 [qw,qx,qy,qz,tx,ty,tz] |

### Depth 双通道编码

**代码位置**: `dataloader.py` → `_load_depth()` (line ~113)

```python
# Ch 0: 原始深度 (米)
arr = np.clip(arr * depth_scale, 0.0, 10.0)

# Ch 1: 逆深度 (1/d)，无效深度处置零
inv_arr = np.zeros_like(arr)
inv_arr[valid] = 1.0 / arr[valid]

return torch.from_numpy(np.stack([arr, inv_arr], axis=0))  # (2, H, W)
```

**设计动机**：
- 原始深度 $d$：近处变化剧烈、远处平坦，编码器难以均匀感知
- 逆深度 $1/d$：线性化深度感知，近处精度增强
- SOTA 方法（CalibNet、RegNet）普遍使用逆深度

---

## Step 1: 特征编码 (Feature Encoding)

**代码位置**: `raft_pose.py` → `forward()` (line ~640)

### 1.1 RGB 编码器 (ImageNet 预训练)

**代码位置**: `modules/pose_extractor.py` → `ResNet18Encoder` (line ~300)

```
输入: image (B, 3, 480, 640)
  │
  conv1 (7×7, stride=2)  → 64ch,  240×320    [ImageNet 预训练权重]
  bn1 + relu
  │
  layer1 (2×BasicBlock)  → 64ch,  240×320    [stride=1]
  layer2 (2×BasicBlock)  → 128ch, 120×160    [stride=2]
  layer3 (2×BasicBlock)  → 256ch, 60×80      [stride=2]
  layer4 (2×BasicBlock)  → 512ch, 60×80      [stride=1, 额外细化]
  │
  conv_out (1×1)         → 256ch, 60×80
  InstanceNorm + ReLU
  │
输出: fmap_rgb (B, 256, 60, 80)
```

**特点**：
- 使用 ImageNet 预训练权重（conv1, bn1, layer1-3），迁移学习
- 去掉原始 ResNet 的 maxpool，总 stride = 2×1×2×2 = **8**
- BatchNorm → InstanceNorm，对 batch size 不敏感
- layer4 是额外添加的细化层（stride=1），不改变分辨率

### 1.2 Depth 编码器 (随机初始化)

**代码位置**: `modules/pose_extractor.py` → `ResNet18Encoder` (line ~300, `in_feat=2`)

```
输入: depth (B, 2, 480, 640)
  │
  conv1 (7×7, stride=2, in=2) → 64ch, 240×320  [随机初始化, 2通道输入]
  bn1 + relu
  │
  layer1 (2×BasicBlock)  → 64ch,  240×320
  layer2 (2×BasicBlock)  → 128ch, 120×160
  layer3 (2×BasicBlock)  → 256ch, 60×80
  layer4 (2×BasicBlock)  → 512ch, 60×80
  │
  conv_out (1×1)         → 256ch, 60×80
  InstanceNorm + ReLU
  │
输出: fmap_depth (B, 256, 60, 80)
```

**特点**：
- 与 RGB 编码器**相同架构**，但**独立权重**（非 Siamese）
- 输入 2 通道（depth + inverse depth），conv1 重新初始化
- 可选：通过 `pretrain_encoder.py` 用对比学习预训练

### 1.3 Context 特征

**代码位置**: `raft_pose.py` → `forward()` (line ~660)

```python
context_feat = self.context_proj(fmap_depth)  # (B, 64, 60, 80)
```

```
fmap_depth (B, 256, 60, 80)
  │
  Conv2d(256, 64, 1) + ReLU
  │
context_feat (B, 64, 60, 80)
```

Context 特征提供全局几何上下文，输入给 PoseUpdateNet 的 ConvGRU。

---

## Step 2: 特征匹配 (Feature Matching)

**代码位置**: `raft_pose.py` → `initialize_correlation()` (line ~262)

支持两种模式，通过 `--matcher_type` 选择：

### 2.1 CorrBlock 模式 (`matcher_type='corr'`)

**代码位置**: `modules/depth_projection.py` → `CorrBlock` (line ~96)

```
fmap_depth (B, 256, 60, 80)    fmap_rgb (B, 256, 60, 80)
        │                              │
        └────── L2 Normalize ──────────┘
                       │
              4D Correlation Volume
              corr[d_h, d_w, r_h, r_w] = cos_sim(depth[d_h,d_w], rgb[r_h,r_w])
              shape: (B, 3600, 3600)
                       │
              ┌────────┴────────┐
              │  金字塔下采样     │
              │  Level 0: 60×80 │
              │  Level 1: 30×40 │
              │  Level 2: 15×20 │
              │  Level 3: 8×10  │
              └────────┬────────┘
                       │
              corr_pyramid (4 levels)
```

**特点**：
- 余弦相似度计算，尺度不变
- 4 层金字塔，支持多尺度匹配
- 局部窗口采样：radius=4 → 每个位置采样 9×9 邻域
- 每个样本输出维度：$C_{corr} = (2 \times 4 + 1)^2 \times 4 = 324$

### 2.2 Cross-Attention Matcher 模式 (`matcher_type='attention'`) ⭐ 新增

**代码位置**: `modules/cross_attention_matcher.py` → `CrossAttentionMatcher` (line ~95)

```
fmap_depth (B, 256, 60, 80)    fmap_rgb (B, 256, 60, 80)
        │                              │
   depth_proj                     rgb_proj
   Conv2d+IN+ReLU                Conv2d+IN+ReLU
   (256→256)                     (256→256)
        │                              │
   + PosEnc2D                    + PosEnc2D
        │                              │
   (B, 256, 60, 80)              (B, 256, 60, 80)
        │                              │
   reshape → (B, 3600, 256)     reshape → (B, 3600, 256)
        │                              │
        │         ┌────────────────────┘
        │         │  作为 Key & Value
        │    ┌────┴────────────────────┐
        │    │  CrossAttentionBlock ×2  │
        │    │                          │
        │    │  Q = LayerNorm(depth)    │
        │    │  K = V = LayerNorm(rgb)  │
        │    │  MultiHeadAttention(8)   │
        │    │  + Residual + LayerNorm  │
        │    │  + FFN(512) + Residual   │
        │    └────┬────────────────────┘
        │         │
        │    attention-enhanced depth
        │    (B, 3600, 256)
        │         │
        │    reshape → (B, 256, 60, 80)
        │         │
        │    out_proj
        │    Conv2d+IN+ReLU
        │         │
        └────┬────┘
             │
    _attention_features (B, 256, 60, 80)  ← 缓存，后续采样复用
```

**CrossAttentionBlock 内部结构**：

```
query (B, 3600, 256)     key_value (B, 3600, 256)
      │                          │
  LayerNorm                  LayerNorm
      │                          │
      └──── MultiHeadAttention ──┘
               │ (8 heads, d=32 each)
          attn_out (B, 3600, 256)
               │
      query + attn_out
               │
          LayerNorm
               │
          + FFN(256→512→256)
               │
          LayerNorm
               │
          out (B, 3600, 256)
```

**特点**：
- **全局感受野**：每个 depth 像素可以 attend 到所有 RGB 像素
- **可学习匹配**：vs CorrBlock 的固定余弦相似度
- **LoFTR 风格**：Pre-Norm + 残差连接 + FFN
- **2D 正弦位置编码**：让 attention 感知空间位置
- 输出维度：$C_{out} = 256$（vs CorrBlock 的 324）
- **推理更快**：一次性计算 attention 并缓存，后续采样只需 bilinear interpolation

**PositionalEncoding2D**：

```python
# 2D 正弦位置编码，分别编码 x 和 y 坐标
# 前半通道编码 x 方向，后半通道编码 y 方向
# 频率: 2π/d_model_half, 4π/d_model_half, 8π/d_model_half, ...
pe[d, h, w] = [sin(w * freq_x), cos(w * freq_x), sin(h * freq_y), cos(h * freq_y), ...]
```

---

## Step 3: 深度投影 (Depth Projection)

**代码位置**: `raft_pose.py` → `sample_correlation_with_poses()` (line ~740)
**核心模块**: `modules/depth_projection.py` → `DepthProjector` (line ~10)

每一轮迭代中，对每个候选 pose，将深度图投影到 RGB 相机平面。

```
输入:
  depth_raw (B, 1, 480, 640)     ← 取 Ch 0 (原始深度)
  pose_samples (B, 37, 4, 4)     ← 37 个候选 pose 矩阵
  intrinsic_depth (B, 3, 3)
  intrinsic_rgb (B, 3, 3)

Step 1: 下采样到特征分辨率
  depth_small = F.interpolate(depth_raw, (60, 80))  → (B, 60, 80)
  intrinsic *= 1/8  (缩放内参匹配分辨率)

Step 2: 像素坐标反投影到 3D
  u_d, v_d = meshgrid(80, 60)
  x_d = (u_d - cx) / fx * depth_small
  y_d = (v_d - cy) / fy * depth_small
  z_d = depth_small
  P_D = stack([x_d, y_d, z_d, 1])  → (B, 4, 60, 80)

Step 3: 用候选 pose 变换到 RGB 坐标系
  P_C = pose_samples @ P_D  → (B, 37, 4, 60, 80)

Step 4: 投影到 RGB 图像平面
  u_proj = fx * (X_C / Z_C) + cx
  v_proj = fy * (Y_C / Z_C) + cy
  valid = (Z_C > 0.01)  ← 深度有效性检查

输出:
  projected_coords (B, 37, 2, 60, 80)  ← (u, v) 投影坐标
```

**特点**：
- 在**特征分辨率** (60×80) 上投影，避免 OOM
- 37 个 pose **批量矩阵乘法**，无 Python 循环
- 无效深度（Z_C ≤ 0）坐标推到 -1e4，由下游 valid_mask 过滤

---

## Step 4: 相关性采样 (Correlation Sampling)

**代码位置**: `raft_pose.py` → `sample_correlation_with_poses()` (line ~790)

### 4.1 CorrBlock 模式

```
projected_coords (B, 37, 2, 60, 80)
        │
  对每个样本 n ∈ [0, 36]:
    centroid = coords[:, n] / scale  (按金字塔层级缩放)
    local_window = centroid + delta_grid  (9×9 邻域, radius=4)
    corr_local = bilinear_sample(corr_pyramid, local_window)
        │
  拼接 4 个金字塔层级:
    (B*37, 81, 60, 80) × 4 levels → (B*37, 324, 60, 80)

输出:
  corr_feats (B*37, 324, 60, 80)
  confidence (B, 37)  ← top-20% 中心相关值的均值
```

### 4.2 Cross-Attention 模式

```
projected_coords (B, 37, 2, 60, 80)
        │
  对每个样本 n ∈ [0, 36]:
    coords_norm = normalize_to[-1, 1](coords[:, n])
    sampled = grid_sample(_attention_features, coords_norm)
        │
  stack all samples:
    (B, 37, 256, 60, 80) → (B*37, 256, 60, 80)

输出:
  corr_feats (B*37, 256, 60, 80)
  confidence (B, 37)  ← 特征范数在有效位置上的均值
```

### 4.3 Coarse-to-Fine 模式 (`--coarse_to_fine`)

**代码位置**: `modules/depth_projection.py` → `CorrBlock.sample_coarse_then_fine()` (line ~400)

```
Phase 1 (粗筛): 对全部 37 个样本在最粗金字塔层评估
  → confidence (B, 37)
  → topk_indices (B, 3)  ← 选 top-3

Phase 2 (精采): 只对 top-3 做完整多层级采样
  → corr_feats_topk (B*3, C_corr, 60, 80)

内存: O(B*37) → O(B*3)，节省 ~92% 采样内存
```

---

## Step 5: Top-K 聚合 (Confidence Aggregation)

**代码位置**: `raft_pose.py` → `_single_iteration()` (line ~490)

```
corr_feats (B*K, C_corr, 60, 80)    confidence (B, K)
        │                                    │
  reshape → (B, K, C_corr, 60, 80)          │
        │                                    │
  topk_weights = softmax(confidence, dim=1)  → (B, K)
        │
  weighted_feats = Σ_k (weight_k × feat_k)  → (B, C_corr, 60, 80)
        │
  direction_encoding = Σ_k (weight_k × dir_k)  → (B, 6)
        │
aggregated_corr (B, C_corr, 60, 80)
direction_encoding (B, 6)  ← [rot_axis(3), trans_dir(3)]
```

**设计动机**：
- 不是只选最好的 1 个样本，而是**加权融合 top-K**
- 权重由 softmax(confidence) 决定，高置信度样本贡献更大
- direction_encoding 告诉网络"这些特征来自哪个方向"，辅助 pose 预测

---

## Step 6: 姿态更新网络 (Pose Update Network)

**代码位置**: `modules/pose_update.py` → `PoseUpdateNet` (line ~200)

```
输入:
  aggregated_corr (B, C_corr, 60, 80)     ← 聚合后的相关性特征
  context_feat (B, 64, 60, 80)            ← 编码器上下文特征
  hidden_state (B, 128, 60, 80) or None   ← ConvGRU 隐藏状态
  direction_encoding (B, 6)               ← 方向编码

Step 1: 特征投影
  corr_proj = corr_proj_net(aggregated_corr)     → (B, 128, 60, 80)
    Conv2d(C_corr, 128, 1) + ReLU
    Conv2d(128, 128, 3) + ReLU

  context_proj = context_proj_net(context_feat)  → (B, 128, 60, 80)
    Conv2d(64, 128, 1) + ReLU

  dir_feat = dir_proj(direction_encoding)        → (B, 128, 1, 1) → broadcast
    Linear(6, 128) + ReLU

Step 2: ConvGRU 更新
  if hidden_state is None:
    hidden_state = init_h(context_feat)  → (B, 128, 60, 80)
      Conv2d(64, 128, 3) + ReLU + Conv2d(128, 128, 3) + ReLU

  hidden_state = ConvGRU(hidden_state, corr_proj + context_proj + dir_feat)
    z = σ(Conv([h; x]))     ← 更新门
    r = σ(Conv([h; x]))     ← 重置门
    q = tanh(Conv([r⊙h; x])) ← 候选状态
    h_new = (1-z)⊙h + z⊙q

Step 3: Pose 回归头
  pose_delta = PoseRegressionHead(hidden_state)
    3× ResidualBlock(128, 128)
    Conv2d(128, 6, 3)  ← 零初始化 (×0.01)

输出:
  pose_delta (B, 6, 60, 80)     ← [rx, ry, rz, tx, ty, tz]
  hidden_state (B, 128, 60, 80) ← 传递给下一轮迭代
```

**PoseRegressionHead 零初始化**：
- `pose_conv.weight *= 0.01`，`pose_conv.bias = 0`
- 未训练时输出接近零 → identity delta → 不破坏初始 pose
- 小随机初始化（而非严格零）保证梯度可以流动

---

## Step 7: 姿态更新 (Pose Update via SE(3))

**代码位置**: `raft_pose.py` → `_single_iteration()` (line ~560)

```
pose_delta (B, 6, 60, 80)
        │
  spatial mean pooling
        │
  pose_delta_avg (B, 6)  ← [rx, ry, rz, tx, ty, tz]
        │
  ┌─────┴─────┐
  │           │
rot_vec     dt
(B, 3)      (B, 3)
  │           │
  │     clamp(-max_trans, max_trans)
  │           │
  │     dt (B, 3)  ← 平移增量 (米)
  │
  clamp(-max_rot, max_rot)
  │
  Rodrigues → quaternion delta
  angle = ‖rot_vec‖
  dq = [cos(θ/2), sinc(θ/2) · rot_vec/2]
  dq = normalize(dq)
  │
  dq (B, 4)  ← 旋转增量 (四元数)
        │
  SE(3) 组合: T_new = T_cur ⊗ Δ
  q_new = qmul(q_cur, dq)
  t_new = t_cur + qrot(q_cur, dt)
  q_new = normalize(q_new)
        │
  current_pose (B, 7)  ← [qw, qx, qy, qz, tx, ty, tz]
```

**关键设计**：
- **max_rot_step / max_trans_step**：限制每步最大修正量，防止发散
- **Rodrigues 公式**：旋转向量 → 四元数，完全可微
- **sinc 的 Taylor 展开**：$\theta \to 0$ 时 $\text{sinc}(\theta) \to 1$，避免 0/0
- **SE(3) 组合**：$T_{new} = T_{cur} \times \Delta$，在李群上更新保证合法性

---

## Step 8: 迭代优化 (Iterative Refinement)

**代码位置**: `raft_pose.py` → `forward()` (line ~680)

```
for it in range(num_iterations):  # 默认 12 轮
    current_pose, hidden_state, rot_vec, dt = _single_iteration(
        current_pose, hidden_state, depth, intrinsic_depth, intrinsic_rgb, context_feat
    )
```

**每轮迭代流程**：

```
current_pose (B, 7)
      │
  ┌───┴───────────────────────────────────────────┐
  │ 1. generate_directional_samples               │
  │    12 方向 × 3 尺度 + 1 恒等 = 37 个候选 pose  │
  │    → pose_samples (B, 37, 4, 4)               │
  │                                               │
  │ 2. sample_correlation_with_poses              │
  │    深度投影 + 特征采样                          │
  │    → corr_feats (B*K, C_corr, 60, 80)         │
  │    → confidence (B, 37)                        │
  │                                               │
  │ 3. Top-K 加权聚合                              │
  │    → aggregated_corr (B, C_corr, 60, 80)      │
  │    → direction_encoding (B, 6)                 │
  │                                               │
  │ 4. PoseUpdateNet                              │
  │    ConvGRU + PoseRegressionHead                │
  │    → pose_delta (B, 6, 60, 80)                │
  │    → hidden_state (B, 128, 60, 80)            │
  │                                               │
  │ 5. SE(3) 姿态更新                              │
  │    → current_pose (B, 7)                       │
  └───────────────────────────────────────────────┘
      │
  (重复 12 轮)
      │
final_pose (B, 7)
```

### 方向采样策略

**代码位置**: `raft_pose.py` → `generate_directional_samples()` (line ~283)

```
12 个方向:
  ±Tx, ±Ty, ±Tz     ← 6 个平移方向
  ±Rx, ±Ry, ±Rz     ← 6 个旋转方向

3 个尺度: 0.25, 1.0, 4.0

12 × 3 = 36 个扰动 + 1 个恒等 = 37 个候选 pose

方向编码 (6D):
  [rot_axis_x, rot_axis_y, rot_axis_z, trans_dir_x, trans_dir_y, trans_dir_z]
  恒等样本: [0, 0, 0, 0, 0, 0]
```

**预计算优化**：方向模板在 `__init__` 中注册为 buffer，每轮迭代只计算 scale-dependent 的四元数扰动，避免 ~100 次张量分配。

### 梯度检查点 (Gradient Checkpointing)

```python
if self.use_checkpoint:
    current_pose, hidden_state, rot_vec, dt = torch_checkpoint(
        self._single_iteration, ...
    )
```

- 前向传播后丢弃中间激活
- 反向传播时重新计算
- 内存：$O(K) \to O(1)$，时间增加 ~20%

---

## 11. 损失函数

**代码位置**: `raft_pose.py` → `compute_loss()` (line ~830), `train.py` → `train_one_epoch()`

### 序列损失 (Sequence Loss)

```python
# 对每一轮迭代的中间 pose 都计算损失
for it in range(num_iterations):
    pred_pose_it = pose_sequence[it + 1]  # 第 it 轮后的 pose
    loss_it = rotation_loss(pred_pose_it, gt_pose) + translation_loss(pred_pose_it, gt_pose)
    # 指数加权: 后期迭代权重更大
    weight = gamma ** (num_iterations - 1 - it)  # gamma=0.8
    total_loss += weight * loss_it
```

### 旋转损失

```python
# 四元数测地线距离 → 角度误差 (度)
rot_error = 2 * arccos(|q_pred · q_gt|) * (180 / π)  # (B,)
loss_rot = rot_error.mean() * rot_weight  # rot_weight=100
```

### 平移损失

```python
trans_error = ‖t_pred - t_gt‖₂  # (B,)
loss_trans = trans_error.mean() * trans_weight  # trans_weight=20
```

### Delta 监督损失 (可选)

```python
# 直接监督预测的 (rot_vec, dt) 与真实 delta 的差距
loss_delta = ‖pred_delta - gt_delta‖₂ * delta_loss_weight
```

### 课程学习 (Curriculum Learning)

```python
# 训练初期: 小噪声 (模型从 GT 附近开始)
# 训练后期: 大噪声 (模型学会处理大误差)
noise_std = curriculum_start + (curriculum_end - curriculum_start) * progress
# progress 在前 80% epoch 从 0 线性增长到 1
```

---

## 12. 参数量统计

| 模块 | 参数量 | 说明 |
|------|--------|------|
| RGB Encoder (ResNet18) | ~11.2M | ImageNet 预训练 |
| Depth Encoder (ResNet18) | ~11.2M | 随机初始化, 2ch 输入 |
| Context Projection | ~16.5K | Conv2d(256→64, 1) |
| CrossAttentionMatcher | ~1.3M | 2× CrossAttnBlock + proj |
| CorrBlock | 0 | 无可学习参数 |
| PoseUpdateNet | ~1.1M | ConvGRU + PoseHead |
| **总计 (attention)** | **~24.8M** | |
| **总计 (corr)** | **~23.5M** | |

### 推理速度对比 (B=2, 12 iter, 480×640)

| Matcher | 推理时间 | 相对速度 |
|---------|---------|---------|
| CorrBlock | 328.8 ms | 1.00× |
| CrossAttention | 176.8 ms | **0.54×** (快 46%) |

---

## 文件索引

| 文件 | 核心内容 |
|------|---------|
| `raft_pose.py` | RAFTPose 主模型、迭代优化循环、pose 采样 |
| `modules/pose_extractor.py` | ResNet18Encoder, BasicEncoder, DepthEncoder |
| `modules/depth_projection.py` | CorrBlock (4D 相关体积), DepthProjector |
| `modules/cross_attention_matcher.py` | CrossAttentionMatcher, PositionalEncoding2D |
| `modules/pose_update.py` | PoseUpdateNet, ConvGRU, PoseRegressionHead |
| `modules/pose_utils.py` | 四元数运算、pose 组合、误差计算 |
| `dataloader.py` | 7Scenes 数据加载、双通道 depth |
| `train.py` | 训练循环、序列损失、课程学习 |
| `pretrain_encoder.py` | 对比学习预训练 encoder |
