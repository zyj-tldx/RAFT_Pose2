# RAFT-Pose 训练与验证流程详解

> 基于当前代码库（`newencoder` 分支），逐步解析 `pretrain_encoder.py` 和 `validate.py` 的完整执行流程。

---

## 目录

1. [预训练脚本概览](#1-预训练脚本概览)
2. [Step 1: 数据加载与增强](#step-1-数据加载与增强)
3. [Step 2: GT 位姿深度投影](#step-2-gt-位姿深度投影)
4. [Step 3: 特征提取](#step-3-特征提取)
5. [Step 4: 对比损失计算](#step-4-对比损失计算)
6. [Step 5: 反向传播与优化](#step-5-反向传播与优化)
7. [Step 6: 验证与保存](#step-6-验证与保存)
8. [验证脚本概览](#8-验证脚本概览)
9. [Step 7: 加载数据与计算 GT](#step-7-加载数据与计算-gt)
10. [Step 8: 模型推理](#step-8-模型推理)
11. [Step 9: 误差计算](#step-9-误差计算)
12. [Step 10: 可视化与输出](#step-10-可视化与输出)
13. [命令行参数速查](#13-命令行参数速查)

---

## 1. 预训练脚本概览

**文件**: `pretrain_encoder.py`

### 核心思想

端到端训练 RAFT-Pose 存在"鸡生蛋"问题：模型需要好的特征来做匹配，但好的特征需要正确的位姿来训练。预训练通过 **GT 位姿** 打破这个循环：

```
在 GT 位姿下，深度投影能精确对齐 RGB-Depth 像素
  → 训练编码器使对齐的像素特征具有高余弦相似度
  → 编码器学会"什么样的 RGB 和 Depth 特征属于同一 3D 点"
  → 后续端到端训练时，即使位姿有误差，编码器仍能提供有意义的匹配信号
```

### 整体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    pretrain_encoder.py 主流程                        │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 数据加载  │ →  │ 数据增强  │ →  │ GT投影   │ →  │ 特征提取  │      │
│  │ + 增强    │    │ (可选)    │    │ depth→RGB│    │ 双编码器  │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                       │             │
│                                                       ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 保存最优  │ ←  │ 验证评估  │ ←  │ 学习率   │ ←  │ 对比损失  │      │
│  │ checkpoint│    │ (无增强)  │    │ 调度     │    │ InfoNCE  │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                     │
│  循环 epochs 轮，每轮包含: train_one_epoch() + validate()           │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 投影分辨率 | 特征图分辨率 (1/8) | 与推理时一致，避免分辨率不匹配 |
| 损失函数 | DenseContrastiveLoss (默认) | 像素级密集监督，比稀疏网格更有效 |
| 负样本策略 | 随机 + 困难负样本 (课程学习) | 困难负样本提升空间判别能力 |
| 只训练编码器 | 冻结 pose 更新网络 | 预训练目标仅是特征对齐 |
| 数据增强 | 光度 + 深度噪声 | 小数据集 (435 samples) 防止过拟合 |

---

## Step 1: 数据加载与增强

**代码位置**: `pretrain_encoder.py` → `main()` (line ~710) + `PretrainAugmentor` (line ~47)

### 1.1 数据集构建

```python
train_dataset = SevenScenesDataset(
    config_path=args.config,       # e.g. configs/allscenes_train.json
    split="train",
    image_size=(480, 640),
    augment=False,                 # 预训练脚本自己管理增强
)
```

每个样本包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `image` | `(3, 480, 640)` | RGB 图像，[0, 1] |
| `depth` | `(2, 480, 640)` | 双通道深度 (raw + inverse) |
| `gt_pose` | `(7,)` | GT 相对位姿 [qw,qx,qy,qz,tx,ty,tz] |
| `intrinsic_rgb` | `(3, 3)` | RGB 相机内参 |
| `intrinsic_depth` | `(3, 3)` | 深度相机内参 |

### 1.2 数据增强 (PretrainAugmentor)

**代码位置**: `pretrain_encoder.py` → `PretrainAugmentor.__call__()` (line ~52)

每个增强以 50% 概率独立应用：

```
输入: image (3, H, W), depth (2, H, W)
  │
  ├── 光度增强 (仅作用于 RGB):
  │   ├── 亮度抖动: image *= [0.7, 1.3]          (50%)
  │   ├── 对比度抖动: (image - mean) * [0.7, 1.3] + mean  (50%)
  │   ├── 饱和度抖动: blend(gray, image, [0.5, 1.0])  (50%)
  │   └── 高斯噪声: image += N(0, 0.02)           (30%)
  │
  ├── 深度增强 (仅作用于 Ch 0 原始深度):
  │   ├── 深度噪声: depth[0] += N(0, 0.02)        (50%)
  │   └── 深度缩放: depth[0] *= [0.98, 1.02]      (30%)
  │
  └── 重新计算逆深度 (Ch 1):
      depth[1, valid] = 1.0 / depth[0, valid]
      depth[1, ~valid] = 0.0
  │
输出: augmented (image, depth)
```

**设计动机**：
- 7Scenes 每个场景仅 ~435 训练样本，增强至关重要
- 深度噪声模拟 LiDAR/ToF 传感器噪声
- 深度缩放模拟尺度不确定性
- 增强后必须重新计算逆深度，保持双通道一致性

---

## Step 2: GT 位姿深度投影

**代码位置**: `pretrain_encoder.py` → `project_depth_to_rgb()` (line ~115)

### 核心步骤

将深度图中的每个像素，通过 GT 相对位姿投影到 RGB 图像平面，找到对应的 RGB 像素位置。

```
深度相机坐标系                              RGB 相机坐标系
     ┌─────────┐                              ┌─────────┐
     │ depth   │     T_rel (GT pose)          │  RGB    │
     │  image  │ ──────────────────────────→  │  image  │
     │         │                              │         │
     │ (u_d,v_d)                            (u_rgb,v_rgb)
     └─────────┘                              └─────────┘

投影流程 (在特征图分辨率 60×80 下进行):
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. 下采样深度到 60×80 (nearest, 保持深度值)                    │
  │    depth_small: (B, 60, 80)                                  │
  │                                                              │
  │ 2. 缩放内参到特征图分辨率 (÷8)                                │
  │    fx_s = fx/8, fy_s = fy/8, cx_s = cx/8, cy_s = cy/8      │
  │                                                              │
  │ 3. 反投影: 像素 → 3D 点 (深度相机坐标系)                       │
  │    x = (u_d - cx_s) / fx_s × depth                          │
  │    y = (v_d - cy_s) / fy_s × depth                          │
  │    z = depth                                                │
  │    P_D = [x, y, z, 1]  →  (B, 4, 60, 80)                   │
  │                                                              │
  │ 4. 位姿变换: 深度相机 → RGB 相机                              │
  │    T_rel = 4×4 矩阵 (从 gt_pose_7d 四元数转换)                │
  │    P_C = T_rel @ P_D  →  (B, 4, 60, 80)                     │
  │                                                              │
  │ 5. 投影到 RGB 平面 (特征图分辨率)                              │
  │    u_proj = fx_s × (X_C / Z_C) + cx_s                      │
  │    v_proj = fy_s × (Y_C / Z_C) + cy_s                      │
  │    projected_coords: (B, 2, 60, 80)  ← (u, v) 坐标           │
  │                                                              │
  │ 6. 有效掩码                                                  │
  │    valid = (depth > 0.01) & (0 ≤ u < W-1) & (0 ≤ v < H-1)  │
  │    valid_mask: (B, 1, 60, 80)                               │
  └──────────────────────────────────────────────────────────────┘
```

**为什么在特征图分辨率投影？**
- 编码器输出是 60×80，对比损失在特征图上计算
- 如果在原始分辨率投影再下采样，会引入插值误差
- 直接在 60×80 投影，与推理时的 `sample_correlation_with_poses()` 完全一致

**四元数 → 旋转矩阵** (Shepperd's method):
```python
R[0,0] = 1 - 2(qy² + qz²)     R[0,1] = 2(qxqy - qwqz)
R[1,0] = 2(qxqy + qwqz)       R[1,1] = 1 - 2(qx² + qz²)
...
```

---

## Step 3: 特征提取

**代码位置**: `pretrain_encoder.py` → `train_one_epoch()` (line ~490)

### 编码器前向传播

```python
# 只运行编码器部分 (不运行 pose 更新网络)
fmap_rgb = model.image_encoder(image)       # (B, 256, 60, 80)
fmap_depth = model.depth_encoder(depth)     # (B, 256, 60, 80)
```

```
image (B, 3, 480, 640)              depth (B, 2, 480, 640)
        │                                    │
   ┌────┴────┐                          ┌────┴────┐
   │  RGB    │                          │  Depth  │
   │ Encoder │                          │ Encoder │
   │ResNet18 │                          │ResNet18 │
   │ImageNet │                          │随机初始化 │
   │预训练    │                          │         │
   └────┬────┘                          └────┬────┘
        │                                    │
   fmap_rgb                            fmap_depth
(B, 256, 60, 80)                    (B, 256, 60, 80)
```

**关键点**：
- 预训练时**只训练编码器参数**，pose 更新网络、相关体积等全部冻结
- RGB 编码器使用 ImageNet 预训练权重（迁移学习）
- Depth 编码器随机初始化（2 通道输入，conv1 权重形状不同）

### 参数冻结策略

```python
# 只收集编码器参数用于优化
encoder_params = []
encoder_keys = ['image_encoder', 'depth_encoder', 'depth_feat_align']
for name, param in model.named_parameters():
    if any(k in name for k in encoder_keys):
        encoder_params.append(param)

optimizer = AdamW(encoder_params, lr=1e-3)  # 非编码器参数不更新
```

---

## Step 4: 对比损失计算

**代码位置**: `pretrain_encoder.py` → `DenseContrastiveLoss.forward()` (line ~355)

支持两种损失函数，通过 `--loss_type` 选择：

### 4.1 InfoNCE Loss (`--loss_type infonce`)

**代码位置**: `pretrain_encoder.py` → `InfoNCELoss` (line ~280)

稀疏网格采样方式，计算效率高但监督信号稀疏：

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 在特征图上均匀采样 ~16×16 = 256 个位置                      │
│    step = max(1, H_feat // 16)                               │
│                                                              │
│ 2. 对每个采样位置 (i, j):                                     │
│    - Query:  fmap_depth[:, :, i, j]     (深度特征)            │
│    - Positive: fmap_rgb 在 projected_coords[i,j] 处双线性采样  │
│    - Negatives: fmap_rgb 在所有 256 个网格位置                 │
│                                                              │
│ 3. 构建相似度矩阵: (B, 256, 256)                              │
│    sim[i, j] = cos(depth_feat_i, rgb_feat_j) / τ             │
│                                                              │
│ 4. 交叉熵损失: 对角线为正样本标签                               │
│    loss = CrossEntropy(sim, labels=diag)                      │
│                                                              │
│ 5. 只计算 valid_mask > 0.5 的位置                             │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Dense Contrastive Loss (`--loss_type dense`, 默认)

**代码位置**: `pretrain_encoder.py` → `DenseContrastiveLoss` (line ~321)

像素级密集监督，每个有效深度像素都计算损失：

```
┌──────────────────────────────────────────────────────────────┐
│ 输入: fmap_rgb (B, 256, 60, 80), fmap_depth (B, 256, 60, 80)│
│       projected_coords (B, 2, 60, 80), valid_mask (B, 1, 60,80)│
│                                                              │
│ 1. L2 归一化特征                                              │
│    fmap_rgb = L2Norm(fmap_rgb)                               │
│    fmap_depth = L2Norm(fmap_depth)                           │
│                                                              │
│ 2. 正样本采样: 通过 grid_sample 在投影坐标处采样 RGB 特征       │
│    rgb_pos = grid_sample(fmap_rgb, projected_coords)          │
│    pos_sim = (fmap_depth · rgb_pos).sum(dim=1) / τ           │
│    → (B, 3600)  每个像素的正样本相似度                         │
│                                                              │
│ 3. 负样本采样 (混合策略):                                      │
│    ┌─────────────────────────────────────────────────────┐    │
│    │ 随机负样本 (n_random 个):                             │    │
│    │   从全图随机选位置，通过 gather + bmm 批量计算相似度   │    │
│    │   rand_neg_sim: (B, n_random, 3600)                 │    │
│    │                                                     │    │
│    │ 困难负样本 (n_hard 个, 课程学习调度):                   │    │
│    │   在正样本位置附近 ±hard_radius 像素内采样              │    │
│    │   这些位置特征相似但空间不同 → 最容易混淆               │    │
│    │   hard_neg_sim: (B, n_hard, 3600)                    │    │
│    └─────────────────────────────────────────────────────┘    │
│                                                              │
│ 4. InfoNCE 损失:                                              │
│    all_sim = cat([pos_sim, rand_neg_sim, hard_neg_sim])       │
│    loss = -pos_sim + logsumexp(all_sim)                       │
│    loss = (loss × valid_mask).sum() / valid_mask.sum()        │
│                                                              │
│ 5. 匹配准确率:                                                │
│    accuracy = (pos_sim > max(neg_sim)) & valid                │
└──────────────────────────────────────────────────────────────┘
```

### 困难负样本课程学习

```
训练进度:  epoch 1 ─────────────────────────────→ epoch 30
           │                                        │
hard_ratio: 0.0 ─────── 线性增长 ──────────────→ 0.75
           │                                        │
负样本构成: 全部随机 (简单)          75% 困难 + 25% 随机
           │                                        │
训练难度:   低 (学习粗粒度匹配)        高 (精细空间判别)
```

**设计动机**：
- 训练初期：编码器还没学好，困难负样本噪声太大，用随机负样本建立基础匹配能力
- 训练后期：编码器已有一定能力，困难负样本迫使模型学习精细的空间判别
- 困难负样本半径 `hard_radius=4` 像素（在 60×80 特征图上），对应原始图像 ~32 像素

### 温度参数 τ

```
τ = 0.07 (默认)

相似度分布:
  τ 小 → 分布尖锐 → 正/负样本区分更明显 → 训练更难但效果更好
  τ 大 → 分布平滑 → 梯度更均匀 → 训练更容易但判别力弱

典型值: CLIP 用 0.07, MoCo 用 0.07, SimCLR 用 0.1-0.5
```

---

## Step 5: 反向传播与优化

**代码位置**: `pretrain_encoder.py` → `train_one_epoch()` (line ~505)

### 单步训练流程

```python
for batch in dataloader:
    # 1. 数据增强
    image, depth = augmentor(image, depth)
    
    # 2. GT 投影
    projected_coords, valid_mask = project_depth_to_rgb(depth, gt_pose, ...)
    
    # 3. 特征提取 (仅编码器)
    fmap_rgb = model.image_encoder(image)
    fmap_depth = model.depth_encoder(depth)
    
    # 4. 对比损失
    loss, accuracy = criterion(fmap_rgb, fmap_depth, projected_coords, valid_mask)
    
    # 5. 反向传播
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(encoder_params, max_norm=1.0)  # 梯度裁剪
    optimizer.step()
```

### 优化器配置

```
优化器: AdamW
  lr = 1e-3
  weight_decay = 1e-5

学习率调度: CosineAnnealingLR
  T_max = epochs
  eta_min = lr × 0.01 = 1e-5
  lr(t) = eta_min + 0.5 × (lr - eta_min) × (1 + cos(πt/T_max))

梯度裁剪: max_norm = 1.0
  防止对比损失中的极端梯度 (如 logsumexp 中的大值)
```

### 训练日志格式

```
  Epoch [15] Step [10/55]  Loss: 2.3456  Acc: 45.2%  Time: 0.182s
  Epoch [15/30] (lr=5.23e-04, hard_ratio=0.50)
    Train → Loss: 2.1234  Acc: 48.5%
    Val   → Loss: 2.3456  Acc: 42.1%
```

---

## Step 6: 验证与保存

**代码位置**: `pretrain_encoder.py` → `validate()` (line ~545) + `main()` (line ~800)

### 验证流程

```python
@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    for batch in dataloader:
        # 不做数据增强
        projected_coords, valid_mask = project_depth_to_rgb(depth, gt_pose, ...)
        fmap_rgb = model.image_encoder(image)
        fmap_depth = model.depth_encoder(depth)
        loss, accuracy = criterion(fmap_rgb, fmap_depth, projected_coords, valid_mask)
```

### Checkpoint 保存策略

```
checkpoints/pretrain_encoder/
  pre_001/
    pretrain_log.txt          ← 完整训练日志
    encoder_pretrained.pth    ← 每个 epoch 的 checkpoint
    encoder_best.pth          ← 验证损失最低的 checkpoint
```

**保存内容**:
```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),       # 完整模型权重
    "optimizer_state_dict": optimizer.state_dict(), # 优化器状态
    "best_val_loss": best_val_loss,
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "args": vars(args),                            # 所有命令行参数
}, path)
```

### 如何使用预训练权重

```bash
python train.py \
    --config configs/chess_train.json \
    --pretrained_encoder checkpoints/pretrain_encoder/pre_027/encoder_best.pth \
    --checkpoint_dir checkpoints/runs_pt1
```

`train.py` 会加载 `encoder_best.pth` 中的 `image_encoder` 和 `depth_encoder` 权重，初始化端到端训练。

---

## 8. 验证脚本概览

**文件**: `validate.py`

### 核心功能

对**单对图像**进行推理验证，输出详细的误差指标和可视化结果。

```
输入:
  --checkpoint    模型权重文件 (.pth)
  --image         RGB 图像 (frame A)
  --depth         深度图 (frame B)
  --pose_image    frame A 的世界位姿 (4×4)
  --pose_depth    frame B 的世界位姿 (4×4)
  --intrinsics    相机内参 fx fy cx cy

输出:
  result.png          三栏对比图 (原图 | GT投影 | 预测投影)
  result.pred.pcd     预测位姿下的彩色点云
  result.gt.pcd       GT 位姿下的彩色点云
  result.json         完整指标 JSON
  test_log.txt        控制台日志
```

### 整体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      validate.py 主流程                              │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 加载图像  │    │ 加载位姿  │    │ 计算 GT  │    │ 加载模型  │      │
│  │ + 深度    │    │ 4×4 矩阵 │    │ T_rel    │    │ checkpoint│     │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                       │             │
│                                                       ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 可视化   │ ←  │ 误差计算  │ ←  │ 世界位姿  │ ←  │ 模型推理  │      │
│  │ + PCD    │    │ rot/trans │    │ 组合     │    │ 12轮迭代  │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 7: 加载数据与计算 GT

**代码位置**: `validate.py` → `main()` (line ~370)

### 7.1 数据加载

```python
image = load_image(args.image, image_size)       # (3, H, W) float32 [0,1]
depth = load_depth(args.depth, image_size)       # (1, H, W) float32 米
T_image = load_pose(args.pose_image)             # (4, 4) float32
T_depth_gt = load_pose(args.pose_depth)          # (4, 4) float32
```

**深度加载细节**:
```python
def load_depth(path, image_size=None, depth_scale=0.001):
    depth = Image.open(path)                     # 16-bit PNG
    arr = np.array(depth, dtype=np.float32) * depth_scale  # raw → meters
    arr = np.clip(arr, 0.0, 10.0)               # 截断到 10m
    return torch.from_numpy(arr).unsqueeze(0)    # (1, H, W)
```

### 7.2 计算 GT 相对位姿

```
已知:
  T_image    = frame A 的世界位姿 (4×4)
  T_depth_gt = frame B 的世界位姿 (4×4)

计算:
  T_rel_gt = T_image^(-1) @ T_depth_gt

含义:
  T_rel_gt 将深度相机坐标系中的点变换到 RGB 相机坐标系
  即: P_rgb = T_rel_gt @ P_depth
```

```
世界坐标系
     │
     ├── T_image ──→ RGB 相机坐标系 (frame A)
     │
     └── T_depth_gt ──→ 深度相机坐标系 (frame B)

T_rel_gt = T_image^(-1) @ T_depth_gt
         = 从深度相机到 RGB 相机的变换
```

### 7.3 位姿格式转换

```
4×4 矩阵 ←→ 7D 向量 [qw, qx, qy, qz, tx, ty, tz]

矩阵 → 7D: pose_matrix_to_7d()
  使用 Shepperd 方法从旋转矩阵提取四元数
  四元数归一化: q = q / ‖q‖

7D → 矩阵: quat_to_rotation_matrix()
  R[0,0] = 1 - 2(qy² + qz²)
  R[0,1] = 2(qxqy - qzqw)
  ...
```

---

## Step 8: 模型推理

**代码位置**: `validate.py` → `main()` (line ~440)

### 8.1 模型加载

```python
checkpoint = torch.load(args.checkpoint, map_location="cpu")

# 从 checkpoint 恢复模型超参数
ckpt_args = checkpoint.get("args", {})
model = RAFTPose(
    image_encoder=ckpt_args.get("image_encoder", "basic"),
    hidden_dim=ckpt_args.get("hidden_dim", 128),
    num_iterations=ckpt_args.get("num_iterations", 12),
    max_rot_step=ckpt_args.get("max_rot_step", 0.3),
    max_trans_step=ckpt_args.get("max_trans_step", 0.3),
    # ... 所有超参数从 checkpoint 恢复
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

**设计**：所有模型超参数保存在 checkpoint 中，验证时无需手动指定。

### 8.2 前向推理

```python
with torch.no_grad():
    pred_pose, pose_sequence = model(
        image=image_b,           # (1, 3, H, W)
        depth=depth_b,           # (1, 1, H, W)
        intrinsic_rgb=intrinsic_b,
        intrinsic_depth=intrinsic_b,
        return_all_poses=True,   # 返回所有迭代的中间 pose
    )
```

**输出**:
```
pred_pose:      (1, 7)           ← 最终预测的相对位姿
pose_sequence:  (1, 13, 7)      ← 初始 pose + 12 轮迭代输出
                [0] = 初始 pose (identity)
                [1] = 第 1 轮迭代后
                ...
                [12] = 第 12 轮迭代后 (最终)
```

### 8.3 世界位姿组合

```python
# 预测的世界位姿: T_pred_world = T_image @ T_rel_pred
T_rel_pred = pose_7d_to_matrix(pred_rel_pose_7d)
T_pred_world = T_image.numpy() @ T_rel_pred

# GT 世界位姿: 就是 T_depth_gt
T_depth_gt_np = T_depth_gt.numpy()
```

---

## Step 9: 误差计算

**代码位置**: `validate.py` → `compute_metrics()` (line ~175)

### 9.1 旋转误差 (Geodesic Distance)

```python
def compute_metrics(pred_pose_7d, gt_pose_7d):
    pred_q = pred_pose_7d[:4]   # (qw, qx, qy, qz)
    gt_q = gt_pose_7d[:4]

    # 归一化
    pred_q = pred_q / ‖pred_q‖
    gt_q = gt_q / ‖gt_q‖

    # 测地线距离
    dot = |pred_q · gt_q|       # 取绝对值 (q 和 -q 表示相同旋转)
    dot = clip(dot, 0, 1)
    rot_error_rad = 2 × arccos(dot)
    rot_error_deg = degrees(rot_error_rad)
```

**数学含义**：
$$\theta = 2 \arccos(|\mathbf{q}_{pred} \cdot \mathbf{q}_{gt}|)$$

这是四元数空间中的测地线距离，范围 $[0°, 180°]$，表示两个旋转之间的最小旋转角。

### 9.2 平移误差

```python
    trans_error = ‖pred_t - gt_t‖₂
```

欧氏距离，单位：米。

### 9.3 双重误差报告

验证脚本报告**两种误差**：

```
1. 相对位姿误差 (Relative Pose Error):
   模型直接预测的 T_rel vs GT 的 T_rel
   → 反映模型本身的预测精度

2. 世界位姿误差 (World Pose Error):
   T_image @ T_rel_pred vs T_depth_gt
   → 反映在实际应用中的精度
   → 当 T_image 有噪声时，世界误差 > 相对误差
```

### 9.4 迭代收敛曲线

```python
for i, p in enumerate(pose_seq_np):
    m = compute_metrics(p, gt_rel_pose_7d)
    print(f"  Iter {i:2d}: rot={m['rotation_error_deg']:8.4f}°  trans={m['translation_error_m']:.6f}m")
```

输出示例：
```
  Iter  0: rot= 45.2341°  trans=0.523412m    ← 初始 (identity)
  Iter  1: rot= 32.1234°  trans=0.412341m
  Iter  2: rot= 21.4567°  trans=0.312345m
  ...
  Iter 11: rot=  5.1234°  trans=0.051234m    ← 最终
  Iter 12: rot=  4.8765°  trans=0.048765m    ← 最终输出
```

---

## Step 10: 可视化与输出

**代码位置**: `validate.py` → `create_comparison_figure()` (line ~260) + `save_colored_pcd()` (line ~300)

### 10.1 三栏对比图

```
┌──────────────┬──────────────┬──────────────────────────────┐
│ Original RGB │ GT World Pose│ Pred World (rot=4.88°,       │
│              │              │  t=0.0488m)                   │
│              │              │                              │
│   原始图像    │  GT位姿投影   │  预测位姿投影                  │
│              │  深度→RGB    │  深度→RGB                    │
│              │  彩色点云     │  彩色点云                     │
└──────────────┴──────────────┴──────────────────────────────┘
```

**投影渲染流程** (`depth_to_colored_pointcloud()`):

```
1. 深度图 → 3D 点云 (深度相机坐标系)
   x = (u - cx) / fx × depth
   y = (v - cy) / fy × depth
   z = depth

2. 变换到 RGB 相机坐标系
   P_rgb = R @ P_depth + t    (使用 T_rel)

3. 投影到 RGB 图像平面
   u_proj = fx × (X/Z) + cx
   v_proj = fy × (Y/Z) + cy

4. 在投影位置采样 RGB 图像颜色
   color = image[v_proj, u_proj]

5. 渲染为图像
   projected_img[v_proj, u_proj] = color
```

**判断标准**：
- GT 投影应该与原始 RGB 图像对齐（深度点云投影后颜色正确）
- 预测投影越接近 GT 投影，说明预测位姿越准确

### 10.2 PCD 点云文件

```
输出: result.pred.pcd, result.gt.pcd

格式: ASCII PCD (Point Cloud Data)
字段: x y z rgb

内容: 3D 点在 RGB 相机坐标系中的坐标 + 从 RGB 图像采样的颜色
用途: 可用 CloudCompare 等工具打开，3D 视角对比 GT vs 预测
```

### 10.3 JSON 指标文件

```json
{
  "checkpoint": "checkpoints/runs_080/model_best.pth",
  "image": "7Scenes/data/chess/seq-01/color_000.png",
  "depth": "7Scenes/data/chess/seq-01/depth_050.png",
  "loss": {
    "total_loss": 0.123456,
    "rot_loss_deg": 4.8765,
    "trans_loss": 0.048765
  },
  "relative_pose_metrics": {
    "rotation_error_deg": 4.8765,
    "translation_error_m": 0.048765
  },
  "world_pose_metrics": {
    "rotation_error_deg": 4.8765,
    "translation_error_m": 0.048765
  },
  "predicted_rel_pose": [0.999, 0.001, -0.002, 0.003, 0.01, -0.02, 0.03],
  "gt_rel_pose": [0.999, 0.001, -0.002, 0.003, 0.01, -0.02, 0.03],
  "convergence": [
    {"iteration": 0, "rotation_error_deg": 45.23, "translation_error_m": 0.523},
    {"iteration": 1, "rotation_error_deg": 32.12, "translation_error_m": 0.412},
    ...
    {"iteration": 12, "rotation_error_deg": 4.88, "translation_error_m": 0.049}
  ]
}
```

---

## 13. 命令行参数速查

### pretrain_encoder.py

```bash
python pretrain_encoder.py \
    --config configs/allscenes_train.json \     # 数据集配置
    --image_encoder resnet18 \                   # 编码器类型: basic/small/resnet18
    --shared_encoder \                           # 使用 Siamese 编码器 (可选)
    --epochs 30 \                                # 训练轮数
    --batch_size 30 \                            # 批大小
    --lr 1e-3 \                                  # 学习率
    --temperature 0.07 \                         # InfoNCE 温度
    --loss_type dense \                          # 损失类型: infonce/dense
    --num_negatives 128 \                        # 负样本数量
    --hard_radius 4 \                            # 困难负样本半径 (像素)
    --hard_ratio 0.75 \                          # 困难负样本比例上限
    --grad_clip 1.0 \                            # 梯度裁剪
    --scheduler cosine \                         # 学习率调度: cosine/none
    --checkpoint_dir checkpoints/pretrain_encoder # 保存目录
```

### validate.py

```bash
python validate.py \
    --checkpoint checkpoints/runs_080/model_best.pth \  # 模型权重
    --image 7Scenes/data/chess/seq-01/color_000.png \   # RGB 图像
    --depth 7Scenes/data/chess/seq-01/depth_050.png \   # 深度图
    --pose_image 7Scenes/data/chess/seq-01/pose_000.txt \ # frame A 位姿
    --pose_depth 7Scenes/data/chess/seq-01/pose_050.txt \ # frame B 位姿
    --intrinsics 585 585 320 240 \                      # 相机内参
    --depth_scale 0.001 \                                # 深度缩放因子
    --image_size 480 640 \                               # 图像尺寸 (可选)
    --output_prefix result \                             # 输出前缀 (可选)
    --checkpoint_dir checkpoints                         # 测试输出目录
```

---

## 附录: 两个脚本的关系

```
pretrain_encoder.py                    train.py                     validate.py
       │                                  │                              │
       │  预训练编码器                      │  端到端训练                    │  单样本验证
       │  (对比学习)                       │  (序列损失)                   │  (推理+可视化)
       │                                  │                              │
       ▼                                  ▼                              ▼
  encoder_best.pth ──────────────────→ runs_XXX/model_best.pth ──────→ result.png
                                       (完整模型权重)                   result.json
                                                                       result.pcd

训练流程:
  1. pretrain_encoder.py  →  encoder_best.pth  (学习特征表示)
  2. train.py --pretrained_encoder encoder_best.pth  →  model_best.pth  (学习位姿估计)
  3. validate.py --checkpoint model_best.pth  →  可视化结果  (评估精度)
```
