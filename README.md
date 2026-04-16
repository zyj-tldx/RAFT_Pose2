# RAFT-Pose

基于 RAFT（Recurrent All-Pairs Field Transforms）架构的相机-LiDAR 外参标定模块，使用多姿态采样和 4D 相关体积进行迭代姿态估计。

## 📁 目录结构

```
RAFT_Pose2/
├── raft_pose.py             # RAFT-Pose 主模型
├── pose_loss.py             # 姿态损失函数（旋转 + 平移）
├── dataloader.py            # 7Scenes 数据集加载器
├── train.py                 # 训练脚本
├── get_model.py             # 模型工厂函数
├── example_usage.py         # 使用示例
├── projection_utils.py      # 投影工具
├── configs/
│   └── chess_train.json     # 训练配置（由脚本生成）
├── scripts/
│   └── generate_pairs.py    # 数据配对生成脚本
├── checkpoints/             # 模型检查点保存目录
├── modules/
│   ├── __init__.py
│   ├── pose_utils.py        # 姿态工具（四元数、se(3) 等）
│   ├── pose_extractor.py    # 特征编码器（BasicEncoder / SmallEncoder / DepthEncoder）
│   ├── depth_projection.py  # 深度投影与 CorrBlock 相关体积
│   └── pose_update.py       # ConvGRU 姿态更新网络
└── 7Scenes/                 # 数据集目录（需自行准备）
```

## 🎯 核心思想

1. **特征提取**：CNN 编码器分别提取 RGB 图像和深度图的特征（stride=8 下采样到 H/8 × W/8）
2. **4D 相关体积**：计算 RGB 与深度特征之间的全对数相关
3. **迭代优化**（每轮迭代）：
   - 在当前姿态估计周围采样 N 个候选姿态
   - 用候选姿态将深度图投影到 RGB 相机平面
   - 从投影坐标采样相关体积，获取匹配特征
   - ConvGRU 更新隐状态 → 预测 7D 姿态增量（四元数 + 平移）
   - 使用 se(3) 李代数更新姿态
4. **输出**：最终姿态估计 `[qw, qx, qy, qz, tx, ty, tz]`

## 🚀 训练

### 环境准备

```bash
conda activate matr2d3d2
```

### Step 1: 编写定义 JSON

在 `configs/` 下创建定义 JSON，指定图像组（场景、序列、帧范围）和配对策略：

```json
{
    "dataset_root": "../7Scenes/data",
    "output": "chess_train.json",
    "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
    "image_size": [480, 640],
    "depth_scale": 0.001,
    "strategy": "random_offset",
    "strategy_args": {
        "min_offset": 1,
        "max_offset": 10,
        "samples_per_group": 200
    },
    "seed": 42,
    "groups": [
        {"scene": "chess", "seq": "seq-01", "frame_range": [0, 79], "split": "train"},
        {"scene": "chess", "seq": "seq-02", "frame_range": [0, 79], "split": "train"},
        {"scene": "chess", "seq": "seq-06", "frame_range": [0, 79], "split": "val"}
    ]
}
```

**定义 JSON 字段说明：**

| 字段 | 说明 |
|------|------|
| `dataset_root` | 7Scenes 数据集根目录（相对于定义 JSON 所在目录） |
| `output` | 输出数据集 JSON 文件名 |
| `strategy` | 配对策略：`random_offset`（随机偏移）或 `all_pairs`（两两全组合） |
| `strategy_args` | 策略参数（见下表） |
| `groups[]` | 图像组列表，每组指定 scene、seq、帧范围和 split |

**策略参数：**

| 参数 | `random_offset` | `all_pairs` |
|------|:-:|:-:|
| `min_offset` | ✅ 最小帧偏移 | — |
| `max_offset` | ✅ 最大帧偏移 | — |
| `samples_per_group` | ✅ 每组生成配对数 | — |

### Step 2: 生成数据集 JSON

```bash
python scripts/generate_pairs.py --definition configs/chess_definition.json
```

### Step 3: 开始训练

```bash
# 基础训练（推荐入门，SmallEncoder）
python train.py \
    --config configs/chess_train.json \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --image_encoder small

# 使用 BasicEncoder（更大模型，精度更高）
python train.py \
    --config configs/chess_train.json \
    --epochs 100 \
    --batch_size 2 \
    --lr 5e-5 \
    --image_encoder basic

# 使用完整 640×480 分辨率
python train.py \
    --config configs/chess_train.json \
    --epochs 100 \
    --batch_size 2 \
    --lr 5e-5 \
    --image_encoder basic \
    --image_size 480 640

# 低分辨率快速验证
python train.py \
    --config configs/chess_train.json \
    --epochs 5 \
    --batch_size 4 \
    --lr 1e-4 \
    --image_size 160 192
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必填 | 训练配置 JSON 文件路径 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 4 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--image_encoder` | `small` | 编码器类型：`small`（128 维，3.3M 参数）或 `basic`（256 维，14M 参数） |
| `--image_size` | `480 640` | 输入图像分辨率（高 宽） |
| `--hidden_dim` | 128 | ConvGRU 隐状态维度 |
| `--num_iterations` | 12 | 迭代优化次数 |
| `--num_pose_samples` | 16 | 每次迭代的姿态采样数 |
| `--save_dir` | `checkpoints` | 检查点保存目录 |
| `--log_dir` | `runs` | TensorBoard 日志目录 |

### 显存参考

| 配置 | 分辨率 | N | 显存峰值（推理） |
|------|--------|---|------------------|
| SmallEncoder, B=2 | 160×192 | 4 | ~50 MB |
| SmallEncoder, B=2 | 640×480 | 4 | ~1118 MB |
| BasicEncoder, B=2 | 640×480 | 4 | ~1170 MB |
| BasicEncoder, B=2 | 640×480 | 16 | ~3967 MB |

> 深度投影已在特征图分辨率（H/8 × W/8）上执行，640×480 分辨率可安全运行。

### 恢复训练

```bash
python train.py \
    --config configs/chess_train.json \
    --resume checkpoints/raft_pose_best.pth
```

### 监控训练

```bash
tensorboard --logdir runs
```

### 验证

```bash
python validate.py \
    --checkpoint checkpoints/model_best.pth \
    --image 7Scenes/data/chess/seq-01/color_000.png \
    --depth 7Scenes/data/chess/seq-01/depth_000.png \
    --pose  7Scenes/data/chess/seq-01/pose_000.txt \
    --intrinsics 585 585 320 240 \
    --output result.png

## 🔧 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_encoder` | str | `'basic'` | 编码器类型（`'basic'` 或 `'small'`）|
| `hidden_dim` | int | 128 | ConvGRU 隐状态维度 |
| `context_dim` | int | 64 | 上下文特征维度 |
| `depth_dim` | int | 32 | 深度特征维度 |
| `corr_levels` | int | 4 | 相关体积金字塔层数 |
| `corr_radius` | int | 4 | 局部相关采样半径 |
| `num_iterations` | int | 12 | 迭代优化次数 |
| `num_pose_samples` | int | 16 | 每次迭代的姿态采样数 |
| `pose_sample_std` | float | 0.01 | 姿态扰动标准差 |

## 💻 推理示例

```python
import torch
from raft_pose import RAFTPose

model = RAFTPose(image_encoder='basic', num_iterations=12).cuda()
model.eval()

B = 1
image = torch.randn(B, 3, 480, 640).cuda()
depth = torch.rand(B, 1, 480, 640).cuda() * 5.0
intrinsic = torch.tensor([[585, 0, 320], [0, 585, 240], [0, 0, 1]]).float()
intrinsic = intrinsic.unsqueeze(0).expand(B, -1, -1).cuda()

with torch.no_grad():
    pose = model(image, depth, intrinsic.clone(), intrinsic.clone())
# pose: (B, 7) — [qw, qx, qy, qz, tx, ty, tz]
```

## 🔬 技术细节

### 姿态表示

- **7D 向量**: `[qw, qx, qy, qz, tx, ty, tz]`
- 四元数（旋转）+ 平移向量
- 使用 se(3) 李代数进行小姿态更新，保证四元数归一化

### 4D 相关体积

- 形状: `(B, H_feat, W_feat, 1, H_feat, W_feat)`
- 多尺度金字塔采样，每个姿态提取 `(2*radius+1)² * num_levels` 维相关特征

### 内存优化

深度投影在特征图分辨率（H/8 × W/8）上执行，采样点数相比原始分辨率减少 64 倍，使得 640×480 分辨率在单卡（8 GB）即可训练。

## 📚 参考文献

- RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
- CMRNext: Camera-LiDAR Matching for Localization and Extrinsic Calibration
- 7Scenes: Scene Coordinate Regression for Camera Localization

## 训练参考

=== 同类方法的数据规模参考 ===

  RAFT (光流)
    参数: 5M, 数据: 229K pairs (FlyingChairs)+FlyingThings, 训练: ~200K steps

  CMRNext (camera-LiDAR)
    参数: ~10M, 数据: KITTI 43K + synthetic, 训练: ~100-200 epochs

  DeepV2D (深度)
    参数: ~10M, 数据: ScanNet 50K+ scenes, 训练: ~100 epochs

  DFC (位姿回归)
    参数: ~5M, 数据: 7Scenes ~5K-10K pairs, 训练: ~200 epochs

  PoseNet (位姿回归)
    参数: ~4M, 数据: 7Scenes ~5K frames, 训练: ~500 epochs

=== 本模型分析 ===

模型参数: 3.15M (Small) / 13.85M (Basic)
当前数据: 474 训练样本
参数/样本比: 6656:1 (严重不足)

=== 数据需求估算 ===

按参数量估计 (参考同类方法):
  Small (3.15M): 建议 3K-10K 样本, 即参数/样本比 300:1 ~ 1000:1
  Basic (13.85M): 建议 10K-50K 样本

但注意: 这个任务不是逐像素回归，而是全局 7D 回归
每个样本提供的监督信号覆盖整个网络 (不是只有局部)
所以实际需要的数据可能比参数量估算的少

=== 7Scenes 完整数据潜力 ===

  窗口 5 帧: ~1,000 训练对
  窗口 10 帧: ~2,000 训练对
  窗口 20 帧: ~4,000 训练对
  窗口 30 帧: ~6,000 训练对

=== 推荐方案 ===

1. 扩展数据到 ~5000-10000 样本
   - 7Scenes chess 场景所有 5 个训练序列的帧对
   - 每帧配前后 10-20 帧生成对
   - 也可以加入 7Scenes 其他 6 个场景

2. 训练轮数: 100-300 epochs (取决于数据量)
   - 数据少 (< 1K): 300-500 epochs, 需要 early stopping
   - 数据中 (3K-5K): 150-200 epochs
   - 数据多 (> 10K): 80-150 epochs

3. 当前最实际的方案:
   - 重新生成 chess 配置: ~5000 样本 (5 seqs × 40帧 × 25对/帧)
   - 训练 200 epochs + early stopping
   - 加入所有 7Scenes 场景: 可达 ~30K 样本
