## 发论文路线建议

### 当前定位

你的核心贡献是 **RAFT-style 迭代精修框架用于 camera-LiDAR 外参标定**，这本身是有新意的——之前没人把 RAFT 的迭代 correlation + ConvGRU 思路用到标定任务上。但 12° 的精度需要提升到 5° 以内才有说服力。

### 推荐路线：两步走

---

#### 第一步：快速提升精度（1-2 周）

不改大架构，用代码改动换 3-5° 的提升：

**1. Cross-Attention 替代 Correlation Volume（最关键）**

这是收益最大的一项改动。当前 CorrBlock 用局部窗口匹配，换成 Transformer cross-attention 后：
- 全局匹配，不受 `corr_radius` 限制
- 不增加显存（attention 是 $O(HW)$ 而非 $O(H^2W^2)$）
- 可以在 1/8 分辨率上实现全局感受野

具体做法：在 modules 下新增一个 `CrossAttentionMatcher`，替换 `CorrBlock`：
- RGB features 作为 Q，Depth features 作为 K/V
- Multi-head attention + positional encoding
- 输出和当前 correlation features 形状兼容，PoseUpdateNet 不需要改

**2. Depth 反距离变换（半小时）**

```python
# 在 dataloader.py 或 raft_pose.py forward 中
depth_inv = 1.0 / (depth.clamp(min=0.1) + 0.5)
```

**3. 数据增强增强（半天）**

当前 `augment` 可能较弱。增加：
- RGB: 颜色抖动、随机高斯噪声
- Depth: 随机 dropout（模拟 LiDAR 稀疏性）、高斯噪声
- 随机裁剪（从 480×640 裁到 384×512 之类）

**预期效果**：Cross-Attention + 数据增强 → Val Rot **5-8°**

---

#### 第二步：论文撰写（同步进行）

**论文标题方向**：
> "RAFT-Calib: Iterative Feature Matching for Camera-LiDAR Extrinsic Calibration via Recurrent All-Pairs Field Transforms"

**核心贡献（Story）**：
1. **新框架**：首次将 RAFT 的迭代精修范式应用到 camera-LiDAR 标定，通过 correlation/attention volume + ConvGRU 实现从粗到精的位姿估计
2. **双编码器设计**：RGB 和 Depth 使用同架构独立编码器，解决多模态特征尺度不匹配问题
3. **Delta 监督**：直接监督每步迭代的位姿修正量，加速收敛
4. **Cross-Attention 匹配**（如果实现了）：全局特征匹配替代局部 correlation，对大基线更鲁棒

**实验设计**：

| 实验 | 内容 |
|------|------|
| 主实验 | 7Scenes 全场景对比 SOTA（CalibNet, RegNet, CMRNext 等） |
| 消融 1 | 编码器对比：BasicEncoder vs ResNet18 vs Dual ResNet18 |
| 消融 2 | 迭代次数：6 vs 12 vs 18 |
| 消融 3 | max_step：0.1 vs 0.15 vs 0.2 |
| 消融 4 | Correlation vs Cross-Attention |
| 消融 5 | Delta supervision 有无 |
| 消融 6 | 预训练 encoder 有无 |
| 鲁棒性 | 不同初始噪声下的收敛曲线 |
| 效率 | 参数量、FLOPs、推理时间对比 |

**对比方法**（需要复现或引用）：
- CalibNet (CVPR 2019)
- RegNet (ICRA 2020)  
- CC (ECCV 2020)
- CMRNext (如果是对标方法)

### 投稿目标

| 会议/期刊 | 截稿周期 | 适合度 |
|-----------|---------|--------|
| ECCV 2026 | 已过 | - |
| ACCV 2026 | ~7月 | ⭐⭐⭐ |
| AAAI 2027 | ~8月 | ⭐⭐ |
| ICRA 2027 | ~9月 | ⭐⭐⭐（机器人方向最对口） |
| CVPR 2027 | ~11月 | ⭐⭐⭐ |
| RA-L | 滚动 | ⭐⭐（快速发表） |

### 建议优先级

1. **立即**：实现 Cross-Attention matcher（这是论文最大的卖点）
2. **同步**：加数据增强 + depth 预处理
3. **然后**：跑实验、做消融
4. **最后**：写论文

需要我帮你设计 Cross-Attention matcher 的具体架构吗？