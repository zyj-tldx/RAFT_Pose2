#!/usr/bin/env python3
"""
visualize_corr.py — 可视化 RGB↔Depth 两帧的特征相关性热力图

为每个样本生成一张综合图:
  Row1: RGB 帧 | Depth 帧(raw) | GT 位姿 per-pixel sim 热力图 | 扰动位姿 sim 热力图
  Row2: 3 个 query 点的 corr slice (RGB query 像素 vs depth 全图的相关性分布,
        红色 + = query 位置, 黄色 × = corr 峰值=最佳匹配) | RGB query 点标记
  Row3: GT vs 扰动 sim 直方图 | 文字说明

用途: 直观理解两帧匹配质量(边缘/纹理高 sim, 平坦/遮挡低 sim),
      GT vs 扰动 sim 差距 = 相关性判别力, corr slice 峰值偏移 = 位姿误差。
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalute_benchmark import load_model_for_eval
from dataloader import SevenScenesDataset
from diagnose_runs104_corr import cos_sim_at_pose, perturb_pose  # 复用几何


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "checkpoints/runs_104/model_best.pth"
    print(f"Device: {device}\nLoading: {ckpt}")
    model, args = load_model_for_eval(ckpt, device)

    ds = SevenScenesDataset(config_path="configs/allscenes_train.json", split="val",
                            image_size=(480, 640))
    out = Path("checkpoints/paper_figs")
    out.mkdir(parents=True, exist_ok=True)

    N_SAMPLES = 3
    indices = [0, 200, 700][:N_SAMPLES]
    PERT_NOISE = 0.15  # rad ≈ 8.6°

    for idx in indices:
        s = ds[idx]
        image = s["image"].unsqueeze(0).to(device)
        depth = s["depth"].unsqueeze(0).to(device)
        gt_pose = s["gt_pose"].unsqueeze(0).to(device)
        K_rgb = s["intrinsic_rgb"].unsqueeze(0).to(device)
        K_depth = s["intrinsic_depth"].unsqueeze(0).to(device)

        with torch.no_grad():
            fmap_rgb = model.image_encoder(image)
            fmap_d = model.depth_encoder(depth).float()
            if fmap_rgb.shape[2:] != fmap_d.shape[2:]:
                fmap_d = F.interpolate(fmap_d, size=fmap_rgb.shape[2:], mode='bilinear', align_corners=False)
            fmap_rgb_n = F.normalize(fmap_rgb.float(), dim=1)
            fmap_d_n = F.normalize(fmap_d, dim=1)
            _, _, fh, fw = fmap_rgb.shape  # 60, 80

            sim_gt, valid_gt = cos_sim_at_pose(fmap_rgb_n, fmap_d_n, depth, gt_pose, K_depth, K_rgb)
            pert_pose = perturb_pose(gt_pose, PERT_NOISE, device)
            sim_pert, valid_pert = cos_sim_at_pose(fmap_rgb_n, fmap_d_n, depth, pert_pose, K_depth, K_rgb)

        sim_gt_2d = sim_gt[0].cpu().numpy()
        sim_pert_2d = sim_pert[0].cpu().numpy()
        vg = valid_gt[0].cpu().numpy()
        vp = valid_pert[0].cpu().numpy()
        sim_gt_2d = np.where(vg, sim_gt_2d, np.nan)
        sim_pert_2d = np.where(vp, sim_pert_2d, np.nan)

        # query 点 (fmap 分辨率, 避开边缘)
        qpoints = [(fh // 4, fw // 4), (fh // 2, fw // 3), (3 * fh // 4, 2 * fw // 3)]

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.40, wspace=0.28)

        # Row 1: RGB | Depth | GT sim | Pert sim
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(image[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        ax.set_title("RGB frame"); ax.axis('off')
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(depth[0, 0].cpu().numpy(), cmap='turbo')
        ax.set_title("Depth frame (raw, m)"); ax.axis('off')
        ax = fig.add_subplot(gs[0, 2])
        im = ax.imshow(sim_gt_2d, cmap='jet', vmin=-0.3, vmax=0.85)
        ax.set_title(f"GT-pose sim (mean={np.nanmean(sim_gt_2d):.3f})"); ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = fig.add_subplot(gs[0, 3])
        im = ax.imshow(sim_pert_2d, cmap='jet', vmin=-0.3, vmax=0.85)
        ax.set_title(f"Perturbed({PERT_NOISE*180/np.pi:.1f}°) sim (mean={np.nanmean(sim_pert_2d):.3f})")
        ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 2: corr slices + RGB query 标记
        for i, (qy, qx) in enumerate(qpoints):
            with torch.no_grad():
                q_feat = fmap_rgb_n[0, :, qy, qx]  # (C,)
                corr_2d = (fmap_d_n[0] * q_feat.view(-1, 1, 1)).sum(0).cpu().numpy()  # (fh,fw)
            peak = np.unravel_index(np.argmax(corr_2d), corr_2d.shape)
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(corr_2d, cmap='viridis')
            ax.plot(qx, qy, 'r+', ms=14, mew=2.5)           # query 位置 (depth 空间近似)
            ax.plot(peak[1], peak[0], 'yx', ms=14, mew=2.5)  # corr 峰值
            ax.set_title(f"corr slice query({qy},{qx})\nred+=query, yellow×=peak")
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax = fig.add_subplot(gs[1, 3])
        ax.imshow(image[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        for qy, qx in qpoints:
            ax.plot(qx * 8, qy * 8, 'r+', ms=16, mew=2.5)   # fmap→原图 ×8
        ax.set_title("RGB query points (×8 to原图)"); ax.axis('off')

        # Row 3: 直方图 + 说明
        ax = fig.add_subplot(gs[2, :2])
        g = sim_gt_2d[~np.isnan(sim_gt_2d)]
        p = sim_pert_2d[~np.isnan(sim_pert_2d)]
        ax.hist(g, bins=50, color='C2', alpha=0.7, label=f"GT (mean={g.mean():.3f})")
        ax.hist(p, bins=50, color='C3', alpha=0.5, label=f"Perturbed (mean={p.mean():.3f})")
        ax.set_xlabel("cosine similarity"); ax.set_ylabel("pixel count")
        ax.legend(); ax.set_title("Sim 分布: GT vs 扰动 (差距=判别力)")

        ax = fig.add_subplot(gs[2, 2:])
        ax.axis('off')
        margin = np.nanmean(sim_gt_2d) - np.nanmean(sim_pert_2d)
        info = (
            f"Sample idx={idx}\n\n"
            f"• GT sim 均值 {np.nanmean(sim_gt_2d):.3f}: 高 sim=边缘/纹理(判别区), 低=平坦/遮挡\n"
            f"• 扰动({PERT_NOISE*180/np.pi:.1f}°) sim 均值 {np.nanmean(sim_pert_2d):.3f}\n"
            f"• 判别 margin = {margin:+.3f}  (>0 表示 GT 更匹配 = 信号有效)\n"
            f"• corr slice: red+=query 位置, yellow×=corr 峰值(最佳匹配);\n"
            f"  GT 对齐时峰值应贴近 query; 偏移量反映该局部位姿误差"
        )
        ax.text(0.0, 0.5, info, fontsize=11, va='center', family='monospace')

        fig.suptitle(f"runs104 RGB↔Depth Feature Correlation — sample {idx}", fontsize=14, y=0.995)
        p = out / f"corr_vis_sample{idx}.png"
        fig.savefig(p, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ {p}  (GT sim={np.nanmean(sim_gt_2d):.3f}, margin={margin:+.3f})")

    print(f"\n✓ 完成，共 {N_SAMPLES} 张: {out}/corr_vis_sample*.png")


if __name__ == "__main__":
    main()
