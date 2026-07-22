#!/usr/bin/env python3
"""
visualize_iter_corr.py — 验证推理时每一 iter 位姿的相关性热力图, 对比 GT

展示迭代精修过程: iter0(identity init)→iterN(final) 每个位姿下的 per-pixel sim 热力图,
与 GT 位姿的 sim 热图并排对比。直观看到相关性从模糊/错位 → 锐化对齐 → 接近 GT。
附收敛曲线: rot error, trans error 与 sim mean 随 iter 变化。
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
from diagnose_runs104_corr import cos_sim_at_pose


def rot_error_deg(q1, q2):
    q1 = q1.flatten()[:4]; q2 = q2.flatten()[:4]
    dot = (q1 * q2).sum().clamp(-1, 1)
    return (2 * torch.acos(dot) * 180 / np.pi).item()


def trans_error_m(p1, p2):
    """两 pose 的平移误差 (m), pose=[qw,qx,qy,qz, tx,ty,tz]。"""
    return (p1.flatten()[4:7] - p2.flatten()[4:7]).norm().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = "checkpoints/runs_104/model_best.pth"
    print(f"Device: {device}\nLoading: {ckpt}")
    model, args = load_model_for_eval(ckpt, device)
    n_iter = args.get("num_iterations", 6)
    #n_iter = 24
    ds = SevenScenesDataset(config_path="configs/allscenes_train.json", split="val", image_size=(480, 640))
    out = Path("checkpoints/paper_figs"); out.mkdir(parents=True, exist_ok=True)

    N_SAMPLES = 1
    indices = [100][:N_SAMPLES]
    VMIN, VMAX = -0.3, 0.85

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

            # 推理, 拿逐 iter 位姿序列
            result = model(image=image, depth=depth, intrinsic_rgb=K_rgb, intrinsic_depth=K_depth,
                           return_all_poses=True, num_iterations=n_iter)
            pred_pose, extra = result
            pose_seq = extra['pose_sequence']  # (1, iters+1, 7)

            # GT sim 热图 (目标)
            sim_gt, valid_gt = cos_sim_at_pose(fmap_rgb_n, fmap_d_n, depth, gt_pose, K_depth, K_rgb)
            sim_gt_2d = np.where(valid_gt[0].cpu().numpy(), sim_gt[0].cpu().numpy(), np.nan)
            gt_mean = np.nanmean(sim_gt_2d)

            # 每 iter 位姿 sim 热图 + rot/trans error
            iters_data = []
            for t in range(pose_seq.shape[1]):
                pose_t = pose_seq[0, t]
                sim_t, valid_t = cos_sim_at_pose(fmap_rgb_n, fmap_d_n, depth, pose_t.unsqueeze(0), K_depth, K_rgb)
                sim_2d = np.where(valid_t[0].cpu().numpy(), sim_t[0].cpu().numpy(), np.nan)
                rerr = rot_error_deg(pose_t, gt_pose[0])
                terr = trans_error_m(pose_t, gt_pose[0])
                iters_data.append((t, sim_2d, rerr, terr, np.nanmean(sim_2d)))

        # ── 可视化 ──────────────────────────────────────────────────────────
        n_panels = len(iters_data) + 1  # iters + GT
        fig = plt.figure(figsize=(2.9 * n_panels, 10.5))
        gs = fig.add_gridspec(3, n_panels, hspace=0.45, wspace=0.12, height_ratios=[2.2, 3, 2])
        half = max(1, n_panels // 2)

        # Row0: 两帧原图 (RGB | Depth)
        ax = fig.add_subplot(gs[0, :half])
        ax.imshow(image[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
        ax.set_title("RGB frame (frame A)", fontsize=10); ax.axis('off')
        ax = fig.add_subplot(gs[0, half:])
        ax.imshow(depth[0, 0].cpu().numpy(), cmap='turbo')
        ax.set_title("Depth frame (frame B, raw m)", fontsize=10); ax.axis('off')

        # Row1: 每 iter 的 sim 热图 + GT
        for col, (t, sim_2d, rerr, terr, smean) in enumerate(iters_data):
            ax = fig.add_subplot(gs[1, col])
            ax.imshow(sim_2d, cmap='jet', vmin=VMIN, vmax=VMAX)
            tag = "iter0 (init)" if t == 0 else (f"iter{t}" if t < len(iters_data) - 1 else f"iter{t} (final)")
            ax.set_title(f"{tag}\nrot={rerr:.1f}° t={terr:.3f}m\nsim={smean:.3f}", fontsize=8.5)
            ax.axis('off')
        ax = fig.add_subplot(gs[1, -1])
        im = ax.imshow(sim_gt_2d, cmap='jet', vmin=VMIN, vmax=VMAX)
        ax.set_title(f"GT (target)\nrot=0° t=0m\nsim={gt_mean:.3f}", fontsize=8.5, color='green')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='cosine sim')

        # Row2: 收敛曲线 (rot / trans / sim vs iter, 三轴)
        ax = fig.add_subplot(gs[2, :])
        xs = [d[0] for d in iters_data]
        rot_y = [d[2] for d in iters_data]
        trans_y = [d[3] for d in iters_data]
        sim_y = [d[4] for d in iters_data]
        ax.plot(xs, rot_y, 'o-', color='C3', lw=2, label='rot error (°)')
        ax.set_xlabel("iteration"); ax.set_ylabel("rot error (°)", color='C3')
        ax.tick_params(axis='y', labelcolor='C3')
        ax2 = ax.twinx()
        ax2.plot(xs, sim_y, 's--', color='C0', lw=1.5, label='sim mean')
        ax2.axhline(gt_mean, color='C0', ls=':', alpha=0.6)
        ax2.set_ylabel("sim mean", color='C0'); ax2.tick_params(axis='y', labelcolor='C0')
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(xs, trans_y, '^-.', color='C2', lw=1.5, label='trans error (m)')
        ax3.set_ylabel("trans error (m)", color='C2'); ax3.tick_params(axis='y', labelcolor='C2')
        ax.set_title(f"Iterative convergence — sample {idx}: "
                     f"rot {rot_y[0]:.1f}→{rot_y[-1]:.1f}°, trans {trans_y[0]:.3f}→{trans_y[-1]:.3f}m, "
                     f"sim {sim_y[0]:.3f}→{sim_y[-1]:.3f}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax.legend(h1 + h2 + h3, l1 + l2 + l3, loc='center right', fontsize=8)

        p = out / f"iter_corr_sample{idx}.png"
        fig.savefig(p, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ {p}  rot {rot_y[0]:.1f}→{rot_y[-1]:.1f}°, trans {trans_y[0]:.3f}→{trans_y[-1]:.3f}m, "
              f"sim {sim_y[0]:.3f}→{sim_y[-1]:.3f} (GT {gt_mean:.3f})")

    print(f"\n✓ 完成 {N_SAMPLES} 张: {out}/iter_corr_sample*.png")


if __name__ == "__main__":
    main()
