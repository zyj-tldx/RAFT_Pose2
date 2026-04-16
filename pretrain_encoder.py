#!/usr/bin/env python3
"""
Pretrain RGB and Depth encoders using contrastive learning at GT pose alignment.

Key idea:
  - At GT pose, depth projection aligns depth pixels with their correct RGB counterparts
  - Train encoders so that aligned RGB-Depth pixel features have high cosine similarity
  - InfoNCE loss: positive = same spatial location, negatives = other locations
  - This breaks the chicken-and-egg problem in end-to-end training

Usage:
    python pretrain_encoder.py \
        --config configs/chess_train.json \
        --epochs 20 \
        --batch_size 8 \
        --lr 1e-3 \
        --temperature 0.07 \
        --checkpoint_dir checkpoints/pretrain_encoder
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Setup import path
RAFT_POSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RAFT_POSE_DIR))

from raft_pose import RAFTPose
from dataloader import SevenScenesDataset


# ─── Utilities ────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_next_run_dir(base_dir="checkpoints", prefix="pre"):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    existing = [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith(prefix + "_")]
    max_n = 0
    for name in existing:
        try:
            n = int(name.split("_")[-1])
            max_n = max(max_n, n)
        except ValueError:
            continue

    next_n = max_n + 1
    run_dir = base / f"{prefix}_{next_n:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir), next_n


# ─── Projection at GT Pose ───────────────────────────────────────────────────

def project_depth_to_rgb(depth, gt_pose_7d, intrinsic_depth, intrinsic_rgb):
    """
    Project depth map to RGB image plane using GT pose (7D: quat + trans).
    
    Args:
        depth: (B, 1, H, W) depth map in meters
        gt_pose_7d: (B, 7) [qw, qx, qy, qz, tx, ty, tz] relative pose
        intrinsic_depth: (B, 3, 3)
        intrinsic_rgb: (B, 3, 3)
    
    Returns:
        projected_coords: (B, 2, H, W) — (u, v) in RGB image plane
        valid_mask: (B, 1, H, W) — 1 where depth > 0 and projection is in-bounds
    """
    B, _, H, W = depth.shape
    device = depth.device
    depth_squeezed = depth.squeeze(1)  # (B, H, W)

    # Pixel coordinates
    u_d = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W).expand(B, H, W)
    v_d = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1).expand(B, H, W)

    # Unproject using depth intrinsic
    fx_d = intrinsic_depth[:, 0, 0].view(B, 1, 1)
    fy_d = intrinsic_depth[:, 1, 1].view(B, 1, 1)
    cx_d = intrinsic_depth[:, 0, 2].view(B, 1, 1)
    cy_d = intrinsic_depth[:, 1, 2].view(B, 1, 1)

    x_d = (u_d - cx_d) / fx_d * depth_squeezed
    y_d = (v_d - cy_d) / fy_d * depth_squeezed
    z_d = depth_squeezed

    # Valid depth mask
    valid_depth = (depth_squeezed > 0.01).float()

    # 3D points in depth camera frame (homogeneous)
    ones = torch.ones_like(z_d)
    P_D = torch.stack([x_d, y_d, z_d, ones], dim=1)  # (B, 4, H, W)

    # Convert 7D pose to 4x4 matrix
    qw, qx, qy, qz = gt_pose_7d[:, 0], gt_pose_7d[:, 1], gt_pose_7d[:, 2], gt_pose_7d[:, 3]
    tx, ty, tz = gt_pose_7d[:, 4], gt_pose_7d[:, 5], gt_pose_7d[:, 6]

    # Quaternion to rotation matrix (Shepperd's method)
    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2*(qy*qy + qz*qz)
    R[:, 0, 1] = 2*(qx*qy - qw*qz)
    R[:, 0, 2] = 2*(qx*qz + qw*qy)
    R[:, 1, 0] = 2*(qx*qy + qw*qz)
    R[:, 1, 1] = 1 - 2*(qx*qx + qz*qz)
    R[:, 1, 2] = 2*(qy*qz - qw*qx)
    R[:, 2, 0] = 2*(qx*qz - qw*qy)
    R[:, 2, 1] = 2*(qy*qz + qw*qx)
    R[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)

    # Build 4x4 extrinsic: T_rel = T_img^-1 @ T_depth
    T = torch.zeros(B, 4, 4, device=device)
    T[:, :3, :3] = R
    T[:, 0, 3] = tx
    T[:, 1, 3] = ty
    T[:, 2, 3] = tz
    T[:, 3, 3] = 1.0

    # Transform: P_C = T @ P_D
    P_D_flat = P_D.view(B, 4, -1)  # (B, 4, H*W)
    P_C_flat = torch.bmm(T, P_D_flat)  # (B, 4, H*W)
    P_C = P_C_flat.view(B, 4, H, W)

    # Project to RGB image plane
    fx_rgb = intrinsic_rgb[:, 0, 0].view(B, 1, 1)
    fy_rgb = intrinsic_rgb[:, 1, 1].view(B, 1, 1)
    cx_rgb = intrinsic_rgb[:, 0, 2].view(B, 1, 1)
    cy_rgb = intrinsic_rgb[:, 1, 2].view(B, 1, 1)

    z_cam = P_C[:, 2].clamp(min=1e-6)
    u_proj = fx_rgb * (P_C[:, 0] / z_cam) + cx_rgb  # (B, H, W)
    v_proj = fy_rgb * (P_C[:, 1] / z_cam) + cy_rgb

    projected_coords = torch.stack([u_proj, v_proj], dim=1)  # (B, 2, H, W)

    # Valid mask: depth > 0 and projection within bounds
    H_rgb, W_rgb = H, W  # Same resolution (both resized in dataloader)
    in_bounds = (
        (u_proj >= 0) & (u_proj < W_rgb - 1) &
        (v_proj >= 0) & (v_proj < H_rgb - 1) &
        (z_cam > 0)
    ).float()
    valid_mask = (valid_depth * in_bounds).unsqueeze(1)  # (B, 1, H, W)

    return projected_coords, valid_mask


# ─── Contrastive Loss ─────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for aligned feature matching.
    
    For each spatial position (i,j) in depth features:
      - Positive: RGB feature at projected location (u,v)
      - Negatives: RGB features at all other sampled positions
    
    Uses cosine similarity (features should be L2-normalized).
    """
    def __init__(self, temperature=0.07, num_negatives=256):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
    
    def forward(self, fmap_rgb, fmap_depth, projected_coords, valid_mask):
        """
        Args:
            fmap_rgb: (B, C, H_feat, W_feat) — encoder output
            fmap_depth: (B, C, H_feat, W_feat) — encoder output
            projected_coords: (B, 2, H_orig, W_orig) — projected coords in original image space
            valid_mask: (B, 1, H_orig, W_orig) — valid projection mask
        """
        B, C, H_feat, W_feat = fmap_rgb.shape
        _, _, H_orig, W_orig = projected_coords.shape
        device = fmap_rgb.device
        downsample = H_orig // H_feat  # typically 8
        
        # Normalize features
        fmap_rgb = F.normalize(fmap_rgb, dim=1)
        fmap_depth = F.normalize(fmap_depth, dim=1)
        
        # Downsample projected coords to feature map resolution
        projected_coords_feat = projected_coords[:, :, ::downsample, ::downsample]
        valid_mask_feat = valid_mask[:, :, ::downsample, ::downsample]
        
        # Sample depth features at grid positions
        step = max(1, H_feat // 16)  # Sample ~16x16 = 256 positions
        grid_h = torch.arange(0, H_feat, step, device=device)
        grid_w = torch.arange(0, W_feat, step, device=device)
        gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')
        
        num_grid = gh.numel()
        grid_h_flat = gh.reshape(-1)
        grid_w_flat = gw.reshape(-1)
        
        depth_feats = fmap_depth[:, :, grid_h_flat, grid_w_flat]  # (B, C, num_grid)
        
        # Projected coords at grid positions (already downsampled, in original pixels)
        # Convert to feature map coords by dividing by downsample
        proj_u = projected_coords_feat[:, 0, grid_h_flat, grid_w_flat] / downsample
        proj_v = projected_coords_feat[:, 1, grid_h_flat, grid_w_flat] / downsample
        
        # Valid mask at grid
        grid_valid = valid_mask_feat[:, 0, grid_h_flat, grid_w_flat]  # (B, num_grid)
        
        # Normalize coords to [-1, 1] for grid_sample (in feature map space)
        proj_u_norm = 2.0 * proj_u / (W_feat - 1) - 1.0
        proj_v_norm = 2.0 * proj_v / (H_feat - 1) - 1.0
        grid_sample_coords = torch.stack([proj_u_norm, proj_v_norm], dim=-1)  # (B, num_grid, 2)
        grid_sample_coords = grid_sample_coords.unsqueeze(2)  # (B, num_grid, 1, 2)
        
        rgb_feats = F.grid_sample(
            fmap_rgb, grid_sample_coords,
            mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(-1)  # (B, C, num_grid)
        
        # Compute cosine similarity between each depth query and ALL RGB features
        # For efficiency, sample a subset of RGB positions as negatives
        # Strategy: use the same grid positions in RGB as negatives
        rgb_neg_feats = fmap_rgb[:, :, grid_h_flat, grid_w_flat]  # (B, C, num_grid)
        
        # Similarity matrix: (B, num_grid, num_grid)
        # sim[i, j] = depth_feats[:, i] · rgb_neg_feats[:, j]
        sim_matrix = torch.bmm(
            depth_feats.permute(0, 2, 1),  # (B, num_grid, C)
            rgb_neg_feats  # (B, C, num_grid)
        ) / self.temperature  # (B, num_grid, num_grid)
        
        # Labels: diagonal (i matches with i, because projected coords are in order)
        labels = torch.arange(num_grid, device=device).unsqueeze(0).expand(B, -1)
        
        # Mask out invalid positions
        # For each batch element, only consider valid grid positions
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        
        for b in range(B):
            valid_idx = grid_valid[b] > 0.5  # (num_grid,)
            if valid_idx.sum() < 2:
                continue
            
            sim_b = sim_matrix[b][valid_idx][:, valid_idx]  # (n_valid, n_valid)
            labels_b = labels[b][valid_idx]
            
            # Relabel: the i-th valid position maps to i-th valid position
            n_valid = valid_idx.sum().item()
            labels_b = torch.arange(n_valid, device=device)
            
            loss_b = F.cross_entropy(sim_b, labels_b)
            total_loss += loss_b
            
            # Accuracy
            preds = sim_b.argmax(dim=1)
            total_correct += (preds == labels_b).sum().item()
            total_count += n_valid
        
        if total_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        avg_loss = total_loss / B
        accuracy = total_correct / total_count * 100.0
        
        return avg_loss, accuracy


# ─── Dense Contrastive Loss (Alternative) ─────────────────────────────────────

class DenseContrastiveLoss(nn.Module):
    """
    Dense pixel-wise contrastive loss.
    For each valid depth pixel, maximize cosine similarity with projected RGB pixel,
    and minimize similarity with random other RGB pixels.
    
    More memory-efficient than full InfoNCE.
    """
    def __init__(self, temperature=0.07, num_negatives=128):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
    
    def forward(self, fmap_rgb, fmap_depth, projected_coords, valid_mask):
        """
        Args:
            fmap_rgb: (B, C, H_feat, W_feat) — encoder output (e.g. H/8 × W/8)
            fmap_depth: (B, C, H_feat, W_feat) — encoder output
            projected_coords: (B, 2, H_orig, W_orig) — projected coords in original image space
            valid_mask: (B, 1, H_orig, W_orig) — valid projection mask in original image space
        """
        B, C, H_feat, W_feat = fmap_rgb.shape
        _, _, H_orig, W_orig = projected_coords.shape
        device = fmap_rgb.device
        downsample = H_orig // H_feat  # typically 8
        
        # Normalize
        fmap_rgb = F.normalize(fmap_rgb, dim=1)
        fmap_depth = F.normalize(fmap_depth, dim=1)
        
        # Downsample projected coords to feature map resolution
        # Use nearest-neighbor to maintain integer alignment
        proj_coords_feat = projected_coords[:, :, ::downsample, ::downsample]  # (B, 2, H_feat, W_feat)
        valid_mask_feat = valid_mask[:, :, ::downsample, ::downsample]  # (B, 1, H_feat, W_feat)
        
        # Normalize coords to [-1, 1] for grid_sample (in feature map coords)
        # projected_coords are in original image pixels, so divide by downsample first
        proj_u_feat = proj_coords_feat[:, 0] / downsample  # (B, H_feat, W_feat)
        proj_v_feat = proj_coords_feat[:, 1] / downsample
        proj_u_norm = 2.0 * proj_u_feat / (W_feat - 1) - 1.0
        proj_v_norm = 2.0 * proj_v_feat / (H_feat - 1) - 1.0
        grid = torch.stack([proj_u_norm, proj_v_norm], dim=-1)  # (B, H_feat, W_feat, 2)
        
        # Sample positive RGB features: grid shape should be (B, H, W, 2)
        rgb_pos = F.grid_sample(
            fmap_rgb, grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (B, C, H, W)
        
        # Positive similarity: (B, H, W)
        pos_sim = (fmap_depth * rgb_pos).sum(dim=1) / self.temperature
        
# Sample negative RGB features at random locations (in feature map space)
        neg_idx_h = torch.randint(0, H_feat, (B, self.num_negatives), device=device)
        neg_idx_w = torch.randint(0, W_feat, (B, self.num_negatives), device=device)

        # Flatten spatial dims for gathering
        fmap_rgb_flat = fmap_rgb.reshape(B, C, -1)  # (B, C, H_feat*W_feat)
        neg_pixel_idx = neg_idx_h * W_feat + neg_idx_w  # (B, num_neg)
        neg_pixel_idx_expanded = neg_pixel_idx.unsqueeze(1).expand(B, C, self.num_negatives)  # (B, C, num_neg)
        rgb_neg = torch.gather(fmap_rgb_flat, 2, neg_pixel_idx_expanded)  # (B, C, num_neg)

        # Negative similarity: fmap_depth (B, C, H_feat, W_feat) x rgb_neg (B, C, num_neg)
        fmap_depth_flat = fmap_depth.reshape(B, C, -1)  # (B, C, H_feat*W_feat)
        neg_sim = torch.bmm(
            fmap_depth_flat.permute(0, 2, 1),  # (B, H_feat*W_feat, C)
            rgb_neg  # (B, C, num_neg)
        )  # (B, H_feat*W_feat, num_neg)
        neg_sim = neg_sim.reshape(B, H_feat, W_feat, self.num_negatives).permute(0, 3, 1, 2)  # (B, num_neg, H_feat, W_feat)
        neg_sim = neg_sim / self.temperature
        
        # Log-sum-exp denominator: log(exp(pos) + sum(exp(neg)))
        # For numerical stability
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+num_neg, H_feat, W_feat)
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(sum(exp(all)))
        log_sum_exp = torch.logsumexp(all_sim, dim=1)  # (B, H_feat, W_feat)
        loss_per_pixel = -pos_sim + log_sum_exp  # (B, H_feat, W_feat)
        
        # Apply valid mask (downsampled to feature map resolution)
        valid_mask_feat = valid_mask[:, :, ::downsample, ::downsample]  # (B, 1, H_feat, W_feat)
        valid_mask_squeezed = valid_mask_feat.squeeze(1)  # (B, H_feat, W_feat)
        loss = (loss_per_pixel * valid_mask_squeezed).sum() / valid_mask_squeezed.sum().clamp(min=1)
        
        # Accuracy (top-1): how often positive has highest similarity
        with torch.no_grad():
            max_neg_sim, _ = neg_sim.max(dim=1)  # (B, H_feat, W_feat)
            correct = ((pos_sim > max_neg_sim) & (valid_mask_squeezed > 0.5)).float()
            accuracy = correct.sum() / valid_mask_squeezed.sum().clamp(min=1) * 100.0
        
        return loss, accuracy


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    log_interval=10, grad_clip=1.0):
    model.train()
    # Freeze pose_update_net — only train encoders + alignment
    for name, param in model.named_parameters():
        if 'pose_update_net' in name:
            param.requires_grad = False
    
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")
    batch_time = AverageMeter("Time")
    
    end = time.time()
    
    for i, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        intrinsic_rgb = batch["intrinsic_rgb"].to(device)
        intrinsic_depth = batch["intrinsic_depth"].to(device)
        gt_pose = batch["gt_pose"].to(device)
        
        # Project depth at GT pose
        projected_coords, valid_mask = project_depth_to_rgb(
            depth, gt_pose, intrinsic_depth, intrinsic_rgb
        )
        
        # Extract features (only encoder forward pass)
        fmap_rgb = model.image_encoder(image)
        fmap_depth_raw = model.depth_encoder(depth)
        fmap_depth = model.depth_feat_align(fmap_depth_raw)
        
        # Compute contrastive loss
        loss, accuracy = criterion(fmap_rgb, fmap_depth, projected_coords, valid_mask)
        
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        optimizer.step()
        
        loss_meter.update(loss.item(), image.size(0))
        acc_meter.update(accuracy, image.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i + 1) % log_interval == 0 or (i + 1) == len(dataloader):
            print(
                f"  Epoch [{epoch}] Step [{i+1}/{len(dataloader)}]  "
                f"Loss: {loss_meter.avg:.4f}  "
                f"Acc: {acc_meter.avg:.1f}%  "
                f"Time: {batch_time.avg:.3f}s"
            )
    
    # Restore requires_grad
    for param in model.parameters():
        param.requires_grad = True
    
    return {
        "train_loss": loss_meter.avg,
        "train_acc": acc_meter.avg,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")
    
    for batch in dataloader:
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        intrinsic_rgb = batch["intrinsic_rgb"].to(device)
        intrinsic_depth = batch["intrinsic_depth"].to(device)
        gt_pose = batch["gt_pose"].to(device)
        
        projected_coords, valid_mask = project_depth_to_rgb(
            depth, gt_pose, intrinsic_depth, intrinsic_rgb
        )
        
        fmap_rgb = model.image_encoder(image)
        fmap_depth_raw = model.depth_encoder(depth)
        fmap_depth = model.depth_feat_align(fmap_depth_raw)
        
        loss, accuracy = criterion(fmap_rgb, fmap_depth, projected_coords, valid_mask)
        
        loss_meter.update(loss.item(), image.size(0))
        acc_meter.update(accuracy, image.size(0))
    
    return {
        "val_loss": loss_meter.avg,
        "val_acc": acc_meter.avg,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain RAFT-Pose Encoders")
    
    # Data
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, nargs=2, default=[480, 640])
    
    # Model
    parser.add_argument("--image_encoder", type=str, default="basic")
    parser.add_argument("--depth_dim", type=int, default=32)
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="cosine")
    
    # Loss
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature. Lower = sharper, harder negatives.")
    parser.add_argument("--num_negatives", type=int, default=128,
                        help="Number of negative samples for dense contrastive loss.")
    parser.add_argument("--loss_type", type=str, default="dense",
                        choices=["infonce", "dense"],
                        help="InfoNCE (grid-based) or Dense (pixel-wise)")
    
    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain_encoder")
    parser.add_argument("--resume", type=str, default=None)
    
    # Other
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    
    # ─── Run directory ────────────────────────────────────────────────────
    run_dir, run_n = get_next_run_dir(args.checkpoint_dir, prefix="pre")
    log_path = Path(run_dir) / "pretrain_log.txt"
    
    def log_print(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    
    log_print(f"Encoder Pretraining")
    log_print(f"Config: {args.config}")
    log_print(f"Run directory: {run_dir}")
    log_print(f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # ─── Data ─────────────────────────────────────────────────────────────
    train_dataset = SevenScenesDataset(
        config_path=args.config,
        split="train",
        image_size=tuple(args.image_size),
        augment=False,
    )
    val_dataset = SevenScenesDataset(
        config_path=args.config,
        split="val",
        image_size=tuple(args.image_size),
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # ─── Model ────────────────────────────────────────────────────────────
    # Build full RAFTPose but only train encoder parts
    model = RAFTPose(
        image_encoder=args.image_encoder,
        depth_dim=args.depth_dim,
        corr_levels=4,
        corr_radius=4,
        num_iterations=6,
    ).to(device)
    
    # Count encoder params
    encoder_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['image_encoder', 'depth_encoder', 'depth_feat_align']):
            encoder_params.append(param)
    
    n_encoder = sum(p.numel() for p in encoder_params)
    n_total = sum(p.numel() for p in model.parameters())
    log_print(f"Encoder parameters: {n_encoder:,} / {n_total:,} total")
    
    # ─── Loss ─────────────────────────────────────────────────────────────
    if args.loss_type == "infonce":
        criterion = InfoNCELoss(temperature=args.temperature)
        log_print(f"Using InfoNCE loss (temperature={args.temperature})")
    else:
        criterion = DenseContrastiveLoss(
            temperature=args.temperature,
            num_negatives=args.num_negatives,
        )
        log_print(f"Using Dense Contrastive loss (temp={args.temperature}, neg={args.num_negatives})")
    
    # ─── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(encoder_params, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    else:
        scheduler = None
    
    # ─── Resume ───────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        # Load only encoder weights
        state = ckpt.get("model_state_dict", ckpt)
        encoder_state = {k: v for k, v in state.items()
                         if any(k.startswith(p) for p in ['image_encoder', 'depth_encoder', 'depth_feat_align'])}
        model.load_state_dict(encoder_state, strict=False)
        log_print(f"Resumed encoder weights from {args.resume}")
    
    # ─── Train ────────────────────────────────────────────────────────────
    log_print(f"\n{'='*60}")
    log_print(f"Pretraining: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    log_print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            log_interval=args.log_interval, grad_clip=args.grad_clip,
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step()
        
        log_print(f"Epoch [{epoch}/{args.epochs}] (lr={optimizer.param_groups[0]['lr']:.2e})")
        log_print(
            f"  Train → Loss: {train_metrics['train_loss']:.4f}  "
            f"Acc: {train_metrics['train_acc']:.1f}%"
        )
        log_print(
            f"  Val   → Loss: {val_metrics['val_loss']:.4f}  "
            f"Acc: {val_metrics['val_acc']:.1f}%"
        )
        
        # Save checkpoint
        is_best = val_metrics["val_loss"] < best_val_loss
        best_val_loss = min(val_metrics["val_loss"], best_val_loss)
        
        ckpt_path = Path(run_dir) / "encoder_pretrained.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        }, str(ckpt_path))
        
        if is_best:
            best_path = Path(run_dir) / "encoder_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, str(best_path))
            log_print(f"  ★ Best model saved (val_loss={best_val_loss:.4f})")
        
        log_print("")
    
    log_print(f"Pretraining complete. Best val_loss: {best_val_loss:.4f}")
    log_print(f"Encoder weights saved to: {run_dir}")
    log_print(f"\nTo use pretrained encoder in end-to-end training:")
    log_print(f"  python train.py --config configs/chess_train.json \\")
    log_print(f"      --pretrained_encoder {run_dir}/encoder_best.pth \\")
    log_print(f"      --checkpoint_dir checkpoints/runs_pt1")


if __name__ == "__main__":
    main()
