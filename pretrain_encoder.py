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


# ─── Data Augmentation ─────────────────────────────────────────────────────

class PretrainAugmentor:
    """
    Data augmentation for encoder pretraining.
    
    Applies photometric and geometric augmentations to RGB and depth images
    to improve generalization on small datasets (435 train samples).
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image, depth):
        """
        Args:
            image: (3, H, W) float32 RGB in [0, 1]
            depth: (1, H, W) float32 depth in meters
        
        Returns:
            augmented (image, depth)
        """
        # ── Photometric augmentations ──
        # Brightness jitter
        if torch.rand(1).item() < self.prob:
            brightness = 0.7 + 0.6 * torch.rand(1).item()  # [0.7, 1.3]
            image = (image * brightness).clamp(0.0, 1.0)
        
        # Contrast jitter
        if torch.rand(1).item() < self.prob:
            mean = image.mean(dim=[1, 2], keepdim=True)
            contrast = 0.7 + 0.6 * torch.rand(1).item()  # [0.7, 1.3]
            image = ((image - mean) * contrast + mean).clamp(0.0, 1.0)
        
        # Saturation jitter (convert to grayscale and blend)
        if torch.rand(1).item() < self.prob:
            gray = image.mean(dim=0, keepdim=True).expand_as(image)
            blend = 0.5 + 0.5 * torch.rand(1).item()  # [0.5, 1.0]
            image = gray * (1 - blend) + image * blend
        
        # Gaussian noise
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(image) * 0.02
            image = (image + noise).clamp(0.0, 1.0)
        
        # ── Depth augmentations ──
        # Depth noise (simulates sensor noise)
        if torch.rand(1).item() < self.prob:
            depth_noise = torch.randn_like(depth) * 0.02  # ±2cm noise
            depth = (depth + depth_noise).clamp(min=0.0)
        
        # Random depth scaling (simulates scale uncertainty)
        if torch.rand(1).item() < 0.3:
            scale = 0.98 + 0.04 * torch.rand(1).item()  # [0.98, 1.02]
            depth = depth * scale
        
        return image, depth


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

def project_depth_to_rgb(depth, gt_pose_7d, intrinsic_depth, intrinsic_rgb,
                         downsample=8):
    """
    Project depth map to RGB image plane using GT pose (7D: quat + trans).
    
    Projects at FEATURE MAP resolution (H/downsample × W/downsample) to match
    the encoder output resolution. This is consistent with how correlation
    sampling works during inference, avoiding resolution mismatch.
    
    Args:
        depth: (B, 1, H, W) depth map in meters
        gt_pose_7d: (B, 7) [qw, qx, qy, qz, tx, ty, tz] relative pose
        intrinsic_depth: (B, 3, 3)
        intrinsic_rgb: (B, 3, 3)
        downsample: spatial downsampling factor (default 8, matching encoder stride)
    
    Returns:
        projected_coords: (B, 2, H_feat, W_feat) — (u, v) in RGB feature map space
        valid_mask: (B, 1, H_feat, W_feat) — 1 where depth > 0 and projection is in-bounds
    """
    B, _, H, W = depth.shape
    device = depth.device
    
    # Downsample depth to feature map resolution (nearest to preserve depth values)
    feat_h, feat_w = H // downsample, W // downsample
    depth_small = F.interpolate(
        depth, size=(feat_h, feat_w), mode='nearest'
    ).squeeze(1)  # (B, feat_h, feat_w)
    
    # Scale intrinsics to feature map resolution
    scale = 1.0 / downsample
    intrinsic_depth_s = intrinsic_depth.clone()
    intrinsic_depth_s[:, 0, 0] *= scale
    intrinsic_depth_s[:, 1, 1] *= scale
    intrinsic_depth_s[:, 0, 2] *= scale
    intrinsic_depth_s[:, 1, 2] *= scale
    
    intrinsic_rgb_s = intrinsic_rgb.clone()
    intrinsic_rgb_s[:, 0, 0] *= scale
    intrinsic_rgb_s[:, 1, 1] *= scale
    intrinsic_rgb_s[:, 0, 2] *= scale
    intrinsic_rgb_s[:, 1, 2] *= scale
    
    # Pixel coordinates at feature map resolution
    u_d = torch.arange(feat_w, dtype=torch.float32, device=device).view(1, 1, feat_w).expand(B, feat_h, feat_w)
    v_d = torch.arange(feat_h, dtype=torch.float32, device=device).view(1, feat_h, 1).expand(B, feat_h, feat_w)

    # Unproject using scaled depth intrinsic
    fx_d = intrinsic_depth_s[:, 0, 0].view(B, 1, 1)
    fy_d = intrinsic_depth_s[:, 1, 1].view(B, 1, 1)
    cx_d = intrinsic_depth_s[:, 0, 2].view(B, 1, 1)
    cy_d = intrinsic_depth_s[:, 1, 2].view(B, 1, 1)

    x_d = (u_d - cx_d) / fx_d * depth_small
    y_d = (v_d - cy_d) / fy_d * depth_small
    z_d = depth_small

    # Valid depth mask
    valid_depth = (depth_small > 0.01).float()

    # 3D points in depth camera frame (homogeneous)
    ones = torch.ones_like(z_d)
    P_D = torch.stack([x_d, y_d, z_d, ones], dim=1)  # (B, 4, feat_h, feat_w)

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
    P_D_flat = P_D.view(B, 4, -1)  # (B, 4, feat_h*feat_w)
    P_C_flat = torch.bmm(T, P_D_flat)  # (B, 4, feat_h*feat_w)
    P_C = P_C_flat.view(B, 4, feat_h, feat_w)

    # Project to RGB image plane (at feature map resolution)
    fx_rgb = intrinsic_rgb_s[:, 0, 0].view(B, 1, 1)
    fy_rgb = intrinsic_rgb_s[:, 1, 1].view(B, 1, 1)
    cx_rgb = intrinsic_rgb_s[:, 0, 2].view(B, 1, 1)
    cy_rgb = intrinsic_rgb_s[:, 1, 2].view(B, 1, 1)

    z_cam = P_C[:, 2].clamp(min=0.01)
    u_proj = fx_rgb * (P_C[:, 0] / z_cam) + cx_rgb  # (B, feat_h, feat_w)
    v_proj = fy_rgb * (P_C[:, 1] / z_cam) + cy_rgb

    projected_coords = torch.stack([u_proj, v_proj], dim=1)  # (B, 2, feat_h, feat_w)

    # Valid mask: depth > 0 and projection within bounds
    in_bounds = (
        (u_proj >= 0) & (u_proj < feat_w - 1) &
        (v_proj >= 0) & (v_proj < feat_h - 1) &
        (z_cam > 0)
    ).float()
    valid_mask = (valid_depth * in_bounds).unsqueeze(1)  # (B, 1, feat_h, feat_w)

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
            projected_coords: (B, 2, H_feat, W_feat) — projected coords already in feature map space
            valid_mask: (B, 1, H_feat, W_feat) — valid projection mask in feature map space
        """
        B, C, H_feat, W_feat = fmap_rgb.shape
        device = fmap_rgb.device
        
        # Normalize features
        fmap_rgb = F.normalize(fmap_rgb, dim=1)
        fmap_depth = F.normalize(fmap_depth, dim=1)
        
        # projected_coords are already in feature map space (no downsample needed)
        
        # Sample depth features at grid positions
        step = max(1, H_feat // 16)  # Sample ~16x16 = 256 positions
        grid_h = torch.arange(0, H_feat, step, device=device)
        grid_w = torch.arange(0, W_feat, step, device=device)
        gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')
        
        num_grid = gh.numel()
        grid_h_flat = gh.reshape(-1)
        grid_w_flat = gw.reshape(-1)
        
        depth_feats = fmap_depth[:, :, grid_h_flat, grid_w_flat]  # (B, C, num_grid)
        
        # Projected coords at grid positions (already in feature map space)
        proj_u = projected_coords[:, 0, grid_h_flat, grid_w_flat]
        proj_v = projected_coords[:, 1, grid_h_flat, grid_w_flat]
        
        # Valid mask at grid
        grid_valid = valid_mask[:, 0, grid_h_flat, grid_w_flat]  # (B, num_grid)
        
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
    Dense pixel-wise contrastive loss with hard negative mining.
    For each valid depth pixel, maximize cosine similarity with projected RGB pixel,
    and minimize similarity with spatially nearby hard negatives.
    
    Hard negatives are sampled in a local window around the positive location,
    forcing the model to learn fine-grained spatial discrimination — exactly
    what pose refinement needs.
    """
    def __init__(self, temperature=0.07, num_negatives=128, hard_radius=4,
                 hard_ratio=0.75, random_ratio=0.25):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.hard_radius = hard_radius  # pixels around positive for hard negatives
        self.hard_ratio_max = hard_ratio    # max fraction of hard negatives (reached at end)
        self.random_ratio = random_ratio  # fraction that are random (far away)
        self._current_hard_ratio = 0.0  # start with all random (easy), ramp up
    
    def set_epoch(self, epoch, total_epochs):
        """Curriculum: linearly ramp hard_ratio from 0 to hard_ratio_max over training."""
        self._current_hard_ratio = self.hard_ratio_max * min(1.0, epoch / (total_epochs * 0.5))
    
    def forward(self, fmap_rgb, fmap_depth, projected_coords, valid_mask):
        """
        Dense contrastive loss on ALL spatial positions.
        Negatives: mix of random (far) + hard (nearby, curriculum-scheduled).
        Memory-efficient: random negs via matmul, hard negs via sparse fill.
        """
        B, C, H_feat, W_feat = fmap_rgb.shape
        device = fmap_rgb.device
        N = H_feat * W_feat
        
        # Normalize
        fmap_rgb = F.normalize(fmap_rgb, dim=1)
        fmap_depth = F.normalize(fmap_depth, dim=1)
        
        # Projected coords in feature map space
        proj_u = projected_coords[:, 0]  # (B, H_feat, W_feat)
        proj_v = projected_coords[:, 1]
        
        # Sample positive RGB features via grid_sample
        proj_u_norm = 2.0 * proj_u / (W_feat - 1) - 1.0
        proj_v_norm = 2.0 * proj_v / (H_feat - 1) - 1.0
        grid = torch.stack([proj_u_norm, proj_v_norm], dim=-1)  # (B, H, W, 2)
        rgb_pos = F.grid_sample(fmap_rgb, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        
        # Positive similarity: (B, N)
        pos_sim = (fmap_depth * rgb_pos).sum(dim=1).reshape(B, N) / self.temperature
        
        # ── Negatives ──
        n_hard = max(0, int(self.num_negatives * self._current_hard_ratio))
        n_random = self.num_negatives - n_hard
        
        fmap_rgb_flat = fmap_rgb.reshape(B, C, N)  # (B, C, N)
        fmap_depth_flat = fmap_depth.reshape(B, C, N)  # (B, C, N)
        
        # --- Random negatives via matmul: (B, C, N) @ (B, N, n_random) -> (B, C, n_random) ---
        rand_idx = torch.randint(0, N, (B, n_random), device=device)
        # One-hot gather: fmap_rgb_flat @ one_hot(rand_idx) 
        # More efficient: just gather and do element-wise
        rand_idx_exp = rand_idx.unsqueeze(1).expand(B, C, -1)
        rgb_rand_neg = torch.gather(fmap_rgb_flat, 2, rand_idx_exp)  # (B, C, n_random)
        # Similarity: (B, n_random) = sum_c depth_flat[b,c,n] * rand_neg[b,c,k] for each n,k
        # depth_flat: (B, C, N), rand_neg: (B, C, n_random)
        # rand_neg_sim[b, k, n] = sum_c fmap_depth_flat[b,c,n] * rgb_rand_neg[b,c,k]
        # = (B, C, N).transpose(1,2) @ (B, C, n_random) -> (B, N, n_random) -> transpose -> (B, n_random, N)
        rand_neg_sim = torch.bmm(fmap_depth_flat.transpose(1, 2), rgb_rand_neg)  # (B, N, n_random)
        rand_neg_sim = rand_neg_sim.transpose(1, 2) / self.temperature  # (B, n_random, N)
        
        neg_sim_list = [rand_neg_sim]
        
        # --- Hard negatives (curriculum-scheduled) ---
        if n_hard > 0:
            # Sample a subset of spatial positions for hard neg computation
            step = max(1, H_feat // 16)
            grid_h = torch.arange(0, H_feat, step, device=device)
            grid_w = torch.arange(0, W_feat, step, device=device)
            gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')
            grid_h_flat = gh.reshape(-1)
            grid_w_flat = gw.reshape(-1)
            n_pos = grid_h_flat.numel()
            
            pos_u = proj_u[:, grid_h_flat, grid_w_flat].clamp(0, W_feat - 1).long()
            pos_v = proj_v[:, grid_h_flat, grid_w_flat].clamp(0, H_feat - 1).long()
            
            hard_off_u = torch.randint(-self.hard_radius, self.hard_radius + 1,
                                       (B, n_hard, n_pos), device=device)
            hard_off_v = torch.randint(-self.hard_radius, self.hard_radius + 1,
                                       (B, n_hard, n_pos), device=device)
            is_zero = (hard_off_u == 0) & (hard_off_v == 0)
            hard_off_u[is_zero] = 1
            
            hard_u = (pos_u.unsqueeze(1) + hard_off_u).clamp(0, W_feat - 1)
            hard_v = (pos_v.unsqueeze(1) + hard_off_v).clamp(0, H_feat - 1)
            hard_idx = hard_v * W_feat + hard_u  # (B, n_hard, n_pos)
            
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_hard, n_pos)
            # (B, n_hard, n_pos, C) -> (B, C, n_hard, n_pos)
            rgb_hard = fmap_rgb_flat[batch_idx, :, hard_idx].permute(0, 3, 1, 2)
            
            # Depth features at sampled positions: (B, C, n_pos)
            depth_sampled = fmap_depth_flat[:, :, grid_h_flat * W_feat + grid_w_flat]
            
            # Hard neg similarity at sampled positions: (B, n_hard, n_pos)
            hard_neg_sim = (depth_sampled.unsqueeze(2) * rgb_hard).sum(dim=1) / self.temperature
            
            # For unsampled positions, hard neg sim = -inf (they don't contribute)
            hard_neg_sim_dense = torch.full((B, n_hard, N), float('-inf'), device=device)
            hard_neg_sim_dense[:, :, grid_h_flat * W_feat + grid_w_flat] = hard_neg_sim
            neg_sim_list.append(hard_neg_sim_dense)
        
        # Combine: (B, num_neg, N)
        neg_sim = torch.cat(neg_sim_list, dim=1)
        
        # InfoNCE loss
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+num_neg, N)
        log_sum_exp = torch.logsumexp(all_sim, dim=1)  # (B, N)
        loss_per_pixel = -pos_sim + log_sum_exp  # (B, N)
        
        valid_mask_flat = valid_mask.reshape(B, N)
        loss = (loss_per_pixel * valid_mask_flat).sum() / valid_mask_flat.sum().clamp(min=1)
        
        # Accuracy
        with torch.no_grad():
            max_neg_sim, _ = neg_sim.max(dim=1)  # (B, N)
            correct = ((pos_sim > max_neg_sim) & (valid_mask_flat > 0.5)).float()
            accuracy = correct.sum() / valid_mask_flat.sum().clamp(min=1) * 100.0
        
        return loss, accuracy


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch,
                    augmentor=None, grad_clip=1.0, log_interval=10):
    model.train()
    
    loss_meter = AverageMeter("Loss")
    acc_meter = AverageMeter("Acc")
    batch_time = AverageMeter("Time")
    
    end = time.time()
    
    for i, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        gt_pose = batch["gt_pose"].to(device)
        intrinsic_depth = batch["intrinsic_depth"].to(device)
        intrinsic_rgb = batch["intrinsic_rgb"].to(device)
        
        # Apply data augmentation
        if augmentor is not None:
            image, depth = augmentor(image, depth)
        
        # Project depth at GT pose
        projected_coords, valid_mask = project_depth_to_rgb(
            depth, gt_pose, intrinsic_depth, intrinsic_rgb
        )
        
        # Extract features (only encoder forward pass)
        fmap_rgb = model.image_encoder(image)
        if model.shared_encoder:
            # Dual encoder: depth has its own encoder (1-channel input)
            fmap_depth = model.depth_encoder(depth)
        else:
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
        if model.shared_encoder:
            # Dual encoder: depth has its own encoder (1-channel input)
            fmap_depth = model.depth_encoder(depth)
        else:
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
    parser.add_argument("--image_encoder", type=str, default="basic",
                        choices=["basic", "small", "resnet18"],
                        help="Image encoder type. 'resnet18' uses ImageNet pretrained weights.")
    parser.add_argument("--shared_encoder", action="store_true",
                        help="Use same encoder for RGB and depth (Siamese).")
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
    parser.add_argument("--hard_radius", type=int, default=4,
                        help="Hard negative sampling radius in pixels.")
    parser.add_argument("--hard_ratio", type=float, default=0.75,
                        help="Fraction of negatives that are hard (nearby).")
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
        shared_encoder=getattr(args, 'shared_encoder', False),
    ).to(device)
    
    # Count encoder params
    encoder_params = []
    encoder_keys = ['image_encoder']
    if model.shared_encoder:
        encoder_keys.append('depth_encoder')
    else:
        encoder_keys.extend(['depth_encoder', 'depth_feat_align'])
    for name, param in model.named_parameters():
        if any(k in name for k in encoder_keys):
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
            hard_radius=args.hard_radius,
            hard_ratio=args.hard_ratio,
        )
        log_print(f"Using Dense Contrastive loss (temp={args.temperature}, neg={args.num_negatives}, "
                  f"hard_radius={args.hard_radius}, hard_ratio={args.hard_ratio})")
    
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
        encoder_prefixes = ['image_encoder', 'depth_encoder']
        if not model.shared_encoder:
            encoder_prefixes.append('depth_feat_align')
        encoder_state = {k: v for k, v in state.items()
                         if any(k.startswith(p) for p in encoder_prefixes)}
        model.load_state_dict(encoder_state, strict=False)
        log_print(f"Resumed encoder weights from {args.resume}")
    
    # ─── Augmentation ─────────────────────────────────────────────────────
    augmentor = PretrainAugmentor(prob=0.5)
    log_print(f"Data augmentation enabled (prob=0.5)")
    
    # ─── Train ────────────────────────────────────────────────────────────
    log_print(f"\n{'='*60}")
    log_print(f"Pretraining: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    log_print(f"Hard negative curriculum: 0 → {args.hard_ratio} over first 50% of epochs")
    log_print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Curriculum: ramp hard negative ratio
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch, args.epochs)
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            log_interval=args.log_interval, grad_clip=args.grad_clip,
            augmentor=augmentor,
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            scheduler.step()
        
        hr = criterion._current_hard_ratio if hasattr(criterion, '_current_hard_ratio') else 0
        log_print(f"Epoch [{epoch}/{args.epochs}] (lr={optimizer.param_groups[0]['lr']:.2e}, hard_ratio={hr:.2f})")
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
