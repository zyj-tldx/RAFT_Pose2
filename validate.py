#!/usr/bin/env python3
"""
RAFT-Pose Validation Script.

Validates a trained model by predicting relative pose between two frames.

Pipeline:
  1. Model predicts T_rel from (image_A, depth_B)
  2. Updated world pose: T_pred = T_image @ T_rel
  3. Compare T_pred vs T_depth_gt (ground-truth world pose of depth frame)

Usage:
    python validate.py \
        --checkpoint checkpoints/model_best.pth \
        --image 7Scenes/data/chess/seq-01/color_000.png \
        --depth 7Scenes/data/chess/seq-01/depth_050.png \
        --pose_image 7Scenes/data/chess/seq-01/pose_000.txt \
        --pose_depth 7Scenes/data/chess/seq-01/pose_050.txt \
        --intrinsics 585 585 320 240 \
        --output_prefix result

Outputs:
    - Console: loss, rotation error (deg), translation error (m)
    - result.png: RGB image with projected colored point cloud overlay
    - result.pred.pcd: Predicted world-pose colored point cloud
    - result.gt.pcd: GT world-pose colored point cloud
    - result.json: Metrics and convergence data
"""

import os
import sys
import argparse
import json

import torch
import numpy as np
from pathlib import Path
from PIL import Image

# Setup import path
RAFT_POSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RAFT_POSE_DIR))

from raft_pose import RAFTPose
from pose_loss import PoseLoss


# ─── Utilities ────────────────────────────────────────────────────────────────

def get_next_test_dir(base_dir="checkpoints", prefix="test"):
    """
    Determine the next available test directory under base_dir.
    
    Scans existing `{prefix}_*` folders and increments the counter.
    Returns (test_dir_path, test_number).
    """
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
    test_dir = base / f"{prefix}_{next_n:03d}"
    test_dir.mkdir(parents=True, exist_ok=True)
    return str(test_dir), next_n


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_image(path, image_size=None):
    """Load RGB image as (3, H, W) float32 tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def load_depth(path, image_size=None, depth_scale=0.001):
    """Load depth map as (1, H, W) float32 tensor in meters."""
    depth = Image.open(path)
    if image_size is not None:
        depth = depth.resize((image_size[1], image_size[0]), Image.NEAREST)
    arr = np.array(depth, dtype=np.float32) * depth_scale
    arr = np.clip(arr, 0.0, 10.0)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def load_pose(path):
    """Load 4x4 pose matrix from text file."""
    pose = np.loadtxt(path, dtype=np.float64)  # (4, 4)
    return torch.from_numpy(pose).float()


def pose_matrix_to_7d(T):
    """Convert 4x4 pose matrix to 7D vector [qw, qx, qy, qz, tx, ty, tz]."""
    R = T[:3, :3]
    t = T[:3, 3]

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz])
    quat = quat / np.linalg.norm(quat)
    return np.concatenate([quat, t])


def quat_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])
    return R


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred_pose_7d, gt_pose_7d):
    """
    Compute rotation and translation errors.

    Args:
        pred_pose_7d: (7,) numpy array [qw, qx, qy, qz, tx, ty, tz]
        gt_pose_7d:   (7,) numpy array [qw, qx, qy, qz, tx, ty, tz]

    Returns:
        dict with rotation_error_deg, translation_error_m
    """
    pred_q = pred_pose_7d[:4]
    gt_q = gt_pose_7d[:4]
    pred_t = pred_pose_7d[4:7]
    gt_t = gt_pose_7d[4:7]

    # Normalize
    pred_q = pred_q / np.linalg.norm(pred_q)
    gt_q = gt_q / np.linalg.norm(gt_q)

    # Geodesic rotation error
    dot = np.abs(np.dot(pred_q, gt_q))
    dot = np.clip(dot, 0.0, 1.0)
    rot_error_rad = 2.0 * np.arccos(dot)
    rot_error_deg = np.degrees(rot_error_rad)

    # Translation error
    trans_error = np.linalg.norm(pred_t - gt_t)

    return {
        "rotation_error_deg": rot_error_deg,
        "translation_error_m": trans_error,
    }


# ─── Visualization ────────────────────────────────────────────────────────────

def depth_to_colored_pointcloud(depth_np, image_np, intrinsic, rel_pose_7d):
    """
    Back-project depth to 3D, transform by T_rel to image cam, project to image
    and sample colors, then render as an image.

    Pipeline:
        1. depth → 3D points in depth camera frame
        2. Apply T_rel: depth cam → image cam
        3. Project to image plane, sample colors from image_np
        4. Render as (H, W, 3) image

    Args:
        depth_np:    (H, W) numpy array, depth in meters
        image_np:   (H, W, 3) numpy array, RGB image in [0, 255]
        intrinsic:  (3, 3) numpy array, image camera intrinsics
        rel_pose_7d: (7,) numpy array [qw, qx, qy, qz, tx, ty, tz]
                     relative pose T_rel from depth cam to image cam

    Returns:
        projected_img: (H, W, 3) numpy array, RGB with point cloud overlay
    """
    H, W = depth_np.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)  # (H, W)

    # Valid depth mask
    valid = depth_np > 0.01

    # Step 1: Back-project to 3D in depth camera frame
    z = depth_np[valid]
    x = (uu[valid] - cx) / fx * z
    y = (vv[valid] - cy) / fy * z
    pts_depth_cam = np.stack([x, y, z], axis=1)  # (N, 3)

    # Step 2: Transform to image camera frame using T_rel
    R = quat_to_rotation_matrix(rel_pose_7d[:4])
    t = rel_pose_7d[4:7]
    pts_img_cam = (R @ pts_depth_cam.T).T + t  # (N, 3)

    # Step 3: Project to image plane
    z_proj = pts_img_cam[:, 2]
    in_front = z_proj > 0.01

    u_proj = (pts_img_cam[:, 0] / z_proj * fx + cx).astype(np.int32)
    v_proj = (pts_img_cam[:, 1] / z_proj * fy + cy).astype(np.int32)

    in_bounds = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
    valid_proj = in_front & in_bounds

    # Step 4: Sample colors from image at projected locations
    u_draw = u_proj[valid_proj]
    v_draw = v_proj[valid_proj]
    colors_draw = image_np[v_draw, u_draw]  # (M, 3) from image

    # Render
    projected_img = np.zeros_like(image_np)
    projected_img[v_draw, u_draw] = colors_draw

    return projected_img


def create_comparison_figure(image_np, depth_np, intrinsic, gt_pose_7d, pred_pose_7d):
    """
    Create a side-by-side comparison: original | GT projection | pred projection.

    Returns:
        comparison_img: (H, W*3, 3) numpy array
    """
    # GT projection
    gt_projected = depth_to_colored_pointcloud(depth_np, image_np, intrinsic, gt_pose_7d)

    # Pred projection
    pred_projected = depth_to_colored_pointcloud(depth_np, image_np, intrinsic, pred_pose_7d)

    # Concatenate horizontally
    comparison = np.concatenate([image_np, gt_projected, pred_projected], axis=1)

    return comparison


def save_colored_pcd(depth_np, image_np, intrinsic, rel_pose_7d, output_path):
    """
    Back-project depth to 3D, transform by relative pose to image camera frame,
    project onto image to sample colors, then save as PCD.

    Pipeline:
        1. depth → 3D points in depth camera frame
        2. Apply T_rel: depth cam → image cam
        3. Project to image plane to find corresponding pixel colors
        4. Save PCD with 3D points (in image cam frame) + sampled colors

    Args:
        depth_np:    (H, W) numpy array, depth in meters
        image_np:   (H, W, 3) numpy array, RGB image in [0, 255]
        intrinsic:  (3, 3) numpy array, image camera intrinsics
        rel_pose_7d: (7,) numpy array [qw, qx, qy, qz, tx, ty, tz]
                     relative pose T_rel from depth cam to image cam
        output_path: str, path to save .pcd file
    """
    H, W = depth_np.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Step 1: Back-project depth to 3D in depth camera frame
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    valid = depth_np > 0.01
    z_depth = depth_np[valid]
    x_depth = (uu[valid] - cx) / fx * z_depth
    y_depth = (vv[valid] - cy) / fy * z_depth
    pts_depth_cam = np.stack([x_depth, y_depth, z_depth], axis=1)  # (N, 3)

    # Step 2: Transform to image camera frame using T_rel
    R = quat_to_rotation_matrix(rel_pose_7d[:4])
    t = rel_pose_7d[4:7]
    pts_img_cam = (R @ pts_depth_cam.T).T + t  # (N, 3)

    # Step 3: Project onto image plane to sample colors
    z_img = pts_img_cam[:, 2]
    in_front = z_img > 0.01  # only points in front of camera

    u_proj = (pts_img_cam[:, 0] / z_img * fx + cx).astype(np.int32)
    v_proj = (pts_img_cam[:, 1] / z_img * fy + cy).astype(np.int32)

    in_bounds = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
    valid_proj = in_front & in_bounds

    pts_final = pts_img_cam[valid_proj]  # (M, 3)
    u_sample = u_proj[valid_proj]
    v_sample = v_proj[valid_proj]

    # Sample colors from image at projected pixel locations
    colors = image_np[v_sample, u_sample].astype(np.float32) / 255.0  # (M, 3) in [0, 1]

    # Step 4: Write ASCII PCD
    n_points = pts_final.shape[0]
    with open(output_path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS x y z rgb\n")
        f.write(f"SIZE 4 4 4 4\n")
        f.write(f"TYPE F F F U\n")
        f.write(f"COUNT 1 1 1 1\n")
        f.write(f"WIDTH {n_points}\n")
        f.write(f"HEIGHT 1\n")
        f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n_points}\n")
        f.write("DATA ascii\n")
        for i in range(n_points):
            px, py, pz = pts_final[i]
            r, g, b = int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255)
            rgb_int = (r << 16) | (g << 8) | b
            f.write(f"{px:.6f} {py:.6f} {pz:.6f} {rgb_int}\n")

    print(f"  PCD saved: {output_path} ({n_points} points)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RAFT-Pose Validation")

    # Input files
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to RGB image from frame A (color_XXX.png)")
    parser.add_argument("--depth", type=str, required=True,
                        help="Path to depth map from frame B (depth_XXX.png)")
    parser.add_argument("--pose_image", type=str, required=True,
                        help="Path to world pose of frame A (pose_XXX.txt)")
    parser.add_argument("--pose_depth", type=str, required=True,
                        help="Path to world pose of frame B (pose_XXX.txt)")

    # Camera intrinsics
    parser.add_argument("--intrinsics", type=float, nargs=4, default=[585.0, 585.0, 320.0, 240.0],
                        metavar=("FX", "FY", "CX", "CY"),
                        help="Camera intrinsics: fx fy cx cy")

    # Options
    parser.add_argument("--depth_scale", type=float, default=0.001,
                        help="Depth scale factor (raw -> meters)")
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Resize image/depth to (H, W). Default: use original size")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output prefix for all result files. Default: auto-generate under checkpoints/test_n/")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Base directory for test outputs (default: checkpoints)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ─── Test directory ───────────────────────────────────────────────────
    test_dir, test_n = get_next_test_dir(args.checkpoint_dir, prefix="test")
    log_path = Path(test_dir) / "test_log.txt"
    print(f"Test directory: {test_dir}")

    def log_print(msg, also_stdout=True):
        """Print to both console and log file."""
        line = msg + "\n"
        if also_stdout:
            print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(line)

    # Auto-generate output_prefix if not specified
    if args.output_prefix is None:
        args.output_prefix = str(Path(test_dir) / "result")

    log_print(f"Test directory: {test_dir}")
    log_print(f"Arguments: {json.dumps(vars(args), indent=2)}")

    # ─── Load data ────────────────────────────────────────────────────────
    log_print(f"Loading data...")
    image_size = tuple(args.image_size) if args.image_size else None

    image = load_image(args.image, image_size)       # (3, H, W)
    depth = load_depth(args.depth, image_size, args.depth_scale)  # (1, H, W)
    T_image = load_pose(args.pose_image)             # (4, 4) world pose of image frame
    T_depth_gt = load_pose(args.pose_depth)          # (4, 4) world pose of depth frame

    H, W = image.shape[1], image.shape[2]
    log_print(f"  Image:      {args.image} -> {image.shape}")
    log_print(f"  Depth:      {args.depth} -> {depth.shape}")
    log_print(f"  Pose image: {args.pose_image}")
    log_print(f"  Pose depth: {args.pose_depth}")

    # Intrinsics
    fx, fy, cx, cy = args.intrinsics
    intrinsic_matrix = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=torch.float32)

    # Compute GT relative pose: T_rel_gt = T_image^(-1) @ T_depth_gt
    T_rel_gt = torch.linalg.inv(T_image) @ T_depth_gt  # (4, 4)
    gt_rel_pose_7d = pose_matrix_to_7d(T_rel_gt.numpy())
    gt_rel_pose_tensor = torch.from_numpy(gt_rel_pose_7d).unsqueeze(0)  # (1, 7)

    log_print(f"  GT relative pose: [{', '.join(f'{v:.6f}' for v in gt_rel_pose_7d)}]")

    # ─── Load model ──────────────────────────────────────────────────────
    log_print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Restore model args from checkpoint
    ckpt_args = checkpoint.get("args", {})
    model = RAFTPose(
        image_encoder=ckpt_args.get("image_encoder", "basic"),
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        context_dim=ckpt_args.get("context_dim", 64),
        depth_dim=ckpt_args.get("depth_dim", 32),
        corr_levels=ckpt_args.get("corr_levels", 4),
        corr_radius=ckpt_args.get("corr_radius", 2),
        num_iterations=ckpt_args.get("num_iterations", 12),
        num_pose_samples=ckpt_args.get("num_pose_samples", 16),
        pose_sample_std=ckpt_args.get("pose_sample_std", 0.01),
        init_pose_noise_std=ckpt_args.get("init_pose_noise_std", 0.05),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log_print(f"  Model parameters: {n_params:,}")
    log_print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")

    # ─── Inference ────────────────────────────────────────────────────────
    log_print(f"\nRunning inference on {device}...")

    # Add batch dimension
    image_b = image.unsqueeze(0).to(device)       # (1, 3, H, W)
    depth_b = depth.unsqueeze(0).to(device)       # (1, 1, H, W)
    intrinsic_b = intrinsic_matrix.unsqueeze(0).to(device)  # (1, 3, 3)

    with torch.no_grad():
        pred_pose, pose_sequence = model(
            image=image_b,
            depth=depth_b,
            intrinsic_rgb=intrinsic_b,
            intrinsic_depth=intrinsic_b,
            return_all_poses=True,
        )

    pred_rel_pose_7d = pred_pose[0].cpu().numpy()  # (7,)
    pose_seq_np = pose_sequence[0].cpu().numpy()  # (num_iters+1, 7)

    # ─── Compute updated world poses ─────────────────────────────────────
    # T_pred_world = T_image @ T_rel_pred
    # T_gt_world   = T_depth_gt (already loaded)
    def pose_7d_to_matrix(pose_7d):
        """Convert 7D [qw,qx,qy,qz,tx,ty,tz] to 4x4 matrix."""
        R = quat_to_rotation_matrix(pose_7d[:4])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pose_7d[4:7]
        return T

    T_rel_pred = pose_7d_to_matrix(pred_rel_pose_7d)
    T_pred_world = T_image.numpy() @ T_rel_pred  # (4, 4)
    T_depth_gt_np = T_depth_gt.numpy()            # (4, 4)

    # World pose errors
    pred_world_7d = pose_matrix_to_7d(T_pred_world)
    gt_world_7d = pose_matrix_to_7d(T_depth_gt_np)
    world_metrics = compute_metrics(pred_world_7d, gt_world_7d)

    # Relative pose errors (what the model directly predicts)
    rel_metrics = compute_metrics(pred_rel_pose_7d, gt_rel_pose_7d)

    # Also compute loss using PoseLoss (on relative pose)
    criterion = PoseLoss(rot_weight=1.0, trans_weight=100.0)
    loss, loss_details = criterion(
        pred_pose.cpu(),             # (1, 7) — relative pose prediction
        gt_rel_pose_tensor,         # (1, 7) — GT relative pose
    )

    # ─── Print results ────────────────────────────────────────────────────
    log_print(f"\n{'='*60}")
    log_print(f"Validation Results")
    log_print(f"{'='*60}")
    log_print(f"  Loss:              {loss_details['total_loss']:.6f}")
    log_print(f"    Rotation loss:   {loss_details['rot_loss_deg']:.4f}°")
    log_print(f"    Translation loss:{loss_details['trans_loss']:.6f} m")
    log_print(f"  --- Relative pose (T_rel) ---")
    log_print(f"    Rotation error:  {rel_metrics['rotation_error_deg']:.4f}°")
    log_print(f"    Translation error:{rel_metrics['translation_error_m']:.6f} m")
    log_print(f"  --- World pose (T_image @ T_rel vs T_depth_gt) ---")
    log_print(f"    Rotation error:  {world_metrics['rotation_error_deg']:.4f}°")
    log_print(f"    Translation error:{world_metrics['translation_error_m']:.6f} m")
    log_print(f"  Predicted T_rel:   [{', '.join(f'{v:.6f}' for v in pred_rel_pose_7d)}]")
    log_print(f"  GT T_rel:          [{', '.join(f'{v:.6f}' for v in gt_rel_pose_7d)}]")

    # Iteration convergence (relative pose)
    log_print(f"\n  Iteration convergence (relative pose):")
    for i, p in enumerate(pose_seq_np):
        m = compute_metrics(p, gt_rel_pose_7d)
        log_print(f"    Iter {i:2d}: rot={m['rotation_error_deg']:8.4f}°  "
              f"trans={m['translation_error_m']:.6f}m")

    # ─── Visualization ────────────────────────────────────────────────────
    log_print(f"\nGenerating visualization...")

    image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # (H, W, 3)
    depth_np = depth.squeeze(0).numpy()  # (H, W)
    intrinsic_np = intrinsic_matrix.numpy()

    # Use relative poses for visualization: depth → image cam → project to image
    comparison = create_comparison_figure(
        image_np, depth_np, intrinsic_np, gt_rel_pose_7d, pred_rel_pose_7d
    )

    # Add text labels
    from PIL import ImageDraw, ImageFont

    comp_img = Image.fromarray(comparison)
    draw = ImageDraw.Draw(comp_img)

    label_h = 30
    draw.rectangle([0, 0, comparison.shape[1], label_h], fill=(0, 0, 0))
    w_third = comparison.shape[1] // 3
    draw.text((10, 5), "Original RGB", fill=(255, 255, 255))
    draw.text((w_third + 10, 5), f"GT World Pose", fill=(255, 255, 255))
    draw.text((2 * w_third + 10, 5),
               f"Pred World (rot={world_metrics['rotation_error_deg']:.2f}°, t={world_metrics['translation_error_m']:.4f}m)",
               fill=(255, 255, 255))

    output_prefix = Path(args.output_prefix)
    output_png = output_prefix.with_suffix(".png")
    output_pred_pcd = output_prefix.with_suffix(".pred.pcd")
    output_gt_pcd = output_prefix.with_suffix(".gt.pcd")
    output_json = output_prefix.with_suffix(".json")

    comp_img.save(str(output_png))
    log_print(f"  Saved to: {output_png}")

    # ─── Save PCD ─────────────────────────────────────────────────────────
    # Use relative pose: depth → image cam, then project to image for colors
    log_print(f"\nSaving PCD files...")
    save_colored_pcd(
        depth_np, image_np, intrinsic_np, pred_rel_pose_7d,
        str(output_pred_pcd),
    )
    save_colored_pcd(
        depth_np, image_np, intrinsic_np, gt_rel_pose_7d,
        str(output_gt_pcd),
    )

    # Also save metrics as JSON
    metrics_path = output_json
    metrics_output = {
        "checkpoint": args.checkpoint,
        "image": args.image,
        "depth": args.depth,
        "pose_image": args.pose_image,
        "pose_depth": args.pose_depth,
        "test_directory": test_dir,
        "loss": loss_details,
        "relative_pose_metrics": rel_metrics,
        "world_pose_metrics": world_metrics,
        "predicted_rel_pose": pred_rel_pose_7d.tolist(),
        "gt_rel_pose": gt_rel_pose_7d.tolist(),
        "predicted_world_pose": pred_world_7d.tolist(),
        "gt_world_pose": gt_world_7d.tolist(),
        "convergence": [
            {
                "iteration": i,
                "rotation_error_deg": compute_metrics(p, gt_rel_pose_7d)["rotation_error_deg"],
                "translation_error_m": compute_metrics(p, gt_rel_pose_7d)["translation_error_m"],
            }
            for i, p in enumerate(pose_seq_np)
        ],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    log_print(f"  Metrics saved to: {metrics_path}")

    log_print(f"\nDone! All outputs saved to: {test_dir}")


if __name__ == "__main__":
    main()
