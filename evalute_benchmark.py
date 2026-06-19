#!/usr/bin/env python3
"""
RAFT-Pose Standardized Benchmark Evaluation Script.

Evaluates a trained model on standard RGB-D pose estimation benchmarks with
comprehensive metrics, per-scene breakdowns, and comparison tables.

Supported Benchmarks:
  - 7Scenes: 7 indoor scenes, standard benchmark for RGB-D relocalization
  - TartanAir: Synthetic scenes with aggressive motion, tests robustness

Metrics Reported:
  - Rotation Error (deg): Geodesic distance between predicted and GT quaternions
  - Translation Error (m): L2 distance between predicted and GT translations
  - Median, Mean, RMSE for both rotation and translation
  - Per-scene breakdown with success rates at standard thresholds
  - Per-iteration convergence analysis (RAFT iterative refinement)

Standard Thresholds (aligned with literature):
  - Rotation: 1°, 5°, 10°
  - Translation: 0.01m, 0.05m, 0.1m

Usage:
    # Evaluate on 7Scenes chess scene (using existing config)
    python evalute_benchmark.py \\
        --checkpoint checkpoints/runs_093/model_best.pth \\
        --config configs/chess_train.json \\
        --split val

    # Evaluate on all 7Scenes scenes with generated benchmark configs
    python evalute_benchmark.py \\
        --checkpoint checkpoints/runs_093/model_best.pth \\
        --benchmark 7scenes \\
        --split val

    # Evaluate on TartanAir tartoffice scene
    python evalute_benchmark.py \\
        --checkpoint checkpoints/runs_093/model_best.pth \\
        --benchmark tartoffice \\
        --split val

    # Evaluate on specific scenes with custom thresholds
    python evalute_benchmark.py \\
        --checkpoint checkpoints/runs_093/model_best.pth \\
        --config configs/chess_train.json \\
        --split val \\
        --rot_thresholds 1 5 10 \\
        --trans_thresholds 0.01 0.05 0.1

Output:
    - Console: formatted results table
    - JSON: full metrics saved to checkpoint_dir/benchmark_results.json
    - CSV: per-sample metrics for further analysis
"""

import os
import sys
import json
import csv
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image

# Setup import path
RAFT_POSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RAFT_POSE_DIR))

from raft_pose import RAFTPose
from dataloader import SevenScenesDataset


# ─── Benchmark Definitions ────────────────────────────────────────────────────

BENCHMARK_DEFS = {
    # 7Scenes standard benchmark — 7 indoor scenes
    "7scenes_chess": {
        "name": "7Scenes Chess",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "chess",
        "sequences": ["seq-01", "seq-02", "seq-03", "seq-04", "seq-05", "seq-06"],
        "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
        "image_size": [480, 640],
        "depth_scale": 0.001,
        "description": "Chess board scene — moderate texture, standard difficulty",
    },
    "7scenes_fire": {
        "name": "7Scenes Fire",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "fire",
        "sequences": ["seq-01", "seq-02"],
        "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
        "image_size": [480, 640],
        "depth_scale": 0.001,
        "description": "Fire extinguisher scene — repetitive texture, challenging",
    },
    "7scenes_heads": {
        "name": "7Scenes Heads",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "heads",
        "sequences": ["seq-01", "seq-02"],
        "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
        "image_size": [480, 640],
        "depth_scale": 0.001,
        "description": "Heads scene — small scene, low texture variation",
    },
    "7scenes_office": {
        "name": "7Scenes Office",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "office",
        "sequences": ["seq-01", "seq-02", "seq-03", "seq-04", "seq-05",
                      "seq-06", "seq-07", "seq-08", "seq-09", "seq-10"],
        "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
        "image_size": [480, 640],
        "depth_scale": 0.001,
        "description": "Office scene — large scene, moderate difficulty",
    },
    "7scenes_pumpkin": {
        "name": "7Scenes Pumpkin",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "pumpkin",
        "sequences": ["seq-01", "seq-02", "seq-03", "seq-04", "seq-05", "seq-06"],
        "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
        "image_size": [480, 640],
        "depth_scale": 0.001,
        "description": "Pumpkin scene — rich texture, moderate difficulty",
    },
    "7scenes_redkitchen": {
        "name": "7Scenes Red Kitchen",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "redkitchen",
        "sequences": ["seq-01", "seq-02", "seq-03", "seq-04", "seq-05", "seq-06",
                      "seq-07", "seq-08", "seq-09", "seq-10", "seq-11", "seq-12"],
        "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
        "image_size": [480, 640],
        "depth_scale": 0.001,
        "description": "Red kitchen scene — largest 7Scenes scene, hard",
    },
    "7scenes_stairs": {
        "name": "7Scenes Stairs",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "stairs",
        "sequences": ["seq-01", "seq-02", "seq-03", "seq-04", "seq-05", "seq-06"],
        "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
        "image_size": [480, 640],
        "depth_scale": 0.001,
        "description": "Stairs scene — repetitive geometry, very challenging",
    },
    # TartanAir scenes — synthetic, aggressive motion
    "tartoffice": {
        "name": "TartanAir Office",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "tartoffice",
        "sequences": ["P0000", "P0004"],
        "camera_intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 319.5},
        "image_size": [640, 640],
        "depth_scale": 0.001,
        "description": "TartanAir synthetic office — aggressive motion, robustness test",
    },
    "tarthouse": {
        "name": "TartanAir House",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "tarthouse",
        "sequences": ["P0000", "P0001", "P0002"],
        "camera_intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 319.5},
        "image_size": [640, 640],
        "depth_scale": 0.001,
        "description": "TartanAir synthetic house — large-scale, challenging motion",
    },
    "tarthospital": {
        "name": "TartanAir Hospital",
        "dataset_root": "/root/autodl-tmp/RAFT_Pose2/7Scenes/data",
        "scene": "tarthospital",
        "sequences": ["P0000"],
        "camera_intrinsics": {"fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 319.5},
        "image_size": [640, 640],
        "depth_scale": 0.001,
        "description": "TartanAir synthetic hospital — very long sequence, stress test",
    },
}

# Named groups of benchmarks
BENCHMARK_GROUPS = {
    "7scenes": ["7scenes_chess", "7scenes_fire", "7scenes_heads", "7scenes_office",
                "7scenes_pumpkin", "7scenes_redkitchen", "7scenes_stairs"],
    "tartanair": ["tartoffice", "tarthouse", "tarthospital"],
    "all": list(BENCHMARK_DEFS.keys()),
}


def _derive_clean_val_sequences():
    """Derive per-scene HELD-OUT val sequences from the training configs.

    Reads val_samples/train_samples from both 7Scenes and TartanAir configs and
    keeps, per scene, only sequences present in val AND absent from train — i.e.
    true sequence-level hold-outs (leakage-free). Scenes with no such sequence
    (e.g. heads: all seqs in train; TartanAir: val/train share sequences) are
    omitted, so --clean naturally skips them.

    The default benchmark evaluates on ALL sequences incl. train ones whose
    frames appear in train_samples (data leakage, inflates numbers ~1.7x).
    --clean restricts to these held-out sequences for a fair generalization
    measurement. Must stay in sync with the training configs' val split.
    """
    holdout = {}
    for cfg_path in ("configs/allscenes_train.json", "configs/tart_scenes_train.json"):
        try:
            with open(cfg_path) as f:
                c = json.load(f)
        except (FileNotFoundError, OSError):
            continue
        train_seqs, val_seqs = {}, {}
        for s in c.get("train_samples", []):
            train_seqs.setdefault(s["image"]["scene"], set()).add(s["image"]["seq"])
        for s in c.get("val_samples", []):
            val_seqs.setdefault(s["image"]["scene"], set()).add(s["image"]["seq"])
        for scene, vs in val_seqs.items():
            clean = sorted(v for v in vs if v not in train_seqs.get(scene, set()))
            if clean:
                holdout[scene] = clean
    return holdout


# scene -> list of held-out val sequences (true sequence-level hold-outs only)
VAL_SEQUENCES = _derive_clean_val_sequences()


# ─── Data Loading ──────────────────────────────────────────────────────────────

class BenchmarkDataset(torch.utils.data.Dataset):
    """
    Direct benchmark dataset that loads from scene directories without
    requiring pre-generated JSON configs.

    Generates frame pairs on-the-fly by sampling consecutive frames with
    configurable stride range.
    """

    def __init__(
        self,
        dataset_root: str,
        scene: str,
        sequences: list,
        camera_intrinsics: dict,
        image_size: list,
        depth_scale: float = 0.001,
        split: str = "val",
        stride_range: tuple = (1, 3),
        max_samples_per_seq: int = 500,
        val_fraction: float = 0.2,
        seed: int = 42,
        image_size_override: tuple = None,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.scene = scene
        self.depth_scale = depth_scale
        self.normalize_image = True

        self.orig_image_size = tuple(image_size)
        self.image_size = tuple(image_size_override) if image_size_override else self.orig_image_size

        # Scale intrinsics if needed
        intr = camera_intrinsics
        if self.image_size != self.orig_image_size:
            sx = self.image_size[1] / self.orig_image_size[1]
            sy = self.image_size[0] / self.orig_image_size[0]
            fx, fy = intr["fx"] * sx, intr["fy"] * sy
            cx, cy = intr["cx"] * sx, intr["cy"] * sy
        else:
            fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

        self.intrinsic_matrix = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)

        # Generate pairs for each sequence
        rng = np.random.RandomState(seed)
        self.samples = []
        sample_id = 0

        for seq in sequences:
            seq_dir = os.path.join(dataset_root, scene, seq)
            if not os.path.isdir(seq_dir):
                print(f"  [Warning] Sequence dir not found: {seq_dir}, skipping")
                continue

            # Count frames
            frame_idx = 0
            while os.path.exists(os.path.join(seq_dir, f"color_{frame_idx:03d}.png")):
                frame_idx += 1
            n_frames = frame_idx

            if n_frames < 2:
                continue

            # Split: use last val_fraction of frames for validation
            n_val = max(1, int(n_frames * val_fraction))
            if split == "val":
                start = n_frames - n_val
                end = n_frames
            else:
                start = 0
                end = n_frames - n_val

            frames = list(range(start, end))
            if len(frames) < 2:
                continue

            # Generate pairs with stride range
            pairs = []
            min_stride, max_stride = stride_range
            for stride in range(min_stride, max_stride + 1):
                for i in range(len(frames) - stride):
                    pairs.append((frames[i], frames[i + stride]))

            # Subsample if too many
            if len(pairs) > max_samples_per_seq:
                indices = rng.choice(len(pairs), max_samples_per_seq, replace=False)
                pairs = [pairs[i] for i in sorted(indices)]

            for fi, fj in pairs:
                self.samples.append({
                    "id": sample_id,
                    "image": {"scene": scene, "seq": seq, "frame": f"{fi:03d}"},
                    "depth": {"scene": scene, "seq": seq, "frame": f"{fj:03d}"},
                })
                sample_id += 1

        print(f"  [{scene}/{split}] {len(self.samples)} pairs, "
              f"image_size={self.image_size}, K=({fx:.1f}, {fy:.1f}, {cx:.1f}, {cy:.1f})")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, scene, seq, frame):
        # Frame A RGB (3, H, W), normalized [0,1].
        path = os.path.join(self.dataset_root, scene, seq, f"color_{frame}.png")
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        if self.normalize_image:
            arr = arr / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_depth(self, scene, seq, frame):
        # Frame B (4, H, W) = [raw_depth, R, G, B]: ch0 depth for projection,
        # ch1:4 frame B RGB encoded by the shared image_encoder (RGB<->RGB matching).
        dpath = os.path.join(self.dataset_root, scene, seq, f"depth_{frame}.png")
        depth = Image.open(dpath)
        depth = depth.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        darr = np.array(depth, dtype=np.float32) * self.depth_scale
        darr = np.clip(darr, 0.0, 10.0)
        cpath = os.path.join(self.dataset_root, scene, seq, f"color_{frame}.png")
        cimg = Image.open(cpath).convert("RGB")
        cimg = cimg.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        carr = np.array(cimg, dtype=np.float32)
        if self.normalize_image:
            carr = carr / 255.0
        return torch.from_numpy(np.stack([darr, carr[..., 0], carr[..., 1], carr[..., 2]], axis=0))

    def _load_pose(self, scene, seq, frame):
        path = os.path.join(self.dataset_root, scene, seq, f"pose_{frame}.txt")
        pose = np.loadtxt(path, dtype=np.float64)
        return torch.from_numpy(pose).float()

    def _compute_relative_pose(self, T_img, T_depth):
        T_rel = torch.linalg.inv(T_img) @ T_depth
        R = T_rel[:3, :3]
        t = T_rel[:3, 3]

        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        quat = torch.stack([qw, qx, qy, qz])
        quat = quat / quat.norm()
        return torch.cat([quat, t])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_ref = sample["image"]
        dep_ref = sample["depth"]

        image = self._load_image(img_ref["scene"], img_ref["seq"], img_ref["frame"])
        depth = self._load_depth(dep_ref["scene"], dep_ref["seq"], dep_ref["frame"])

        T_img = self._load_pose(img_ref["scene"], img_ref["seq"], img_ref["frame"])
        T_depth = self._load_pose(dep_ref["scene"], dep_ref["seq"], dep_ref["frame"])
        gt_pose = self._compute_relative_pose(T_img, T_depth)

        return {
            "image": image,
            "depth": depth,
            "intrinsic_rgb": self.intrinsic_matrix.clone(),
            "intrinsic_depth": self.intrinsic_matrix.clone(),
            "gt_pose": gt_pose,
            "sample_id": sample["id"],
            "image_frame": img_ref["frame"],
            "depth_frame": dep_ref["frame"],
        }


# ─── Metrics ───────────────────────────────────────────────────────────────────

def compute_rotation_error_deg(pred_quat, gt_quat):
    """Geodesic rotation error in degrees. Works with tensors or numpy arrays."""
    if isinstance(pred_quat, torch.Tensor):
        pred_q = F.normalize(pred_quat, dim=-1)
        gt_q = F.normalize(gt_quat, dim=-1)
        dot = torch.sum(pred_q * gt_q, dim=-1).abs().clamp(0.0, 1.0)
        return 2.0 * torch.acos(dot) * (180.0 / 3.14159265358979)
    else:
        pred_q = pred_quat / np.linalg.norm(pred_quat, axis=-1, keepdims=True)
        gt_q = gt_quat / np.linalg.norm(gt_quat, axis=-1, keepdims=True)
        dot = np.abs(np.sum(pred_q * gt_q, axis=-1))
        dot = np.clip(dot, 0.0, 1.0)
        return 2.0 * np.arccos(dot) * (180.0 / np.pi)


def compute_translation_error_m(pred_trans, gt_trans):
    """L2 translation error in meters."""
    if isinstance(pred_trans, torch.Tensor):
        return torch.norm(pred_trans - gt_trans, dim=-1)
    else:
        return np.linalg.norm(pred_trans - gt_trans, axis=-1)


def compute_statistics(errors):
    """Compute comprehensive statistics for an array of errors."""
    errors = np.array(errors)
    if len(errors) == 0:
        return {"count": 0, "mean": float('nan'), "median": float('nan'),
                "rmse": float('nan'), "std": float('nan'),
                "min": float('nan'), "max": float('nan'), "p95": float('nan')}
    return {
        "count": len(errors),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "p95": float(np.percentile(errors, 95)),
    }


def compute_success_rate(errors, thresholds):
    """Compute fraction of samples below each threshold."""
    if len(errors) == 0:
        return {t: float('nan') for t in thresholds}
    return {t: float(np.mean(np.array(errors) < t) * 100.0) for t in thresholds}


# ─── Model Loading ─────────────────────────────────────────────────────────────

def load_model_for_eval(checkpoint_path, device, overrides=None):
    """Load RAFTPose model from checkpoint with optional overrides."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})

    if overrides is None:
        overrides = {}

    def get_arg(key, default=None):
        return overrides.get(key, ckpt_args.get(key, default))

    model = RAFTPose(
        image_encoder=get_arg("image_encoder", "basic"),
        hidden_dim=get_arg("hidden_dim", 128),
        context_dim=get_arg("context_dim", 64),
        depth_dim=get_arg("depth_dim", 32),
        corr_levels=get_arg("corr_levels", 4),
        corr_radius=get_arg("corr_radius", 2),
        num_iterations=get_arg("num_iterations", 6),
        pose_sample_std=get_arg("pose_sample_std", 0.05),
        init_pose_noise_std=get_arg("init_pose_noise_std", 0.0),
        use_checkpoint=False,
        use_amp=False,
        coarse_to_fine=get_arg("coarse_to_fine", False),
        corr_temperature=get_arg("corr_temperature", 1.0),
        max_rot_step=get_arg("max_rot_step", 0.3),
        max_trans_step=get_arg("max_trans_step", 0.3),
        shared_encoder=get_arg("shared_encoder", False),
        matcher_type=get_arg("matcher_type", "corr"),
        matcher_heads=get_arg("matcher_heads", 8),
        matcher_blocks=get_arg("matcher_blocks", 2),
        matcher_ffn_dim=get_arg("matcher_ffn_dim", 512),
        matcher_dropout=get_arg("matcher_dropout", 0.1),
        coarse_iters=get_arg("coarse_iters", 3),
    )

    # Load weights with compatibility handling
    model_state = checkpoint["model_state_dict"]
    # Strip torch.compile's '_orig_mod.' prefix if the checkpoint was saved
    # from a compiled model (keys like '_orig_mod.image_encoder...').
    if any(k.startswith("_orig_mod.") for k in model_state):
        model_state = {(k[10:] if k.startswith("_orig_mod.") else k): v
                       for k, v in model_state.items()}
        print("  [Compat] Stripped _orig_mod. prefix (torch.compile checkpoint)")
    new_state = model.state_dict()

    # Filter mismatched shapes
    mismatch_keys = []
    for k in list(model_state.keys()):
        if k in new_state and model_state[k].shape != new_state[k].shape:
            mismatch_keys.append(k)
            del model_state[k]
        elif k not in new_state:
            del model_state[k]

    if mismatch_keys:
        print(f"  [Compat] Skipped {len(mismatch_keys)} mismatched layers: {mismatch_keys[:5]}...")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  [Compat] Missing keys ({len(missing)}): {missing[:5]}...")

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    epoch = checkpoint.get("epoch", "N/A")
    best_val_loss = checkpoint.get("best_val_loss", "N/A")

    print(f"  Model loaded: {n_params:,} params, epoch={epoch}, "
          f"best_val_loss={best_val_loss}")

    return model, ckpt_args


# ─── Evaluation Engine ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(model, dataset, device, batch_size=1, num_iterations=None):
    """
    Run evaluation on a dataset, collecting per-sample metrics.

    Returns:
        results: list of dicts with per-sample metrics
        convergence_data: list of per-iteration errors (if available)
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )

    all_rot_errors = []
    all_trans_errors = []
    all_gt_poses = []
    all_pred_poses = []
    per_sample_results = []
    convergence_data = []  # per-iteration errors

    total_time = 0.0

    for batch_idx, batch in enumerate(dataloader):
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        intrinsic_rgb = batch["intrinsic_rgb"].to(device)
        intrinsic_depth = batch["intrinsic_depth"].to(device)
        gt_pose = batch["gt_pose"].to(device)
        sample_ids = batch["sample_id"].numpy()
        img_frames = batch["image_frame"]
        dep_frames = batch["depth_frame"]

        B = image.size(0)
        start_time = time.time()

        # Run model
        result = model(
            image=image,
            depth=depth,
            intrinsic_rgb=intrinsic_rgb,
            intrinsic_depth=intrinsic_depth,
            return_all_poses=True,
        )

        # Extract predictions
        if isinstance(result, tuple):
            pred_pose = result[0]  # (B, 7)
            extra = result[1] if len(result) > 1 else {}
        else:
            pred_pose = result
            extra = {}

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        total_time += elapsed

        # Compute errors
        pred_quat = pred_pose[:, :4]
        gt_quat = gt_pose[:, :4]
        pred_trans = pred_pose[:, 4:7]
        gt_trans = gt_pose[:, 4:7]

        rot_errors = compute_rotation_error_deg(pred_quat, gt_quat).cpu().numpy()
        trans_errors = compute_translation_error_m(pred_trans, gt_trans).cpu().numpy()

        # Per-sample results
        for i in range(B):
            all_rot_errors.append(rot_errors[i])
            all_trans_errors.append(trans_errors[i])
            all_gt_poses.append(gt_pose[i].cpu().numpy())
            all_pred_poses.append(pred_pose[i].cpu().numpy())

            per_sample_results.append({
                "sample_id": int(sample_ids[i]),
                "image_frame": img_frames[i] if isinstance(img_frames[i], str) else str(img_frames[i]),
                "depth_frame": dep_frames[i] if isinstance(dep_frames[i], str) else str(dep_frames[i]),
                "rotation_error_deg": float(rot_errors[i]),
                "translation_error_m": float(trans_errors[i]),
            })

        # Per-iteration convergence (if available)
        if 'pose_sequence' in extra:
            pose_seq = extra['pose_sequence']  # (B, K+1, 7)
            if isinstance(pose_seq, torch.Tensor) and pose_seq.dim() == 3:
                for i in range(B):
                    seq_rot_errs = []
                    seq_trans_errs = []
                    for k in range(pose_seq.size(1)):
                        p = pose_seq[i, k]
                        r_err = compute_rotation_error_deg(
                            p[:4].unsqueeze(0), gt_quat[i:i+1]
                        ).item()
                        t_err = compute_translation_error_m(
                            p[4:7].unsqueeze(0), gt_trans[i:i+1]
                        ).item()
                        seq_rot_errs.append(r_err)
                        seq_trans_errs.append(t_err)
                    convergence_data.append({
                        "sample_id": int(sample_ids[i]),
                        "rot_errors_per_iter": seq_rot_errs,
                        "trans_errors_per_iter": seq_trans_errs,
                    })

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"    [{batch_idx+1}/{len(dataloader)}] "
                  f"avg_rot={np.mean(all_rot_errors):.2f}°, "
                  f"avg_trans={np.mean(all_trans_errors):.4f}m, "
                  f"time={total_time:.1f}s")

    return per_sample_results, convergence_data


# ─── Reporting ──────────────────────────────────────────────────────────────────

def format_results_table(scene_name, rot_stats, trans_stats,
                         rot_success, trans_success,
                         rot_thresholds, trans_thresholds):
    """Format a single scene's results as a readable table row."""
    row = (
        f"  {scene_name:<25s} │ "
        f"{rot_stats['median']:6.2f} {rot_stats['mean']:6.2f} {rot_stats['rmse']:6.2f} │ "
        f"{trans_stats['median']:7.4f} {trans_stats['mean']:7.4f} {trans_stats['rmse']:7.4f} │ "
        f"{rot_stats['p95']:6.2f} │ "
        f"{trans_stats['p95']:7.4f} │ "
        f"{rot_stats['count']}"
    )
    return row


def print_full_report(all_results, rot_thresholds, trans_thresholds, output_path=None):
    """Print comprehensive benchmark results."""

    lines = []
    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗")
    lines.append("║                         RAFT-Pose Benchmark Evaluation Results                                  ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣")
    lines.append("║                                                                                                ║")
    lines.append("║  Rotation Error (°)                                                                           ║")
    lines.append("║  ┌─────────────────────────┬──────────────────────────────┬─────────────┬───────────┐           ║")
    lines.append("║  │ Scene                   │   Median    Mean    RMSE     │    P95      │  Samples  │           ║")
    lines.append("║  ├─────────────────────────┼──────────────────────────────┼─────────────┼───────────┤           ║")

    for r in all_results:
        name = r["scene_name"]
        rs = r["rot_stats"]
        ts = r["trans_stats"]
        lines.append(
            f"║  │ {name:<23s} │ {rs['median']:7.2f} {rs['mean']:7.2f} {rs['rmse']:7.2f} │ "
            f"{rs['p95']:9.2f}  │ {rs['count']:>7d}  │           ║"
        )

    lines.append("║  └─────────────────────────┴──────────────────────────────┴─────────────┴───────────┘           ║")
    lines.append("║                                                                                                ║")
    lines.append("║  Translation Error (m)                                                                         ║")
    lines.append("║  ┌─────────────────────────┬──────────────────────────────┬─────────────┬───────────┐           ║")
    lines.append("║  │ Scene                   │   Median    Mean    RMSE     │    P95      │  Samples  │           ║")
    lines.append("║  ├─────────────────────────┼──────────────────────────────┼─────────────┼───────────┤           ║")

    for r in all_results:
        name = r["scene_name"]
        ts = r["trans_stats"]
        lines.append(
            f"║  │ {name:<23s} │ {ts['median']:7.4f} {ts['mean']:7.4f} {ts['rmse']:7.4f} │ "
            f"{ts['p95']:9.4f}  │ {ts['count']:>7d}  │           ║"
        )

    lines.append("║  └─────────────────────────┴──────────────────────────────┴─────────────┴───────────┘           ║")

    # Success rates at thresholds
    lines.append("║                                                                                                ║")
    lines.append("║  Success Rate (%) — Rotation (°)                                                               ║")
    rot_thresh_str = "  ".join(f"{'<'+str(t)+'°':>8s}" for t in rot_thresholds)
    lines.append(f"║  ┌─────────────────────────┬{('─'*9)*len(rot_thresholds)}┐                                    ║")
    lines.append(f"║  │ Scene                   │ {rot_thresh_str} │                                    ║")
    lines.append(f"║  ├─────────────────────────┼{('─'*9)*len(rot_thresholds)}┤                                    ║")

    for r in all_results:
        name = r["scene_name"]
        rs = r["rot_success"]
        vals_str = "  ".join(f"{rs[t]:7.1f}%" for t in rot_thresholds)
        lines.append(f"║  │ {name:<23s} │ {vals_str} │                                    ║")

    lines.append(f"║  └─────────────────────────┴{('─'*9)*len(rot_thresholds)}┘                                    ║")

    # Translation success rates
    lines.append("║                                                                                                ║")
    lines.append("║  Success Rate (%) — Translation (m)                                                            ║")
    trans_thresh_str = "  ".join(f"{'<'+str(t)+'m':>8s}" for t in trans_thresholds)
    lines.append(f"║  ┌─────────────────────────┬{('─'*9)*len(trans_thresholds)}┐                                    ║")
    lines.append(f"║  │ Scene                   │ {trans_thresh_str} │                                    ║")
    lines.append(f"║  ├─────────────────────────┼{('─'*9)*len(trans_thresholds)}┤                                    ║")

    for r in all_results:
        name = r["scene_name"]
        ts = r["trans_success"]
        vals_str = "  ".join(f"{ts[t]:7.1f}%" for t in trans_thresholds)
        lines.append(f"║  │ {name:<23s} │ {vals_str} │                                    ║")

    lines.append(f"║  └─────────────────────────┴{('─'*9)*len(trans_thresholds)}┘                                    ║")

    # Aggregate summary
    all_rot = np.concatenate([np.array(r["rot_errors"]) for r in all_results])
    all_trans = np.concatenate([np.array(r["trans_errors"]) for r in all_results])

    lines.append("║                                                                                                ║")
    lines.append("║  ─── Overall Summary ───                                                                       ║")
    lines.append(f"║  Total samples:          {len(all_rot):>8d}                                                          ║")
    lines.append(f"║  Rotation  — Median: {np.median(all_rot):6.2f}°  Mean: {np.mean(all_rot):6.2f}°  RMSE: {np.sqrt(np.mean(all_rot**2)):6.2f}°                        ║")
    lines.append(f"║  Translation — Median: {np.median(all_trans):6.4f}m  Mean: {np.mean(all_trans):6.4f}m  RMSE: {np.sqrt(np.mean(all_trans**2)):6.4f}m                       ║")
    lines.append("║                                                                                                ║")
    lines.append("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝")

    full_report = "\n".join(lines)
    print(full_report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(full_report)
        print(f"\nReport saved to: {output_path}")

    return full_report


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="RAFT-Pose Standardized Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single config (e.g., chess val set)
  python evalute_benchmark.py --checkpoint ckpt.pth --config configs/chess_train.json --split val

  # Predefined benchmarks
  python evalute_benchmark.py --checkpoint ckpt.pth --benchmark 7scenes_chess
  python evalute_benchmark.py --checkpoint ckpt.pth --benchmark 7scenes
  python evalute_benchmark.py --checkpoint ckpt.pth --benchmark tartanair
  python evalute_benchmark.py --checkpoint ckpt.pth --benchmark all

Available benchmarks: 7scenes_chess, 7scenes_fire, 7scenes_heads, 7scenes_office,
  7scenes_pumpkin, 7scenes_redkitchen, 7scenes_stairs, tartoffice, tarthouse,
  tarthospital, 7scenes (all 7 scenes), tartanair (3 scenes), all (10 scenes)
        """,
    )

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (default: 1)")

    # Data source: either config file or predefined benchmark
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--config", type=str,
                            help="Path to dataset JSON config file (same format as training)")
    data_group.add_argument("--benchmark", type=str,
                            help="Predefined benchmark name or group "
                                 "(7scenes_chess, tartoffice, 7scenes, tartanair, all)")

    # Config-based options
    parser.add_argument("--split", type=str, default="val",
                        help="Data split to evaluate (default: val)")
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Override image size (H W) for evaluation")

    # Benchmark-based options
    parser.add_argument("--stride_range", type=int, nargs=2, default=[1, 3],
                        metavar=("MIN", "MAX"),
                        help="Frame stride range for benchmark pair generation (default: 1 3)")
    parser.add_argument("--max_samples_per_seq", type=int, default=500,
                        help="Max samples per sequence (default: 500)")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of frames used for validation (default: 0.2)")

    # Metric thresholds
    parser.add_argument("--rot_thresholds", type=float, nargs="+",
                        default=[1.0, 5.0, 10.0],
                        help="Rotation error thresholds in degrees (default: 1 5 10)")
    parser.add_argument("--trans_thresholds", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1],
                        help="Translation error thresholds in meters (default: 0.01 0.05 0.1)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: alongside checkpoint)")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save per-sample metrics as CSV")

    # Model overrides
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--image_encoder", type=str, default=None)
    parser.add_argument("--shared_encoder", action="store_true", default=None)
    parser.add_argument("--matcher_type", type=str, default=None)
    parser.add_argument("--coarse_iters", type=int, default=None,
                        help="Override coarse_iters (number of 1/8 coarse iterations). "
                             "Set to num_iterations to DISABLE the fine stage "
                             "(fine-ON vs fine-OFF ablation). Default: from checkpoint.")
    parser.add_argument("--clean", action="store_true",
                        help="Evaluate ONLY on held-out val sequences (per-scene sequences "
                             "in val_samples but NOT train_samples). Avoids the ~1.7x "
                             "inflation from training-frame leakage in the default "
                             "(all-sequences) benchmark. Scenes with no clean hold-out "
                             "(heads, TartanAir) are skipped automatically.")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("RAFT-Pose Standardized Benchmark Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    # ─── Load Model ────────────────────────────────────────────────────────
    print("\n─── Loading Model ───")
    overrides = {}
    if args.num_iterations is not None:
        overrides["num_iterations"] = args.num_iterations
    if args.image_encoder is not None:
        overrides["image_encoder"] = args.image_encoder
    if args.shared_encoder:
        overrides["shared_encoder"] = True
    if args.matcher_type is not None:
        overrides["matcher_type"] = args.matcher_type
    if args.coarse_iters is not None:
        overrides["coarse_iters"] = args.coarse_iters

    model, ckpt_args = load_model_for_eval(args.checkpoint, device, overrides)

    # ─── Determine Benchmarks ──────────────────────────────────────────────
    benchmarks_to_run = []

    if args.config:
        # Single config file mode
        benchmarks_to_run.append({
            "name": Path(args.config).stem,
            "type": "config",
            "config_path": args.config,
            "split": args.split,
        })
    elif args.benchmark:
        # Resolve benchmark name/group
        if args.benchmark in BENCHMARK_GROUPS:
            bench_names = BENCHMARK_GROUPS[args.benchmark]
        elif args.benchmark in BENCHMARK_DEFS:
            bench_names = [args.benchmark]
        else:
            print(f"Error: Unknown benchmark '{args.benchmark}'")
            print(f"  Available: {list(BENCHMARK_DEFS.keys())}")
            print(f"  Groups: {list(BENCHMARK_GROUPS.keys())}")
            sys.exit(1)

        for bn in bench_names:
            bdef = BENCHMARK_DEFS[bn]
            benchmarks_to_run.append({
                "name": bdef["name"],
                "type": "definition",
                "definition": bdef,
                "stride_range": tuple(args.stride_range),
                "max_samples_per_seq": args.max_samples_per_seq,
                "val_fraction": args.val_fraction,
            })

    # ─── Output Directory ──────────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # --clean writes to a separate dir so it never clobbers the default
        # (all-sequences) benchmark report.
        subdir = "benchmark_eval_clean" if args.clean else "benchmark_eval"
        output_dir = Path(args.checkpoint).parent / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ─── Run Evaluation ────────────────────────────────────────────────────
    all_results = []

    for bench in benchmarks_to_run:
        print(f"\n{'─'*60}")
        print(f"Evaluating: {bench['name']}")
        print(f"{'─'*60}")

        # Build dataset
        if bench["type"] == "config":
            dataset = SevenScenesDataset(
                config_path=bench["config_path"],
                split=bench["split"],
                image_size=tuple(args.image_size) if args.image_size else None,
                augment=False,
            )
        else:
            bdef = bench["definition"]
            # --clean: restrict to held-out val sequences (leakage-free). Scenes
            # with no clean hold-out get [] -> skipped by the len==0 check below.
            seqs = bdef["sequences"]
            if args.clean:
                seqs = VAL_SEQUENCES.get(bdef["scene"], [])
                print(f"  [--clean] {bdef['scene']}: held-out val seqs = {seqs}")
            dataset = BenchmarkDataset(
                dataset_root=bdef["dataset_root"],
                scene=bdef["scene"],
                sequences=seqs,
                camera_intrinsics=bdef["camera_intrinsics"],
                image_size=bdef["image_size"],
                depth_scale=bdef["depth_scale"],
                split="val",
                stride_range=bench["stride_range"],
                max_samples_per_seq=bench["max_samples_per_seq"],
                val_fraction=bench["val_fraction"],
                image_size_override=tuple(args.image_size) if args.image_size else None,
            )

        if len(dataset) == 0:
            print(f"  ⚠ No samples found for {bench['name']}, skipping")
            continue

        # Evaluate
        per_sample, convergence = evaluate_dataset(
            model, dataset, device, batch_size=args.batch_size
        )

        # Compute statistics
        rot_errors = [s["rotation_error_deg"] for s in per_sample]
        trans_errors = [s["translation_error_m"] for s in per_sample]

        rot_stats = compute_statistics(rot_errors)
        trans_stats = compute_statistics(trans_errors)
        rot_success = compute_success_rate(rot_errors, args.rot_thresholds)
        trans_success = compute_success_rate(trans_errors, args.trans_thresholds)

        result = {
            "scene_name": bench["name"],
            "rot_errors": rot_errors,
            "trans_errors": trans_errors,
            "rot_stats": rot_stats,
            "trans_stats": trans_stats,
            "rot_success": rot_success,
            "trans_success": trans_success,
            "per_sample": per_sample,
            "convergence": convergence,
            "n_samples": len(per_sample),
        }
        all_results.append(result)

        # Print scene summary
        print(f"\n  Results for {bench['name']}:")
        print(f"    Rotation  — Median: {rot_stats['median']:.2f}°, "
              f"Mean: {rot_stats['mean']:.2f}°, RMSE: {rot_stats['rmse']:.2f}°")
        print(f"    Translation — Median: {trans_stats['median']:.4f}m, "
              f"Mean: {trans_stats['mean']:.4f}m, RMSE: {trans_stats['rmse']:.4f}m")
        print(f"    Samples: {len(per_sample)}")

    # ─── Full Report ───────────────────────────────────────────────────────
    if not all_results:
        print("\n⚠ No results to report.")
        return

    report_path = output_dir / "benchmark_report.txt"
    print_full_report(
        all_results,
        args.rot_thresholds,
        args.trans_thresholds,
        output_path=report_path,
    )

    # ─── Save JSON Results ─────────────────────────────────────────────────
    json_output = {
        "checkpoint": args.checkpoint,
        "benchmark_args": {
            "rot_thresholds": args.rot_thresholds,
            "trans_thresholds": args.trans_thresholds,
            "stride_range": args.stride_range,
            "image_size_override": args.image_size,
        },
        "scenes": [],
        "overall": {},
    }

    for r in all_results:
        scene_data = {
            "name": r["scene_name"],
            "n_samples": r["n_samples"],
            "rotation_stats": r["rot_stats"],
            "translation_stats": r["trans_stats"],
            "rotation_success_rate_%": r["rot_success"],
            "translation_success_rate_%": r["trans_success"],
        }

        # Convergence summary (average across samples)
        if r["convergence"]:
            n_iters = len(r["convergence"][0]["rot_errors_per_iter"])
            avg_rot_per_iter = [
                np.mean([c["rot_errors_per_iter"][k] for c in r["convergence"]])
                for k in range(n_iters)
            ]
            avg_trans_per_iter = [
                np.mean([c["trans_errors_per_iter"][k] for c in r["convergence"]])
                for k in range(n_iters)
            ]
            scene_data["convergence"] = {
                "avg_rot_error_per_iter_deg": avg_rot_per_iter,
                "avg_trans_error_per_iter_m": avg_trans_per_iter,
            }

        json_output["scenes"].append(scene_data)

    # Overall
    all_rot = np.concatenate([np.array(r["rot_errors"]) for r in all_results])
    all_trans = np.concatenate([np.array(r["trans_errors"]) for r in all_results])
    json_output["overall"] = {
        "total_samples": len(all_rot),
        "rotation": compute_statistics(all_rot),
        "translation": compute_statistics(all_trans),
        "rotation_success_rate_%": compute_success_rate(all_rot, args.rot_thresholds),
        "translation_success_rate_%": compute_success_rate(all_trans, args.trans_thresholds),
    }

    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=float)
    print(f"\nJSON results saved to: {json_path}")

    # ─── Save CSV ──────────────────────────────────────────────────────────
    if args.save_csv:
        csv_path = output_dir / "per_sample_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "scene", "sample_id", "image_frame", "depth_frame",
                "rotation_error_deg", "translation_error_m",
            ])
            for r in all_results:
                for s in r["per_sample"]:
                    writer.writerow([
                        r["scene_name"],
                        s["sample_id"],
                        s["image_frame"],
                        s["depth_frame"],
                        f"{s['rotation_error_deg']:.6f}",
                        f"{s['translation_error_m']:.6f}",
                    ])
        print(f"CSV results saved to: {csv_path}")

    # ─── Convergence Plot Data ─────────────────────────────────────────────
    for r in all_results:
        if r["convergence"]:
            conv_path = output_dir / f"convergence_{r['scene_name'].replace(' ', '_')}.json"
            with open(conv_path, "w") as f:
                json.dump(r["convergence"], f, indent=2, default=float)

    print(f"\n{'='*80}")
    print(f"Benchmark evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
