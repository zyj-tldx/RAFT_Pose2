#!/usr/bin/env python3
"""
Generate training dataset JSON for RAFT-Pose from a definition JSON.

Two-step workflow:
  1. Write a definition JSON that specifies image groups (scene, seq, frame range)
  2. Run this script to generate the dataset JSON with paired samples

Definition JSON format (input):
{
    "dataset_root": "7Scenes/data",
    "output": "configs/chess_train.json",
    "camera_intrinsics": {"fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0},
    "image_size": [480, 640],
    "depth_scale": 0.001,
    "strategy": "random_offset",
    "strategy_args": {
        "max_offset": 10,
        "min_offset": 1,
        "samples_per_group": 200
    },
    "groups": [
        {
            "scene": "chess",
            "seq": "seq-01",
            "frame_range": [0, 79],
            "split": "train"
        },
        {
            "scene": "chess",
            "seq": "seq-02",
            "frame_range": [0, 39],
            "split": "train"
        },
        {
            "scene": "chess",
            "seq": "seq-02",
            "frame_range": [40, 79],
            "split": "val"
        }
    ]
}

Supported strategies:
  - "random_offset": randomly sample frame pairs with offset in [min_offset, max_offset]
  - "all_pairs":     exhaustive C(n,2) pairing within each group

Output dataset JSON format (unchanged):
{
    "dataset_root": "...",
    "camera_intrinsics": {...},
    "image_size": [H, W],
    "depth_scale": 0.001,
    "train_samples": [{"id": 0, "image": {...}, "depth": {...}}, ...],
    "val_samples": [...]
}

GT pose is computed dynamically during training: T_rel = T_img^(-1) @ T_depth.
"""

import os
import json
import random
import argparse
from pathlib import Path
from itertools import combinations


# ─── Pairing Strategies ──────────────────────────────────────────────────────

def pairs_random_offset(frames, min_offset=1, max_offset=10,
                        samples_per_group=200):
    """
    Random offset pairing within a group of frame indices.

    For each sample, pick a random frame i, then pick frame j = i ± offset
    where offset is in [min_offset, max_offset].

    Args:
        frames: list of frame indices (int)
        min_offset: minimum frame offset
        max_offset: maximum frame offset
        samples_per_group: number of pairs to generate

    Returns:
        list of (frame_i, frame_j) tuples
    """
    frame_set = set(frames)
    pairs = []
    attempts = 0
    max_attempts = samples_per_group * 10

    while len(pairs) < samples_per_group and attempts < max_attempts:
        attempts += 1
        fi = random.choice(frames)
        offset = random.randint(min_offset, max_offset)
        if random.random() < 0.5:
            offset = -offset
        fj = fi + offset
        if fj in frame_set:
            pairs.append((fi, fj))

    return pairs


def pairs_all_pairs(frames):
    """
    Exhaustive pairing: all C(n,2) combinations within a group.

    Args:
        frames: list of frame indices (int)

    Returns:
        list of (frame_i, frame_j) tuples
    """
    return list(combinations(frames, 2))


# ─── Main ─────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "random_offset": pairs_random_offset,
    "all_pairs": pairs_all_pairs,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate RAFT-Pose dataset JSON from a definition JSON"
    )
    parser.add_argument(
        "--definition", type=str, required=True,
        help="Path to the definition JSON file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (overrides definition JSON's 'output' field)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides definition JSON's seed)"
    )

    args = parser.parse_args()

    # Load definition JSON
    def_path = Path(args.definition)
    if not def_path.exists():
        raise FileNotFoundError(f"Definition JSON not found: {def_path}")

    with open(def_path, 'r') as f:
        definition = json.load(f)

    # Set random seed
    seed = args.seed if args.seed is not None else definition.get("seed", 42)
    random.seed(seed)

    # Resolve paths relative to definition JSON location
    def_dir = def_path.parent
    dataset_root = definition.get("dataset_root", "7Scenes/data")
    if not os.path.isabs(dataset_root):
        dataset_root = str(def_dir / dataset_root)
    dataset_root = os.path.abspath(dataset_root)

    output_path = args.output or definition.get("output", "dataset.json")
    if not os.path.isabs(output_path):
        output_path = str(def_dir / output_path)

    # Read config fields
    intrinsics = definition.get("camera_intrinsics", {
        "fx": 585.0, "fy": 585.0, "cx": 320.0, "cy": 240.0
    })
    image_size = definition.get("image_size", [480, 640])
    depth_scale = definition.get("depth_scale", 0.001)

    strategy_name = definition.get("strategy", "random_offset")
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )
    strategy_fn = STRATEGIES[strategy_name]
    strategy_args = definition.get("strategy_args", {})

    groups = definition.get("groups", [])
    if not groups:
        raise ValueError("No groups defined in the definition JSON")

    # ── Generate pairs per group ──────────────────────────────────────────
    train_samples = []
    val_samples = []

    print(f"Strategy: {strategy_name}")
    print(f"Strategy args: {strategy_args}")
    print(f"Groups: {len(groups)}")
    print()

    for gi, group in enumerate(groups):
        scene = group["scene"]
        seq = group["seq"]
        frame_start, frame_end = group["frame_range"]
        split = group.get("split", "train")

        # Generate frame index list
        frames = list(range(frame_start, frame_end + 1))
        print(f"  Group {gi}: scene={scene}, seq={seq}, "
              f"frames=[{frame_start}, {frame_end}] ({len(frames)} frames), "
              f"split={split}")

        # Generate pairs
        if strategy_name == "random_offset":
            spg = strategy_args.get("samples_per_group", 200)
            min_off = strategy_args.get("min_offset", 1)
            max_off = strategy_args.get("max_offset", 10)
            raw_pairs = strategy_fn(
                frames,
                min_offset=min_off,
                max_offset=max_off,
                samples_per_group=spg
            )
        elif strategy_name == "all_pairs":
            raw_pairs = strategy_fn(frames)
        else:
            raw_pairs = strategy_fn(frames, **strategy_args)

        # Convert to sample dicts
        for fi, fj in raw_pairs:
            sample = {
                "image": {
                    "scene": scene,
                    "seq": seq,
                    "frame": f"{fi:03d}"
                },
                "depth": {
                    "scene": scene,
                    "seq": seq,
                    "frame": f"{fj:03d}"
                }
            }
            if split == "val":
                val_samples.append(sample)
            else:
                train_samples.append(sample)

        print(f"    -> {len(raw_pairs)} pairs generated")

    # Add IDs
    train_samples = [{"id": i, **s} for i, s in enumerate(train_samples)]
    val_samples = [{"id": i, **s} for i, s in enumerate(val_samples)]

    print()
    print(f"Total train samples: {len(train_samples)}")
    print(f"Total val samples:   {len(val_samples)}")

    # ── Build output config ───────────────────────────────────────────────
    config = {
        "dataset_root": dataset_root,
        "camera_intrinsics": intrinsics,
        "image_size": image_size,
        "depth_scale": depth_scale,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "metadata": {
            "definition_file": str(def_path),
            "strategy": strategy_name,
            "strategy_args": strategy_args,
            "total_train": len(train_samples),
            "total_val": len(val_samples),
            "seed": seed,
        }
    }

    # Write output JSON
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to: {out_path}")

    # Print sample entries
    if train_samples:
        print(f"\nSample train entries:")
        for s in train_samples[:3]:
            print(f"  {s}")
    if val_samples:
        print(f"\nSample val entries:")
        for s in val_samples[:3]:
            print(f"  {s}")


if __name__ == "__main__":
    main()
