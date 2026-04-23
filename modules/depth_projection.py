"""
Depth projection and correlation sampling for RAFT-Pose module.
Projects depth maps using multiple poses and samples from correlation volume.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthProjector(nn.Module):
    """
    Projects 3D points from depth map to camera image plane using extrinsic parameters.
    Supports multiple pose transformations for correlation sampling.
    """
    def __init__(self):
        super(DepthProjector, self).__init__()
    
    def forward(self, depth_map, extrinsics, intrinsic_depth, intrinsic_rgb):
        """
        Project 3D points from depth map to camera image plane using multiple extrinsics.
        
        Args:
            depth_map (torch.Tensor): Depth map of shape (B, H, W)
            extrinsics (torch.Tensor): Extrinsic matrices of shape (B, N, 4, 4), 
                                      where N is the number of pose samples
            intrinsic_depth (torch.Tensor): Intrinsic matrix of depth camera, shape (B, 3, 3)
            intrinsic_rgb (torch.Tensor): Intrinsic matrix of RGB camera, shape (B, 3, 3)
        
        Returns:
            torch.Tensor: Projected coordinates of shape (B, N, 2, H, W), 
                          where the channels are u and v coordinates
        """
        B, H, W = depth_map.shape
        N = extrinsics.shape[1]
        device = depth_map.device
        
        # Create pixel coordinates
        u_d = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W).expand(B, H, W)
        v_d = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1).expand(B, H, W)
        
        # Normalize coordinates using depth intrinsic
        fx_d = intrinsic_depth[:, 0, 0].view(B, 1, 1)
        fy_d = intrinsic_depth[:, 1, 1].view(B, 1, 1)
        cx_d = intrinsic_depth[:, 0, 2].view(B, 1, 1)
        cy_d = intrinsic_depth[:, 1, 2].view(B, 1, 1)
        
        x_d = (u_d - cx_d) / fx_d
        y_d = (v_d - cy_d) / fy_d
        
        # 3D points in depth camera coordinate system (homogeneous)
        X_d = x_d * depth_map
        Y_d = y_d * depth_map
        Z_d = depth_map
        ones = torch.ones_like(Z_d)
        
        P_D = torch.stack([X_d, Y_d, Z_d, ones], dim=1)  # (B, 4, H, W)
        
        # Transform to RGB camera coordinate system for all N extrinsics
        # P_D: (B, 4, H, W) -> (B, 1, 4, H*W)
        P_D_flat = P_D.view(B, 4, -1).unsqueeze(1)  # (B, 1, 4, H*W)
        
        # extrinsics: (B, N, 4, 4)
        # Reshape for batch matrix multiplication
        P_C_flat = torch.matmul(extrinsics, P_D_flat)  # (B, N, 4, H*W)
        P_C = P_C_flat.view(B, N, 4, H, W)  # (B, N, 4, H, W)
        
        # Project to image plane using RGB intrinsic
        fx_rgb = intrinsic_rgb[:, 0, 0].view(B, 1, 1, 1)
        fy_rgb = intrinsic_rgb[:, 1, 1].view(B, 1, 1, 1)
        cx_rgb = intrinsic_rgb[:, 0, 2].view(B, 1, 1, 1)
        cy_rgb = intrinsic_rgb[:, 1, 2].view(B, 1, 1, 1)
        
        # Depth validity check: only project points with positive depth
        # after transformation (Z_C > 0 means in front of camera).
        # Without this, depth=0 or Z_C≈0 causes NaN via division by zero,
        # which clamp() cannot fix (clamp(NaN) = NaN).
        Z_C = P_C[:, :, 2]  # (B, N, H, W)
        valid_depth = (Z_C > 0.01).float()  # (B, N, H, W)
        
        # Safe division: clamp prevents NaN/Inf for positive but tiny depths
        u_proj = fx_rgb * (P_C[:, :, 0] / Z_C.clamp(min=0.01)) + cx_rgb  # (B, N, H, W)
        v_proj = fy_rgb * (P_C[:, :, 1] / Z_C.clamp(min=0.01)) + cy_rgb  # (B, N, H, W)
        
        # Push invalid coordinates far out-of-bounds so downstream
        # valid_mask in CorrBlock naturally filters them out
        u_proj = u_proj * valid_depth + (-1e4) * (1 - valid_depth)
        v_proj = v_proj * valid_depth + (-1e4) * (1 - valid_depth)
        
        # Stack into (B, N, 2, H, W)
        projected_coords = torch.stack([u_proj, v_proj], dim=2)
        
        return projected_coords


class CorrBlock(nn.Module):
    """
    4D correlation volume for feature matching.
    Computes all-pairs correlation between two feature maps.
    Uses cosine similarity (L2-normalized features) for scale-invariant matching.
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, temperature=1.0):
        """
        Args:
            fmap1: Feature map 1, shape (B, C, H, W)
            fmap2: Feature map 2, shape (B, C, H, W)
            num_levels: Number of pyramid levels for multi-scale correlation
            radius: Radius for local correlation sampling
            temperature: Softmax temperature for correlation sharpening.
                temperature=1.0: raw cosine similarity (default)
                temperature<1.0: sharper peaks (more discriminative)
                temperature>1.0: smoother landscape (less overshoot)
        """
        super(CorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.temperature = temperature
        self.corr_pyramid = []
        
        # Compute all-pairs correlation using normalized features (cosine similarity)
        # This ensures confidence is scale-invariant and decreases monotonically
        # with pose error, unlike unnormalized dot products.
        corr = CorrBlock.corr(fmap1, fmap2)
        
        # Apply temperature: divide logits by temperature before storing
        # This smooths (T>1) or sharpens (T<1) the correlation landscape
        if temperature != 1.0:
            corr = corr / temperature
        
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

        # ── Cache indices and delta grid (Optimization 5) ─────────
        # These are constant across all iterations since the correlation
        # pyramid is built once from fmap1/fmap2 and never changes.
        # Caching saves ~2ms/iter by avoiding redundant index creation.
        H_feat, W_feat = h1, w1
        B_size = batch

        # Shared spatial indices for gather operations
        h_idx = torch.arange(H_feat, device=fmap1.device).view(1, H_feat, 1).expand(B_size, H_feat, W_feat)
        w_idx = torch.arange(W_feat, device=fmap1.device).view(1, 1, W_feat).expand(B_size, H_feat, W_feat)
        h_idx = torch.clamp(h_idx, 0, H_feat - 1)
        w_idx = torch.clamp(w_idx, 0, W_feat - 1)
        corr_idx = (h_idx * W_feat + w_idx).reshape(B_size * H_feat * W_feat)
        b_idx = torch.arange(B_size, device=fmap1.device).unsqueeze(1).expand(B_size, H_feat * W_feat).reshape(B_size * H_feat * W_feat)

        self.register_buffer('_cached_corr_idx', corr_idx)
        self.register_buffer('_cached_b_idx', b_idx)
        self._cached_H_feat = H_feat
        self._cached_W_feat = W_feat
        self._cached_B_size = B_size

        # Local grid delta for correlation sampling
        r = radius
        dx = torch.linspace(-r, r, 2 * r + 1, dtype=fmap1.dtype, device=fmap1.device)
        dy = torch.linspace(-r, r, 2 * r + 1, dtype=fmap1.dtype, device=fmap1.device)
        delta_grid = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)  # (2r+1, 2r+1, 2)
        self.register_buffer('_cached_delta_grid', delta_grid)

        # Pre-compute per-level gathered correlation tensors
        # These are constant since corr_pyramid doesn't change across iterations
        min_size = 2 * r + 1
        self._cached_corr_gathered = {}
        for lvl in range(num_levels):
            corr_lvl = self.corr_pyramid[lvl]
            h2_i, w2_i = corr_lvl.shape[-2], corr_lvl.shape[-1]
            if h2_i < min_size or w2_i < min_size:
                continue
            corr_5d = corr_lvl.view(B_size, H_feat * W_feat, 1, h2_i, w2_i)
            self._cached_corr_gathered[lvl] = corr_5d[b_idx, corr_idx]  # (B*H*W, 1, h2_i, w2_i)
    
    def __call__(self, coords):
        """
        Sample correlation volume at given coordinates.

        Args:
            coords: Sampling coordinates of shape (B, N, 2, H, W), where N is the number
                    of coordinate sets (one per pose sample)

        Returns:
            Correlation features of shape (B, N*C_corr, H, W), where C_corr is the
            number of correlation channels per pose sample
        """
        r = self.radius
        B, N, C_coord, H, W = coords.shape

        # corr_pyramid[i] shape: (B*h1*w1, 1, h2, w2) at level i
        # We need to sample at coords for each of B*N queries, each of spatial size (H, W)
        # The corr volume maps (h1,w1) -> (h2,w2) correlation
        # Here h1==h2==H_feat, w1==w2==W_feat (same feature map size)

        # Get feature map size from corr pyramid
        corr0 = self.corr_pyramid[0]
        B_q, C_q, H_feat, W_feat = corr0.shape  # B_q = B * H_feat * W_feat

        # coords: (B, N, 2, H, W) -> (B*N, 2, H, W) -> (B*N, H, W, 2)
        coords_flat = coords.permute(0, 1, 3, 4, 2).reshape(B * N, H, W, 2)

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]  # (B_q, 1, h2_i, w2_i)
            h2_i, w2_i = corr.shape[-2], corr.shape[-1]
            scale = 2 ** i

            # Create local grid around each coordinate
            dx = torch.linspace(-r, r, 2 * r + 1, dtype=corr.dtype, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, dtype=corr.dtype, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)  # (2r+1, 2r+1, 2)

            # coords_flat: (B*N, H, W, 2) -> scale -> (B*N, H, W, 1, 1, 2)
            centroid = coords_flat / scale
            centroid = centroid.unsqueeze(3).unsqueeze(3)  # (B*N, H, W, 1, 1, 2)

            # delta: (1, 1, 1, 2r+1, 2r+1, 2)
            delta = delta.view(1, 1, 1, 2 * r + 1, 2 * r + 1, 2)

            # sampling coords: (B*N, H, W, 2r+1, 2r+1, 2)
            sampling_coords = centroid + delta

            # Reshape for grid_sample: (B*N*H*W, 2r+1, 2r+1, 2)
            sampling_coords = sampling_coords.reshape(B * N * H * W, 2 * r + 1, 2 * r + 1, 2).contiguous()

            # corr: (B_q, 1, h2_i, w2_i) -> expand to (B*N*H*W, 1, h2_i, w2_i)
            # Each spatial position (h,w) in the output corresponds to query position (h,w)
            # in the original feature map, which is corr index b*H*W + h*W + w
            # We need to index corr for each (b, h, w) in the output
            # corr is indexed as: for batch b, pixel (h,w) -> corr[b*H*W + h*W + w]
            # So we expand corr to (B*N*H*W, 1, h2_i, w2_i) by repeating

            # Build index: for each (b, n, h, w) -> corr index = b * H * W + h * W + w
            # But H, W here are the output spatial dims, which should match H_feat, W_feat
            # after encoder downsampling (H/8, W/8)
            # The coords are already in feature-map space since they come from depth projection
            # at feature resolution

            # Simple approach: repeat corr for each of B*N queries
            # corr: (B_q, 1, h2_i, w2_i) -> (1, B_q, 1, h2_i, w2_i) -> (B*N, B_q, 1, h2_i, w2_i)
            # This is too memory-intensive. Instead, use the fact that each query pixel
            # (h,w) corresponds to corr index b*H_feat*W_feat + h*W_feat + w

            # Reshape corr to (B, H_feat*W_feat, 1, h2_i, w2_i)
            corr_5d = corr.view(B, H_feat * W_feat, 1, h2_i, w2_i)

            # For each (b, n, h, w), we want corr_5d[b, h*W_feat+w, ...]
            # Build gather index: (B*N, H, W) -> (B*N*H*W,)
            h_idx = torch.arange(H, device=coords.device).view(1, H, 1).expand(B * N, H, W)
            w_idx = torch.arange(W, device=coords.device).view(1, 1, W).expand(B * N, H, W)
            # Clamp to feature map size
            h_idx = torch.clamp(h_idx, 0, H_feat - 1)
            w_idx = torch.clamp(w_idx, 0, W_feat - 1)
            corr_idx = (h_idx * W_feat + w_idx).reshape(B * N * H * W)  # (B*N*H*W,)

            # Gather: (B, H_feat*W_feat, 1, h2_i, w2_i) -> select corr_idx for each b
            b_idx = torch.arange(B, device=coords.device).view(B, 1).expand(B, N * H * W).reshape(B * N * H * W)
            corr_sampled = corr_5d[b_idx, corr_idx]  # (B*N*H*W, 1, h2_i, w2_i)

            # Bilinear sampling
            corr_sampled = CorrBlock.bilinear_sampler(corr_sampled, sampling_coords)
            # corr_sampled: (B*N*H*W, 1, 2r+1, 2r+1) -> (B*N, H, W, (2r+1)^2)
            corr_sampled = corr_sampled.squeeze(1)  # remove channel dim
            corr_sampled = corr_sampled.view(B * N, H, W, -1)
            out_pyramid.append(corr_sampled)

        # Concatenate all pyramid levels
        out = torch.cat(out_pyramid, dim=-1)  # (B*N, H, W, C_corr)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B*N, C_corr, H, W)
        out = out.view(B, N, -1, H, W)  # (B, N, C_corr, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous().reshape(B, N * out.shape[2], H, W)

        return out.float()

    def sample_per_pose(self, coords):
        """
        Sample correlation volume per pose sample and compute confidence scores.

        OPTIMIZED: Uses cached indices and corr_gathered from __init__
        (Optimization 5). Avoids re-creating corr_idx, b_idx, delta_grid,
        and per-level gather operations every call.

        The gather index corr_idx = h * W_feat + w depends only on spatial
        position, NOT on the pose sample index n. By gathering once per
        pyramid level and sharing across N grid_sample calls, we reduce
        peak memory from (B*N*H*W, 1, h2, w2) to (B*H*W, 1, h2, w2).

        Args:
            coords: Sampling coordinates of shape (B, N, 2, H, W)

        Returns:
            corr_feats: Per-sample correlation features of shape (B*N, C_corr, H, W)
            confidence: Per-sample confidence scores of shape (B, N)
        """
        r = self.radius
        B, N, C_coord, H, W = coords.shape

        H_feat, W_feat = self._cached_H_feat, self._cached_W_feat
        corr_idx = self._cached_corr_idx
        b_idx = self._cached_b_idx
        delta_grid = self._cached_delta_grid

        # Pre-compute per-sample coordinates and validity masks
        coords_per_sample = []
        valid_masks = []
        for n in range(N):
            coords_n = coords[:, n].permute(0, 2, 3, 1)  # (B, H, W, 2)
            coords_per_sample.append(coords_n)
            valid_mask = (
                (coords_n[..., 0] >= 0) & (coords_n[..., 0] <= W_feat - 1) &
                (coords_n[..., 1] >= 0) & (coords_n[..., 1] <= H_feat - 1)
            )  # (B, H, W)
            valid_masks.append(valid_mask)

        # --- Per-level: use cached corr_gathered ---
        sample_out_pyramids = [[] for _ in range(N)]
        min_size = 2 * r + 1

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            h2_i, w2_i = corr.shape[-2], corr.shape[-1]
            scale = 2 ** i

            if h2_i < min_size or w2_i < min_size:
                for n in range(N):
                    sample_out_pyramids[n].append(
                        torch.zeros(B, H, W, (2 * r + 1) ** 2,
                                    dtype=corr.dtype, device=coords.device)
                    )
                continue

            # Use cached corr_gathered for this level
            if i in self._cached_corr_gathered:
                corr_gathered = self._cached_corr_gathered[i]
            else:
                corr_5d = corr.view(B, H_feat * W_feat, 1, h2_i, w2_i)
                corr_gathered = corr_5d[b_idx, corr_idx]

            delta = delta_grid.view(1, 1, 1, 2 * r + 1, 2 * r + 1, 2)

            for n in range(N):
                centroid = coords_per_sample[n] / scale
                centroid = centroid.unsqueeze(3).unsqueeze(3)
                sampling_coords = centroid + delta
                sampling_coords = sampling_coords.reshape(B * H * W, 2 * r + 1, 2 * r + 1, 2)

                corr_local = CorrBlock.bilinear_sampler(corr_gathered, sampling_coords)
                corr_local = corr_local.squeeze(1).view(B, H, W, -1)
                sample_out_pyramids[n].append(corr_local)

        # --- Assemble per-sample outputs ---
        sample_corr_feats = []
        sample_confidence = []

        for n in range(N):
            out = torch.cat(sample_out_pyramids[n], dim=-1)
            out = out * valid_masks[n].unsqueeze(-1).float()
            corr_feat_n = out.permute(0, 3, 1, 2).contiguous()
            sample_corr_feats.append(corr_feat_n)

            center_val = sample_out_pyramids[n][0][:, :, :, r * (2 * r + 1) + r]
            center_val = center_val * valid_masks[n].float()

            # ── Top-K mean confidence (instead of global mean) ──
            center_val_flat = center_val.view(B, -1)  # (B, H*W)
            valid_flat = valid_masks[n].view(B, -1).float()  # (B, H*W)
            k = max(1, int(0.2 * H * W))
            n_valid = valid_flat.sum(dim=1, keepdim=True).clamp(min=1)
            k = min(k, int(n_valid.min().item()))

            center_val_masked = center_val_flat.clone()
            center_val_masked[valid_flat < 0.5] = float('-inf')
            topk_vals = center_val_masked.topk(k, dim=1).values
            finite_mask = topk_vals.isfinite()
            conf_n = (topk_vals * finite_mask.float()).sum(dim=1) / finite_mask.sum(dim=1).clamp(min=1)
            sample_confidence.append(conf_n)

        corr_feats = torch.stack(sample_corr_feats, dim=1).reshape(B * N, -1, H, W)
        confidence = torch.stack(sample_confidence, dim=1)

        return corr_feats, confidence

    def sample_coarse_then_fine(self, coords, top_k=3):
        """
        Two-phase coarse-to-fine correlation sampling for memory efficiency.

        Phase 1 (Coarse): Evaluate ALL N samples using the coarsest pyramid level(s).
                           Compute confidence to select top-K candidates.
                           Cost: O(B*N) with tiny spatial dims — very cheap.
        Phase 2 (Fine):   Perform full multi-level sampling on only the top-K samples.
                           Cost: O(B*K) at full resolution — K << N saves memory.

        OPTIMIZED: Uses cached corr_gathered and delta_grid from __init__
        (Optimization 5) to avoid recomputing constant tensors each call.

        Args:
            coords: Sampling coordinates of shape (B, N, 2, H, W)
            top_k: Number of top-confidence samples for fine sampling

        Returns:
            corr_feats_topk: Correlation features for top-K samples,
                             shape (B*K, C_corr, H, W)
            confidence: Confidence scores for all N samples, shape (B, N)
            topk_indices: Indices of selected top-K samples, shape (B, K)
        """
        r = self.radius
        B, N, C_coord, H, W = coords.shape

        # Use cached feature map dimensions
        H_feat, W_feat = self._cached_H_feat, self._cached_W_feat

        # ============================================================
        # Phase 1: Coarse confidence estimation for ALL N samples
        # ============================================================
        min_size = 2 * r + 1
        coarsest_level = self.num_levels - 1
        for lvl in range(self.num_levels - 1, -1, -1):
            h_lvl, w_lvl = self.corr_pyramid[lvl].shape[-2:]
            if h_lvl >= min_size and w_lvl >= min_size:
                coarsest_level = lvl
                break
        corr_coarse = self.corr_pyramid[coarsest_level]
        h2_c, w2_c = corr_coarse.shape[-2], corr_coarse.shape[-1]
        scale_c = 2 ** coarsest_level

        # Use cached indices (avoid re-creating every call)
        corr_idx = self._cached_corr_idx  # (B*H*W,)
        b_idx = self._cached_b_idx        # (B*H*W,)
        delta_grid = self._cached_delta_grid
        delta = delta_grid.view(1, 1, 1, 2 * r + 1, 2 * r + 1, 2)

        with torch.no_grad():
            # Use pre-cached corr_gathered for coarsest level if available
            if coarsest_level in self._cached_corr_gathered:
                corr_gathered = self._cached_corr_gathered[coarsest_level]
            elif h2_c >= min_size and w2_c >= min_size:
                corr_5d = corr_coarse.view(B, H_feat * W_feat, 1, h2_c, w2_c)
                corr_gathered = corr_5d[b_idx, corr_idx]
            else:
                corr_gathered = None

            if corr_gathered is not None:
                # ── Per-sample batched grid_sample (avoids 200+MB expand) ──
                # Benchmark showed: expand+grid_sample ≈ 10ms,
                #                   per-sample loop × 37   ≈ 4ms
                # The grid_sample kernel itself is only ~1ms; the bottleneck
                # is allocating corr_expanded (203MB) and sampling_coords (110MB).
                # By looping over N samples and reusing corr_gathered (5.5MB),
                # we save ~6ms and ~300MB peak VRAM per iteration.

                all_center_vals = []
                all_valid_counts = []

                for n in range(N):
                    # coords for this sample: (B, H, W, 2)
                    coords_n = coords[:, n].detach().permute(0, 2, 3, 1)  # (B, H, W, 2)

                    centroid = coords_n / scale_c
                    centroid = centroid.unsqueeze(3).unsqueeze(3)  # (B, H, W, 1, 1, 2)
                    sampling_coords = centroid + delta  # (B, H, W, 2r+1, 2r+1, 2)
                    sampling_coords = sampling_coords.reshape(B * H * W, 2 * r + 1, 2 * r + 1, 2)

                    # Reuse cached corr_gathered directly — no expand needed!
                    corr_local = CorrBlock.bilinear_sampler(corr_gathered, sampling_coords)
                    corr_local = corr_local.squeeze(1)  # (B*H*W, 2r+1, 2r+1)
                    corr_local_flat = corr_local.reshape(B * H * W, -1)  # (B*H*W, (2r+1)^2)

                    # Center correlation value
                    center_val = corr_local_flat[:, r * (2 * r + 1) + r].view(B, H, W)

                    # Validity mask for this sample
                    valid_mask = (
                        (coords_n[..., 0] >= 0) & (coords_n[..., 0] <= W_feat - 1) &
                        (coords_n[..., 1] >= 0) & (coords_n[..., 1] <= H_feat - 1)
                    )
                    center_val = center_val * valid_mask.float()

                    # ── Top-K mean confidence (instead of global mean) ──
                    # Global mean dilutes signal: ~50% flat regions contribute noise.
                    # Top-K mean uses only the highest-correlation spatial positions,
                    # which are typically edges/corners with discriminative features.
                    # This dramatically improves signal-to-noise ratio.
                    center_val_flat = center_val.view(B, -1)  # (B, H*W)
                    valid_flat = valid_mask.view(B, -1).float()  # (B, H*W)
                    n_valid = valid_flat.sum(dim=1)  # (B,)

                    # Guard: if no valid pixels exist, confidence = 0
                    # (avoids -inf * 0.0 = NaN from topk on all-invalid data)
                    has_valid = (n_valid > 0).float()  # (B,)
                    n_valid_safe = n_valid.clamp(min=1)  # (B,) for k computation

                    k = max(1, int(0.2 * H * W))  # top 20% of spatial positions
                    k = min(k, int(n_valid_safe.min().item()))  # don't exceed valid count

                    # Set invalid positions to -inf so they're never selected
                    center_val_masked = center_val_flat.clone()
                    center_val_masked[valid_flat < 0.5] = float('-inf')
                    topk_vals = center_val_masked.topk(k, dim=1).values  # (B, k)
                    # Replace -inf with 0 before summing to avoid NaN
                    topk_vals_safe = topk_vals.clamp(min=0.0)
                    topk_sum = topk_vals_safe.sum(dim=1)  # (B,)
                    topk_count = n_valid_safe  # (B,)
                    conf_n = (topk_sum / topk_count) * has_valid  # (B,) zero when no valid pixels

                    all_center_vals.append(conf_n)

                # Stack: (B, N)
                confidence = torch.stack(all_center_vals, dim=1)  # (B, N)
            else:
                confidence = torch.zeros(B, N, device=coords.device)

            # Select top-K candidates
            K = min(top_k, N)
            _, topk_indices = confidence.topk(K, dim=1, largest=True)  # (B, K)

        # ============================================================
        # Phase 2: Full multi-level sampling on top-K only
        # ============================================================
        # Gather top-K coordinates: (B, K, 2, H, W)
        topk_exp = topk_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, H, W)
        coords_topk = torch.gather(coords, 1, topk_exp)  # (B, K, 2, H, W)

        # Full multi-level sampling for top-K samples using cached corr_gathered
        sample_out_pyramids = [[] for _ in range(K)]

        # Pre-compute per-sample coords and validity masks
        coords_topk_list = []
        valid_masks_topk = []
        for k in range(K):
            coords_k = coords_topk[:, k].permute(0, 2, 3, 1)  # (B, H, W, 2)
            coords_topk_list.append(coords_k)
            valid_mask = (
                (coords_k[..., 0] >= 0) & (coords_k[..., 0] <= W_feat - 1) &
                (coords_k[..., 1] >= 0) & (coords_k[..., 1] <= H_feat - 1)
            )
            valid_masks_topk.append(valid_mask)

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            h2_i, w2_i = corr.shape[-2], corr.shape[-1]
            scale = 2 ** i

            if h2_i < min_size or w2_i < min_size:
                for k in range(K):
                    sample_out_pyramids[k].append(
                        torch.zeros(B, H, W, (2 * r + 1) ** 2,
                                    dtype=corr.dtype, device=coords.device)
                    )
                continue

            # Use cached corr_gathered for this level
            if i in self._cached_corr_gathered:
                corr_gathered_fine = self._cached_corr_gathered[i]
            else:
                corr_5d = corr.view(B, H_feat * W_feat, 1, h2_i, w2_i)
                corr_gathered_fine = corr_5d[b_idx, corr_idx]

            delta_scaled = delta_grid.view(1, 1, 1, 2 * r + 1, 2 * r + 1, 2)

            for k in range(K):
                centroid = coords_topk_list[k] / scale
                centroid = centroid.unsqueeze(3).unsqueeze(3)
                sampling_coords = centroid + delta_scaled
                sampling_coords = sampling_coords.reshape(B * H * W, 2 * r + 1, 2 * r + 1, 2)

                corr_local = CorrBlock.bilinear_sampler(corr_gathered_fine, sampling_coords)
                corr_local = corr_local.squeeze(1).view(B, H, W, -1)
                sample_out_pyramids[k].append(corr_local)

        # Assemble outputs for top-K samples
        sample_corr_feats = []
        for k in range(K):
            out = torch.cat(sample_out_pyramids[k], dim=-1)  # (B, H, W, C_corr)
            out = out * valid_masks_topk[k].unsqueeze(-1).float()  # mask invalid
            corr_feat_k = out.permute(0, 3, 1, 2).contiguous()  # (B, C_corr, H, W)
            sample_corr_feats.append(corr_feat_k)

        # Stack: (B, K, C_corr, H, W) -> (B*K, C_corr, H, W)
        corr_feats_topk = torch.stack(sample_corr_feats, dim=1).reshape(B * K, -1, H, W)

        return corr_feats_topk, confidence, topk_indices

    @staticmethod
    def corr(fmap1, fmap2):
        """
        Compute all-pairs cosine similarity between two feature maps.
        
        Features are L2-normalized per spatial position before computing
        dot products, yielding cosine similarities in [-1, 1].
        This makes the correlation scale-invariant: confidence reflects
        feature alignment quality rather than feature magnitude.
        
        Args:
            fmap1: Feature map 1, shape (B, C, H, W)
            fmap2: Feature map 2, shape (B, C, H, W)
        
        Returns:
            Correlation volume of shape (B, H1, W1, 1, H2, W2)
        """
        batch, dim, ht, wd = fmap1.shape
        
        # L2-normalize features per spatial position for cosine similarity
        fmap1_norm = F.normalize(fmap1, dim=1)  # (B, C, H, W)
        fmap2_norm = F.normalize(fmap2, dim=1)  # (B, C, H, W)
        
        fmap1_flat = fmap1_norm.view(batch, dim, ht * wd)
        fmap2_flat = fmap2_norm.view(batch, dim, ht * wd)
        
        corr = torch.matmul(fmap1_flat.transpose(1, 2), fmap2_flat)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr
    
    @staticmethod
    def bilinear_sampler(img, coords, mode="bilinear", mask_mode=""):
        """
        Bilinear sampler for correlation volume.
        
        Args:
            img: Input tensor of shape (B, C, H, W)
            coords: Sampling coordinates of shape (B, H, W, 2)
        
        Returns:
            Sampled features of shape (B, C, H, W)
        """
        H, W = img.shape[-2:]
        
        # Normalize coordinates to [-1, 1]
        x = coords[..., 0] / (W - 1) * 2 - 1
        y = coords[..., 1] / (H - 1) * 2 - 1
        
        # Clamp to prevent NaN from grid_sample with align_corners=True
        # (out-of-bounds coords produce NaN in some PyTorch versions)
        x = torch.clamp(x, -1.0, 1.0)
        y = torch.clamp(y, -1.0, 1.0)
        
        coords_norm = torch.stack([x, y], dim=-1).contiguous()
        # Use non-cuDNN path to avoid CUDNN_STATUS_NOT_SUPPORTED errors
        # with certain input shape combinations (e.g. large batch, small spatial dims)
        with torch.backends.cudnn.flags(enabled=False):
            sampled = F.grid_sample(img.contiguous(), coords_norm, mode=mode, align_corners=True)
        
        return sampled


class PoseCorrSampler(nn.Module):
    """
    Combines depth projection and correlation sampling for pose estimation.
    """
    def __init__(self, corr_block):
        """
        Args:
            corr_block: CorrBlock instance for 4D correlation volume
        """
        super(PoseCorrSampler, self).__init__()
        self.corr_block = corr_block
        self.depth_projector = DepthProjector()
    
    def forward(self, depth_map, extrinsics, intrinsic_depth, intrinsic_rgb):
        """
        Project depth map using multiple poses and sample correlation volume.
        
        Args:
            depth_map: Depth map of shape (B, H, W)
            extrinsics: Extrinsic matrices of shape (B, N, 4, 4)
            intrinsic_depth: Depth camera intrinsic, shape (B, 3, 3)
            intrinsic_rgb: RGB camera intrinsic, shape (B, 3, 3)
        
        Returns:
            corr_feat: Correlation features of shape (B, N*C_corr, H, W)
            projected_coords: Projected coordinates of shape (B, N, 2, H, W)
        """
        # Project depth map using multiple extrinsics
        projected_coords = self.depth_projector(
            depth_map, extrinsics, intrinsic_depth, intrinsic_rgb
        )
        
        # Sample correlation volume at projected coordinates
        corr_feat = self.corr_block(projected_coords)
        
        return corr_feat, projected_coords
