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
        
        u_proj = fx_rgb * (P_C[:, :, 0] / (P_C[:, :, 2] + 1e-8)) + cx_rgb  # (B, N, H, W)
        v_proj = fy_rgb * (P_C[:, :, 1] / (P_C[:, :, 2] + 1e-8)) + cy_rgb  # (B, N, H, W)
        
        # Stack into (B, N, 2, H, W)
        projected_coords = torch.stack([u_proj, v_proj], dim=2)
        
        return projected_coords


class CorrBlock(nn.Module):
    """
    4D correlation volume for feature matching.
    Computes all-pairs correlation between two feature maps.
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        """
        Args:
            fmap1: Feature map 1, shape (B, C, H, W)
            fmap2: Feature map 2, shape (B, C, H, W)
            num_levels: Number of pyramid levels for multi-scale correlation
            radius: Radius for local correlation sampling
        """
        super(CorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        
        # Compute all-pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
    
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

        OPTIMIZED: The gather index corr_idx = h * W_feat + w depends only on spatial
        position, NOT on the pose sample index n. This means all N pose samples query
        the SAME correlation slices. By gathering once per pyramid level and sharing
        the result across N grid_sample calls, we reduce peak memory from
        (B*N*H*W, 1, h2, w2) to (B*H*W, 1, h2, w2) — an N× reduction.

        For N=8 at 480×640 input: 703 MB → 88 MB per pyramid level.

        Out-of-bounds projected coordinates are masked to zero (ignored) instead of
        being clamped to boundary values, avoiding noisy features from invalid regions.

        Args:
            coords: Sampling coordinates of shape (B, N, 2, H, W)

        Returns:
            corr_feats: Per-sample correlation features of shape (B*N, C_corr, H, W)
            confidence: Per-sample confidence scores of shape (B, N),
                        computed as mean correlation response over valid pixels only
        """
        r = self.radius
        B, N, C_coord, H, W = coords.shape

        corr0 = self.corr_pyramid[0]
        B_q, C_q, H_feat, W_feat = corr0.shape

        # --- Pre-compute shared indices (identical for all N samples) ---
        # For query pixel (h, w), the correlation slice index is h * W_feat + w
        # This does NOT depend on which pose sample we're evaluating
        h_idx = torch.arange(H, device=coords.device).view(1, H, 1).expand(B, H, W)
        w_idx = torch.arange(W, device=coords.device).view(1, 1, W).expand(B, H, W)
        h_idx = torch.clamp(h_idx, 0, H_feat - 1)
        w_idx = torch.clamp(w_idx, 0, W_feat - 1)
        corr_idx = (h_idx * W_feat + w_idx).reshape(B * H * W)  # (B*H*W,) — NOT (B*N*H*W,)
        b_idx = torch.arange(B, device=coords.device).unsqueeze(1).expand(B, H * W).reshape(B * H * W)

        # Pre-compute local grid delta (shared across all levels and samples)
        dx = torch.linspace(-r, r, 2 * r + 1, dtype=corr0.dtype, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, dtype=corr0.dtype, device=coords.device)
        delta_grid = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)  # (2r+1, 2r+1, 2)

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

        # --- Per-level: gather ONCE, grid_sample for each sample ---
        # sample_out_pyramids[n] = list of per-level tensors (B, H, W, (2r+1)^2)
        sample_out_pyramids = [[] for _ in range(N)]

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            h2_i, w2_i = corr.shape[-2], corr.shape[-1]
            scale = 2 ** i

            # Skip pyramid levels that are too small for the sampling window
            min_size = 2 * r + 1
            if h2_i < min_size or w2_i < min_size:
                for n in range(N):
                    sample_out_pyramids[n].append(
                        torch.zeros(B, H, W, (2 * r + 1) ** 2,
                                    dtype=corr.dtype, device=coords.device)
                    )
                continue

            # === GATHER ONCE — the key optimization ===
            # (B*H*W, 1, h2_i, w2_i) instead of (B*N*H*W, 1, h2_i, w2_i)
            # For N=8: 88 MB instead of 703 MB!
            corr_5d = corr.view(B, H_feat * W_feat, 1, h2_i, w2_i)
            corr_gathered = corr_5d[b_idx, corr_idx]  # (B*H*W, 1, h2_i, w2_i)

            # === Grid sample for each pose sample (sharing corr_gathered) ===
            delta = delta_grid.view(1, 1, 1, 2 * r + 1, 2 * r + 1, 2)

            for n in range(N):
                centroid = coords_per_sample[n] / scale  # (B, H, W, 2)
                centroid = centroid.unsqueeze(3).unsqueeze(3)  # (B, H, W, 1, 1, 2)
                sampling_coords = centroid + delta  # (B, H, W, 2r+1, 2r+1, 2)
                sampling_coords = sampling_coords.reshape(B * H * W, 2 * r + 1, 2 * r + 1, 2)

                corr_local = CorrBlock.bilinear_sampler(corr_gathered, sampling_coords)
                corr_local = corr_local.squeeze(1).view(B, H, W, -1)  # (B, H, W, (2r+1)^2)
                sample_out_pyramids[n].append(corr_local)

        # --- Assemble per-sample outputs ---
        sample_corr_feats = []
        sample_confidence = []

        for n in range(N):
            # Concatenate pyramid levels: (B, H, W, C_corr)
            out = torch.cat(sample_out_pyramids[n], dim=-1)

            # Mask invalid (out-of-bounds) pixels
            out = out * valid_masks[n].unsqueeze(-1).float()

            # (B, C_corr, H, W)
            corr_feat_n = out.permute(0, 3, 1, 2).contiguous()
            sample_corr_feats.append(corr_feat_n)

            # Confidence: mean correlation at level-0 center, over valid pixels only
            center_val = sample_out_pyramids[n][0][:, :, :, r * (2 * r + 1) + r]  # (B, H, W)
            center_val = center_val * valid_masks[n].float()
            valid_count = valid_masks[n].sum(dim=[1, 2]).float().clamp(min=1.0)  # (B,)
            conf_n = center_val.sum(dim=[1, 2]) / valid_count  # (B,)
            sample_confidence.append(conf_n)

        # Stack: (B, N, C_corr, H, W) -> (B*N, C_corr, H, W)
        corr_feats = torch.stack(sample_corr_feats, dim=1).reshape(B * N, -1, H, W)
        confidence = torch.stack(sample_confidence, dim=1)  # (B, N)

        return corr_feats, confidence

    def sample_coarse_then_fine(self, coords, top_k=3):
        """
        Two-phase coarse-to-fine correlation sampling for memory efficiency.

        Phase 1 (Coarse): Evaluate ALL N samples using the coarsest pyramid level(s).
                           Compute confidence to select top-K candidates.
                           Cost: O(B*N) with tiny spatial dims — very cheap.
        Phase 2 (Fine):   Perform full multi-level sampling on only the top-K samples.
                           Cost: O(B*K) at full resolution — K << N saves memory.

        Memory analysis (480×640 input, 8× downsample → 60×80 feat, 4 corr levels):
          - Coarsest level: 8×10, gather = B*H*W*1*8*10 ≈ 0.05 MB per sample → negligible
          - Fine sampling: only K=3 instead of N=36 samples at 60×80 → 12× memory reduction

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

        corr0 = self.corr_pyramid[0]
        B_q, C_q, H_feat, W_feat = corr0.shape

        # ============================================================
        # Phase 1: Coarse confidence estimation for ALL N samples
        # ============================================================
        # Use the coarsest level for cheap confidence scoring
        coarsest_level = self.num_levels - 1
        corr_coarse = self.corr_pyramid[coarsest_level]
        h2_c, w2_c = corr_coarse.shape[-2], corr_coarse.shape[-1]
        scale_c = 2 ** coarsest_level

        # Shared spatial indices (same for all samples)
        h_idx = torch.arange(H, device=coords.device).view(1, H, 1).expand(B, H, W)
        w_idx = torch.arange(W, device=coords.device).view(1, 1, W).expand(B, H, W)
        h_idx = torch.clamp(h_idx, 0, H_feat - 1)
        w_idx = torch.clamp(w_idx, 0, W_feat - 1)
        corr_idx = (h_idx * W_feat + w_idx).reshape(B * H * W)  # (B*H*W,)
        b_idx = torch.arange(B, device=coords.device).unsqueeze(1).expand(B, H * W).reshape(B * H * W)

        # Local grid delta
        dx = torch.linspace(-r, r, 2 * r + 1, dtype=corr_coarse.dtype, device=coords.device)
        dy = torch.linspace(-r, r, 2 * r + 1, dtype=corr_coarse.dtype, device=coords.device)
        delta_grid = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)  # (2r+1, 2r+1, 2)
        delta = delta_grid.view(1, 1, 1, 2 * r + 1, 2 * r + 1, 2)

        # Gather ONCE at coarsest level — shared across all N samples
        min_size = 2 * r + 1
        if h2_c >= min_size and w2_c >= min_size:
            corr_5d = corr_coarse.view(B, H_feat * W_feat, 1, h2_c, w2_c)
            corr_gathered = corr_5d[b_idx, corr_idx]  # (B*H*W, 1, h2_c, w2_c) — very small!
        else:
            corr_gathered = None

        # ── Evaluate confidence for ALL N samples in parallel ──
        # Expand corr_gathered for B*N queries: (B*H*W, 1, h2_c, w2_c) -> (B*N*H*W, 1, h2_c, w2_c)
        # Coarsest level is tiny, so this expansion is cheap.
        if corr_gathered is not None:
            corr_expanded = corr_gathered.unsqueeze(1).expand(-1, N, -1, -1, -1)
            corr_expanded = corr_expanded.reshape(B * N * H * W, 1, h2_c, w2_c)

            # All N coords at once: (B, N, 2, H, W) -> (B*N, H, W, 2)
            coords_flat = coords.permute(0, 1, 3, 4, 2).reshape(B * N, H, W, 2)

            centroid = coords_flat / scale_c
            centroid = centroid.unsqueeze(3).unsqueeze(3)  # (B*N, H, W, 1, 1, 2)
            sampling_coords = centroid + delta  # (B*N, H, W, 2r+1, 2r+1, 2)
            sampling_coords = sampling_coords.reshape(B * N * H * W, 2 * r + 1, 2 * r + 1, 2)

            corr_local = CorrBlock.bilinear_sampler(corr_expanded, sampling_coords)
            corr_local = corr_local.squeeze(1).view(B * N, H, W, -1)  # (B*N, H, W, (2r+1)^2)

            # Validity masks: (B*N, H, W)
            valid_masks = (
                (coords_flat[..., 0] >= 0) & (coords_flat[..., 0] <= W_feat - 1) &
                (coords_flat[..., 1] >= 0) & (coords_flat[..., 1] <= H_feat - 1)
            )

            # Center correlation value per sample
            center_vals = corr_local[:, :, :, r * (2 * r + 1) + r]  # (B*N, H, W)
            center_vals = center_vals * valid_masks.float()
            valid_counts = valid_masks.sum(dim=[1, 2]).float().clamp(min=1.0)  # (B*N,)
            confidence = (center_vals.sum(dim=[1, 2]) / valid_counts).view(B, N)  # (B, N)
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

        # Full multi-level sampling for top-K samples
        # Hybrid: coarse evaluation was parallel (cheap), fine sampling uses
        # the shared corr_gathered trick to save memory (K=3 loop is negligible).
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

            # Gather once per level — shared across K samples (memory-efficient)
            corr_5d = corr.view(B, H_feat * W_feat, 1, h2_i, w2_i)
            corr_gathered_fine = corr_5d[b_idx, corr_idx]  # (B*H*W, 1, h2_i, w2_i)

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
        Compute all-pairs correlation between two feature maps.
        
        Args:
            fmap1: Feature map 1, shape (B, C, H, W)
            fmap2: Feature map 2, shape (B, C, H, W)
        
        Returns:
            Correlation volume of shape (B, H1, W1, 1, H2, W2)
        """
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
    
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
