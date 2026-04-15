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
            sampling_coords = sampling_coords.reshape(B * N * H * W, 2 * r + 1, 2 * r + 1, 2)

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
        out = out.permute(0, 1, 3, 4, 2).reshape(B, N * out.shape[2], H, W)

        return out.float()

    def sample_per_pose(self, coords):
        """
        Sample correlation volume per pose sample and compute confidence scores.

        Unlike __call__ which concatenates all samples into one tensor, this returns
        individual features and confidence scores for each sample.

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

        coords_flat = coords.permute(0, 1, 3, 4, 2).reshape(B * N, H, W, 2)

        # Validity mask: True where projected coords are within feature map bounds
        # coords_flat: (B*N, H, W, 2) -> u is [...,0], v is [...,1]
        valid_mask = (
            (coords_flat[..., 0] >= 0) & (coords_flat[..., 0] <= W_feat - 1) &
            (coords_flat[..., 1] >= 0) & (coords_flat[..., 1] <= H_feat - 1)
        )  # (B*N, H, W)

        out_pyramid = []
        # Also collect center correlation (confidence) from level 0
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            h2_i, w2_i = corr.shape[-2], corr.shape[-1]
            scale = 2 ** i

            # Skip pyramid levels that are too small for the sampling window
            # grid_sample on very small feature maps (e.g. 1x2) with align_corners
            # produces NaN in gradients. Minimum size: 2*r+1 in each dimension.
            min_size = 2 * r + 1
            if h2_i < min_size or w2_i < min_size:
                # Fill with zeros for this level
                out_pyramid.append(torch.zeros(B * N, H, W, (2 * r + 1) ** 2,
                                                dtype=corr.dtype, device=coords.device))
                continue

            dx = torch.linspace(-r, r, 2 * r + 1, dtype=corr.dtype, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, dtype=corr.dtype, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid = coords_flat / scale
            centroid = centroid.unsqueeze(3).unsqueeze(3)
            delta = delta.view(1, 1, 1, 2 * r + 1, 2 * r + 1, 2)
            sampling_coords = centroid + delta
            sampling_coords = sampling_coords.reshape(B * N * H * W, 2 * r + 1, 2 * r + 1, 2)

            corr_5d = corr.view(B, H_feat * W_feat, 1, h2_i, w2_i)

            h_idx = torch.arange(H, device=coords.device).view(1, H, 1).expand(B * N, H, W)
            w_idx = torch.arange(W, device=coords.device).view(1, 1, W).expand(B * N, H, W)
            h_idx = torch.clamp(h_idx, 0, H_feat - 1)
            w_idx = torch.clamp(w_idx, 0, W_feat - 1)
            corr_idx = (h_idx * W_feat + w_idx).reshape(B * N * H * W)

            b_idx = torch.arange(B, device=coords.device).view(B, 1).expand(B, N * H * W).reshape(B * N * H * W)
            corr_sampled = corr_5d[b_idx, corr_idx]

            corr_sampled = CorrBlock.bilinear_sampler(corr_sampled, sampling_coords)
            corr_sampled = corr_sampled.squeeze(1)
            corr_sampled = corr_sampled.view(B * N, H, W, -1)
            out_pyramid.append(corr_sampled)

        # Concatenate all pyramid levels
        out = torch.cat(out_pyramid, dim=-1)  # (B*N, H, W, C_corr)
        corr_feats = out.permute(0, 3, 1, 2).contiguous()  # (B*N, C_corr, H, W)

        # Zero out invalid (out-of-bounds) pixels
        # valid_mask: (B*N, H, W) -> (B*N, 1, H, W) broadcast over C_corr
        corr_feats = corr_feats * valid_mask.unsqueeze(1).float()

        # Compute confidence: mean correlation response at level 0 (finest)
        # Only over valid pixels to avoid bias from out-of-bounds regions
        center_val = out_pyramid[0][:, :, :, r * (2 * r + 1) + r]  # (B*N, H, W)
        center_val = center_val * valid_mask.float()  # mask invalid pixels

        valid_count = valid_mask.sum(dim=[1, 2]).float().clamp(min=1.0)  # (B*N,)
        confidence = center_val.sum(dim=[1, 2]) / valid_count  # (B*N,)
        confidence = confidence.view(B, N)  # (B, N)

        return corr_feats, confidence

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
        
        coords_norm = torch.stack([x, y], dim=-1)
        sampled = F.grid_sample(img, coords_norm, mode=mode, align_corners=True)
        
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
