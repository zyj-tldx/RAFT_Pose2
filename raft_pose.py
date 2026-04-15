"""
RAFT-Pose: Pose estimation using optical flow architecture with multi-pose sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from modules subdirectory — support both package and direct execution
try:
    from .modules.pose_utils import (
        quaternion_to_matrix, matrix_to_quaternion,
        compose_pose, apply_pose_update, generate_pose_samples,
        compute_pose_error, sampled_poses_to_matrices
    )
    from .modules.pose_extractor import BasicEncoder, DepthEncoder, SmallEncoder
    from .modules.depth_projection import DepthProjector, CorrBlock, PoseCorrSampler
    from .modules.pose_update import PoseUpdateNet
except ImportError:
    from modules.pose_utils import (
        quaternion_to_matrix, matrix_to_quaternion,
        compose_pose, apply_pose_update, generate_pose_samples,
        compute_pose_error, sampled_poses_to_matrices
    )
    from modules.pose_extractor import BasicEncoder, DepthEncoder, SmallEncoder
    from modules.depth_projection import DepthProjector, CorrBlock, PoseCorrSampler
    from modules.pose_update import PoseUpdateNet



class RAFTPose(nn.Module):
    """
    RAFT-Pose model for camera-LiDAR extrinsic calibration.
    
    Architecture:
    1. Extract features from RGB image and depth map using CNN encoders
    2. Build 4D correlation volume between features
    3. Iterate for K iterations:
       - Generate N pose samples around current pose estimate
       - Project depth map using N poses and sample correlation volume
       - Use ConvGRU to update hidden state
       - Predict pose delta (7D vector) from correlation features
       - Update pose estimate using se(3) Lie algebra
    4. Return final pose estimate and intermediate poses
    """
    def __init__(
        self,
        image_encoder='basic',
        hidden_dim=128,
        context_dim=64,
        depth_dim=32,
        corr_levels=4,
        corr_radius=4,
        num_iterations=12,
        num_pose_samples=16,
        pose_sample_std=0.01,
        init_pose_noise_std=0.05
    ):
        """
        Args:
            image_encoder: Type of image feature encoder ('basic' or 'small')
            hidden_dim: Dimension of hidden state in ConvGRU
            context_dim: Dimension of context features from image encoder
            depth_dim: Dimension of depth features from depth encoder
            corr_levels: Number of pyramid levels for correlation volume
            corr_radius: Radius for local correlation sampling
            num_iterations: Number of iterations for pose refinement
            num_pose_samples: Number of pose samples for correlation lookup
            pose_sample_std: Standard deviation for pose perturbation sampling
            init_pose_noise_std: Standard deviation for initial pose noise
        """
        super(RAFTPose, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.depth_dim = depth_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.num_iterations = num_iterations
        self.num_pose_samples = num_pose_samples
        self.pose_sample_std = pose_sample_std
        self.init_pose_noise_std = init_pose_noise_std
        self.downsample_factor = 8  # Total stride of image encoder (3 layers × stride 2)
        
        # Image feature encoder
        if image_encoder == 'basic':
            self.image_encoder = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0.1)
            self.fmap_dim = 256
        elif image_encoder == 'small':
            self.image_encoder = SmallEncoder(output_dim=128, norm_fn='instance', dropout=0.1)
            self.fmap_dim = 128
        else:
            raise ValueError(f"Unknown image encoder: {image_encoder}")
        
        # Depth feature encoder
        self.depth_encoder = DepthEncoder(
            output_dim=depth_dim,
            fourier_levels=-1
        )
        
        # Feature alignment layer: align depth features to match RGB feature dimension
        # This ensures balanced correlation computation
        self.depth_feat_align = nn.Sequential(
            nn.Conv2d(depth_dim, self.fmap_dim, 1),
            nn.InstanceNorm2d(self.fmap_dim),
            nn.ReLU(inplace=True)
        )
        
        # Context feature projector (for depth features)
        self.context_proj = nn.Sequential(
            nn.Conv2d(depth_dim, context_dim, 1),
            nn.ReLU(inplace=True)
        )
        
        # Pose update network
        # corr_dim is now per-sample C_corr (not N*C_corr), so the network
        # is decoupled from num_pose_samples — training and inference can use
        # different numbers of samples.
        per_sample_corr_dim = (2 * corr_radius + 1) ** 2 * corr_levels
        self.pose_update_net = PoseUpdateNet(
            hidden_dim=hidden_dim,
            corr_dim=per_sample_corr_dim,
            context_dim=context_dim,
            num_layers=3
        )
        
        # Correlation block placeholder (initialized in forward)
        self.corr_block = None
        
        # Depth projector for batch pose projection
        self.depth_projector = DepthProjector()
    
    def initialize_correlation(self, fmap_rgb, fmap_depth):
        """
        Initialize 4D correlation volume between RGB and depth features.
        
        Args:
            fmap_rgb: RGB features of shape (B, C, H, W)
            fmap_depth: Depth features of shape (B, C, H, W)
        """
        self.corr_block = CorrBlock(fmap_rgb, fmap_depth, 
                                     num_levels=self.corr_levels, 
                                     radius=self.corr_radius)
    
    def initialize_pose(self, batch_size, device):
        """
        Initialize pose estimate with small random noise.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            init_pose: Initial pose estimate of shape (B, 7)
        """
        # Identity pose: quaternion [1, 0, 0, 0] + translation [0, 0, 0]
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        identity_trans = torch.tensor([0.0, 0.0, 0.0], device=device)
        identity_pose = torch.cat([identity_quat, identity_trans]).unsqueeze(0)
        
        init_pose = identity_pose.expand(batch_size, -1).clone()
        
        # Add small noise for robustness
        if self.init_pose_noise_std > 0:
            quat_noise = torch.randn(batch_size, 4, device=device) * self.init_pose_noise_std
            quat_noise[:, 0] = 1.0  # Keep quaternion close to identity
            quat_noise = F.normalize(quat_noise, dim=1)
            
            trans_noise = torch.randn(batch_size, 3, device=device) * self.init_pose_noise_std
            
            init_pose = torch.cat([quat_noise, trans_noise], dim=1)
        
        return init_pose
    
    def forward(
        self, 
        image, 
        depth, 
        intrinsic_rgb, 
        intrinsic_depth, 
        init_pose=None,
        return_all_poses=False
    ):
        """
        Forward pass of RAFT-Pose model.
        
        Args:
            image: RGB image of shape (B, 3, H, W)
            depth: Depth map of shape (B, 1, H, W)
            intrinsic_rgb: RGB camera intrinsic matrix, shape (B, 3, 3)
            intrinsic_depth: Depth camera (LiDAR) intrinsic matrix, shape (B, 3, 3)
            init_pose: Initial pose estimate of shape (B, 7), optional
            return_all_poses: Whether to return all intermediate poses
        
        Returns:
            final_pose: Final pose estimate of shape (B, 7)
            pose_sequence: All intermediate poses (if return_all_poses=True)
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Extract features
        fmap_rgb = self.image_encoder(image)  # (B, 256, H, W)
        fmap_depth = self.depth_encoder(depth)  # (B, 32, H, W)
        
        # Align depth features to match RGB feature dimension for better correlation
        fmap_depth_aligned = self.depth_feat_align(fmap_depth)  # (B, 256, H, W)
        
        context_feat = self.context_proj(fmap_depth)  # (B, context_dim, H, W)
        
        # Initialize correlation volume with aligned features
        # Note: fmap_rgb and fmap_depth_aligned should have compatible shapes
        # Ensure they are the same spatial size
        if fmap_rgb.shape[2:] != fmap_depth_aligned.shape[2:]:
            fmap_depth_aligned = F.interpolate(fmap_depth_aligned, size=fmap_rgb.shape[2:], 
                                               mode='bilinear', align_corners=False)
        self.initialize_correlation(fmap_rgb, fmap_depth_aligned)
        
        # Initialize pose estimate
        if init_pose is None:
            current_pose = self.initialize_pose(batch_size, device)
        else:
            current_pose = init_pose
        
        # Initialize hidden state for ConvGRU
        hidden_state = None
        
        # Track all poses if requested
        if return_all_poses:
            pose_sequence = [current_pose.clone()]
        
        # Iterative pose refinement
        for it in range(self.num_iterations):
            # Generate pose samples around current estimate
            q_cur, t_cur = current_pose[:, :4], current_pose[:, 4:7]
            sampled_poses, _ = generate_pose_samples(
                (q_cur, t_cur),
                num_samples=self.num_pose_samples,
                std_rot=self.pose_sample_std,
                std_trans=self.pose_sample_std
            )
            
            # Convert sampled poses to matrix format
            pose_samples = sampled_poses_to_matrices(sampled_poses)  # (B, N, 4, 4)
            
            # Sample correlation volume — returns per-sample features + confidence
            corr_feats, confidence = self.sample_correlation_with_poses(
                pose_samples,
                depth,
                intrinsic_depth,
                intrinsic_rgb
            )  # corr_feats: (B*N, C_corr, H_feat, W_feat), confidence: (B, N)

            # Ensure spatial size matches context_feat
            feat_h, feat_w = context_feat.shape[2], context_feat.shape[3]
            if corr_feats.shape[2:] != (feat_h, feat_w):
                corr_feats = F.interpolate(
                    corr_feats, size=(feat_h, feat_w),
                    mode='bilinear', align_corners=False
                )

            # Select best sample by confidence score
            best_idx = confidence.argmax(dim=1)  # (B,)
            B_size = corr_feats.shape[0] // confidence.shape[1]  # B
            N_size = confidence.shape[1]

            # Gather the best sample's correlation features: (B, C_corr, H, W)
            best_corr = corr_feats.view(B_size, N_size, -1, feat_h, feat_w)
            best_corr = best_corr[torch.arange(B_size, device=device), best_idx]  # (B, C_corr, H, W)

            # Predict pose delta using only the best sample (ConvGRU runs once)
            pose_delta, hidden_state = self.pose_update_net(
                best_corr,
                context_feat,
                hidden_state
            )  # pose_delta: (B, 7, H, W)
            
            # Aggregate pose delta predictions (spatial pooling)
            pose_delta_avg = pose_delta.mean(dim=[2, 3])  # (B, 7)
            
            # Update pose estimate using se(3) Lie algebra
            q_cur, t_cur = current_pose[:, :4], current_pose[:, 4:7]
            dq, dt = pose_delta_avg[:, :4], pose_delta_avg[:, 4:7]
            q_new, t_new = apply_pose_update((q_cur, t_cur), (dq, dt))
            q_new = F.normalize(q_new, dim=1)
            current_pose = torch.cat([q_new, t_new], dim=1)  # (B, 7)
            
            # Track pose
            if return_all_poses:
                pose_sequence.append(current_pose.clone())
        
        if return_all_poses:
            pose_sequence = torch.stack(pose_sequence, dim=1)  # (B, num_iters+1, 7)
            return current_pose, pose_sequence
        else:
            return current_pose
    
    def sample_correlation_with_poses(
        self,
        pose_samples,
        depth,
        intrinsic_depth,
        intrinsic_rgb
    ):
        """
        Sample correlation volume using multiple pose transformations.

        Projects depth at FEATURE MAP resolution (H/8 × W/8) instead of
        original image resolution to avoid OOM.

        Returns per-sample correlation features and confidence scores.

        Args:
            pose_samples: Pose matrices of shape (B, N, 4, 4)
            depth: Depth map of shape (B, 1, H, W)
            intrinsic_depth: Depth camera intrinsic, shape (B, 3, 3)
            intrinsic_rgb: RGB camera intrinsic, shape (B, 3, 3)

        Returns:
            corr_feats: Per-sample correlation features (B*N, C_corr, H_feat, W_feat)
            confidence: Per-sample confidence scores (B, N)
        """
        B, C, H, W = depth.shape
        device = depth.device

        if self.corr_block is None:
            corr_dim = (2 * self.corr_radius + 1) ** 2 * self.corr_levels
            N = pose_samples.shape[1]
            feat_h, feat_w = H // self.downsample_factor, W // self.downsample_factor
            dummy_feats = torch.randn(B * N, corr_dim, feat_h, feat_w, device=device)
            dummy_conf = torch.zeros(B, N, device=device)
            return dummy_feats, dummy_conf

        # Downsample depth to feature map resolution
        downsample = self.downsample_factor
        feat_h, feat_w = H // downsample, W // downsample

        depth_small = F.interpolate(
            depth, size=(feat_h, feat_w), mode='nearest'
        ).squeeze(1)  # (B, feat_h, feat_w)

        # Scale intrinsics to match feature map resolution
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

        # Batch depth projection at feature map resolution
        projected_coords = self.depth_projector(
            depth_small, pose_samples, intrinsic_depth_s, intrinsic_rgb_s
        )  # (B, N, 2, feat_h, feat_w)

        # Sample correlation volume per pose — returns (B*N, C_corr, H, W) + (B, N)
        corr_feats, confidence = self.corr_block.sample_per_pose(projected_coords)

        return corr_feats, confidence
    
    def compute_loss(self, pred_pose, gt_pose, reduction='mean'):
        """
        Compute pose estimation loss.
        
        Args:
            pred_pose: Predicted pose of shape (B, 7)
            gt_pose: Ground truth pose of shape (B, 7)
            reduction: Reduction method ('mean', 'sum', or 'none')
        
        Returns:
            loss: Pose error (rotation error + position error)
        """
        # Compute rotation error (geodesic distance via quaternion)
        pred_q, pred_t = pred_pose[:, :4], pred_pose[:, 4:7]
        gt_q, gt_t = gt_pose[:, :4], gt_pose[:, 4:7]
        error_dict = compute_pose_error((pred_q, pred_t), (gt_q, gt_t))
        rot_error = error_dict['rotation_error']  # (B,)
        
        # Compute translation error
        trans_error = error_dict['translation_error']  # (B,)
        
        # Combined loss
        loss = rot_error + trans_error  # (B,)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


def build_raft_pose(config):
    """
    Factory function to build RAFT-Pose model from configuration dict.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        model: RAFT-Pose model instance
    """
    return RAFTPose(
        image_encoder=config.get('image_encoder', 'basic'),
        hidden_dim=config.get('hidden_dim', 128),
        context_dim=config.get('context_dim', 64),
        depth_dim=config.get('depth_dim', 32),
        corr_levels=config.get('corr_levels', 4),
        corr_radius=config.get('corr_radius', 4),
        num_iterations=config.get('num_iterations', 12),
        num_pose_samples=config.get('num_pose_samples', 16),
        pose_sample_std=config.get('pose_sample_std', 0.01),
        init_pose_noise_std=config.get('init_pose_noise_std', 0.05)
    )
