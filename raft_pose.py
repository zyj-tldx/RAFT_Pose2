"""
RAFT-Pose: Pose estimation using optical flow architecture with multi-pose sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

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
        pose_sample_std=0.01,
        init_pose_noise_std=0.0,
        top_k=3,
        use_checkpoint=False,
        use_amp=False,
        coarse_to_fine=False
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
            pose_sample_std: Standard deviation for pose perturbation sampling
            init_pose_noise_std: Standard deviation for initial pose noise
            top_k: Number of top-confidence samples to aggregate (default 3).
                   Uses confidence-weighted average of top-K correlation features.
                   Set to 1 for best-only selection (original behavior).
            use_checkpoint: Use gradient checkpointing to save memory
            use_amp: Use automatic mixed precision
            coarse_to_fine: If True, use coarse-to-fine sampling strategy:
                   Phase 1: evaluate ALL N samples on coarsest pyramid level (cheap)
                   Phase 2: full multi-level sampling on only top-K samples (saves memory)
                   Reduces peak memory from O(B*N) to O(B*K) for fine sampling.
        """
        super(RAFTPose, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.depth_dim = depth_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.num_iterations = num_iterations
        self.num_pose_samples = 36  # Fixed: 12 directions × 3 magnitude scales
        self.pose_sample_std = pose_sample_std
        self.init_pose_noise_std = init_pose_noise_std
        self.top_k = min(top_k, self.num_pose_samples)  # top_k cannot exceed num_pose_samples
        self.downsample_factor = 8  # Total stride of image encoder (3 layers × stride 2)
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp
        self.coarse_to_fine = coarse_to_fine
        
        # Image feature encoder
        if image_encoder == 'basic':
            self.image_encoder = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0.1,
                                              use_checkpoint=use_checkpoint)
            self.fmap_dim = 256
        elif image_encoder == 'small':
            self.image_encoder = SmallEncoder(output_dim=128, norm_fn='instance', dropout=0.1,
                                              use_checkpoint=use_checkpoint)
            self.fmap_dim = 128
        else:
            raise ValueError(f"Unknown image encoder: {image_encoder}")
        
        # Depth feature encoder
        self.depth_encoder = DepthEncoder(
            output_dim=depth_dim,
            fourier_levels=-1,
            use_checkpoint=use_checkpoint
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
        # is decoupled from the number of pose samples — training and inference
        # can handle variable numbers of samples.
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

    def generate_directional_samples(self, current_pose, depth, intrinsic_depth,
                                     intrinsic_rgb, base_step_rot=0.02,
                                     base_step_trans=0.02):
        """
        Generate pose samples along fixed directional perturbations.

        Sampling strategy: 12 directions × 3 magnitude scales = 36 samples + 1 identity.

        The 12 base directions cover full 6-DOF:
          ±Rx, ±Ry, ±Rz, ±Tx, ±Ty, ±Tz
        Each direction is sampled at 3 magnitude scales:
          scale 1: base_step × 0.5  (fine)
          scale 2: base_step × 1.0  (medium)
          scale 3: base_step × 2.0  (coarse)
        This provides both local refinement and large-range exploration.

        An identity (zero perturbation) sample is appended so the model
        can learn to "stay put" when the pose is already good.

        Args:
            current_pose: Current pose estimate of shape (B, 7)
            depth: Depth map of shape (B, 1, H, W)
            intrinsic_depth: Depth camera intrinsic, shape (B, 3, 3)
            intrinsic_rgb: RGB camera intrinsic, shape (B, 3, 3)
            base_step_rot: Base rotation step in radians
            base_step_trans: Base translation step in meters

        Returns:
            pose_samples: Pose matrices of shape (B, N, 4, 4)
            confidence: Confidence scores of shape (B, N)
            corr_feats: Correlation features of shape (B*N, C_corr, H_feat, W_feat)
            direction_vecs: Direction encoding of shape (B, N, 6)
                           Each row is [rot_axis_x, rot_axis_y, rot_axis_z,
                                        trans_dir_x, trans_dir_y, trans_dir_z]
        """
        B = current_pose.shape[0]
        device = current_pose.device

        q_cur, t_cur = current_pose[:, :4], current_pose[:, 4:7]

        # ── 36-sample mode: 12 directions × 3 magnitudes ──────────
        # 12 base directions: full 6-DOF ±
        base_directions = [
            ([0, 0, 0], [1, 0, 0]),   # +Tx
            ([0, 0, 0], [-1, 0, 0]),  # -Tx
            ([0, 0, 0], [0, 1, 0]),   # +Ty
            ([0, 0, 0], [0, -1, 0]),  # -Ty
            ([0, 0, 0], [0, 0, 1]),   # +Tz
            ([0, 0, 0], [0, 0, -1]),  # -Tz
            ([1, 0, 0], [0, 0, 0]),   # +Rx
            ([-1, 0, 0], [0, 0, 0]),  # -Rx
            ([0, 1, 0], [0, 0, 0]),   # +Ry
            ([0, -1, 0], [0, 0, 0]),  # -Ry
            ([0, 0, 1], [0, 0, 0]),   # +Rz
            ([0, 0, -1], [0, 0, 0]),  # -Rz
        ]
        # 3 magnitude scales: fine / medium / coarse
        scales = [0.25, 1.0, 4.0]

        # Build per-sample (rot_axis, trans_dir, scale) tuples
        # Order: for each direction, iterate all 3 scales
        # [dir0_s0, dir0_s1, dir0_s2, dir1_s0, dir1_s1, dir1_s2, ...]
        rot_axes_list = []
        trans_dirs_list = []
        scale_list = []
        for rot_dir, trans_dir in base_directions:
            for s in scales:
                rot_axes_list.append(rot_dir)
                trans_dirs_list.append(trans_dir)
                scale_list.append(s)

        actual_N = len(rot_axes_list)  # 36

        # Build tensors: (actual_N, 3)
        rot_axes = torch.tensor(rot_axes_list, device=device, dtype=torch.float32)
        trans_dirs = torch.tensor(trans_dirs_list, device=device, dtype=torch.float32)
        sample_scales = torch.tensor(scale_list, device=device, dtype=torch.float32)  # (36,)

        # Normalize rotation axes (identity for zero-axes translation directions)
        rot_norms = rot_axes.norm(dim=1, keepdim=True).clamp(min=1e-8)
        rot_axes_normed = rot_axes / rot_norms

        # Per-sample rotation angles: base_step_rot * scale
        rot_angles = base_step_rot * sample_scales  # (36,)
        half_angles = rot_angles / 2.0  # (36,)
        cos_ha = torch.cos(half_angles)
        sin_ha = torch.sin(half_angles)

        # Build per-sample quaternions: (36, 4)
        all_dq = torch.zeros(actual_N, 4, device=device)
        all_dq[:, 0] = cos_ha
        all_dq[:, 1] = rot_axes_normed[:, 0] * sin_ha
        all_dq[:, 2] = rot_axes_normed[:, 1] * sin_ha
        all_dq[:, 3] = rot_axes_normed[:, 2] * sin_ha

        # Build per-sample translations: (36, 3)
        # scale applied to translation magnitude
        all_dt = trans_dirs * (base_step_trans * sample_scales.unsqueeze(1))  # (36, 3)

        # Direction encoding: use the original (unscaled) direction + scale info
        # Embed scale into direction vector by multiplying
        dir_vecs_raw = torch.cat([rot_axes, trans_dirs], dim=1)  # (36, 6)

        # ── Append identity (zero perturbation) sample ────────────────
        # This ensures the model can "see" the correlation at the current pose.
        # Without it, all samples are perturbations AWAY from the current pose,
        # so the network cannot learn to "stay put" when the pose is already good.
        # The identity sample acts as an anchor for stable convergence.
        identity_dq = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # quat identity
        identity_dt = torch.zeros(1, 3, device=device)  # zero translation
        identity_dir = torch.zeros(1, 6, device=device)  # zero direction encoding

        all_dq = torch.cat([all_dq, identity_dq], dim=0)  # (actual_N+1, 4)
        all_dt = torch.cat([all_dt, identity_dt], dim=0)  # (actual_N+1, 3)
        dir_vecs_raw = torch.cat([dir_vecs_raw, identity_dir], dim=0)  # (actual_N+1, 6)
        actual_N = actual_N + 1  # now includes identity

        # ── Common: apply perturbations and sample correlation ─────────
        # Expand for batch
        all_dq = all_dq.unsqueeze(0).expand(B, -1, -1)  # (B, actual_N, 4)
        all_dt = all_dt.unsqueeze(0).expand(B, -1, -1)  # (B, actual_N, 3)

        # Apply all perturbations at once using vectorized operations
        q_cur_exp = q_cur.unsqueeze(1)  # (B, 1, 4)
        t_cur_exp = t_cur.unsqueeze(1)  # (B, 1, 3)

        # Quaternion multiply: q_new = q_cur * dq (in camera frame)
        sampled_quats = self._batch_quaternion_multiply(q_cur_exp, all_dq)  # (B, actual_N, 4)
        sampled_quats = F.normalize(sampled_quats, dim=-1)

        # Translation update: t_new = t_cur + q_cur @ dt
        rotated_dt = self._batch_quaternion_apply(q_cur_exp, all_dt)  # (B, actual_N, 3)
        sampled_trans = t_cur_exp + rotated_dt  # (B, actual_N, 3)

        # Convert to matrices: (B, actual_N, 4, 4)
        pose_list = [(sampled_quats[:, i], sampled_trans[:, i]) for i in range(actual_N)]
        pose_samples = sampled_poses_to_matrices(pose_list)

        # Evaluate correlation confidence for all samples in parallel
        sample_result = self.sample_correlation_with_poses(
            pose_samples, depth, intrinsic_depth, intrinsic_rgb
        )
        if self.coarse_to_fine:
            corr_feats, confidence, topk_indices = sample_result
        else:
            corr_feats, confidence, topk_indices = sample_result[0], sample_result[1], None

        # Build direction encoding vectors: (B, actual_N, 6)
        dir_vecs = dir_vecs_raw.unsqueeze(0).expand(B, -1, -1)  # (B, actual_N, 6)

        return pose_samples, confidence, corr_feats, dir_vecs, topk_indices

    @staticmethod
    def _batch_quaternion_multiply(q1, q2):
        """
        Batch quaternion multiplication: q1 * q2.

        Args:
            q1: (B, N, 4) or (B, 1, 4)
            q2: (B, N, 4) or (B, 1, 4)

        Returns:
            q_out: (B, N, 4)
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def _batch_quaternion_apply(q, v):
        """
        Batch quaternion rotation: rotate vector v by quaternion q.

        Args:
            q: (B, N, 4) unit quaternions
            v: (B, N, 3) vectors

        Returns:
            v_rot: (B, N, 3) rotated vectors
        """
        # q * v * q^(-1) where q^(-1) = [w, -x, -y, -z] for unit quaternion
        q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)  # (B, N, 4)
        q_inv = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)  # (B, N, 4)

        # q * (v as quaternion)
        qv = RAFTPose._batch_quaternion_multiply(q, q_v)
        # (qv) * q^(-1)
        result = RAFTPose._batch_quaternion_multiply(qv, q_inv)

        return result[..., 1:]  # return vector part only
    
    def _single_iteration(self, current_pose, hidden_state, depth, intrinsic_depth,
                           intrinsic_rgb, context_feat):
        """
        Single iteration of pose refinement — designed to be gradient-checkpointable.

        Contains all heavy computation: depth projection, correlation sampling,
        feature aggregation, ConvGRU update, and pose delta prediction.

        When wrapped with torch.utils.checkpoint, intermediate activations are
        discarded after forward and recomputed during backward, reducing autograd
        memory from O(num_iterations) to O(1).

        Args:
            current_pose: Current pose estimate (B, 7)
            hidden_state: ConvGRU hidden state (B, hidden_dim, H, W), or None
            depth: Depth map (B, 1, H_orig, W_orig)
            intrinsic_depth: Depth camera intrinsic (B, 3, 3)
            intrinsic_rgb: RGB camera intrinsic (B, 3, 3)
            context_feat: Context features from encoder (B, context_dim, H, W)

        Returns:
            current_pose: Updated pose estimate (B, 7)
            hidden_state: Updated ConvGRU hidden state (B, hidden_dim, H, W)
            rot_vec: Raw rotation vector prediction (B, 3)
            dt: Translation delta prediction (B, 3)
        """
        batch_size = current_pose.shape[0]
        device = current_pose.device

        # Generate directional pose samples
        sample_result = self.generate_directional_samples(
            current_pose, depth, intrinsic_depth, intrinsic_rgb,
            base_step_rot=self.pose_sample_std,
            base_step_trans=self.pose_sample_std
        )
        pose_samples, confidence, corr_feats, dir_vecs, topk_indices = sample_result

        # Select best sample by confidence score
        best_idx = confidence.argmax(dim=1)  # (B,)
        N_size = pose_samples.shape[1]

        # Gather correlation features and aggregate top-K by confidence
        feat_h, feat_w = context_feat.shape[2], context_feat.shape[3]
        if corr_feats.shape[2:] != (feat_h, feat_w):
            corr_feats = F.interpolate(
                corr_feats, size=(feat_h, feat_w),
                mode='bilinear', align_corners=False
            )

        if self.coarse_to_fine and topk_indices is not None:
            # ── Coarse-to-fine mode ──────────────────────────────
            K = topk_indices.shape[1]
            all_corr = corr_feats.view(batch_size, K, -1, feat_h, feat_w)  # (B, K, C_corr, H, W)

            topk_conf = confidence.gather(1, topk_indices)  # (B, K)
            topk_weights = F.softmax(topk_conf, dim=1)  # (B, K)

            if K > 1:
                weighted_feats = (all_corr * topk_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)).sum(dim=1)
                aggregated_corr = weighted_feats  # (B, C_corr, H, W)

                topk_dirs = dir_vecs.gather(1, topk_indices.unsqueeze(2).expand(-1, -1, 6))  # (B, K, 6)
                direction_encoding = (topk_dirs * topk_weights.unsqueeze(2)).sum(dim=1)  # (B, 6)
            else:
                aggregated_corr = all_corr[:, 0]  # (B, C_corr, H, W)
                direction_encoding = dir_vecs.gather(1, topk_indices.unsqueeze(2).expand(-1, -1, 6))[:, 0]  # (B, 6)
        else:
            # ── Original mode: all N samples have full features ──
            all_corr = corr_feats.view(batch_size, N_size, -1, feat_h, feat_w)  # (B, N, C_corr, H, W)

            if self.top_k > 1 and N_size > 1:
                K = min(self.top_k, N_size)
                sorted_indices = confidence.argsort(dim=1, descending=True)  # (B, N)
                topk_idx_orig = sorted_indices[:, :K]  # (B, K)

                topk_feats = all_corr.gather(
                    1, topk_idx_orig.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                    .expand(-1, -1, all_corr.shape[2], feat_h, feat_w)
                )  # (B, K, C_corr, H, W)

                topk_conf = confidence.gather(1, topk_idx_orig)  # (B, K)
                topk_weights = F.softmax(topk_conf, dim=1)  # (B, K)

                weighted_feats = (topk_feats * topk_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)).sum(dim=1)
                aggregated_corr = weighted_feats  # (B, C_corr, H, W)

                topk_dirs = dir_vecs.gather(1, topk_idx_orig.unsqueeze(2).expand(-1, -1, 6))  # (B, K, 6)
                direction_encoding = (topk_dirs * topk_weights.unsqueeze(2)).sum(dim=1)  # (B, 6)
            else:
                aggregated_corr = all_corr[torch.arange(batch_size, device=device), best_idx]  # (B, C_corr, H, W)
                direction_encoding = dir_vecs[torch.arange(batch_size, device=device), best_idx]  # (B, 6)

        # Predict pose delta using aggregated correlation features + direction encoding
        pose_delta, hidden_state = self.pose_update_net(
            aggregated_corr,
            context_feat,
            hidden_state,
            direction_encoding=direction_encoding
        )  # pose_delta: (B, 6, H, W)

        # Aggregate pose delta predictions (spatial pooling)
        pose_delta_avg = pose_delta.mean(dim=[2, 3])  # (B, 6)

        # Convert rotation vector (rx, ry, rz) to quaternion delta
        rot_vec = pose_delta_avg[:, :3]  # (B, 3)
        dt = pose_delta_avg[:, 3:6]      # (B, 3)

        # Rodrigues' formula: rotation vector → quaternion (fully differentiable)
        angle = rot_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        scale = (angle.clamp(max=0.5) / angle).detach()  # (B, 1)
        rot_vec_scaled = rot_vec * scale  # (B, 3)

        half_angle = rot_vec_scaled.norm(dim=1, keepdim=True).clamp(min=1e-8) / 2.0  # (B, 1)
        cos_ha = torch.cos(half_angle)  # (B, 1)
        sinc_factor = torch.sin(half_angle) / half_angle  # (B, 1)
        dq = torch.cat([cos_ha, sinc_factor * rot_vec_scaled / 2.0], dim=1)  # (B, 4)
        dq = F.normalize(dq, dim=1)

        # Update pose estimate: T_new = T_cur * Delta
        q_cur, t_cur = current_pose[:, :4], current_pose[:, 4:7]
        q_new, t_new = apply_pose_update((q_cur, t_cur), (dq, dt))
        q_new = F.normalize(q_new, dim=1)
        current_pose = torch.cat([q_new, t_new], dim=1)  # (B, 7)

        return current_pose, hidden_state, rot_vec, dt

    def forward(
        self, 
        image, 
        depth, 
        intrinsic_rgb, 
        intrinsic_depth, 
        init_pose=None,
        return_all_poses=False,
        return_all_deltas=False
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
            return_all_deltas: Whether to return all predicted deltas (rot_vec, dt) per iteration
        
        Returns:
            final_pose: Final pose estimate of shape (B, 7)
            pose_sequence: All intermediate poses (if return_all_poses=True)
            delta_sequence: Dict with 'rot_vec' and 'dt' per iteration (if return_all_deltas=True)
        """
        batch_size = image.shape[0]
        device = image.device
        
        # Extract features (use AMP for encoders if enabled)
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            fmap_rgb = self.image_encoder(image)  # (B, fmap_dim, H, W)
            fmap_depth = self.depth_encoder(depth)  # (B, depth_dim, H, W)
        # Ensure float32 for correlation and downstream ops
        fmap_rgb = fmap_rgb.float()
        fmap_depth = fmap_depth.float()
        
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
        
        # Track all deltas if requested
        if return_all_deltas:
            rot_vec_sequence = []
            dt_sequence = []
        
        # Iterative pose refinement (with gradient checkpointing)
        for it in range(self.num_iterations):
            if self.use_checkpoint:
                current_pose, hidden_state, rot_vec, dt = torch_checkpoint(
                    self._single_iteration,
                    current_pose, hidden_state, depth, intrinsic_depth, intrinsic_rgb,
                    context_feat,
                    use_reentrant=False
                )
            else:
                current_pose, hidden_state, rot_vec, dt = self._single_iteration(
                    current_pose, hidden_state, depth, intrinsic_depth, intrinsic_rgb,
                    context_feat
                )

            # Track deltas (lightweight, outside checkpoint)
            if return_all_deltas:
                rot_vec_sequence.append(rot_vec.clone())
                dt_sequence.append(dt.clone())

            # Track pose
            if return_all_poses:
                pose_sequence.append(current_pose.clone())
        
        # Build return values
        result_pose = current_pose
        extra = {}
        
        if return_all_poses:
            extra['pose_sequence'] = torch.stack(pose_sequence, dim=1)  # (B, num_iters+1, 7)
        
        if return_all_deltas:
            extra['delta_sequence'] = {
                'rot_vec': torch.stack(rot_vec_sequence, dim=1),  # (B, K, 3)
                'dt': torch.stack(dt_sequence, dim=1),  # (B, K, 3)
            }
        
        if extra:
            return result_pose, extra
        else:
            return result_pose
    
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

        if self.coarse_to_fine:
            # Coarse-to-fine: cheap coarse scoring for all N, full sampling for top-K only
            corr_feats_topk, confidence, topk_indices = self.corr_block.sample_coarse_then_fine(
                projected_coords, top_k=self.top_k
            )
            return corr_feats_topk, confidence, topk_indices
        else:
            # Original: full multi-level sampling for all N samples
            corr_feats, confidence = self.corr_block.sample_per_pose(projected_coords)
            return corr_feats, confidence, None
    
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
        pose_sample_std=config.get('pose_sample_std', 0.01),
        init_pose_noise_std=config.get('init_pose_noise_std', 0.05),
        top_k=config.get('top_k', 3),
        use_checkpoint=config.get('use_checkpoint', False),
        use_amp=config.get('use_amp', False),
        coarse_to_fine=config.get('coarse_to_fine', False)
    )
