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
    from .modules.pose_extractor import BasicEncoder, DepthEncoder, SmallEncoder, ResNet18Encoder
    from .modules.depth_projection import DepthProjector, CorrBlock, PoseCorrSampler
    from .modules.cross_attention_matcher import CrossAttentionMatcher
    from .modules.pose_update import PoseUpdateNet
except ImportError:
    from modules.pose_utils import (
        quaternion_to_matrix, matrix_to_quaternion,
        compose_pose, apply_pose_update, generate_pose_samples,
        compute_pose_error, sampled_poses_to_matrices
    )
    from modules.pose_extractor import BasicEncoder, DepthEncoder, SmallEncoder, ResNet18Encoder
    from modules.depth_projection import DepthProjector, CorrBlock, PoseCorrSampler
    from modules.cross_attention_matcher import CrossAttentionMatcher
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
        coarse_to_fine=False,
        corr_temperature=1.0,
        max_rot_step=0.1,
        max_trans_step=0.1,
        shared_encoder=False,
        matcher_type='corr',
        matcher_heads=8,
        matcher_blocks=2,
        matcher_ffn_dim=512,
        matcher_dropout=0.1,
        coarse_iters=3,
        max_rot_step_fine=0.05,
        max_trans_step_fine=0.05,
    ):
        """
        Args:
            image_encoder: Type of image feature encoder ('basic' or 'small')
            shared_encoder: If True, use the same encoder for both RGB and depth
                           (Siamese). Depth is repeated to 3 channels before encoding.
                           This ensures features are in the same space without alignment.
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
        self.num_pose_samples_actual = 37  # 36 + 1 identity
        self.pose_sample_std = pose_sample_std
        self.corr_temperature = corr_temperature
        self.max_rot_step = max_rot_step
        self.max_trans_step = max_trans_step
        self.shared_encoder = shared_encoder
        self.matcher_type = matcher_type
        self.matcher_heads = matcher_heads
        self.matcher_blocks = matcher_blocks
        self.matcher_ffn_dim = matcher_ffn_dim
        self.matcher_dropout = matcher_dropout

        # ── Precompute pose sample templates (Optimization 1) ──────────
        # These are constant across all iterations, so compute once.
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
        scales = [0.25, 1.0, 4.0]

        rot_axes_list, trans_dirs_list, scale_list = [], [], []
        for rot_dir, trans_dir in base_directions:
            for s in scales:
                rot_axes_list.append(rot_dir)
                trans_dirs_list.append(trans_dir)
                scale_list.append(s)

        rot_axes = torch.tensor(rot_axes_list, dtype=torch.float32)  # (36, 3)
        trans_dirs = torch.tensor(trans_dirs_list, dtype=torch.float32)  # (36, 3)
        sample_scales = torch.tensor(scale_list, dtype=torch.float32)  # (36,)

        # Direction encoding vectors (unscaled)
        dir_vecs_raw = torch.cat([rot_axes, trans_dirs], dim=1)  # (36, 6)

        self.register_buffer('_precomputed_dir_vecs_raw', dir_vecs_raw)  # (36, 6)
        self.register_buffer('_precomputed_sample_scales', sample_scales)  # (36,)
        self.register_buffer('_precomputed_rot_axes', rot_axes)  # (36, 3)
        self.register_buffer('_precomputed_trans_dirs', trans_dirs)  # (36, 3)
        self.init_pose_noise_std = init_pose_noise_std
        self.top_k = min(top_k, self.num_pose_samples)  # top_k cannot exceed num_pose_samples
        self.downsample_factor = 8  # Total stride of image encoder (3 layers × stride 2)
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp
        self.coarse_to_fine = coarse_to_fine
        
        # ── Single shared RGB encoder for BOTH frames (true Siamese, weight-shared) ──
        # Matching is RGB<->RGB appearance (discriminative via learned edges/texture),
        # which fixes the cross-modality gap of encoding a depth MAP separately.
        # Frame B's depth is NOT encoded — it is used only for geometric projection
        # (depth_projector samples correlation at depth-projected locations).
        # 'depth' tensor is (B, 4, H, W) = [raw_depth, R, G, B]; ch1:4 (frame B RGB)
        # feeds this same encoder; ch0 feeds depth_projector.
        if image_encoder == 'basic':
            self.image_encoder = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0.1,
                                              use_checkpoint=use_checkpoint)
            self.fmap_dim = 256
        elif image_encoder == 'small':
            self.image_encoder = SmallEncoder(output_dim=128, norm_fn='instance', dropout=0.1,
                                              use_checkpoint=use_checkpoint)
            self.fmap_dim = 128
        elif image_encoder == 'resnet18':
            self.image_encoder = ResNet18Encoder(output_dim=256, norm_fn='instance', dropout=0.1,
                                                 pretrained=True, use_checkpoint=use_checkpoint)
            self.fmap_dim = 256
        else:
            raise ValueError(f"Unknown image encoder: {image_encoder}")
        # No separate depth encoder, no feature alignment (both frames are RGB).
        self.depth_feat_align = None
        # shared_encoder flag retained for backward-compat (forward keys on it);
        # it is effectively always True now.
        self.shared_encoder = True
        # Context feature projector
        if shared_encoder:
            # Context from depth features (same dim as fmap_dim)
            self.context_proj = nn.Sequential(
                nn.Conv2d(self.fmap_dim, context_dim, 1),
                nn.ReLU(inplace=True)
            )
        else:
            self.context_proj = nn.Sequential(
                nn.Conv2d(depth_dim, context_dim, 1),
                nn.ReLU(inplace=True)
            )
        
        # Pose update network
        # corr_dim is now per-sample C_corr (not N*C_corr), so the network
        # is decoupled from the number of pose samples — training and inference
        # can handle variable numbers of samples.
        if matcher_type == 'attention':
            per_sample_corr_dim = self.fmap_dim  # Cross-attention outputs fmap_dim channels
        else:
            per_sample_corr_dim = (2 * corr_radius + 1) ** 2 * corr_levels
        self.per_sample_corr_dim = per_sample_corr_dim
        self.pose_update_net = PoseUpdateNet(
            hidden_dim=hidden_dim,
            corr_dim=per_sample_corr_dim,
            context_dim=context_dim,
            num_layers=3
        )
        
        # Correlation / cross-attention matcher
        if matcher_type == 'attention':
            # Register CrossAttentionMatcher as a persistent sub-module
            # so its parameters are included in model.parameters() and optimized.
            # Use a large max size for positional encoding; actual features are
            # sliced to the correct spatial size at runtime.
            self.corr_block = CrossAttentionMatcher(
                C_depth=self.fmap_dim,
                C_rgb=self.fmap_dim,
                H=128,  # max_h for positional encoding (actual H is sliced)
                W=160,  # max_w for positional encoding (actual W is sliced)
                num_heads=self.matcher_heads,
                feature_dim=self.fmap_dim,
                ffn_dim=self.matcher_ffn_dim,
                num_blocks=self.matcher_blocks,
                dropout=self.matcher_dropout,
                temperature=self.corr_temperature,
                num_levels=self.corr_levels,
                radius=self.corr_radius,
            )
        else:
            # CorrBlock has no learnable parameters, create dynamically in forward
            self.corr_block = None
        
        # Depth projector for batch pose projection
        self.depth_projector = DepthProjector()

        # ── Fine refinement stage (1/4 resolution dynamic alignment) ────────
        # Number of coarse iterations; remaining iterations use fine stage
        self.coarse_iters = min(coarse_iters, num_iterations)
        if self.coarse_iters < num_iterations:
            # Residual feature projection: fmap_dim → hidden_dim
            self.fine_feat_proj = nn.Sequential(
                nn.Conv2d(self.fmap_dim, hidden_dim, 1),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                nn.InstanceNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            # Similarity score projection: 1 → hidden_dim
            self.fine_sim_proj = nn.Sequential(
                nn.Conv2d(1, hidden_dim, 1),
                nn.ReLU(inplace=True),
            )
            # Merge projection: 2*hidden_dim → per_sample_corr_dim
            self.fine_corr_proj = nn.Sequential(
                nn.Conv2d(2 * hidden_dim, self.per_sample_corr_dim, 1),
                nn.ReLU(inplace=True),
            )
            # 1/4 context projection (mirrors context_proj, fmap_dim→context_dim)
            self.context_proj_1_4 = nn.Sequential(
                nn.Conv2d(self.fmap_dim, context_dim, 1),
                nn.ReLU(inplace=True),
            )
            # Dedicated 1/4 fine update network (resolution-agnostic PoseUpdateNet).
            # Separate from pose_update_net so the fine stage keeps its own 1/4
            # hidden state and learns small-step behaviour independently.
            self.fine_update_net = PoseUpdateNet(
                hidden_dim=hidden_dim,
                corr_dim=self.per_sample_corr_dim,
                context_dim=context_dim,
                num_layers=3,
            )
            # Genuinely fine per-iteration step limits (configurable)
            self.max_rot_step_fine = max_rot_step_fine
            self.max_trans_step_fine = max_trans_step_fine
    
    def initialize_correlation(self, fmap_rgb, fmap_depth):
        """
        Initialize correlation or cross-attention matcher between RGB and depth features.
        
        For 'corr': builds 4D correlation volume (CorrBlock).
        For 'attention': runs the persistent CrossAttentionMatcher (created in __init__).
        
        Args:
            fmap_rgb: RGB features of shape (B, C, H, W)
            fmap_depth: Depth features of shape (B, C, H, W)
        """
        if self.matcher_type == 'attention':
            # CrossAttentionMatcher is already registered in __init__.
            # Just run forward to compute and cache attention features.
            self.corr_block(fmap_depth, fmap_rgb)
        else:
            # CorrBlock has no learnable parameters, safe to create dynamically
            self.corr_block = CorrBlock(fmap_depth, fmap_rgb, 
                                         num_levels=self.corr_levels, 
                                         radius=self.corr_radius,
                                         temperature=self.corr_temperature)
    
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

        OPTIMIZED: Pose sample templates (rot_axes, trans_dirs, dir_vecs) are
        precomputed in __init__ as persistent buffers. This method only computes
        the scale-dependent dq/dt at each call, avoiding ~100 tensor allocations
        per iteration. Expected saving: ~8ms/iter (from 26ms → ~18ms).
        """
        B = current_pose.shape[0]
        device = current_pose.device

        q_cur, t_cur = current_pose[:, :4], current_pose[:, 4:7]

        # ── Use precomputed templates ─────────────────────────────
        sample_scales = self._precomputed_sample_scales  # (36,)
        rot_axes = self._precomputed_rot_axes              # (36, 3)
        trans_dirs = self._precomputed_trans_dirs           # (36, 3)
        dir_vecs_base = self._precomputed_dir_vecs_raw      # (36, 6)

        # Normalize rotation axes
        rot_norms = rot_axes.norm(dim=1, keepdim=True).clamp(min=1e-8)
        rot_axes_normed = rot_axes / rot_norms

        # Compute scale-dependent quaternion perturbations
        half_angles = base_step_rot * sample_scales / 2.0
        cos_ha = torch.cos(half_angles)
        sin_ha = torch.sin(half_angles)

        all_dq = torch.stack([
            cos_ha,
            rot_axes_normed[:, 0] * sin_ha,
            rot_axes_normed[:, 1] * sin_ha,
            rot_axes_normed[:, 2] * sin_ha,
        ], dim=-1)  # (36, 4)

        # Scale-dependent translation perturbations
        all_dt = trans_dirs * (base_step_trans * sample_scales.unsqueeze(1))  # (36, 3)

        # Append identity sample (index 36)
        identity_dq = dir_vecs_base.new_tensor([[1.0, 0.0, 0.0, 0.0]])
        identity_dt = dir_vecs_base.new_zeros(1, 3)
        identity_dir = dir_vecs_base.new_zeros(1, 6)

        all_dq = torch.cat([all_dq, identity_dq], dim=0)   # (37, 4)
        all_dt = torch.cat([all_dt, identity_dt], dim=0)   # (37, 3)
        dir_vecs_raw = torch.cat([dir_vecs_base, identity_dir], dim=0)  # (37, 6)
        actual_N = all_dq.shape[0]  # 37

        # Expand for batch: (B, 37, 4) and (B, 37, 3)
        all_dq = all_dq.unsqueeze(0).expand(B, -1, -1)
        all_dt = all_dt.unsqueeze(0).expand(B, -1, -1)

        # Apply perturbations using vectorized quaternion math
        q_cur_exp = q_cur.unsqueeze(1)  # (B, 1, 4)
        t_cur_exp = t_cur.unsqueeze(1)  # (B, 1, 3)

        sampled_quats = self._batch_quaternion_multiply(q_cur_exp, all_dq)
        sampled_quats = F.normalize(sampled_quats, dim=-1)

        rotated_dt = self._batch_quaternion_apply(q_cur_exp, all_dt)
        sampled_trans = t_cur_exp + rotated_dt

        # Vectorized conversion to 4×4 matrices (no Python loop)
        pose_samples = self._vectorized_poses_to_matrices(sampled_quats, sampled_trans)

        # Sample correlation volume
        sample_result = self.sample_correlation_with_poses(
            pose_samples, depth, intrinsic_depth, intrinsic_rgb
        )
        if self.coarse_to_fine:
            corr_feats, confidence, topk_indices = sample_result
        else:
            corr_feats, confidence, topk_indices = sample_result[0], sample_result[1], None

        # Direction encoding: (B, actual_N, 6)
        dir_vecs = dir_vecs_raw.unsqueeze(0).expand(B, -1, -1)

        return pose_samples, confidence, corr_feats, dir_vecs, topk_indices

    @staticmethod
    def _vectorized_poses_to_matrices(quats, trans):
        """
        Vectorized conversion of quaternions + translations to 4×4 matrices.
        Avoids the Python for-loop in sampled_poses_to_matrices.

        Args:
            quats: (B, N, 4) normalized quaternions
            trans: (B, N, 3) translations

        Returns:
            matrices: (B, N, 4, 4) transformation matrices
        """
        w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        T = quats.new_zeros(quats.shape[:-1] + (4, 4))
        T[..., 0, 0] = 1 - 2 * (yy + zz)
        T[..., 0, 1] = 2 * (xy - wz)
        T[..., 0, 2] = 2 * (xz + wy)
        T[..., 1, 0] = 2 * (xy + wz)
        T[..., 1, 1] = 1 - 2 * (xx + zz)
        T[..., 1, 2] = 2 * (yz - wx)
        T[..., 2, 0] = 2 * (xz - wy)
        T[..., 2, 1] = 2 * (yz + wx)
        T[..., 2, 2] = 1 - 2 * (xx + yy)
        T[..., :3, 3] = trans
        T[..., 3, 3] = 1.0

        return T

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

        # Apply predicted delta to current pose
        current_pose, rot_vec, dt = self._apply_pose_delta(
            current_pose, pose_delta,
            max_rot_step=self.max_rot_step,
            max_trans_step=self.max_trans_step,
        )

        return current_pose, hidden_state, rot_vec, dt

    def _apply_pose_delta(self, current_pose, pose_delta, max_rot_step, max_trans_step):
        """
        Convert predicted 6D delta map to pose update via Rodrigues + quaternion compose.

        Shared by both coarse and fine iteration stages.

        Args:
            current_pose: (B, 7) current pose estimate
            pose_delta: (B, 6, H, W) predicted delta field
            max_rot_step: max rotation step per iteration (radians)
            max_trans_step: max translation step per iteration (meters)

        Returns:
            current_pose: (B, 7) updated pose
            rot_vec: (B, 3) raw rotation vector (before Rodrigues)
            dt: (B, 3) translation delta
        """
        # Aggregate pose delta predictions (spatial pooling)
        pose_delta_avg = pose_delta.mean(dim=[2, 3])  # (B, 6)

        rot_vec = pose_delta_avg[:, :3]  # (B, 3)
        dt = pose_delta_avg[:, 3:6]      # (B, 3)

        # Clamp magnitudes to prevent divergence
        rot_vec = rot_vec.clamp(-max_rot_step, max_rot_step)
        dt = dt.clamp(-max_trans_step, max_trans_step)

        # Rodrigues' formula: rotation vector → quaternion (fully differentiable)
        angle = rot_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        scale = (angle.clamp(max=max_rot_step) / angle).detach()  # (B, 1)
        rot_vec_scaled = rot_vec * scale  # (B, 3)

        half_angle = rot_vec_scaled.norm(dim=1, keepdim=True).clamp(min=1e-8) / 2.0  # (B, 1)
        cos_ha = torch.cos(half_angle)  # (B, 1)
        small_ha = (half_angle.abs() < 1e-4)
        sinc_factor = torch.where(
            small_ha,
            torch.ones_like(half_angle),
            torch.sin(half_angle) / half_angle
        )
        dq = torch.cat([cos_ha, sinc_factor * rot_vec_scaled / 2.0], dim=1)  # (B, 4)
        dq = F.normalize(dq, dim=1)

        # Update pose estimate: T_new = T_cur * Delta
        q_cur, t_cur = current_pose[:, :4], current_pose[:, 4:7]
        q_new, t_new = apply_pose_update((q_cur, t_cur), (dq, dt))
        q_new = F.normalize(q_new, dim=1)
        q_norm = q_new.norm(dim=1, keepdim=True).clamp(min=1e-8)
        q_new = q_new / q_norm
        current_pose = torch.cat([q_new, t_new], dim=1)  # (B, 7)

        return current_pose, rot_vec, dt

    def _single_iteration_fine(self, current_pose, hidden_state_fine, depth,
                               intrinsic_depth, intrinsic_rgb):
        """
        Fine refinement iteration: 1/4 resolution dynamic feature alignment.

        With only frame-A depth + the forward projection g(p_d)=p_rgb
        available (no frame-B depth → no inverse map), the geometrically
        correct grid_sample warp is to pull RGB features BACK onto the
        depth (frame-A) grid: grid_sample(fmap_rgb, grid)[p_d] = fmap_rgb[g(p_d)]
        = feature of point P (at depth pixel p_d) as seen in the RGB frame.
        The residual fmap_depth[p_d] − fmap_rgb_warped[p_d] then compares the
        same 3D point across the two frames and vanishes at the correct pose.

        Runs the update at 1/4 resolution (dedicated fine_update_net + its own
        1/4 hidden state), preserving the 4× spatial precision. No directional
        sampling — direct residual signal. Genuinely small step limits.
        """
        batch_size = current_pose.shape[0]
        device = current_pose.device

        # ── 1. Depth projection at 1/4 resolution ────────────────────────
        _, _, H_orig, W_orig = depth.shape
        fine_h, fine_w = H_orig // 4, W_orig // 4  # 120×160 for 480×640 input

        # Use raw depth channel (Ch 0)
        depth_raw = depth[:, 0:1, :, :]  # (B, 1, H, W)
        depth_fine = F.interpolate(
            depth_raw, size=(fine_h, fine_w), mode='nearest'
        ).squeeze(1)  # (B, fine_h, fine_w)

        # Scale intrinsics to 1/4
        intr_d = intrinsic_depth.clone()
        intr_d[:, 0, 0] *= 0.25
        intr_d[:, 1, 1] *= 0.25
        intr_d[:, 0, 2] *= 0.25
        intr_d[:, 1, 2] *= 0.25
        intr_r = intrinsic_rgb.clone()
        intr_r[:, 0, 0] *= 0.25
        intr_r[:, 1, 1] *= 0.25
        intr_r[:, 0, 2] *= 0.25
        intr_r[:, 1, 2] *= 0.25

        # Current pose → 4×4 matrix
        q_cur, t_cur = current_pose[:, :4], current_pose[:, 4:7]
        T_cur = self._vectorized_poses_to_matrices(
            q_cur.unsqueeze(1), t_cur.unsqueeze(1)
        ).squeeze(1)  # (B, 4, 4)

        # Project depth at 1/4 resolution: coords_1_4[p_d] = g(p_d) (RGB coord)
        projected = self.depth_projector(
            depth_fine, T_cur.unsqueeze(1), intr_d, intr_r
        )  # (B, 1, 2, fine_h, fine_w)
        coords_1_4 = projected[:, 0]  # (B, 2, fine_h, fine_w)

        # ── 2. Warp RGB features onto depth grid (correct direction) ──────
        u = coords_1_4[:, 0].clamp(0, fine_w - 1) / max(fine_w - 1, 1) * 2 - 1
        v = coords_1_4[:, 1].clamp(0, fine_h - 1) / max(fine_h - 1, 1) * 2 - 1
        grid = torch.stack([u, v], dim=-1).clamp(-1.0, 1.0)  # (B, fine_h, fine_w, 2)

        # Pull RGB features to the depth grid: output[p_d] = fmap_rgb[g(p_d)]
        fmap_rgb_warped = F.grid_sample(
            self._fmap_rgb_1_4, grid,
            mode='bilinear', align_corners=True, padding_mode='zeros'
        )  # (B, fmap_dim, fine_h, fine_w)

        # Validity mask: out-of-bounds projections have no correspondence.
        # (depth_projector already pushes Z_C<=0 points to -1e4 → out-of-bounds.)
        inb = ((coords_1_4[:, 0] >= 0) & (coords_1_4[:, 0] <= fine_w - 1) &
               (coords_1_4[:, 1] >= 0) & (coords_1_4[:, 1] <= fine_h - 1))
        valid_mask = inb.float().unsqueeze(1)  # (B, 1, fine_h, fine_w)

        # ── 3. Compute residual and similarity (masked) ───────────────────
        # Same 3D point P across frames → diff ≈ 0, sim ≈ 1 at correct pose.
        feat_diff = (self._fmap_depth_1_4 - fmap_rgb_warped) * valid_mask

        rgb_norm = F.normalize(self._fmap_rgb_1_4, dim=1)
        warped_norm = F.normalize(fmap_rgb_warped, dim=1)
        similarity = ((rgb_norm * warped_norm).sum(dim=1, keepdim=True)) * valid_mask

        # ── 4. Project to corr_dim (stay at 1/4 — no downsample) ──────────
        diff_proj = self.fine_feat_proj(feat_diff)    # (B, hidden_dim, fine_h, fine_w)
        sim_proj = self.fine_sim_proj(similarity)      # (B, hidden_dim, fine_h, fine_w)
        align_feat = torch.cat([diff_proj, sim_proj], dim=1)  # (B, 2*hidden_dim, fine_h, fine_w)

        align_corr = self.fine_corr_proj(align_feat)  # (B, corr_dim, fine_h, fine_w)

        # ── 5. Feed to dedicated 1/4 fine update net (+ own hidden state) ──
        context_feat_1_4 = self.context_proj_1_4(self._fmap_depth_1_4)
        pose_delta, hidden_state_fine = self.fine_update_net(
            align_corr,
            context_feat_1_4,
            hidden_state_fine,
            direction_encoding=None,  # No directional probing in fine stage
        )

        # ── 6. Apply delta with fine-grained step limits ───────────────────
        current_pose, rot_vec, dt = self._apply_pose_delta(
            current_pose, pose_delta,
            max_rot_step=self.max_rot_step_fine,
            max_trans_step=self.max_trans_step_fine,
        )

        return current_pose, hidden_state_fine, rot_vec, dt

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
        Forward pass of RAFT-Pose model with staged iterative refinement.

        Stage 1 (iterations 0..coarse_iters-1): 1/8 correlation volume + 37-sample probing
        Stage 2 (iterations coarse_iters..num_iterations-1): 1/4 dynamic feature alignment

        Args:
            image: RGB image of shape (B, 3, H, W)
            depth: Depth map of shape (B, 2, H, W)  [raw_depth, inv_depth]
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

        # Extract multi-scale features with the SHARED encoder (both frames RGB).
        # NOTE: names fmap_rgb_* (frame A) / fmap_depth_* (frame B) are kept for
        # minimal diff, but both are now RGB-appearance features -> RGB<->RGB matching.
        # depth[:, 0:1] (raw) is consumed by depth_projector in the iteration stages.
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            enc_result = self.image_encoder(image)              # frame A RGB
            fmap_rgb_1_8 = enc_result[0].float()
            fmap_rgb_1_4 = enc_result[1].float()

            enc_result_d = self.image_encoder(depth[:, 1:4])    # frame B RGB (shared weights)
            fmap_depth_1_8 = enc_result_d[0].float()
            fmap_depth_1_4 = enc_result_d[1].float()

        # Save 1/4 features for fine stage
        self._fmap_rgb_1_4 = fmap_rgb_1_4
        self._fmap_depth_1_4 = fmap_depth_1_4

        if self.shared_encoder:
            fmap_depth_aligned = fmap_depth_1_8
        else:
            fmap_depth_aligned = self.depth_feat_align(fmap_depth_1_8)

        context_feat = self.context_proj(fmap_depth_1_8)  # (B, context_dim, H/8, W/8)

        # Initialize correlation volume with aligned 1/8 features
        if fmap_rgb_1_8.shape[2:] != fmap_depth_aligned.shape[2:]:
            fmap_depth_aligned = F.interpolate(fmap_depth_aligned, size=fmap_rgb_1_8.shape[2:],
                                               mode='bilinear', align_corners=False)
        self.initialize_correlation(fmap_rgb_1_8, fmap_depth_aligned)
        
        # Initialize pose estimate
        if init_pose is None:
            current_pose = self.initialize_pose(batch_size, device)
        else:
            current_pose = init_pose
        
        # Initialize hidden states for ConvGRU
        hidden_state = None        # coarse stage, 1/8 resolution
        hidden_state_fine = None   # fine stage, 1/4 resolution (separate)
        
        # Track all poses if requested
        if return_all_poses:
            pose_sequence = [current_pose.clone()]
        
        # Track all deltas if requested
        if return_all_deltas:
            rot_vec_sequence = []
            dt_sequence = []
        
        # Iterative pose refinement — staged: coarse then fine
        for it in range(self.num_iterations):
            if it < self.coarse_iters:
                # Stage 1: Coarse — 1/8 correlation + 37-sample probing (existing logic)
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
            else:
                # Stage 2: Fine — 1/4 dynamic feature alignment (own hidden state)
                if self.use_checkpoint:
                    current_pose, hidden_state_fine, rot_vec, dt = torch_checkpoint(
                        self._single_iteration_fine,
                        current_pose, hidden_state_fine, depth, intrinsic_depth, intrinsic_rgb,
                        use_reentrant=False
                    )
                else:
                    current_pose, hidden_state_fine, rot_vec, dt = self._single_iteration_fine(
                        current_pose, hidden_state_fine, depth, intrinsic_depth, intrinsic_rgb
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

        # Use only Ch 0 (raw depth in meters) for geometric projection
        depth_raw = depth[:, 0:1, :, :]  # (B, 1, H, W)
        depth_small = F.interpolate(
            depth_raw, size=(feat_h, feat_w), mode='nearest'
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
        coarse_to_fine=config.get('coarse_to_fine', False),
        coarse_iters=config.get('coarse_iters', 3),
        max_rot_step_fine=config.get('max_rot_step_fine', 0.05),
        max_trans_step_fine=config.get('max_trans_step_fine', 0.05),
    )
