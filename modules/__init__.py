"""
RAFT-Pose module initialization.
"""

try:
    from .pose_utils import (
        normalize_quaternion,
        quaternion_multiply,
        quaternion_to_matrix,
        matrix_to_quaternion,
        compose_pose,
        decompose_pose,
        apply_pose_transform,
        apply_pose_update,
        quaternion_apply,
        generate_pose_samples,
        compute_pose_error,
        sampled_poses_to_tensor,
        sampled_poses_to_matrices
    )

    from .pose_extractor import (
        ResidualBlock,
        BasicEncoder,
        SmallEncoder,
        DepthEncoder
    )

    from .depth_projection import (
        DepthProjector,
        CorrBlock,
        PoseCorrSampler
    )

    from .pose_update import (
        ConvGRU,
        PoseRegressionHead,
        PoseUpdateNet
    )
except ImportError:
    from pose_utils import (
        normalize_quaternion,
        quaternion_multiply,
        quaternion_to_matrix,
        matrix_to_quaternion,
        compose_pose,
        decompose_pose,
        apply_pose_transform,
        apply_pose_update,
        quaternion_apply,
        generate_pose_samples,
        compute_pose_error,
        sampled_poses_to_tensor,
        sampled_poses_to_matrices
    )

    from pose_extractor import (
        ResidualBlock,
        BasicEncoder,
        SmallEncoder,
        DepthEncoder
    )

    from depth_projection import (
        DepthProjector,
        CorrBlock,
        PoseCorrSampler
    )

    from pose_update import (
        ConvGRU,
        PoseRegressionHead,
        PoseUpdateNet
    )

__all__ = [
    'normalize_quaternion',
    'quaternion_multiply',
    'quaternion_to_matrix',
    'matrix_to_quaternion',
    'compose_pose',
    'decompose_pose',
    'apply_pose_transform',
    'apply_pose_update',
    'quaternion_apply',
    'generate_pose_samples',
    'compute_pose_error',
    'sampled_poses_to_tensor',
    'sampled_poses_to_matrices',
    'ResidualBlock',
    'BasicEncoder',
    'SmallEncoder',
    'DepthEncoder',
    'DepthProjector',
    'CorrBlock',
    'PoseCorrSampler',
    'ConvGRU',
    'PoseRegressionHead',
    'PoseUpdateNet'
]
