"""
Pose utilities for the RAFT-Pose module.
Handles pose representation (quaternion + translation) and transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_quaternion(q):
    """
    Normalize quaternion to unit length.
    
    Args:
        q: Tensor of shape (..., 4) representing quaternion (w, x, y, z)
    
    Returns:
        Normalized quaternion of same shape
    """
    norm = torch.norm(q, dim=-1, keepdim=True)
    # Avoid division by zero
    norm = torch.where(norm < 1e-8, torch.ones_like(norm), norm)
    return q / norm


def quaternion_to_matrix(q):
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Tensor of shape (..., 4) representing normalized quaternion (w, x, y, z)
    
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Compute rotation matrix elements
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    R = torch.empty(q.shape[:-1] + (3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    
    return R


def matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: Tensor of shape (..., 3, 3) representing rotation matrix
    
    Returns:
        Quaternion of shape (..., 4) representing (w, x, y, z)
    """
    # Extract diagonal and off-diagonal elements
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    # Initialize quaternion
    q = torch.empty(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    
    # Compute quaternion based on trace
    case1 = trace > 0
    case2 = ~case1
    
    # Case 1: trace > 0
    s1 = torch.sqrt(trace + 1.0) * 2.0
    w1 = 0.25 * s1
    x1 = (R[..., 2, 1] - R[..., 1, 2]) / s1
    y1 = (R[..., 0, 2] - R[..., 2, 0]) / s1
    z1 = (R[..., 1, 0] - R[..., 0, 1]) / s1
    
    # Case 2: find largest diagonal element
    # Case 2a: R[0,0] is largest
    s2a = torch.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) * 2.0
    w2a = (R[..., 2, 1] - R[..., 1, 2]) / s2a
    x2a = 0.25 * s2a
    y2a = (R[..., 0, 1] + R[..., 1, 0]) / s2a
    z2a = (R[..., 0, 2] + R[..., 2, 0]) / s2a
    
    # Case 2b: R[1,1] is largest
    s2b = torch.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]) * 2.0
    w2b = (R[..., 0, 2] - R[..., 2, 0]) / s2b
    x2b = (R[..., 0, 1] + R[..., 1, 0]) / s2b
    y2b = 0.25 * s2b
    z2b = (R[..., 1, 2] + R[..., 2, 1]) / s2b
    
    # Case 2c: R[2,2] is largest
    s2c = torch.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]) * 2.0
    w2c = (R[..., 1, 0] - R[..., 0, 1]) / s2c
    x2c = (R[..., 0, 2] + R[..., 2, 0]) / s2c
    y2c = (R[..., 1, 2] + R[..., 2, 1]) / s2c
    z2c = 0.25 * s2c
    
    # Select appropriate case
    mask2a = (R[..., 0, 0] >= R[..., 1, 1]) & (R[..., 0, 0] >= R[..., 2, 2])
    mask2b = (R[..., 1, 1] >= R[..., 0, 0]) & (R[..., 1, 1] >= R[..., 2, 2])
    mask2c = ~mask2a & ~mask2b
    
    # Assemble quaternion
    q[..., 0] = torch.where(case1, w1, torch.where(mask2a, w2a, torch.where(mask2b, w2b, w2c)))
    q[..., 1] = torch.where(case1, x1, torch.where(mask2a, x2a, torch.where(mask2b, x2b, x2c)))
    q[..., 2] = torch.where(case1, y1, torch.where(mask2a, y2a, torch.where(mask2b, y2b, y2c)))
    q[..., 3] = torch.where(case1, z1, torch.where(mask2a, z2a, torch.where(mask2b, z2b, z2c)))
    
    return normalize_quaternion(q)


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Args:
        q1: Tensor of shape (..., 4) representing (w, x, y, z)
        q2: Tensor of shape (..., 4) representing (w, x, y, z)
    
    Returns:
        Quaternion product q1 * q2 of shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)


def compose_pose(q, t):
    """
    Compose rotation (quaternion) and translation into a 4x4 transformation matrix.
    
    Args:
        q: Tensor of shape (..., 4) representing normalized quaternion (w, x, y, z)
        t: Tensor of shape (..., 3) representing translation (x, y, z)
    
    Returns:
        4x4 transformation matrix of shape (..., 4, 4)
    """
    # Normalize quaternion
    q = normalize_quaternion(q)
    
    # Get rotation matrix
    R = quaternion_to_matrix(q)
    
    # Create transformation matrix
    T = torch.empty(q.shape[:-1] + (4, 4), device=q.device, dtype=q.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, :3] = 0
    T[..., 3, 3] = 1
    
    return T


def decompose_pose(T):
    """
    Decompose 4x4 transformation matrix into quaternion and translation.
    
    Args:
        T: Tensor of shape (..., 4, 4) representing transformation matrix
    
    Returns:
        q: Quaternion of shape (..., 4) representing (w, x, y, z)
        t: Translation of shape (..., 3) representing (x, y, z)
    """
    # Extract rotation matrix and convert to quaternion
    R = T[..., :3, :3]
    q = matrix_to_quaternion(R)
    
    # Extract translation
    t = T[..., :3, 3]
    
    return q, t


def apply_pose_transform(T, points):
    """
    Apply 4x4 transformation matrix to points.
    
    Args:
        T: Tensor of shape (..., 4, 4) or (B, 4, 4) representing transformation matrix
        points: Tensor of shape (..., N, 3) or (B, 3, N) or (B, N, 3) representing 3D points
    
    Returns:
        Transformed points of same shape as input
    """
    # Convert to homogeneous coordinates if needed
    if points.shape[-1] == 3:
        points_hom = torch.cat([
            points,
            torch.ones_like(points[..., :1])
        ], dim=-1)  # (..., N, 4)
        # Transpose for matrix multiplication
        points_hom = points_hom.transpose(-1, -2)  # (..., 4, N)
    elif points.shape[0] == 3:
        points_hom = torch.cat([
            points,
            torch.ones_like(points[:1])
        ], dim=0)  # (4, N)
    else:
        points_hom = points
    
    # Apply transformation
    if T.dim() == 2 and points_hom.dim() == 2:
        transformed = torch.mm(T, points_hom)
    else:
        transformed = torch.matmul(T, points_hom)
    
    # Return non-homogeneous coordinates
    return transformed[..., :3, :]


def apply_pose_update(pose, delta_pose, in_camera_frame=True):
    """
    Apply a pose update (delta) to the current pose.
    
    Args:
        pose: Current pose as tuple (q, t) where q is (..., 4) and t is (..., 3)
        delta_pose: Update as tuple (dq, dt) where dq is (..., 4) and dt is (..., 3)
        in_camera_frame: If True, update is applied in camera frame (right multiplication)
                        If False, update is applied in world frame (left multiplication)
    
    Returns:
        Updated pose as tuple (q_new, t_new)
    """
    q, t = pose
    dq, dt = delta_pose
    
    # Normalize quaternions
    q = normalize_quaternion(q)
    dq = normalize_quaternion(dq)
    
    if in_camera_frame:
        # Update in camera frame: T_new = T * Delta
        q_new = quaternion_multiply(q, dq)
        t_new = t + quaternion_apply(q, dt)
    else:
        # Update in world frame: T_new = Delta * T
        q_new = quaternion_multiply(dq, q)
        t_new = quaternion_apply(dq, t) + dt
    
    return q_new, t_new


def quaternion_apply(q, v):
    """
    Apply quaternion rotation to vector.
    
    Args:
        q: Tensor of shape (..., 4) representing normalized quaternion (w, x, y, z)
        v: Tensor of shape (..., 3) representing vector (x, y, z)
    
    Returns:
        Rotated vector of shape (..., 3)
    """
    # Convert vector to quaternion
    q_v = torch.cat([
        torch.zeros_like(v[..., :1]),
        v
    ], dim=-1)  # (..., 4)
    
    # Compute q * v * q^(-1)
    q_inv = torch.cat([
        q[..., :1],
        -q[..., 1:]
    ], dim=-1)  # Quaternion inverse for unit quaternion
    
    rotated = quaternion_multiply(quaternion_multiply(q, q_v), q_inv)
    
    return rotated[..., 1:]  # Return only vector part


def pose_to_vector(q, t):
    """
    Convert pose (quaternion + translation) to 7D vector.
    
    Args:
        q: Tensor of shape (..., 4) representing quaternion (w, x, y, z)
        t: Tensor of shape (..., 3) representing translation (x, y, z)
    
    Returns:
        7D pose vector of shape (..., 7) representing [w, x, y, z, tx, ty, tz]
    """
    return torch.cat([q, t], dim=-1)


def vector_to_pose(pose_vec):
    """
    Convert 7D vector to pose (quaternion + translation).
    
    Args:
        pose_vec: Tensor of shape (..., 7) representing [w, x, y, z, tx, ty, tz]
    
    Returns:
        q: Quaternion of shape (..., 4) representing (w, x, y, z)
        t: Translation of shape (..., 3) representing (x, y, z)
    """
    q = pose_vec[..., :4]
    t = pose_vec[..., 4:7]
    
    # Normalize quaternion
    q = normalize_quaternion(q)
    
    return q, t


def identity_pose(batch_size=1, device='cpu'):
    """
    Create identity pose.
    
    Args:
        batch_size: Number of poses to create
        device: Device to create tensors on
    
    Returns:
        Tuple (q, t) where q is (B, 4) and t is (B, 3)
    """
    q = torch.zeros(batch_size, 4, device=device)
    q[:, 0] = 1.0  # w = 1
    t = torch.zeros(batch_size, 3, device=device)
    return q, t


def generate_pose_samples(pose, num_samples, std_rot=0.1, std_trans=0.1, in_camera_frame=True):
    """
    Generate pose samples around a given pose for correlation sampling.
    
    Args:
        pose: Current pose as tuple (q, t) where q is (..., 4) and t is (..., 3)
        num_samples: Number of samples to generate
        std_rot: Standard deviation for rotation perturbations (radians)
        std_trans: Standard deviation for translation perturbations
        in_camera_frame: If True, perturbations are in camera frame
    
    Returns:
        sampled_poses: List of pose tuples [(q_i, t_i), ...]
        perturbations: List of perturbation tuples [(dq_i, dt_i), ...]
    """
    q, t = pose
    device = q.device
    
    # Generate random perturbations
    if q.dim() > 1:
        batch_size = q.shape[0]
    else:
        batch_size = 1
    
    # Random rotation perturbations (small angles)
    dq_random = torch.randn(batch_size, num_samples, 4, device=device)
    dq_random[:, :, 0] = 1.0  # Set w component to 1 (identity-like)
    dq_random = dq_random / dq_random.norm(dim=-1, keepdim=True)  # Normalize
    dq_random[:, :, 0] = torch.clamp(dq_random[:, :, 0], max=1.0, min=torch.cos(torch.tensor(std_rot)))
    dq_random = normalize_quaternion(dq_random.view(-1, 4)).view(batch_size, num_samples, 4)
    
    # Random translation perturbations
    dt_random = torch.randn(batch_size, num_samples, 3, device=device) * std_trans
    
    # Generate sampled poses
    sampled_poses = []
    perturbations = []
    
    for i in range(num_samples):
        dq_i = dq_random[:, i]  # (B, 4)
        dt_i = dt_random[:, i]  # (B, 3)
        
        # Apply perturbation
        q_i, t_i = apply_pose_update((q, t), (dq_i, dt_i), in_camera_frame)
        
        sampled_poses.append((q_i, t_i))
        perturbations.append((dq_i, dt_i))
    
    return sampled_poses, perturbations


def compute_pose_error(pose1, pose2):
    """
    Compute error between two poses.
    
    Args:
        pose1: First pose as tuple (q1, t1)
        pose2: Second pose as tuple (q2, t2)
    
    Returns:
        error_dict: Dictionary containing rotation_error (radians) and translation_error
    """
    q1, t1 = pose1
    q2, t2 = pose2
    
    # Compute rotation: q_err = q2^(-1) * q1
    q2_inv = torch.cat([q2[..., :1], -q2[..., 1:]], dim=-1)
    q_err = quaternion_multiply(q2_inv, q1)
    
    # Convert to rotation angle
    # For quaternion q = [w, x, y, z], angle = 2 * acos(|w|)
    w = torch.clamp(torch.abs(q_err[..., 0]), max=1.0)
    rotation_error = 2 * torch.acos(w)
    
    # Compute translation error
    translation_error = torch.norm(t2 - t1, dim=-1)
    
    return {
        'rotation_error': rotation_error,
        'translation_error': translation_error
    }


def sampled_poses_to_tensor(sampled_poses):
    """
    Convert sampled poses from list format to tensor format.
    
    Args:
        sampled_poses: List of [(q_i, t_i), ...] where q_i: (B, 4), t_i: (B, 3)
    
    Returns:
        pose_tensor: (B, N, 7) where pose_tensor[b, n, :] = [w, x, y, z, tx, ty, tz]
    """
    if len(sampled_poses) == 0:
        raise ValueError("sampled_poses is empty")
    
    num_samples = len(sampled_poses)
    batch_size = sampled_poses[0][0].shape[0]
    device = sampled_poses[0][0].device
    
    poses = []
    for q_i, t_i in sampled_poses:
        pose_7d = torch.cat([q_i, t_i], dim=1)  # (B, 7)
        poses.append(pose_7d)
    
    pose_tensor = torch.stack(poses, dim=1)  # (B, N, 7)
    return pose_tensor


def sampled_poses_to_matrices(sampled_poses):
    """
    Convert sampled poses to 4x4 transformation matrices.
    
    Args:
        sampled_poses: List of [(q_i, t_i), ...] where q_i: (B, 4), t_i: (B, 3)
    
    Returns:
        matrices: (B, N, 4, 4) transformation matrices
    """
    if len(sampled_poses) == 0:
        raise ValueError("sampled_poses is empty")
    
    num_samples = len(sampled_poses)
    
    matrices = []
    for q_i, t_i in sampled_poses:
        matrix = compose_pose(q_i, t_i)  # (B, 4, 4)
        matrices.append(matrix)
    
    matrices_tensor = torch.stack(matrices, dim=1)  # (B, N, 4, 4)
    return matrices_tensor
