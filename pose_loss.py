"""
Pose loss functions for RAFT-Pose training.

Provides modular loss components for camera-LiDAR extrinsic calibration:
- GeodesicRotationLoss: Rotation error via quaternion geodesic distance
- TranslationLoss: Translation error via L2 distance
- UncertaintyLoss: Negative log-likelihood for uncertainty estimation
- PoseLoss: Combined loss with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicRotationLoss(nn.Module):
    """
    Geodesic distance loss between two sets of quaternions.

    Computes the minimal rotation angle: theta = 2 * arccos(|q1 · q2|)
    which is invariant to quaternion double-cover (q and -q represent the same rotation).
    """

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super(GeodesicRotationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_quat, gt_quat):
        """
        Args:
            pred_quat: Predicted quaternion (B, 4) in (w, x, y, z) format
            gt_quat: Ground truth quaternion (B, 4) in (w, x, y, z) format

        Returns:
            Angular error in radians, shape depends on reduction
        """
        # Normalize quaternions
        pred_quat = F.normalize(pred_quat, dim=1)
        gt_quat = F.normalize(gt_quat, dim=1)

        # Absolute dot product (handles q / -q ambiguity)
        dot = torch.sum(pred_quat * gt_quat, dim=1)  # (B,)
        dot = torch.clamp(torch.abs(dot), min=0.0, max=1.0)

        # Geodesic distance: theta = 2 * arccos(|dot|)
        angle = 2.0 * torch.acos(dot)  # (B,), range [0, pi]

        return self._reduce(angle)

    def _reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # 'none'


class TranslationLoss(nn.Module):
    """
    L2 distance loss between two translation vectors.
    """

    def __init__(self, reduction='mean'):
        super(TranslationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_trans, gt_trans):
        """
        Args:
            pred_trans: Predicted translation (B, 3)
            gt_trans: Ground truth translation (B, 3)

        Returns:
            L2 distance, shape depends on reduction
        """
        error = torch.norm(pred_trans - gt_trans, dim=1)  # (B,)
        return self._reduce(error)

    def _reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class UncertaintyLoss(nn.Module):
    """
    Negative log-likelihood loss for uncertainty estimation.

    Encourages the model to predict uncertainty that reflects actual error:
    L = 0.5 * (error^2 / sigma^2 + log(sigma^2))
    """

    def __init__(self, reduction='mean'):
        super(UncertaintyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, error, log_sigma):
        """
        Args:
            error: Absolute error (B,) or (B, D)
            log_sigma: Predicted log variance (B,) or (B, D)

        Returns:
            Uncertainty loss value
        """
        # L = 0.5 * (exp(-log_sigma) * error^2 + log_sigma)
        sigma_sq = torch.exp(log_sigma)
        loss = 0.5 * (error ** 2 / sigma_sq + log_sigma)
        return self._reduce(loss)

    def _reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PoseLoss(nn.Module):
    """
    Combined pose estimation loss with configurable weights.

    Loss = rot_weight * GeodesicRotationLoss
         + trans_weight * TranslationLoss
         + uncertainty_weight * UncertaintyLoss (optional)

    Usage:
        loss_fn = PoseLoss(rot_weight=1.0, trans_weight=100.0)
        loss, details = loss_fn(pred_pose, gt_pose)
    """

    def __init__(self, rot_weight=1.0, trans_weight=1.0, uncertainty_weight=0.0):
        """
        Args:
            rot_weight: Weight for rotation loss (radians)
            trans_weight: Weight for translation loss (meters)
            uncertainty_weight: Weight for uncertainty loss (0 to disable)
        """
        super(PoseLoss, self).__init__()

        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.uncertainty_weight = uncertainty_weight

        self.rot_loss_fn = GeodesicRotationLoss(reduction='mean')
        self.trans_loss_fn = TranslationLoss(reduction='mean')

        if uncertainty_weight > 0:
            self.uncertainty_loss_fn = UncertaintyLoss(reduction='mean')
        else:
            self.uncertainty_loss_fn = None

    def forward(self, pred_pose, gt_pose, pred_log_sigma=None):
        """
        Compute combined pose loss.

        Args:
            pred_pose: Predicted pose (B, 7) — [w, x, y, z, tx, ty, tz]
            gt_pose: Ground truth pose (B, 7) — [w, x, y, z, tx, ty, tz]
            pred_log_sigma: Optional predicted log variance (B, 7) for uncertainty

        Returns:
            total_loss: Weighted sum of all losses (scalar)
            details: Dict with individual loss components for logging
        """
        # Split into rotation and translation
        pred_quat = pred_pose[:, :4]
        pred_trans = pred_pose[:, 4:7]
        gt_quat = gt_pose[:, :4]
        gt_trans = gt_pose[:, 4:7]

        # Rotation loss (radians)
        rot_loss = self.rot_loss_fn(pred_quat, gt_quat)

        # Translation loss (meters)
        trans_loss = self.trans_loss_fn(pred_trans, gt_trans)

        # Combined
        total_loss = self.rot_weight * rot_loss + self.trans_weight * trans_loss

        # Uncertainty loss (optional)
        uncertainty_loss = torch.tensor(0.0, device=pred_pose.device)
        if self.uncertainty_loss_fn is not None and pred_log_sigma is not None:
            rot_error = self.rot_loss_fn(pred_quat, gt_quat, reduction='none')
            trans_error = self.trans_loss_fn(pred_trans, gt_trans, reduction='none')
            all_errors = torch.cat([rot_error.unsqueeze(1), trans_error], dim=1)
            uncertainty_loss = self.uncertainty_loss_fn(all_errors, pred_log_sigma)
            total_loss = total_loss + self.uncertainty_weight * uncertainty_loss

        # Build details dict for logging
        details = {
            'total_loss': total_loss.item(),
            'rot_loss': rot_loss.item(),
            'rot_loss_deg': rot_loss.item() * (180.0 / 3.14159265358979),
            'trans_loss': trans_loss.item(),
        }
        if self.uncertainty_loss_fn is not None:
            details['uncertainty_loss'] = uncertainty_loss.item()

        return total_loss, details

    def forward_sequence(self, pose_sequence, gt_pose, gamma=0.8):
        """
        Compute sequence loss over all iterative refinement steps.

        Each iteration's predicted pose is supervised against the GT pose,
        with exponentially increasing weights for later iterations (RAFT-style).

        Args:
            pose_sequence: Predicted poses (B, K+1, 7) — initial pose + K iteration outputs.
                           pose_sequence[:, 0] is the initial (identity) pose, skipped.
            gt_pose: Ground truth pose (B, 7) — [w, x, y, z, tx, ty, tz]
            gamma: Exponential weighting factor in (0, 1].
                   Later iterations receive higher weight: w_i = gamma^(K-1-i).
                   gamma=1.0 means all iterations weighted equally.
                   gamma=0.8 means the last iteration is weighted ~2.4x more than the first (for K=4).

        Returns:
            total_loss: Weighted sum of per-iteration losses (scalar)
            details: Dict with average loss components for logging
        """
        K = pose_sequence.shape[1] - 1  # Number of refinement iterations
        iter_poses = pose_sequence[:, 1:]  # (B, K, 7) — skip initial identity pose

        gt_quat = gt_pose[:, :4]
        gt_trans = gt_pose[:, 4:7]

        total_loss = torch.tensor(0.0, device=gt_pose.device)
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        weight_sum = 0.0

        for i in range(K):
            # Exponential weighting: later iterations matter more
            weight = gamma ** (K - 1 - i)

            pred_i = iter_poses[:, i]  # (B, 7)
            pred_quat = pred_i[:, :4]
            pred_trans = pred_i[:, 4:7]

            rot_loss = self.rot_loss_fn(pred_quat, gt_quat)
            trans_loss = self.trans_loss_fn(pred_trans, gt_trans)
            iter_loss = self.rot_weight * rot_loss + self.trans_weight * trans_loss

            total_loss = total_loss + weight * iter_loss
            total_rot_loss += weight * rot_loss.item()
            total_trans_loss += weight * trans_loss.item()
            weight_sum += weight

        # Normalize by total weight
        total_rot_loss /= weight_sum
        total_trans_loss /= weight_sum

        details = {
            'total_loss': total_loss.item() / weight_sum,
            'rot_loss': total_rot_loss,
            'rot_loss_deg': total_rot_loss * (180.0 / 3.14159265358979),
            'trans_loss': total_trans_loss,
        }

        return total_loss / weight_sum, details


def quat_conjugate(q):
    """Quaternion conjugate (= inverse for unit quaternions). q: (..., 4) in (w,x,y,z)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_multiply(q1, q2):
    """Hamilton product of two quaternions. (..., 4) in (w,x,y,z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_rotate(q, v):
    """Rotate vector v by quaternion q. q: (..., 4), v: (..., 3)."""
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_inv = quat_conjugate(q)
    return quat_multiply(quat_multiply(q, q_v), q_inv)[..., 1:]


def quat_to_rotvec(q):
    """
    Convert quaternion to rotation vector (axis * angle).
    q: (..., 4) in (w, x, y, z).
    Returns: rot_vec (..., 3)
    """
    q = F.normalize(q, dim=-1)
    # Ensure w >= 0 for canonical form
    sign = torch.sign(q[..., :1])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign

    sin_half = q[..., 1:].norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_half = q[..., :1]
    half_angle = torch.atan2(sin_half, cos_half)  # (..., 1)
    # rot_vec = axis * 2 * half_angle = (q_xyz / sin_half) * 2 * half_angle
    sinc = torch.sin(half_angle) / half_angle  # → 1 when half_angle → 0
    rot_vec = q[..., 1:] / sin_half * half_angle * sinc
    return rot_vec


class DeltaPoseLoss(nn.Module):
    """
    Direct supervision on the predicted pose delta (rot_vec, dt) per iteration.

    For each iteration i, the "ideal" delta that would correct the remaining error is:
        true_dq = inverse(current_pose_quat) * gt_quat
        true_dt = inverse(current_pose_R) @ (gt_trans - current_pose_trans)

    We then convert true_dq → true_rot_vec for L1 comparison with the raw model output.
    This gives the model a strong, direct learning signal instead of only end-pose supervision.

    Usage:
        delta_loss_fn = DeltaPoseLoss(rot_weight=1.0, trans_weight=1.0)
        loss, details = delta_loss_fn(delta_sequence, pose_sequence, gt_pose, gamma=0.8)
    """

    def __init__(self, rot_weight=1.0, trans_weight=1.0):
        super(DeltaPoseLoss, self).__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight

    def forward(self, delta_sequence, pose_sequence, gt_pose, gamma=0.8):
        """
        Args:
            delta_sequence: Dict with:
                'rot_vec': (B, K, 3) — predicted rotation vectors per iteration
                'dt': (B, K, 3) — predicted translation deltas per iteration
            pose_sequence: (B, K+1, 7) — poses at each step (index 0 = init_pose)
            gt_pose: (B, 7) — ground truth pose [w, x, y, z, tx, ty, tz]
            gamma: exponential weighting (same as sequence loss)

        Returns:
            total_loss: scalar
            details: dict with logging info
        """
        pred_rot_vecs = delta_sequence['rot_vec']  # (B, K, 3)
        pred_dts = delta_sequence['dt']             # (B, K, 3)
        B, K, _ = pred_rot_vecs.shape

        gt_quat = gt_pose[:, :4]    # (B, 4)
        gt_trans = gt_pose[:, 4:7]  # (B, 3)

        total_loss = torch.tensor(0.0, device=gt_pose.device)
        total_rot_loss = 0.0
        total_trans_loss = 0.0
        weight_sum = 0.0

        for i in range(K):
            weight = gamma ** (K - 1 - i)

            # Current pose at this iteration (before applying delta)
            cur_pose = pose_sequence[:, i, :]  # (B, 7)
            cur_quat = F.normalize(cur_pose[:, :4], dim=1)  # (B, 4)
            cur_trans = cur_pose[:, 4:7]  # (B, 3)

            # Compute true delta: T_delta = T_cur^{-1} * T_gt
            # Rotation: true_dq = conj(cur_quat) * gt_quat
            true_dq = quat_multiply(quat_conjugate(cur_quat), gt_quat)  # (B, 4)
            true_dq = F.normalize(true_dq, dim=1)

            # Convert true_dq to rotation vector for direct comparison
            true_rot_vec = quat_to_rotvec(true_dq)  # (B, 3)

            # Translation: true_dt = R_cur^{-1} @ (gt_trans - cur_trans)
            residual_trans = gt_trans - cur_trans  # (B, 3)
            true_dt = quat_rotate(quat_conjugate(cur_quat), residual_trans)  # (B, 3)

            # L1 loss on rotation vector
            rot_loss = F.l1_loss(pred_rot_vecs[:, i], true_rot_vec)
            # L1 loss on translation delta
            trans_loss = F.l1_loss(pred_dts[:, i], true_dt)

            total_loss = total_loss + weight * (self.rot_weight * rot_loss + self.trans_weight * trans_loss)
            total_rot_loss += weight * rot_loss.item()
            total_trans_loss += weight * trans_loss.item()
            weight_sum += weight

        total_rot_loss /= weight_sum
        total_trans_loss /= weight_sum

        details = {
            'delta_rot_loss': total_rot_loss,
            'delta_trans_loss': total_trans_loss,
            'delta_total_loss': (total_loss / weight_sum).item(),
        }

        return total_loss / weight_sum, details
