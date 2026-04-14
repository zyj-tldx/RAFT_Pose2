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
