"""
Pose update network with ConvGRU for iterative pose refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRU(nn.Module):
    """
    Convolutional Gated Recurrent Unit for iterative refinement.
    """
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        """
        Args:
            hidden_dim: Number of hidden channels
            input_dim: Number of input channels
            kernel_size: Size of convolution kernel
        """
        super(ConvGRU, self).__init__()
        padding = kernel_size // 2
        
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize ConvGRU parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, h, x):
        """
        Args:
            h: Hidden state of shape (B, hidden_dim, H, W)
            x: Input features of shape (B, input_dim, H, W)
        
        Returns:
            Updated hidden state of shape (B, hidden_dim, H, W)
        """
        # Concatenate hidden state and input
        hx = torch.cat([h, x], dim=1)
        
        # Update gate
        z = torch.sigmoid(self.convz(hx))
        
        # Reset gate
        r = torch.sigmoid(self.convr(hx))
        
        # Candidate hidden state
        hr = torch.cat([r * h, x], dim=1)
        q = torch.tanh(self.convq(hr))
        
        # Update hidden state
        h_new = (1 - z) * h + z * q
        
        return h_new


class ResidualBlock(nn.Module):
    """
    Residual block for feature refinement.
    """
    def __init__(self, in_dim, out_dim, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        
        self.downsample = None
        if stride != 1 or in_dim != out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 1, stride, 0),
                nn.BatchNorm2d(out_dim)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out += residual
        out = self.relu(out)
        
        return out


class PoseRegressionHead(nn.Module):
    """
    Pose regression head that predicts 6D pose updates (rotation vector + translation).

    Outputs a 6D vector: [rx, ry, rz, tx, ty, tz]
    - (rx, ry, rz): rotation vector (axis * angle), converted to quaternion delta
    - (tx, ty, tz): translation delta in meters

    With zero initialization (bias=0), the output is all zeros, which maps to
    identity quaternion delta + zero translation — i.e. no pose change.
    This ensures the untrained model does not corrupt the initial pose.
    """
    def __init__(self, hidden_dim, num_layers=3):
        """
        Args:
            hidden_dim: Number of hidden channels
            num_layers: Number of residual blocks
        """
        super(PoseRegressionHead, self).__init__()
        
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.conv_blocks.append(ResidualBlock(in_dim, out_dim))
        
        # Output layer: 6 channels (3 rotation + 3 translation)
        self.pose_conv = nn.Conv2d(hidden_dim, 6, 3, 1, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize pose regression head parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name and name != 'pose_conv.weight' and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name and name != 'pose_conv.bias':
                nn.init.zeros_(param)
        
        # Zero initialization: output = 0 → identity delta (no pose change)
        nn.init.zeros_(self.pose_conv.weight)
        nn.init.zeros_(self.pose_conv.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input features of shape (B, hidden_dim, H, W)
        
        Returns:
            pose_delta: Raw 6D pose update of shape (B, 6, H, W)
                       [rx, ry, rz, tx, ty, tz]
        """
        for block in self.conv_blocks:
            x = block(x)
        
        pose_delta = self.pose_conv(x)  # (B, 6, H, W)
        return pose_delta


class PoseUpdateNet(nn.Module):
    """
    Pose update network with ConvGRU for iterative refinement.

    corr_dim is now the per-sample correlation dimension (C_corr), not N*C_corr.
    The network processes one sample at a time; the caller handles multi-sample
    selection via confidence scores.
    """
    def __init__(self, hidden_dim=128, corr_dim=256, context_dim=64, num_layers=3):
        """
        Args:
            hidden_dim: Dimension of ConvGRU hidden state
            corr_dim: Dimension of per-sample correlation features (C_corr)
            context_dim: Dimension of context features from encoder
            num_layers: Number of residual blocks in pose regression head
        """
        super(PoseUpdateNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.corr_dim = corr_dim
        self.context_dim = context_dim
        
        # Project correlation features to hidden dimension
        self.corr_proj = nn.Sequential(
            nn.Conv2d(corr_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Project context features to hidden dimension
        self.context_proj = nn.Sequential(
            nn.Conv2d(context_dim, hidden_dim, 1),
            nn.ReLU(inplace=True)
        )
        
        # Project 6D direction encoding to hidden dimension
        # Direction encoding: [rot_axis(3), trans_dir(3)] tells the network
        # which sampling direction produced the correlation features
        self.dir_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # ConvGRU for iterative refinement
        self.gru = ConvGRU(hidden_dim, hidden_dim, kernel_size=3)
        
        # Initialize hidden state
        self.init_h = nn.Sequential(
            nn.Conv2d(context_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Pose regression head
        self.pose_head = PoseRegressionHead(hidden_dim, num_layers=num_layers)
    
    def forward(self, corr_feat, context_feat, hidden_state=None, direction_encoding=None):
        """
        Args:
            corr_feat: Correlation features of shape (B, corr_dim, H, W)
            context_feat: Context features from encoder of shape (B, context_dim, H, W)
            hidden_state: Previous hidden state of shape (B, hidden_dim, H, W), optional
            direction_encoding: Direction encoding of shape (B, 6), optional.
                [rot_axis_x, rot_axis_y, rot_axis_z, trans_dir_x, trans_dir_y, trans_dir_z]
                Tells the network which sampling direction produced the correlation features.
        
        Returns:
            pose_delta: Predicted pose update of shape (B, 6, H, W)
            hidden_state: Updated hidden state of shape (B, hidden_dim, H, W)
        """
        # Project correlation features
        corr_proj = self.corr_proj(corr_feat)
        
        # Project context features
        context_proj = self.context_proj(context_feat)
        
        # Inject direction encoding if provided
        # Project 6D direction vector to hidden_dim and broadcast spatially
        if direction_encoding is not None:
            B = direction_encoding.shape[0]
            H, W = corr_proj.shape[2], corr_proj.shape[3]
            # (B, 6) -> (B, hidden_dim) -> (B, hidden_dim, 1, 1) -> broadcast
            dir_feat = self.dir_proj(direction_encoding)  # (B, hidden_dim)
            dir_feat = dir_feat.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_dim, 1, 1)
            dir_feat = dir_feat.expand(-1, -1, H, W)  # (B, hidden_dim, H, W)
        else:
            dir_feat = 0
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_h(context_feat)
        
        # Update hidden state with ConvGRU
        hidden_state = self.gru(hidden_state, corr_proj + context_proj + dir_feat)
        
        # Predict pose update
        pose_delta = self.pose_head(hidden_state)
        
        return pose_delta, hidden_state
