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
    Pose regression head that predicts 7D pose updates (quaternion + translation).
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
        
        # Output layer: 7 channels for pose (quaternion w,x,y,z + translation x,y,z)
        self.pose_conv = nn.Conv2d(hidden_dim, 7, 3, 1, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize pose regression head parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name and name != 'pose_conv.weight' and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name and name != 'pose_conv.bias':
                nn.init.zeros_(param)
        
        # Initialize output layer to predict small updates
        nn.init.xavier_normal_(self.pose_conv.weight)
        nn.init.zeros_(self.pose_conv.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input features of shape (B, hidden_dim, H, W)
        
        Returns:
            pose_delta: Pose update of shape (B, 7, H, W)
        """
        for block in self.conv_blocks:
            x = block(x)
        
        pose_delta = self.pose_conv(x)
        
        # Normalize quaternion to unit length (avoid inplace op for autograd)
        quat_delta = F.normalize(pose_delta[:, :4, :, :], dim=1)
        trans_delta = pose_delta[:, 4:, :, :]
        pose_delta = torch.cat([quat_delta, trans_delta], dim=1)
        
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
    
    def forward(self, corr_feat, context_feat, hidden_state=None):
        """
        Args:
            corr_feat: Correlation features of shape (B, corr_dim, H, W)
            context_feat: Context features from encoder of shape (B, context_dim, H, W)
            hidden_state: Previous hidden state of shape (B, hidden_dim, H, W), optional
        
        Returns:
            pose_delta: Predicted pose update of shape (B, 7, H, W)
            hidden_state: Updated hidden state of shape (B, hidden_dim, H, W)
        """
        # Project correlation features
        corr_proj = self.corr_proj(corr_feat)
        
        # Project context features
        context_proj = self.context_proj(context_feat)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_h(context_feat)
        
        # Update hidden state with ConvGRU
        hidden_state = self.gru(hidden_state, corr_proj + context_proj)
        
        # Predict pose update
        pose_delta = self.pose_head(hidden_state)
        
        return pose_delta, hidden_state
