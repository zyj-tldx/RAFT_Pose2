"""
Feature extractors for RAFT-Pose module.
Extracts features from images and depth maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, planes_mid=None, norm_fn='instance', stride=1):
        super(ResidualBlock, self).__init__()
        planes_mid = planes if planes_mid is None else planes_mid

        self.conv1 = nn.Conv2d(in_planes, planes_mid, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes_mid, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8 if planes >= 8 else 1

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes_mid)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes_mid)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes_mid)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    """
    Basic CNN feature encoder with additional refinement layers.

    Architecture:
        conv1 (7x7, stride=2)  → 64ch
        layer1 (2x ResBlock, stride=1)  → output_dim
        layer2 (2x ResBlock, stride=2)  → output_dim
        layer3 (2x ResBlock, stride=2)  → output_dim
        layer4 (2x ResBlock, stride=1)  → output_dim   ← new
        layer5 (2x ResBlock, stride=1)  → output_dim   ← new
        conv_out (1x1)                   → output_dim
    """
    def __init__(self, output_dim=256, norm_fn='instance', dropout=0.0, in_feat=3,
                 use_checkpoint=False):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv2d(in_feat, 64, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64) if norm_fn == 'instance' else nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(output_dim, stride=1)
        self.layer2 = self._make_layer(output_dim, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        # Additional refinement layers (no downsampling)
        self.layer4 = self._make_layer(output_dim, stride=1)
        self.layer5 = self._make_layer(output_dim, stride=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, planes_mid=dim, stride=stride, norm_fn=self.norm_fn)
        layer2 = ResidualBlock(dim, dim, planes_mid=dim, stride=1, norm_fn=self.norm_fn)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.norm1(self.conv1(x)))
        if self.use_checkpoint:
            out = torch_checkpoint(self.layer1, out, use_reentrant=False)
            out = torch_checkpoint(self.layer2, out, use_reentrant=False)
            out = torch_checkpoint(self.layer3, out, use_reentrant=False)
            out = torch_checkpoint(self.layer4, out, use_reentrant=False)
            out = torch_checkpoint(self.layer5, out, use_reentrant=False)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        return out


class SmallEncoder(nn.Module):
    """
    Smaller CNN feature encoder for efficiency.
    """
    def __init__(self, output_dim=128, norm_fn='instance', dropout=0.0, in_feat=3,
                 use_checkpoint=False):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv2d(in_feat, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=32) if norm_fn == 'instance' else nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(output_dim, stride=1)
        self.layer2 = self._make_layer(output_dim, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, planes_mid=dim // 2, stride=stride, norm_fn=self.norm_fn)
        layer2 = ResidualBlock(dim, dim, planes_mid=dim // 2, stride=1, norm_fn=self.norm_fn)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.norm1(self.conv1(x)))
        if self.use_checkpoint:
            out = torch_checkpoint(self.layer1, out, use_reentrant=False)
            out = torch_checkpoint(self.layer2, out, use_reentrant=False)
            out = torch_checkpoint(self.layer3, out, use_reentrant=False)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        return out


class DepthEncoder(nn.Module):
    """
    Encoder for depth maps.
    Can include Fourier features for better geometric representation.
    """
    def __init__(self, output_dim=128, norm_fn='instance', dropout=0.0, fourier_levels=-1,
                 use_checkpoint=False):
        super(DepthEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.fourier_levels = fourier_levels
        self.use_checkpoint = use_checkpoint
        
        # Determine input channels
        self.in_feat = 1
        if fourier_levels >= 0:
            self.in_feat += 2 * fourier_levels  # Fourier features
        
        self.conv1 = nn.Conv2d(self.in_feat, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=32) if norm_fn == 'instance' else nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(output_dim, stride=1)
        self.layer2 = self._make_layer(output_dim, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, planes_mid=dim // 2, stride=stride, norm_fn=self.norm_fn)
        layer2 = ResidualBlock(dim, dim, planes_mid=dim // 2, stride=1, norm_fn=self.norm_fn)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def _add_fourier_features(self, depth):
        """
        Add Fourier positional encoding to depth map.
        """
        B, C, H, W = depth.shape
        device = depth.device
        
        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        yy = yy.view(1, 1, H, W).expand(B, -1, -1, -1)
        xx = xx.view(1, 1, H, W).expand(B, -1, -1, -1)
        
        # Compute Fourier features
        fourier_feat = []
        for level in range(1, self.fourier_levels + 1):
            freq = 2.0 ** level
            fourier_feat.append(torch.sin(freq * yy))
            fourier_feat.append(torch.sin(freq * xx))
        
        fourier_feat = torch.cat(fourier_feat, dim=1)  # (B, 2*L, H, W)
        
        # Concatenate with depth
        return torch.cat([depth, fourier_feat], dim=1)

    def forward(self, x):
        if self.fourier_levels >= 0:
            x = self._add_fourier_features(x)
        
        out = self.relu1(self.norm1(self.conv1(x)))
        if self.use_checkpoint:
            out = torch_checkpoint(self.layer1, out, use_reentrant=False)
            out = torch_checkpoint(self.layer2, out, use_reentrant=False)
            out = torch_checkpoint(self.layer3, out, use_reentrant=False)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        return out
