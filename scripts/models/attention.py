"""
attention.py
============
Attention modules for the Mars Gully U-Net.
Includes:
  - ChannelAttention  (SE-Net style)
  - SpatialAttention
  - CBAM              (Convolutional Block Attention Module, Woo et al. 2018)
  - AttentionGate     (Schlemper et al. 2019, used in decoder skip connections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Channel Attention (Squeeze-and-Excitation)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Args:
        in_channels (int): Number of input feature channels.
        reduction   (int): Reduction ratio for the bottleneck FC layers.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ---------------------------------------------------------------------------
# Spatial Attention
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """
    Spatial attention via channel-wise average + max pooling followed by a
    7×7 convolution.

    Args:
        kernel_size (int): Convolution kernel size (7 recommended).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale


# ---------------------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------------------

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Applies channel attention then spatial attention sequentially.

    Args:
        in_channels   (int): Number of input feature channels.
        reduction     (int): Channel attention reduction ratio.
        spatial_kernel(int): Spatial attention kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------------
# Attention Gate (decoder skip-connection gating)
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Attention Gate as in Schlemper et al. (2019) "Attention U-Net".

    Computes a soft attention map from the gating signal (decoder) and the
    skip connection (encoder) to suppress irrelevant activations.

    Args:
        F_g (int): Channels in the gating signal (from decoder).
        F_l (int): Channels in the skip connection (from encoder).
        F_int (int): Number of intermediate channels.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, g: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder  (B, F_g, H, W)
            x: Skip connection from encoder (B, F_l, H, W)

        Returns:
            Attention-weighted skip connection (B, F_l, H, W)
        """
        # Upsample gating signal to match skip spatial dimensions
        g_up = F.interpolate(
            self.W_g(g), size=x.shape[2:], mode="bilinear", align_corners=False
        )
        x_proj = self.W_x(x)
        combined = self.relu(g_up + x_proj)
        alpha = self.psi(combined)           # (B, 1, H, W)
        return x * alpha


# ---------------------------------------------------------------------------
# Self-Attention (lightweight non-local block) — optional
# ---------------------------------------------------------------------------

class NonLocalBlock(nn.Module):
    """
    Lightweight non-local self-attention block (Wang et al. 2018).
    Uses embedded Gaussian formulation with 1×1 convolutions.

    Args:
        in_channels (int): Input channels.
        sub_sample  (bool): Whether to sub-sample key/value by 2× pooling.
    """

    def __init__(self, in_channels: int, sub_sample: bool = True):
        super().__init__()
        inter = max(in_channels // 2, 1)
        self.theta = nn.Conv2d(in_channels, inter, 1)
        self.phi = nn.Conv2d(in_channels, inter, 1)
        self.g = nn.Conv2d(in_channels, inter, 1)
        self.W = nn.Sequential(
            nn.Conv2d(inter, in_channels, 1),
            nn.BatchNorm2d(in_channels),
        )
        self.sub_sample = sub_sample
        if sub_sample:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//2)

        phi_x = self.pool(x) if self.sub_sample else x
        g_x = self.pool(x) if self.sub_sample else x
        HW2 = phi_x.shape[2] * phi_x.shape[3]

        phi = self.phi(phi_x).view(B, -1, HW2)          # (B, C//2, HW2)
        g = self.g(g_x).view(B, -1, HW2).permute(0, 2, 1)  # (B, HW2, C//2)

        attn = torch.softmax(torch.bmm(theta, phi), dim=-1)  # (B, HW, HW2)
        out = torch.bmm(attn, g).permute(0, 2, 1).view(B, -1, H, W)
        return x + self.W(out)
