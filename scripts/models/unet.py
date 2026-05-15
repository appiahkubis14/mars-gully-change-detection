"""
U-Net with ResNet34/50 Encoder and Attention Gates
No pretrained weights — Mars data is far from ImageNet.

Architecture:
  Encoder: ResNet34 (default) — 4 downsampling blocks
  Decoder: Bilinear upsampling + skip connections
  Attention: Spatial attention gate at each decoder level
  Output: Single-channel sigmoid probability map

Input:  (B, C, H, W)  C = in_channels (default 8)
Output: (B, 1, H, W)  sigmoid probability [0, 1]
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("model.unet")


# ─── Building blocks ──────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Standard residual block for the encoder."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ─── Attention Gate ───────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate (Schlemper et al. 2019, MedIA).
    Learns to focus on relevant features (gully pixels).

    g  : gating signal from decoder (coarser)
    x  : skip connection from encoder (finer)
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=True),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g may be smaller than x — upsample g
        g_up = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=True)
        g1 = self.W_g(g_up)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # attended features


# ─── Encoder ──────────────────────────────────────────────────────────────────

class MarsResNetEncoder(nn.Module):
    """
    ResNet-like encoder (34-layer style) without pretrained weights.
    Outputs feature maps at 4 scales.
    """

    def __init__(self, in_channels: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # /2

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)  # /4

        self.layer1 = self._make_layer(64, 64, n_blocks=3)            # /4
        self.layer2 = self._make_layer(64, 128, n_blocks=4, stride=2) # /8
        self.layer3 = self._make_layer(128, 256, n_blocks=6, stride=2)# /16
        self.layer4 = self._make_layer(256, 512, n_blocks=3, stride=2)# /32

        self.out_channels = [64, 64, 128, 256, 512]

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns feature maps at increasing depth: [e0, e1, e2, e3, e4]"""
        e0 = self.stem(x)       # C=64,  H/2
        e1 = self.layer1(self.pool(e0))  # C=64,  H/4
        e2 = self.layer2(e1)    # C=128, H/8
        e3 = self.layer3(e2)    # C=256, H/16
        e4 = self.layer4(e3)    # C=512, H/32
        return [e0, e1, e2, e3, e4]


# ─── Decoder Block ────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(g_ch=in_ch, x_ch=skip_ch, inter_ch=skip_ch // 2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        if self.use_attention:
            skip = self.attn(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── U-Net ────────────────────────────────────────────────────────────────────

class MarsGullyUNet(nn.Module):
    """
    Mars Gully Detection U-Net.

    Parameters
    ----------
    in_channels    : number of input feature channels (from feature stack)
    out_channels   : 1 (binary segmentation)
    use_attention  : whether to use attention gates in decoder
    dropout        : dropout probability in bottleneck
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        use_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = MarsResNetEncoder(in_channels)

        # [64, 64, 128, 256, 512]
        enc_ch = self.encoder.out_channels

        self.bottleneck = nn.Sequential(
            ConvBnRelu(enc_ch[4], 512),
            nn.Dropout2d(dropout),
            ConvBnRelu(512, 512)
        )

        self.dec4 = DecoderBlock(512, enc_ch[3], 256, use_attention)
        self.dec3 = DecoderBlock(256, enc_ch[2], 128, use_attention)
        self.dec2 = DecoderBlock(128, enc_ch[1], 64,  use_attention)
        self.dec1 = DecoderBlock(64,  enc_ch[0], 32,  use_attention)

        # Final upsampling to input resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBnRelu(32, 16),
            nn.Conv2d(16, out_channels, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0, e1, e2, e3, e4 = self.encoder(x)

        b = self.bottleneck(e4)

        d4 = self.dec4(b, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        out = self.final_up(d1)
        return torch.sigmoid(out)


def build_model(cfg: Dict) -> MarsGullyUNet:
    """Build model from config."""
    train_cfg = cfg.get("training", {})
    model = MarsGullyUNet(
        in_channels=train_cfg.get("in_channels", 8),
        out_channels=train_cfg.get("out_channels", 1),
        use_attention=True,
        dropout=0.1
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {n_params/1e6:.2f}M parameters")
    return model


if __name__ == "__main__":
    # Quick shape test
    model = MarsGullyUNet(in_channels=8)
    x = torch.randn(2, 8, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (2, 1, 512, 512), f"Unexpected output shape: {y.shape}"
    print("✓ U-Net shape test passed")
