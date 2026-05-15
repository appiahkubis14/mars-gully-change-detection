"""
test_model.py
=============
Unit tests for the Mars Gully U-Net and attention modules.

Run with: pytest tests/test_model.py -v
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class TestAttentionModules:

    def test_channel_attention_shape(self):
        """ChannelAttention output should have same shape as input."""
        from scripts.models.attention import ChannelAttention
        m = ChannelAttention(in_channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == x.shape

    def test_spatial_attention_shape(self):
        """SpatialAttention output should match input shape."""
        from scripts.models.attention import SpatialAttention
        m = SpatialAttention()
        x = torch.randn(2, 32, 64, 64)
        out = m(x)
        assert out.shape == x.shape

    def test_cbam_shape(self):
        """CBAM output should match input shape."""
        from scripts.models.attention import CBAM
        m = CBAM(in_channels=128)
        x = torch.randn(2, 128, 16, 16)
        out = m(x)
        assert out.shape == x.shape

    def test_attention_gate_shape(self):
        """AttentionGate output should match skip connection shape."""
        from scripts.models.attention import AttentionGate
        ag = AttentionGate(F_g=128, F_l=256, F_int=64)
        g = torch.randn(2, 128, 16, 16)
        x = torch.randn(2, 256, 16, 16)
        out = ag(g, x)
        assert out.shape == x.shape

    def test_cbam_values_bounded(self):
        """CBAM spatial weights are sigmoid-bounded → output ≤ input magnitude."""
        from scripts.models.attention import CBAM
        m = CBAM(in_channels=16)
        m.eval()
        x = torch.ones(1, 16, 8, 8)
        with torch.no_grad():
            out = m(x)
        # Attention weights are in (0,1], so output ≤ input
        assert (out.abs() <= x.abs() + 1e-4).all()


# ---------------------------------------------------------------------------
# U-Net forward pass
# ---------------------------------------------------------------------------

class TestUNet:

    def _build_cfg(self, in_channels=8):
        """Minimal config dict for build_model."""
        return {
            "training": {
                "in_channels": in_channels,
                "encoder_channels": [32, 64, 128, 256],
                "decoder_channels": [128, 64, 32],
                "num_classes": 1,
                "dropout": 0.1,
            }
        }

    def test_output_shape_matches_input(self):
        """U-Net output (H, W) should match input (H, W)."""
        from scripts.models.unet import build_model
        cfg = self._build_cfg()
        model = build_model(cfg).eval()
        x = torch.randn(2, 8, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 256, 256)

    def test_output_shape_non_square(self):
        """U-Net should handle non-square inputs."""
        from scripts.models.unet import build_model
        cfg = self._build_cfg()
        model = build_model(cfg).eval()
        x = torch.randn(1, 8, 256, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 256, 512)

    def test_output_shape_small_patch(self):
        """U-Net should work on 64×64 patches."""
        from scripts.models.unet import build_model
        cfg = self._build_cfg()
        model = build_model(cfg).eval()
        x = torch.randn(4, 8, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1, 64, 64)

    def test_no_nan_in_output(self):
        """U-Net forward pass should not produce NaN."""
        from scripts.models.unet import build_model
        cfg = self._build_cfg()
        model = build_model(cfg).eval()
        x = torch.randn(2, 8, 128, 128)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_grad_flows(self):
        """Backward pass should not raise errors."""
        from scripts.models.unet import build_model
        from scripts.models.losses import DiceLoss
        cfg = self._build_cfg()
        model = build_model(cfg).train()
        loss_fn = DiceLoss()
        x = torch.randn(2, 8, 128, 128, requires_grad=False)
        targets = (torch.rand(2, 1, 128, 128) > 0.85).float()
        logits = model(x)
        loss = loss_fn(logits, targets)
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_parameter_count_reasonable(self):
        """Model should have between 1 M and 50 M trainable parameters."""
        from scripts.models.unet import build_model
        cfg = self._build_cfg()
        model = build_model(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 1_000_000 < n_params < 50_000_000, f"Unexpected param count: {n_params:,}"

    def test_different_in_channels(self):
        """Model should work for in_channels = 4 and 12."""
        from scripts.models.unet import build_model
        for c in [4, 12]:
            cfg = self._build_cfg(in_channels=c)
            model = build_model(cfg).eval()
            x = torch.randn(1, c, 64, 64)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 1, 64, 64)

    def test_inference_mode_no_dropout(self):
        """In eval mode, two forward passes should give identical results."""
        from scripts.models.unet import build_model
        cfg = self._build_cfg()
        model = build_model(cfg).eval()
        x = torch.randn(1, 8, 64, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

class TestKalmanFilter:

    def test_smooth_monotone_series(self):
        """Kalman smoother should track a monotone increasing series."""
        from scripts.change.kalman_smooth import smooth_area_series
        times = [2005.0, 2008.0, 2011.0, 2014.0, 2017.0]
        areas = [1.0, 1.5, 2.0, 2.5, 3.0]
        result = smooth_area_series(times, areas)
        smoothed = result["smoothed_areas"]
        # smoothed should generally increase
        assert smoothed[-1] > smoothed[0]

    def test_smooth_length_matches_input(self):
        """Output length should match input length."""
        from scripts.change.kalman_smooth import smooth_area_series
        times = [2005.0, 2009.0, 2013.0]
        areas = [0.5, 0.8, 0.6]
        result = smooth_area_series(times, areas)
        assert len(result["smoothed_areas"]) == 3
        assert len(result["velocities"]) == 3

    def test_tracker_streaming(self):
        """KalmanAreaTracker should accept sequential updates."""
        from scripts.change.kalman_smooth import KalmanAreaTracker
        tracker = KalmanAreaTracker()
        for t, a in zip([2010.0, 2012.0, 2015.0], [1.0, 1.2, 1.8]):
            tracker.update(t, a)
        summary = tracker.summary()
        assert len(summary["smoothed_areas"]) == 3
        assert summary["total_gain_ha"] >= 0
