"""
test_features.py
================
Unit tests for feature stack and texture/morphometric modules.

Run with: pytest tests/test_features.py -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Feature stack
# ---------------------------------------------------------------------------

class TestFeatureStack:

    def test_output_channels(self):
        """Feature stack should have the configured number of channels."""
        from scripts.features.feature_stack import build_feature_stack
        H, W = 64, 64
        hirise = np.random.rand(3, H, W).astype(np.float32)
        ctx = np.random.rand(1, H, W).astype(np.float32)
        dem = np.random.rand(1, H, W).astype(np.float32)
        stack = build_feature_stack(hirise=hirise, ctx=ctx, dem=dem)
        # expect ≥ 8 channels (3 HiRISE + indices + texture + CTX + slope)
        assert stack.shape[0] >= 8
        assert stack.shape[1] == H
        assert stack.shape[2] == W

    def test_no_nan_in_stack(self):
        """Feature stack should not contain NaN."""
        from scripts.features.feature_stack import build_feature_stack
        hirise = np.random.rand(3, 32, 32).astype(np.float32)
        ctx = np.random.rand(1, 32, 32).astype(np.float32)
        dem = np.random.rand(1, 32, 32).astype(np.float32)
        stack = build_feature_stack(hirise=hirise, ctx=ctx, dem=dem)
        assert np.isfinite(stack).all()


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

class TestTexture:

    def test_glcm_output_shape(self):
        """GLCM texture output should have (n_features, H, W) shape."""
        from scripts.features.texture import compute_glcm_texture
        band = np.random.rand(64, 64).astype(np.float32)
        out = compute_glcm_texture(band, window_size=7)
        assert out.ndim == 3
        assert out.shape[1] == 64
        assert out.shape[2] == 64

    def test_glcm_finite(self):
        """GLCM output should be finite."""
        from scripts.features.texture import compute_glcm_texture
        band = np.random.rand(32, 32).astype(np.float32)
        out = compute_glcm_texture(band, window_size=5)
        assert np.isfinite(out).all()

    def test_multi_scale_shape(self):
        """Multi-scale GLCM output should have shape (n_scales * n_feats, H, W)."""
        from scripts.features.texture import compute_multiscale_texture
        band = np.random.rand(64, 64).astype(np.float32)
        out = compute_multiscale_texture(band, window_sizes=[3, 5, 7])
        assert out.ndim == 3
        assert out.shape[1] == 64


# ---------------------------------------------------------------------------
# Blending weights
# ---------------------------------------------------------------------------

class TestBlending:

    def test_hanning_shape(self):
        """Hanning window should be square with correct size."""
        from scripts.inference.blend import hanning_weight_map
        w = hanning_weight_map(256)
        assert w.shape == (256, 256)

    def test_hanning_positive(self):
        """All hanning weights should be strictly positive (clipped from 0)."""
        from scripts.inference.blend import hanning_weight_map
        w = hanning_weight_map(128)
        assert (w > 0).all()

    def test_hanning_max_at_centre(self):
        """Centre pixel should have maximum weight."""
        from scripts.inference.blend import hanning_weight_map
        w = hanning_weight_map(64)
        mid = 32
        centre = w[mid, mid]
        assert centre == w.max()

    def test_gaussian_shape(self):
        """Gaussian window should have correct shape and be positive."""
        from scripts.inference.blend import gaussian_weight_map
        w = gaussian_weight_map(128)
        assert w.shape == (128, 128)
        assert (w > 0).all()


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestLosses:

    def _dummy_batch(self, h=32, w=32, bs=4):
        import torch
        logits = torch.randn(bs, 1, h, w)
        targets = (torch.rand(bs, 1, h, w) > 0.8).float()
        return logits, targets

    def test_focal_loss_scalar(self):
        """FocalLoss forward should return a scalar tensor."""
        import torch
        from scripts.models.losses import FocalLoss
        loss_fn = FocalLoss()
        logits, targets = self._dummy_batch()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_dice_loss_scalar(self):
        """DiceLoss forward should return a scalar in (0, 1]."""
        from scripts.models.losses import DiceLoss
        loss_fn = DiceLoss()
        logits, targets = self._dummy_batch()
        loss = loss_fn(logits, targets)
        assert 0.0 < loss.item() <= 1.01

    def test_combined_loss_positive(self):
        """CombinedLoss should return a positive scalar."""
        from scripts.models.losses import CombinedLoss
        loss_fn = CombinedLoss(dice_weight=0.5)
        logits, targets = self._dummy_batch()
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_tversky_loss_scalar(self):
        """TverskyLoss forward should return a scalar."""
        from scripts.models.losses import TverskyLoss
        loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
        logits, targets = self._dummy_batch()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_perfect_prediction_low_loss(self):
        """Loss should be near 0 for perfect predictions."""
        import torch
        from scripts.models.losses import DiceLoss
        loss_fn = DiceLoss()
        targets = torch.ones(2, 1, 16, 16)
        logits = torch.full((2, 1, 16, 16), 10.0)   # very high → sigmoid ≈ 1
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.05


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_perfect_iou(self):
        """Identical prediction and target should give IoU ≈ 1."""
        from scripts.validation.metrics import pixel_metrics
        target = np.ones(100, dtype=np.float32)
        probs = np.ones(100, dtype=np.float32)
        m = pixel_metrics(probs, target, threshold=0.5)
        assert m["iou"] > 0.99

    def test_all_wrong_iou(self):
        """Completely wrong prediction should give IoU ≈ 0."""
        from scripts.validation.metrics import pixel_metrics
        target = np.ones(100, dtype=np.float32)
        probs = np.zeros(100, dtype=np.float32)
        m = pixel_metrics(probs, target, threshold=0.5)
        assert m["iou"] < 0.01

    def test_roc_auc_range(self):
        """ROC AUC should be in [0, 1]."""
        from scripts.validation.metrics import roc_curve_data
        probs = np.random.rand(1000).astype(np.float32)
        targets = (np.random.rand(1000) > 0.7).astype(np.float32)
        roc = roc_curve_data(probs, targets, n_thresholds=20)
        assert 0.0 <= roc["auc"] <= 1.0

    def test_pr_ap_range(self):
        """Average Precision should be in [0, 1]."""
        from scripts.validation.metrics import pr_curve_data
        probs = np.random.rand(1000).astype(np.float32)
        targets = (np.random.rand(1000) > 0.8).astype(np.float32)
        pr = pr_curve_data(probs, targets, n_thresholds=20)
        assert 0.0 <= pr["ap"] <= 1.0
