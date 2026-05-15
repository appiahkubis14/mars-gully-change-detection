"""
test_preprocess.py
==================
Unit tests for preprocessing modules.

Run with: pytest tests/test_preprocess.py -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_output_range(self):
        """Output should be clipped to [0, 1]."""
        from scripts.preprocess.normalize import percentile_normalize
        arr = np.random.randn(3, 64, 64).astype(np.float32) * 1000
        out = percentile_normalize(arr)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        from scripts.preprocess.normalize import percentile_normalize
        arr = np.ones((4, 32, 32), dtype=np.float32) * 500
        out = percentile_normalize(arr)
        assert out.shape == arr.shape

    def test_constant_band_no_nan(self):
        """Constant band (all same value) should not produce NaN."""
        from scripts.preprocess.normalize import percentile_normalize
        arr = np.full((2, 16, 16), 42.0, dtype=np.float32)
        out = percentile_normalize(arr)
        assert not np.isnan(out).any()

    def test_nan_handling(self):
        """NaN inputs should produce finite outputs."""
        from scripts.preprocess.normalize import percentile_normalize
        arr = np.random.rand(2, 32, 32).astype(np.float32)
        arr[0, 0, 0] = np.nan
        out = percentile_normalize(arr)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class TestComposite:

    def test_median_correct(self):
        """Pixel-wise median of 3 identical arrays should equal the input."""
        from scripts.preprocess.composite import pixel_median_composite
        a = np.ones((2, 16, 16), dtype=np.float32) * 3.0
        b = np.ones((2, 16, 16), dtype=np.float32) * 3.0
        c = np.ones((2, 16, 16), dtype=np.float32) * 3.0
        out = pixel_median_composite([a, b, c])
        np.testing.assert_array_almost_equal(out, a)

    def test_output_shape(self):
        """Output shape should match input shape."""
        from scripts.preprocess.composite import pixel_median_composite
        arrays = [np.random.rand(3, 32, 32).astype(np.float32) for _ in range(5)]
        out = pixel_median_composite(arrays)
        assert out.shape == (3, 32, 32)

    def test_single_input(self):
        """Single input should pass through unchanged."""
        from scripts.preprocess.composite import pixel_median_composite
        arr = np.random.rand(2, 8, 8).astype(np.float32)
        out = pixel_median_composite([arr])
        np.testing.assert_array_almost_equal(out, arr)


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

class TestSpectralIndices:

    def _make_band_dict(self):
        """Create a dummy band dictionary."""
        rng = np.random.RandomState(42)
        return {
            "RED": rng.rand(16, 16).astype(np.float32) + 0.1,
            "NIR": rng.rand(16, 16).astype(np.float32) + 0.1,
            "SWIR": rng.rand(16, 16).astype(np.float32) + 0.1,
        }

    def test_rsi_range(self):
        """RSI (Redness Spectral Index) should be finite."""
        from scripts.features.spectral_indices import compute_rsi
        bands = self._make_band_dict()
        out = compute_rsi(bands["RED"], bands["NIR"])
        assert np.isfinite(out).all()

    def test_no_nan_in_indices(self):
        """compute_all_indices should return finite arrays."""
        from scripts.features.spectral_indices import compute_all_indices
        bands = self._make_band_dict()
        indices = compute_all_indices(bands)
        for name, arr in indices.items():
            assert np.isfinite(arr).all(), f"{name} contains non-finite values"

    def test_output_shape_matches_input(self):
        """All spectral index outputs should have (H, W) shape."""
        from scripts.features.spectral_indices import compute_all_indices
        bands = self._make_band_dict()
        indices = compute_all_indices(bands)
        for name, arr in indices.items():
            assert arr.shape == (16, 16), f"{name} has unexpected shape {arr.shape}"


# ---------------------------------------------------------------------------
# Morphometrics
# ---------------------------------------------------------------------------

class TestMorphometrics:

    def test_slope_non_negative(self):
        """Slope should be ≥ 0 everywhere."""
        from scripts.features.morphometric import compute_slope
        dem = np.random.rand(32, 32).astype(np.float32) * 1000
        slope = compute_slope(dem, resolution=463.0)
        assert (slope >= 0).all()

    def test_aspect_range(self):
        """Aspect should be in [0, 360)."""
        from scripts.features.morphometric import compute_aspect
        dem = np.random.rand(32, 32).astype(np.float32) * 1000
        aspect = compute_aspect(dem, resolution=463.0)
        assert aspect.min() >= 0.0
        assert aspect.max() < 360.0

    def test_flat_dem_zero_slope(self):
        """Flat DEM should have zero slope everywhere."""
        from scripts.features.morphometric import compute_slope
        dem = np.full((16, 16), 500.0, dtype=np.float32)
        slope = compute_slope(dem, resolution=463.0)
        np.testing.assert_array_almost_equal(slope, 0.0, decimal=3)
