"""
test_downloaders.py
===================
Unit tests for data downloader modules.

Run with: pytest tests/test_downloaders.py -v
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import tempfile

# ---------------------------------------------------------------------------
# HiRISE downloader
# ---------------------------------------------------------------------------

class TestHiRISEDownloader(unittest.TestCase):

    def test_obs_id_to_url(self):
        """obs_id_to_url should construct the correct PDS URL."""
        from scripts.downloaders.download_hirise import obs_id_to_url
        url = obs_id_to_url("ESP_012345_1385")
        self.assertIn("ESP_012345_1385", url)
        self.assertIn("hirise-pds.lpl.arizona.edu", url)

    def test_obs_id_to_url_format(self):
        """URL should end with .JP2 or a directory path."""
        from scripts.downloaders.download_hirise import obs_id_to_url
        url = obs_id_to_url("PSP_001234_1385")
        self.assertIsInstance(url, str)
        self.assertTrue(url.startswith("https://"))

    @patch("scripts.downloaders.download_hirise.requests")
    def test_download_skips_existing(self, mock_requests):
        """Downloader should skip files that already exist."""
        from scripts.downloaders.download_hirise import download_hirise_file
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "test.JP2"
            dest.touch()   # create an existing file
            result = download_hirise_file("https://example.com/test.JP2", dest)
            mock_requests.get.assert_not_called()


# ---------------------------------------------------------------------------
# CTX downloader
# ---------------------------------------------------------------------------

class TestCTXDownloader(unittest.TestCase):

    def test_ode_url_construction(self):
        """ODE API URL should include bbox parameters."""
        from scripts.downloaders.download_ctx import build_ode_query_url
        url = build_ode_query_url(
            target="Mars", bbox=(-50.0, 125.0, -45.0, 135.0)
        )
        self.assertIn("Mars", url)
        self.assertIn("125.0", url)

    @patch("scripts.downloaders.download_ctx.requests.get")
    def test_empty_ode_response(self, mock_get):
        """Empty ODE response should return an empty list."""
        from scripts.downloaders.download_ctx import parse_ode_response
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ODEResults": {"Products": []}}
        result = parse_ode_response(mock_resp)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# MOLA downloader
# ---------------------------------------------------------------------------

class TestMOLADownloader(unittest.TestCase):

    def test_mola_url_is_string(self):
        """MOLA global mosaic URL should be a non-empty string."""
        from scripts.downloaders.download_mola import MOLA_GLOBAL_URL
        self.assertIsInstance(MOLA_GLOBAL_URL, str)
        self.assertTrue(MOLA_GLOBAL_URL.startswith("https://"))

    def test_bbox_subset_args(self):
        """bbox_to_gdal_args should return a list of strings."""
        from scripts.downloaders.download_mola import bbox_to_gdal_args
        args = bbox_to_gdal_args(west=-50.0, south=125.0, east=-45.0, north=135.0)
        self.assertIsInstance(args, list)
        self.assertTrue(all(isinstance(a, str) for a in args))


# ---------------------------------------------------------------------------
# StepCheckpoint (used by all downloaders)
# ---------------------------------------------------------------------------

class TestStepCheckpoint(unittest.TestCase):

    def test_mark_and_check(self):
        """Marking a step done should make is_done return True."""
        from scripts.utils import StepCheckpoint
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            ckpt_path = f.name
        try:
            ckpt = StepCheckpoint(ckpt_path)
            self.assertFalse(ckpt.is_done("step_a"))
            ckpt.mark_done("step_a")
            ckpt2 = StepCheckpoint(ckpt_path)  # reload from disk
            self.assertTrue(ckpt2.is_done("step_a"))
        finally:
            os.unlink(ckpt_path)

    def test_reset(self):
        """Reset should clear all completed steps."""
        from scripts.utils import StepCheckpoint
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            ckpt_path = f.name
        try:
            ckpt = StepCheckpoint(ckpt_path)
            ckpt.mark_done("step_x")
            ckpt.reset()
            self.assertFalse(ckpt.is_done("step_x"))
        finally:
            os.unlink(ckpt_path)


if __name__ == "__main__":
    unittest.main()
