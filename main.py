#!/usr/bin/env python3
"""
main.py  —  Mars Gully Digital Twin pipeline orchestrator.

Usage
-----
    python main.py --step all
    python main.py --step download
    python main.py --step preprocess
    python main.py --step features
    python main.py --step labels
    python main.py --step train [--resume last|best|<path>]
    python main.py --step infer
    python main.py --step threshold
    python main.py --step change
    python main.py --step export
    python main.py --step dashboard
    python main.py --step validate [--temporal]
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO",
                  log_file: str = "data/outputs/pipeline.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    # UTF-8 safe stdout for Windows
    import io
    if sys.platform == "win32":
        try:
            safe_out = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8",
                errors="replace", line_buffering=True)
        except AttributeError:
            safe_out = sys.stdout
    else:
        safe_out = sys.stdout
    handlers = [
        logging.StreamHandler(safe_out),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt, handlers=handlers,
    )


log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _load_cfg(cfg_path: str) -> dict:
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _ckpt(subdir: str = "data/models"):
    from scripts.utils import StepCheckpoint
    return StepCheckpoint(subdir)


def _name_to_short(name: str) -> str:
    """Convert site name to short key: 'Gasa Crater' -> 'gasa'"""
    return name.lower().split()[0]


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _discover_hirise(base: Path) -> Dict[str, Path]:
    """
    Return {obs_id: path} for all downloaded HiRISE files.
    Finds any usable image (tif preferred over JP2) > 1 MB.
    Priority: RED.tif > RED.JP2 > COLOR.tif > COLOR.JP2 > MRGB.JP2
    """
    result: Dict[str, Path] = {}
    if not base.exists():
        return result
    for obs_dir in sorted(base.iterdir()):
        if not obs_dir.is_dir():
            continue
        obs_id = obs_dir.name
        chosen = None
        for suffix in ["RED.tif", "RED.JP2", "COLOR.tif", "COLOR.JP2",
                       "MRGB.tif", "MRGB.JP2"]:
            p = obs_dir / f"{obs_id}_{suffix}"
            if p.exists() and p.stat().st_size > 1_000_000:
                chosen = p
                break
        if chosen:
            result[obs_id] = chosen
    return result


def _discover_ctx(base: Path) -> Dict[str, Path]:
    """
    Return {site_name: ctx_tif} for each site.
    Auto-converts .IMG files to GeoTIFF if .tif not already present.
    """
    result: Dict[str, Path] = {}
    if not base.exists():
        return result
    for site_dir in sorted(base.iterdir()):
        if not site_dir.is_dir() or site_dir.name.startswith("."):
            continue
        site = site_dir.name

        # Convert any IMG files that don't have a .tif yet
        for img_path in sorted(site_dir.glob("*.IMG")):
            tif_path = img_path.with_suffix(".tif")
            if not tif_path.exists():
                try:
                    from scripts.downloaders.download_ctx import convert_img_to_geotiff
                    convert_img_to_geotiff(img_path, tif_path)
                except Exception as e:
                    log.warning(f"Could not convert {img_path.name}: {e}")

        # Pick the first non-empty .tif (simple size check, no rasterio open)
        tifs = sorted(site_dir.glob("*.tif"))
        chosen_tif = None
        for tif in tifs:
            if tif.stat().st_size > 10_000:  # > 10 KB = not empty/corrupt
                chosen_tif = tif
                break
        if chosen_tif:
            result[site] = chosen_tif
            log.info(f"  CTX for {site}: {chosen_tif.name} "
                     f"({chosen_tif.stat().st_size/1e6:.1f} MB)")
        else:
            log.warning(f"No valid CTX tif in {site_dir} "
                        f"(found {len(tifs)} tif files)")
    return result


def _discover_mola(base: Path) -> Dict[str, Path]:
    """Return {site_name: mola_subset_tif}."""
    subset_dir = base / "subsets"
    result: Dict[str, Path] = {}
    if not subset_dir.exists():
        return result
    for p in sorted(subset_dir.glob("*.tif")):
        # Filename: Gasa_Crater_mola.tif -> key: gasa
        key = p.stem.replace("_mola", "").replace("_Crater", "").lower()
        result[key] = p
    return result


def _site_keys(config: dict) -> List[str]:
    return [k for k, v in config.get("study_area", {}).items()
            if isinstance(v, dict)]


# ---------------------------------------------------------------------------
# Step: download
# ---------------------------------------------------------------------------

def run_download(cfg: str, **kwargs) -> None:
    from scripts.downloaders.download_hirise import download_hirise_all
    from scripts.downloaders.download_ctx    import download_ctx_all
    from scripts.downloaders.download_mola   import download_mola_all

    log.info("=== STEP: download ===")
    config = _load_cfg(cfg)
    ckpt = _ckpt()

    download_hirise_all(config, Path("data/raw/hirise"), ckpt)
    download_ctx_all   (config, Path("data/raw/ctx"),    ckpt)
    download_mola_all  (config, Path("data/raw/mola"),   ckpt)


# ---------------------------------------------------------------------------
# Step: preprocess
# ---------------------------------------------------------------------------

def run_preprocess(cfg: str, **kwargs) -> None:
    from scripts.preprocess.coregister import coregister_site
    from scripts.preprocess.project    import project_all
    from scripts.preprocess.normalize  import normalize_all
    from scripts.preprocess.composite  import composite_by_year

    log.info("=== STEP: preprocess ===")
    config = _load_cfg(cfg)
    ckpt   = _ckpt()

    # _discover_hirise returns flat {obs_id: path}
    hirise_flat = _discover_hirise(Path("data/raw/hirise"))
    ctx_all     = _discover_ctx   (Path("data/raw/ctx"))
    mola_all    = _discover_mola  (Path("data/raw/mola"))

    log.info(f"Found: {len(hirise_flat)} HiRISE obs, "
             f"{len(ctx_all)} CTX sites, {len(mola_all)} MOLA subsets")

    # Build a name->key map: "gasa" matches "Gasa Crater" etc.
    def _name_to_short(name: str) -> str:
        return name.lower().split()[0]  # "Gasa Crater" -> "gasa"

    # Build per-site HiRISE mapping from config known_images
    hirise_by_site: Dict[str, Dict[str, Path]] = {}
    for site_key in _site_keys(config):
        site_name = config["study_area"][site_key].get("name", site_key)
        short = _name_to_short(site_name)
        known = (config["data_sources"]["hirise"]
                 .get("known_images", {}).get(short, []))
        site_obs = {obs_id: hirise_flat[obs_id]
                    for obs_id in known if obs_id in hirise_flat}
        hirise_by_site[site_key] = site_obs
        if site_obs:
            log.info(f"  {site_name}: {len(site_obs)} HiRISE obs matched")

    coreg_dir = Path("data/processed/coregistered")
    proj_dir  = Path("data/processed/projected")
    norm_dir  = Path("data/processed/normalised")
    comp_dir  = Path("data/processed/composites")

    # --- 1. Coregistration ---
    coregistered: Dict[str, Dict[str, Path]] = {}
    for site_key in _site_keys(config):
        site_name    = config["study_area"][site_key].get("name", site_key)
        short_name   = _name_to_short(site_name)   # 'gasa','palikir','russell'
        hirise_paths = hirise_by_site.get(site_key, {})
        ctx_path     = ctx_all.get(short_name)      # ctx_all keyed by dir name

        if not hirise_paths:
            log.warning(f"No HiRISE obs for {site_name} — skipping coregistration")
            coregistered[site_key] = {}
            continue
        if not ctx_path:
            log.warning(f"No CTX tif for {site_name} — skipping coregistration")
            coregistered[site_key] = hirise_paths
            continue

        log.info(f"Coregistering {site_name} "
                 f"({len(hirise_paths)} obs to {ctx_path.name})...")
        try:
            result = coregister_site(
                site_name    = site_name,
                hirise_paths = hirise_paths,
                ctx_path     = ctx_path,
                output_dir   = coreg_dir / site_key,
                cfg          = config,
                checkpoint   = ckpt,
            )
            coregistered[site_key] = result or hirise_paths
        except Exception as e:
            log.warning(f"Coregistration failed for {site_name}: {e}")
            coregistered[site_key] = hirise_paths

    # --- 2. Reprojection ---
    all_files: Dict[str, Path] = {}
    for site_key, obs_dict in coregistered.items():
        for obs_id, p in obs_dict.items():
            all_files[f"{site_key}_{obs_id}"] = p
    # ctx_all keyed by dir name (gasa/palikir/russell), need to match to study_area key
    ctx_key_map = {_name_to_short(config["study_area"][k].get("name","")): k
                   for k in _site_keys(config)}
    for ctx_site, p in ctx_all.items():
        study_key = ctx_key_map.get(ctx_site, ctx_site)
        all_files[f"{study_key}_ctx"] = p
    for mola_site, p in mola_all.items():
        study_key = ctx_key_map.get(mola_site, mola_site)
        all_files[f"{study_key}_mola"] = p

    if not all_files:
        log.warning("No files to reproject — check download step output")
    else:
        log.info(f"Reprojecting {len(all_files)} files to Mars 6m grid...")
        try:
            projected = project_all(
                file_map   = all_files,
                output_dir = proj_dir,
                cfg        = config,
                checkpoint = ckpt,
            )
        except Exception as e:
            log.warning(f"Reprojection partial: {e}")
            projected = all_files

        # --- 3. Normalisation ---
        log.info(f"Normalising {len(projected)} files...")
        try:
            normalize_all(
                file_map   = projected,
                output_dir = norm_dir,
                cfg        = config,
                checkpoint = ckpt,
            )
        except Exception as e:
            log.warning(f"Normalisation partial: {e}")

        # --- 4. Temporal composites ---
        norm_files = list(norm_dir.glob("*.tif")) if norm_dir.exists() else []
        if norm_files:
            date_map: Dict[str, List[Path]] = {}
            for p in norm_files:
                key = re.sub(r"_(PSP|ESP).*", "", p.stem)
                date_map.setdefault(key, []).append(p)
            log.info(f"Building composites for {len(date_map)} groups...")
            try:
                composite_by_year(
                    date_file_map = date_map,
                    output_dir    = comp_dir,
                    checkpoint    = ckpt,
                )
            except Exception as e:
                log.warning(f"Compositing partial: {e}")

    log.info("Preprocess step complete.")


# ---------------------------------------------------------------------------
# Step: features
# ---------------------------------------------------------------------------

def run_features(cfg: str, **kwargs) -> None:
    from scripts.features.feature_stack import build_site_features

    log.info("=== STEP: features ===")
    config = _load_cfg(cfg)
    ckpt   = _ckpt()

    norm_dir = Path("data/processed/normalised")
    comp_dir = Path("data/processed/composites")
    mola_all = _discover_mola(Path("data/raw/mola"))
    ctx_all  = _discover_ctx(Path("data/raw/ctx"))
    feat_dir = Path("data/processed/feature_stacks")

    # Build a map: site_key -> all normalised files for that site
    # Normalised files are named: {site_key}_*_projected_norm.tif
    # where site_key is primary/secondary/tertiary
    def _norm_files_for_site(site_key: str) -> Dict[str, Path]:
        """Return all normalised HiRISE files for a site."""
        result = {}
        for search_dir in [norm_dir, comp_dir]:
            if not search_dir.exists():
                continue
            for p in sorted(search_dir.glob(f"{site_key}_*.tif")):
                # Skip MOLA and CTX projected files — those go in separately
                stem = p.stem.lower()
                if "mola" in stem or "_ctx_" in stem:
                    continue
                result[p.stem] = p
        return result

    # Also look for composite files
    def _composite_for_site(site_key: str) -> Optional[Path]:
        for search_dir in [comp_dir, norm_dir]:
            if not search_dir.exists():
                continue
            for p in sorted(search_dir.glob(f"composite_{site_key}*.tif")):
                return p
        return None

    def _norm_ctx_for_site(site_key: str) -> Optional[Path]:
        """Return the normalised CTX projected file for a site."""
        if norm_dir.exists():
            matches = list(norm_dir.glob(f"{site_key}_ctx_projected_norm.tif"))
            if matches:
                return matches[0]
        return None

    def _norm_mola_for_site(site_key: str) -> Optional[Path]:
        """Return the normalised MOLA projected file for a site."""
        if norm_dir.exists():
            matches = list(norm_dir.glob(f"{site_key}_mola_projected_norm.tif"))
            if matches:
                return matches[0]
        return None

    for site_key in _site_keys(config):
        site_name  = config["study_area"][site_key].get("name", site_key)
        short_name = _name_to_short(site_name)

        hirise_paths = _norm_files_for_site(site_key)
        composite    = _composite_for_site(site_key)
        ctx_norm     = _norm_ctx_for_site(site_key)
        mola_norm    = _norm_mola_for_site(site_key)

        # Fall back to raw CTX/MOLA paths if normalised versions not found
        ctx_path  = ctx_norm  or ctx_all.get(short_name)
        mola_path = mola_norm or mola_all.get(short_name)

        # Use composite as the primary HiRISE input if available
        if composite and composite.stem not in hirise_paths:
            hirise_paths[composite.stem] = composite

        if not hirise_paths and not ctx_path:
            log.warning(f"No input data for {site_name} features — skipping")
            continue

        log.info(
            f"Building feature stack for {site_name}: "
            f"{len(hirise_paths)} HiRISE, "
            f"CTX={'yes' if ctx_path else 'no'}, "
            f"MOLA={'yes' if mola_path else 'no'}"
        )
        try:
            build_site_features(
                site_name    = site_name,
                hirise_paths = hirise_paths,
                ctx_path     = ctx_path,
                mola_path    = mola_path,
                output_dir   = feat_dir,
                cfg          = config,
                checkpoint   = ckpt,
            )
        except Exception as e:
            log.warning(f"Feature stack failed for {site_name}: {e}")

    log.info("Features step complete.")


# ---------------------------------------------------------------------------
# Step: labels
# ---------------------------------------------------------------------------

def run_labels(cfg: str, **kwargs) -> None:
    from scripts.labels.transfer_labels import generate_labels_for_site
    from scripts.labels.augment_labels  import extract_patches, save_patches
    import numpy as np

    log.info("=== STEP: labels ===")
    config    = _load_cfg(cfg)
    ckpt      = _ckpt()
    mola_all  = _discover_mola(Path("data/raw/mola"))
    label_dir = Path("data/processed/labels")
    patch_dir = Path("data/processed/patches")

    for site_key in _site_keys(config):
        site_cfg  = config["study_area"][site_key]
        site_name  = site_cfg.get("name", site_key)
        short_name = _name_to_short(site_name)
        mola_path  = mola_all.get(short_name) or mola_all.get(site_key)

        if not mola_path:
            log.warning(f"No MOLA subset for {site_name} (tried '{short_name}')")
            continue

        # Load slope/aspect from MOLA
        try:
            import rasterio
            from scripts.features.morphometric import compute_slope, compute_aspect
            with rasterio.open(mola_path) as src:
                dem = src.read(1).astype("float32")
            resolution = config["preprocessing"].get("target_resolution_m", 463.0)
            slope  = compute_slope(dem,  resolution=resolution)
            aspect = compute_aspect(dem, resolution=resolution)
        except Exception as e:
            log.warning(f"Could not compute slope/aspect for {site_name}: {e}")
            h, w = 64, 64
            slope  = np.zeros((h, w), dtype="float32")
            aspect = np.zeros((h, w), dtype="float32")

        log.info(f"Generating labels for {site_name} "
                 f"(slope shape: {slope.shape})...")
        try:
            generate_labels_for_site(
                site_name    = site_name,
                target_shape = slope.shape,
                slope_map    = slope,
                aspect_map   = aspect,
                output_dir   = label_dir,
                cfg          = config,
                checkpoint   = ckpt,
            )
        except Exception as e:
            log.warning(f"Label generation failed for {site_name}: {e}")

    # Extract .npy patches from feature stacks + label masks for PyTorch dataset
    log.info("Extracting training patches...")
    try:
        from scripts.labels.augment_labels import extract_patches, save_patches
        feat_dir      = Path("data/processed/feature_stacks")
        label_dir     = Path("data/processed/labels")
        img_patch_dir = Path("data/processed/patches/images")
        msk_patch_dir = Path("data/processed/patches/masks")
        img_patch_dir.mkdir(parents=True, exist_ok=True)
        msk_patch_dir.mkdir(parents=True, exist_ok=True)

        patch_size   = config.get("training", {}).get("patch_size", 512)
        patch_stride = config.get("training", {}).get("patch_stride", 256)

        for feat_file in sorted(feat_dir.glob("*.npz")):
            site_prefix = feat_file.stem.split("_")[0]
            site_name_cap = {"primary": "Gasa", "secondary": "Palikir",
                             "tertiary": "Russell"}.get(site_prefix, site_prefix)
            mask_files = list(label_dir.glob(f"*{site_name_cap}*.tif"))
            if not mask_files:
                continue
            try:
                npz = np.load(feat_file)
                # Key is "stack" (set in save_feature_stack)
                key = "stack" if "stack" in npz else list(npz.keys())[0]
                feat_data = npz[key]
                import rasterio as _rio
                import cv2 as _cv2
                with _rio.open(mask_files[0]) as _r:
                    mask_data = _r.read(1)
                # Resize mask to match feature stack spatial dims
                fh, fw = feat_data.shape[1], feat_data.shape[2]
                if mask_data.shape != (fh, fw):
                    mask_data = _cv2.resize(
                        mask_data.astype(np.float32), (fw, fh),
                        interpolation=_cv2.INTER_NEAREST
                    ).astype(np.uint8)
                img_patches, msk_patches = extract_patches(
                    feat_data, mask_data,
                    patch_size=patch_size, stride=patch_stride,
                    min_fg=0.001  # low threshold - gullies are rare
                )
                if img_patches:
                    n_saved = save_patches(img_patches, msk_patches,
                                          img_patch_dir, msk_patch_dir,
                                          prefix=feat_file.stem)
                    log.info(f"  {feat_file.name}: {n_saved} patches saved")
                else:
                    log.debug(f"  {feat_file.name}: no fg patches found")
            except Exception as _e:
                log.warning(f"Patch extraction failed for {feat_file.name}: {_e}")

        log.info(f"Total patches: {len(list(img_patch_dir.glob('*.npy')))}")
    except Exception as e:
        log.warning(f"Patch extraction step failed: {e}")

    log.info("Labels step complete.")


# ---------------------------------------------------------------------------
# Step: train
# ---------------------------------------------------------------------------

def run_train(cfg: str, resume: str = None, **kwargs) -> None:
    from scripts.train.train_unet import train
    log.info("=== STEP: train ===")
    train(cfg_path=cfg, resume=resume)


# ---------------------------------------------------------------------------
# Step: infer
# ---------------------------------------------------------------------------

def run_infer(cfg: str, **kwargs) -> None:
    from scripts.inference.predict import run_inference
    log.info("=== STEP: infer ===")
    run_inference(cfg)


# ---------------------------------------------------------------------------
# Step: threshold
# ---------------------------------------------------------------------------

def run_threshold(cfg: str, **kwargs) -> None:
    from scripts.inference.threshold import run_threshold as _thr
    log.info("=== STEP: threshold ===")
    _thr(cfg)


# ---------------------------------------------------------------------------
# Step: change
# ---------------------------------------------------------------------------

def run_change(cfg: str, **kwargs) -> None:
    from scripts.change.detect_changes    import run_change_detection
    from scripts.change.kalman_smooth     import run_kalman_smoothing
    from scripts.change.activity_tracking import run_activity_tracking
    log.info("=== STEP: change ===")
    run_change_detection(cfg)
    run_kalman_smoothing(cfg)
    run_activity_tracking(cfg)


# ---------------------------------------------------------------------------
# Step: export
# ---------------------------------------------------------------------------

def run_export(cfg: str, **kwargs) -> None:
    from scripts.export.to_geotiff   import run_geotiff_export
    from scripts.export.to_geojson   import run_geojson_export
    from scripts.export.to_csv       import run_csv_export
    from scripts.export.stac_catalog import run_stac_export
    log.info("=== STEP: export ===")
    run_geotiff_export(cfg)
    run_geojson_export(cfg)
    run_csv_export(cfg)
    run_stac_export(cfg)


# ---------------------------------------------------------------------------
# Step: dashboard
# ---------------------------------------------------------------------------

def run_dashboard(cfg: str, **kwargs) -> None:
    from scripts.dashboard.time_series_plot import generate_all_plots
    from scripts.dashboard.folium_map       import build_dashboard
    log.info("=== STEP: dashboard ===")
    generate_all_plots(cfg)
    build_dashboard(cfg)


# ---------------------------------------------------------------------------
# Step: validate
# ---------------------------------------------------------------------------

def run_validate(cfg: str, temporal: bool = False, **kwargs) -> None:
    log.info("=== STEP: validate ===")
    if temporal:
        from scripts.validation.temporal_cv import run_temporal_cv
        run_temporal_cv(cfg)
    else:
        from scripts.validation.spatial_cv import run_spatial_cv
        run_spatial_cv(cfg)


# ---------------------------------------------------------------------------
# Step registry + pipeline order
# ---------------------------------------------------------------------------

STEPS = {
    "download":   run_download,
    "preprocess": run_preprocess,
    "features":   run_features,
    "labels":     run_labels,
    "train":      run_train,
    "infer":      run_infer,
    "threshold":  run_threshold,
    "change":     run_change,
    "export":     run_export,
    "dashboard":  run_dashboard,
    "validate":   run_validate,
}

PIPELINE_ORDER = [
    "download", "preprocess", "features", "labels",
    "train", "infer", "threshold", "change",
    "export", "dashboard", "validate",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Mars Gully Digital Twin — Pipeline Orchestrator",
    )
    parser.add_argument("--step", "-s", default="all",
                        choices=list(STEPS.keys()) + ["all"])
    parser.add_argument("--cfg",  "-c", default="config.yaml")
    parser.add_argument("--resume", "-r", default=None)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--site",  default=None)
    parser.add_argument("--date",  default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    if not Path(args.cfg).exists():
        log.error(f"Config file not found: {args.cfg}")
        sys.exit(1)

    kwargs = dict(cfg=args.cfg, resume=args.resume,
                  temporal=args.temporal, site=args.site, date=args.date)

    if args.step == "all":
        log.info("Running full pipeline ...")
        for step_name in PIPELINE_ORDER:
            log.info("-" * 60)
            try:
                STEPS[step_name](**kwargs)
            except Exception as exc:
                log.error(f"Step '{step_name}' failed: {exc}", exc_info=True)
                log.error("Fix the error and re-run with --step <step_name>")
                sys.exit(1)
        log.info("Pipeline complete.")
    else:
        try:
            STEPS[args.step](**kwargs)
        except Exception as exc:
            log.error(f"Step '{args.step}' failed: {exc}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()