#!/usr/bin/env python3
"""
main.py
=======
CLI orchestrator for the Mars Gully Digital Twin pipeline.

Usage
-----
    python main.py --step all
    python main.py --step download
    python main.py --step preprocess
    python main.py --step features
    python main.py --step labels
    python main.py --step train
    python main.py --step train --resume last
    python main.py --step infer
    python main.py --step threshold
    python main.py --step change
    python main.py --step export
    python main.py --step dashboard
    python main.py --step validate
    python main.py --step validate --temporal

Options
-------
    --cfg        Path to config.yaml (default: config.yaml)
    --step       Pipeline step to run (see above)
    --resume     Checkpoint to resume training from: 'last' | 'best' | <path>
    --temporal   Use temporal hold-out for validation (instead of spatial CV)
    --site       Restrict processing to a single study area site
    --date       Filter to a specific year/date string
    --log-level  Logging verbosity: DEBUG | INFO | WARNING (default: INFO)
"""

import argparse
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: str = "data/outputs/pipeline.log") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, handlers=handlers)


log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

STEP_HELP = """
Available pipeline steps:
  download    Download raw HiRISE, CTX, CRISM, MOLA data
  preprocess  Coregister, project, normalise, composite
  features    Build multi-sensor feature stacks
  labels      Transfer/synthesise gully labels
  train       Train the U-Net segmentation model
  infer       Run sliding-window inference on feature stacks
  threshold   Post-process probability maps to binary masks
  change      Multi-temporal change detection + Kalman smoothing + tracking
  export      Export COG GeoTIFFs, GeoJSON, CSV reports, STAC catalog
  dashboard   Generate interactive Folium HTML dashboard
  validate    Spatial block cross-validation (add --temporal for hold-out)
  all         Run the full pipeline end-to-end
"""


def run_download(cfg: str, site: str = None, **kwargs) -> None:
    from scripts.downloaders.download_hirise import download_hirise
    from scripts.downloaders.download_ctx import download_ctx
    from scripts.downloaders.download_mola import download_mola
    log.info("=== STEP: download ===")
    download_hirise(cfg)
    download_ctx(cfg)
    download_mola(cfg)


def run_preprocess(cfg: str, **kwargs) -> None:
    from scripts.preprocess.project import reproject_all
    from scripts.preprocess.normalize import normalize_all
    from scripts.preprocess.coregister import coregister_all
    from scripts.preprocess.composite import composite_all
    log.info("=== STEP: preprocess ===")
    coregister_all(cfg)
    reproject_all(cfg)
    normalize_all(cfg)
    composite_all(cfg)


def run_features(cfg: str, **kwargs) -> None:
    from scripts.features.feature_stack import build_all_feature_stacks
    log.info("=== STEP: features ===")
    build_all_feature_stacks(cfg)


def run_labels(cfg: str, **kwargs) -> None:
    from scripts.labels.transfer_labels import run_label_transfer
    from scripts.labels.augment_labels import run_label_augmentation
    log.info("=== STEP: labels ===")
    run_label_transfer(cfg)
    run_label_augmentation(cfg)


def run_train(cfg: str, resume: str = None, **kwargs) -> None:
    from scripts.train.train_unet import train
    log.info("=== STEP: train ===")
    train(cfg_path=cfg, resume=resume)


def run_infer(cfg: str, **kwargs) -> None:
    from scripts.inference.predict import run_inference
    log.info("=== STEP: infer ===")
    run_inference(cfg)


def run_threshold(cfg: str, **kwargs) -> None:
    from scripts.inference.threshold import run_threshold as _threshold
    log.info("=== STEP: threshold ===")
    _threshold(cfg)


def run_change(cfg: str, **kwargs) -> None:
    from scripts.change.detect_changes import run_change_detection
    from scripts.change.kalman_smooth import run_kalman_smoothing
    from scripts.change.activity_tracking import run_activity_tracking
    log.info("=== STEP: change detection ===")
    run_change_detection(cfg)
    run_kalman_smoothing(cfg)
    run_activity_tracking(cfg)


def run_export(cfg: str, **kwargs) -> None:
    from scripts.export.to_geotiff import run_geotiff_export
    from scripts.export.to_geojson import run_geojson_export
    from scripts.export.to_csv import run_csv_export
    from scripts.export.stac_catalog import run_stac_export
    log.info("=== STEP: export ===")
    run_geotiff_export(cfg)
    run_geojson_export(cfg)
    run_csv_export(cfg)
    run_stac_export(cfg)


def run_dashboard(cfg: str, **kwargs) -> None:
    from scripts.dashboard.time_series_plot import generate_all_plots
    from scripts.dashboard.folium_map import build_dashboard
    log.info("=== STEP: dashboard ===")
    generate_all_plots(cfg)
    out_path = build_dashboard(cfg)
    log.info(f"Dashboard ready → {out_path}")


def run_validate(cfg: str, temporal: bool = False, **kwargs) -> None:
    log.info("=== STEP: validate ===")
    if temporal:
        from scripts.validation.temporal_cv import run_temporal_cv
        run_temporal_cv(cfg)
    else:
        from scripts.validation.spatial_cv import run_spatial_cv
        run_spatial_cv(cfg)


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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=STEP_HELP,
    )
    parser.add_argument(
        "--step", "-s",
        default="all",
        choices=list(STEPS.keys()) + ["all"],
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--cfg", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--resume", "-r",
        default=None,
        help="Resume training from: 'last' | 'best' | /path/to/checkpoint.pth",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporal hold-out for --step validate",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="Restrict to a specific study area site (e.g. 'gasa')",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Filter to a specific date/year string (e.g. '2015')",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(level=args.log_level)

    if not Path(args.cfg).exists():
        log.error(f"Config file not found: {args.cfg}")
        sys.exit(1)

    kwargs = {
        "cfg": args.cfg,
        "resume": args.resume,
        "temporal": args.temporal,
        "site": args.site,
        "date": args.date,
    }

    if args.step == "all":
        log.info("Running full pipeline …")
        for step_name in PIPELINE_ORDER:
            try:
                log.info(f"{'─'*60}")
                STEPS[step_name](**kwargs)
            except Exception as exc:
                log.error(f"Step '{step_name}' failed: {exc}", exc_info=True)
                log.error("Pipeline aborted. Fix the error and re-run with --step <step>.")
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
