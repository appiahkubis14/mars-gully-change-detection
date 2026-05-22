"""
train_unet.py
=============
Full training loop for the Mars Gully U-Net.

Features
--------
* Mixed-precision (torch.cuda.amp) for memory efficiency
* Gradient accumulation for effective large batch sizes
* ReduceLROnPlateau scheduler
* Early stopping on validation IoU
* TensorBoard logging (loss, IoU, F1, LR per epoch)
* Best + latest checkpoint saving
* Resume from last checkpoint automatically

Usage (CLI via main.py)
-----------------------
    python main.py --step train
    python main.py --step train --resume last
    python main.py --step train --resume data/models/unet_best.pth
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from scripts.dataset.mars_dataset import build_dataloaders
from scripts.models.unet import build_model
from scripts.models.losses import build_loss
from scripts.utils import load_config, StepCheckpoint, Timer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Binary IoU, averaged over the batch."""
    probs = torch.sigmoid(preds).detach()
    binary = (probs > threshold).float()
    targets = targets.float()
    intersection = (binary * targets).sum(dim=(1, 2, 3))
    union = binary.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = ((intersection + eps) / (union + eps)).mean().item()
    return iou


def _f1_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """Binary F1, averaged over the batch."""
    probs = torch.sigmoid(preds).detach()
    binary = (probs > threshold).float()
    targets = targets.float()
    tp = (binary * targets).sum(dim=(1, 2, 3))
    fp = (binary * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - binary) * targets).sum(dim=(1, 2, 3))
    f1 = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()
    return f1


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    optimizer.zero_grad()

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast('cpu' if not torch.cuda.is_available() else 'cuda'):
            logits = model(images)
            loss = criterion(logits, masks) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        total_iou += _iou_score(logits, masks)
        total_f1 += _f1_score(logits, masks)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "f1": total_f1 / n,
    }


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast('cpu' if not torch.cuda.is_available() else 'cuda'):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()
        total_iou += _iou_score(logits, masks)
        total_f1 += _f1_score(logits, masks)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "f1": total_f1 / n,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    state: dict,
    path: Path,
    is_best: bool,
    best_path: Path,
) -> None:
    torch.save(state, path)
    if is_best:
        import shutil
        shutil.copy2(path, best_path)
        log.info(f"New best model saved -> {best_path}")


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    device: torch.device,
) -> Dict:
    log.info(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg_path: str = "config.yaml", resume: Optional[str] = None) -> None:
    """
    Run the full training loop.

    Args:
        cfg_path : Path to config.yaml.
        resume   : 'last', 'best', or explicit checkpoint path.
                   If None, uses cfg.checkpoint.resume_from.
    """
    cfg = load_config(cfg_path)
    t_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]

    # ------------------------------------------------------------------ dirs
    model_dir = Path(ckpt_cfg["save_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["logging"].get("tensorboard_dir", "data/outputs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = model_dir / "unet_last.pth"
    best_ckpt = model_dir / "unet_best.pth"
    history_path = model_dir / "training_history.json"

    # ------------------------------------------------------------------ device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on {device}")
    if device.type == "cpu":
        # CPU training: reduce batch size and warn about speed
        cfg.setdefault("training", {})
        original_bs = cfg["training"].get("batch_size", 8)
        if original_bs > 2:
            cfg["training"]["batch_size"] = 2
            log.info(f"CPU mode: batch_size reduced {original_bs}->2 for speed")
        log.info("Tip: training on CPU is slow. Consider Google Colab for GPU.")
    log.info(f"Training on {device}")

    # ------------------------------------------------------------------ data
    log.info("Building dataloaders ...")
    patch_dir    = Path(cfg.get("paths", {}).get("patches_dir", "data/processed/patches"))
    img_dir      = patch_dir / "images"
    msk_dir      = patch_dir / "masks"

    if not img_dir.exists() or not any(img_dir.glob("*.npy")):
        log.error(
            f"No patch images found in {img_dir}. "
            "Run '--step labels' to extract patches first."
        )
        return

    # Override batch size for CPU (multiprocessing deadlocks on Windows with workers>0)
    if device.type == "cpu":
        cfg.setdefault("training", {})["batch_size"] = 2
        log.info("CPU: batch_size=2, num_workers=0")

    train_loader, val_loader = build_dataloaders(
        cfg,
        train_img_dir=img_dir,
        train_msk_dir=msk_dir,
    )

    # ------------------------------------------------------------------ model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model params: {total_params:,}")

    # ------------------------------------------------------------------ loss / opt
    criterion = build_loss(t_cfg)
    opt_cfg = t_cfg["optimizer"]
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg.get("learning_rate", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )
    sched_cfg = t_cfg["scheduler"]
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",          # maximise validation IoU
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 10),
        min_lr=sched_cfg.get("min_lr", 1e-7),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------ resume
    start_epoch = 0
    best_iou = 0.0
    history: list = []

    resume_from = resume or ckpt_cfg.get("resume_from", "last")
    ckpt_path: Optional[Path] = None
    if resume_from == "last" and last_ckpt.exists():
        ckpt_path = last_ckpt
    elif resume_from == "best" and best_ckpt.exists():
        ckpt_path = best_ckpt
    elif resume_from and Path(resume_from).exists():
        ckpt_path = Path(resume_from)

    if ckpt_path:
        meta = _load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        start_epoch = meta.get("epoch", 0) + 1
        best_iou = meta.get("best_iou", 0.0)
        history = meta.get("history", [])
        log.info(f"Resumed at epoch {start_epoch}, best IoU = {best_iou:.4f}")

    # ------------------------------------------------------------------ tensorboard
    writer = SummaryWriter(log_dir=str(log_dir))

    # ------------------------------------------------------------------ loop
    max_epochs = t_cfg["training"]["epochs"]
    patience = t_cfg["training"]["early_stopping_patience"]
    grad_accum = t_cfg["training"].get("gradient_accumulation_steps", 1)
    grad_clip = t_cfg["training"].get("gradient_clip_norm", 1.0)
    save_every = ckpt_cfg.get("save_every_epochs", 5)
    no_improve = 0
    timer = Timer()

    log.info(f"Starting training from epoch {start_epoch} to {max_epochs}")

    for epoch in range(start_epoch, max_epochs):
        timer.start()

        train_metrics = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, grad_accum, grad_clip,
        )
        val_metrics = _validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_metrics["iou"])
        elapsed = timer.elapsed

        # ---- logging ----
        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"Train loss={train_metrics['loss']:.4f} IoU={train_metrics['iou']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} IoU={val_metrics['iou']:.4f} F1={val_metrics['f1']:.4f} | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )

        writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("iou", {"train": train_metrics["iou"], "val": val_metrics["iou"]}, epoch)
        writer.add_scalars("f1", {"train": train_metrics["f1"], "val": val_metrics["f1"]}, epoch)
        writer.add_scalar("lr", lr_now, epoch)

        is_best = val_metrics["iou"] > best_iou
        if is_best:
            best_iou = val_metrics["iou"]
            no_improve = 0
        else:
            no_improve += 1

        # ---- history ----
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr_now,
            "best_iou": best_iou,
        }
        history.append(epoch_record)

        # ---- checkpoint ----
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou": best_iou,
            "history": history,
            "cfg": cfg,
        }
        _save_checkpoint(state, last_ckpt, is_best, best_ckpt)

        if (epoch + 1) % save_every == 0:
            periodic = model_dir / f"unet_epoch{epoch:04d}.pth"
            torch.save(state, periodic)

        # ---- early stopping ----
        if no_improve >= patience:
            log.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # ------------------------------------------------------------------ wrap-up
    writer.close()
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training complete. Best Val IoU = {best_iou:.4f}")
    log.info(f"History saved -> {history_path}")
    log.info(f"Best model     -> {best_ckpt}")