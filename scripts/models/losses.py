"""
losses.py
=========
Loss functions for Mars gully binary segmentation.

Includes:
  - FocalLoss          — handles class imbalance without sampling
  - DiceLoss           — region overlap; stable on tiny positive areas
  - CombinedLoss       — weighted sum of Focal + Dice (default for training)
  - TverskyLoss        — generalised Dice with configurable FP/FN weights
  - BoundaryLoss       — penalises errors near gully edges (optional addon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss (Lin et al. 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma (float): Focusing exponent.  2.0 is the standard choice.
        alpha (float): Weighting for the positive class [0, 1].
        reduction (str): 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W) or (B, H, W) — raw (pre-sigmoid) values.
            targets : (B, 1, H, W) or (B, H, W) — binary masks in {0, 1}.
        """
        logits = logits.squeeze(1) if logits.ndim == 4 else logits
        targets = targets.squeeze(1) if targets.ndim == 4 else targets
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    Args:
        smooth  (float): Laplace smoothing to avoid 0/0.
        from_logits (bool): If True, applies sigmoid internally.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.squeeze(1) if logits.ndim == 4 else logits
        targets = targets.squeeze(1) if targets.ndim == 4 else targets
        targets = targets.float()

        probs = torch.sigmoid(logits) if self.from_logits else logits

        # Flatten spatial dims
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


# ---------------------------------------------------------------------------
# Combined Focal + Dice Loss
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Weighted combination of Focal and Dice losses.

    L = (1 - w_dice) * FocalLoss + w_dice * DiceLoss

    Args:
        focal_gamma  (float): Focal loss gamma.
        focal_alpha  (float): Focal loss alpha.
        dice_weight  (float): Weight for the Dice component [0, 1].
        smooth       (float): Dice smoothing constant.
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.dice = DiceLoss(smooth=smooth)
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (1.0 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Tversky Loss
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalises Dice with configurable FP / FN penalties.

    T(alpha, beta) = TP / (TP + alpha*FP + beta*FN)
    Loss = 1 - T

    Setting alpha=beta=0.5 recovers Dice Loss.
    Use beta > 0.5 to penalise false negatives (missed gullies) more.

    Args:
        alpha (float): FP weight.
        beta  (float): FN weight.
        smooth(float): Smoothing constant.
        from_logits (bool): Apply sigmoid internally.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        from_logits: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.squeeze(1) if logits.ndim == 4 else logits
        targets = targets.squeeze(1) if targets.ndim == 4 else targets
        targets = targets.float()

        probs = torch.sigmoid(logits) if self.from_logits else logits
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1.0 - targets)).sum(dim=1)
        FN = ((1.0 - probs) * targets).sum(dim=1)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        return (1.0 - tversky).mean()


# ---------------------------------------------------------------------------
# Boundary Loss (optional addon for edge-sensitive training)
# ---------------------------------------------------------------------------

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that applies extra weight to pixels near gully edges.

    Uses a distance-transformed edge map derived from the target mask.
    Requires scipy at runtime (imported lazily).

    Args:
        theta0 (int): Distance threshold in pixels for the edge region.
    """

    def __init__(self, theta0: int = 3):
        super().__init__()
        self.theta0 = theta0

    @staticmethod
    def _edge_map(mask: torch.Tensor, theta0: int) -> torch.Tensor:
        """Dilate the mask boundary to create an edge-weight map on CPU."""
        from scipy.ndimage import distance_transform_edt
        import numpy as np

        mask_np = mask.cpu().numpy().astype(np.uint8)
        B, H, W = mask_np.shape
        weight_maps = []
        for b in range(B):
            dist_fg = distance_transform_edt(mask_np[b])
            dist_bg = distance_transform_edt(1 - mask_np[b])
            edge = (dist_fg < theta0) | (dist_bg < theta0)
            w = np.ones((H, W), dtype=np.float32)
            w[edge] = 2.0
            weight_maps.append(w)
        return torch.from_numpy(np.stack(weight_maps)).to(mask.device)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.squeeze(1) if logits.ndim == 4 else logits
        targets = targets.squeeze(1) if targets.ndim == 4 else targets
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        weights = self._edge_map(targets.long(), self.theta0)
        return (bce * weights).mean()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_loss(cfg: dict) -> nn.Module:
    """
    Build a loss function from the training section of config.yaml.

    Supported cfg['loss']['type'] values:
      'focal', 'dice', 'combined', 'tversky', 'boundary'

    Example config block::

        loss:
          type: combined
          focal_gamma: 2.0
          focal_alpha: 0.25
          dice_weight: 0.5
    """
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "combined").lower()

    if loss_type == "focal":
        return FocalLoss(
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", 0.25),
        )
    elif loss_type == "dice":
        return DiceLoss(smooth=loss_cfg.get("smooth", 1.0))
    elif loss_type == "combined":
        return CombinedLoss(
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
        )
    elif loss_type == "tversky":
        return TverskyLoss(
            alpha=loss_cfg.get("tversky_alpha", 0.3),
            beta=loss_cfg.get("tversky_beta", 0.7),
        )
    elif loss_type == "boundary":
        return BoundaryLoss(theta0=loss_cfg.get("boundary_theta0", 3))
    else:
        raise ValueError(f"Unknown loss type: {loss_type!r}")
