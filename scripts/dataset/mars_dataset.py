"""
Mars Gully PyTorch Dataset
Loads image and mask patches with augmentation.
Handles class imbalance (gully pixels are rare ~1-5% of area).
"""

import sys
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.utils import get_logger

log = get_logger("dataset.mars")


class MarsGullyDataset(Dataset):
    """
    PyTorch Dataset for Mars gully detection.

    Loads pre-extracted patches (C, H, W) images and (H, W) binary masks.
    Optionally applies online augmentation via albumentations.

    Parameters
    ----------
    img_dir  : directory of .npy image patches (C, H, W), float32 [0, 1]
    msk_dir  : directory of .npy mask patches (H, W), uint8 {0, 1}
    transform: optional albumentations transform
    in_channels: expected number of input channels (will pad or truncate)
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        msk_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        in_channels: int = 8,
        augment: bool = True,
        cfg: Optional[Dict] = None
    ):
        self.img_dir = Path(img_dir)
        self.msk_dir = Path(msk_dir)
        self.transform = transform
        self.in_channels = in_channels
        self.augment = augment
        self.cfg = cfg or {}

        self.img_paths = sorted(self.img_dir.glob("*.npy"))
        self.msk_paths = sorted(self.msk_dir.glob("*.npy"))

        # Match files by stem
        img_stems = {p.stem: p for p in self.img_paths}
        msk_stems = {p.stem: p for p in self.msk_paths}
        common = sorted(set(img_stems) & set(msk_stems))

        self.img_paths = [img_stems[s] for s in common]
        self.msk_paths = [msk_stems[s] for s in common]

        if len(self.img_paths) == 0:
            raise ValueError(
                f"No matching patches found in {img_dir} and {msk_dir}. "
                "Run --step labels first."
            )

        # Compute foreground fraction for each patch (for WeightedSampler)
        self._fg_fractions = self._compute_fg_fractions()
        log.info(
            f"Dataset: {len(self)} patches, "
            f"mean fg={np.mean(self._fg_fractions):.3f}, "
            f"in_channels={in_channels}"
        )

    def _compute_fg_fractions(self) -> np.ndarray:
        """Compute foreground fraction for each mask patch."""
        fracs = np.zeros(len(self.msk_paths), dtype=np.float32)
        for i, p in enumerate(self.msk_paths):
            try:
                m = np.load(p)
                fracs[i] = m.mean()
            except Exception:
                fracs[i] = 0.0
        return fracs

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.img_paths[idx]).astype(np.float32)  # (C, H, W)
        msk = np.load(self.msk_paths[idx]).astype(np.float32)  # (H, W)

        # Ensure correct number of channels
        img = self._fix_channels(img)

        # Replace NaN
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        img = np.clip(img, 0.0, 1.0)
        msk = np.clip(msk, 0.0, 1.0)

        # Albumentations augmentation (expects HWC format)
        if self.transform is not None and self.augment:
            img_hwc = img.transpose(1, 2, 0)  # (H, W, C)
            transformed = self.transform(image=img_hwc, mask=msk)
            img = transformed["image"].transpose(2, 0, 1)  # back to (C, H, W)
            msk = transformed["mask"]

        # Simple augmentation without albumentations
        elif self.augment:
            img, msk = self._simple_augment(img, msk)

        img_tensor = torch.from_numpy(img.copy()).float()
        msk_tensor = torch.from_numpy(msk.copy()).float().unsqueeze(0)  # (1, H, W)

        return img_tensor, msk_tensor

    def _fix_channels(self, img: np.ndarray) -> np.ndarray:
        """Pad or truncate channels to match in_channels."""
        C, H, W = img.shape
        if C == self.in_channels:
            return img
        elif C > self.in_channels:
            return img[:self.in_channels]
        else:
            # Pad by repeating last channel
            pad = np.repeat(img[-1:], self.in_channels - C, axis=0)
            return np.concatenate([img, pad], axis=0)

    def _simple_augment(
        self, img: np.ndarray, msk: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple augmentation: random flip and rotation."""
        aug_cfg = self.cfg.get("training", {}).get("augmentation", {})
        if not aug_cfg.get("enabled", True):
            return img, msk

        # Random horizontal flip
        if aug_cfg.get("flip_horizontal", True) and random.random() > 0.5:
            img = img[:, :, ::-1]
            msk = msk[:, ::-1]

        # Random vertical flip
        if aug_cfg.get("flip_vertical", True) and random.random() > 0.5:
            img = img[:, ::-1, :]
            msk = msk[::-1, :]

        # Random 90-degree rotation
        rotations = aug_cfg.get("rotation", [0, 90, 180, 270])
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            msk = np.rot90(msk, k=k, axes=(0, 1)).copy()

        return img, msk

    def get_weighted_sampler(self, weight_bg: float = 0.3) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler that oversamples gully patches.

        Parameters
        ----------
        weight_bg : weight for background-only patches (relative to fg weight=1)
        """
        weights = np.where(self._fg_fractions > 0.01, 1.0, weight_bg)
        weights = weights / weights.sum()
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(self),
            replacement=True
        )
        return sampler


def get_albumentations_transform(cfg: Dict):
    """
    Build albumentations augmentation pipeline from config.
    Returns None if albumentations not installed.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        log.warning("albumentations not installed  -  using simple augmentation")
        return None

    aug_cfg = cfg.get("training", {}).get("augmentation", {})
    transforms = []

    if aug_cfg.get("flip_horizontal", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug_cfg.get("flip_vertical", True):
        transforms.append(A.VerticalFlip(p=0.5))
    if aug_cfg.get("rotation", []):
        transforms.append(A.RandomRotate90(p=0.5))
    if aug_cfg.get("brightness_contrast", True):
        bl = aug_cfg.get("brightness_limit", 0.2)
        cl = aug_cfg.get("contrast_limit", 0.2)
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=bl, contrast_limit=cl, p=0.3
        ))

    return A.Compose(transforms) if transforms else None


def build_dataloaders(
    cfg: Dict,
    train_img_dir: Path,
    train_msk_dir: Path,
    val_img_dir: Optional[Path] = None,
    val_msk_dir: Optional[Path] = None,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.
    If val directories not provided, split train data.
    """
    train_cfg = cfg.get("training", {}).get("training", {})
    batch_size = train_cfg.get("batch_size", 8)
    in_channels = cfg.get("training", {}).get("in_channels", 8)

    transform = get_albumentations_transform(cfg)

    full_dataset = MarsGullyDataset(
        train_img_dir, train_msk_dir,
        transform=transform,
        in_channels=in_channels,
        augment=True,
        cfg=cfg
    )

    if val_img_dir and val_msk_dir and val_img_dir.exists():
        train_ds = full_dataset
        val_ds = MarsGullyDataset(
            val_img_dir, val_msk_dir,
            transform=None, in_channels=in_channels, augment=False, cfg=cfg
        )
    else:
        # Split by index
        n = len(full_dataset)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        indices = list(range(n))
        random.shuffle(indices)

        from torch.utils.data import Subset
        train_ds = Subset(full_dataset, indices[:n_train])
        val_ds = Subset(
            MarsGullyDataset(
                train_img_dir, train_msk_dir,
                transform=None, in_channels=in_channels, augment=False, cfg=cfg
            ),
            indices[n_train:]
        )

    sampler = full_dataset.get_weighted_sampler() if isinstance(train_ds, MarsGullyDataset) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,  # 0 = main process only (Windows fix)
        pin_memory=False,  # no CUDA pin on CPU
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 0 = main process only (Windows fix)
        pin_memory=False  # no CUDA pin on CPU
    )

    log.info(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
    log.info(f"Batch size: {batch_size} | Loaders ready")
    return train_loader, val_loader