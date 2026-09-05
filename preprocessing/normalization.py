"""
preprocessing/normalization.py

Hounsfield Unit (HU) windowing and intensity normalization for CT imaging.
Default: Abdominal soft-tissue window [-135, 215] HU, normalized to [0.0, 1.0].
Includes extension point for multi-window ablation (soft-tissue, bone, lung).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


# Standard clinical CT windows: (min_hu, max_hu)
STANDARD_WINDOWS = {
    # Abdominal soft-tissue (Window Width: 350, Window Level: 40) -> [-135, 215]
    "abdominal_soft_tissue": (-135.0, 215.0),
    # Bone window (Width: 1500, Level: 400) -> [-350, 1150]
    "bone": (-350.0, 1150.0),
    # Lung window (Width: 1500, Level: -600) -> [-1350, 150]
    "lung": (-1350.0, 150.0),
    # Mediastinal window (Width: 350, Level: 50) -> [-125, 225]
    "mediastinal": (-125.0, 225.0),
}


def apply_hu_window(
    volume: Union[np.ndarray, torch.Tensor],
    min_hu: float = -135.0,
    max_hu: float = 215.0,
    normalize: bool = True,
    norm_range: Tuple[float, float] = (0.0, 1.0),
) -> Union[np.ndarray, torch.Tensor]:
    """
    Clips CT voxel values to [min_hu, max_hu] and optionally normalizes to norm_range.

    Args:
        volume: 3D or 4D array of raw HU values.
        min_hu: Lower bound for clipping (HU).
        max_hu: Upper bound for clipping (HU).
        normalize: Whether to scale linearly to norm_range.
        norm_range: Target output intensity range, default (0.0, 1.0).

    Returns:
        Windowed and normalized volume (same type as input).
    """
    if isinstance(volume, torch.Tensor):
        clipped = torch.clamp(volume, min=min_hu, max=max_hu)
        if not normalize:
            return clipped
        low, high = norm_range
        normalized = (clipped - min_hu) / (max_hu - min_hu)
        if (low, high) != (0.0, 1.0):
            normalized = normalized * (high - low) + low
        return normalized

    elif isinstance(volume, np.ndarray):
        clipped = np.clip(volume, min_hu, max_hu).astype(np.float32)
        if not normalize:
            return clipped
        low, high = norm_range
        normalized = (clipped - min_hu) / (max_hu - min_hu)
        if (low, high) != (0.0, 1.0):
            normalized = normalized * (high - low) + low
        return normalized.astype(np.float32)

    else:
        raise TypeError(f"Unsupported volume type: {type(volume)}")


def apply_multi_window(
    volume: Union[np.ndarray, torch.Tensor],
    window_dict: Optional[Dict[str, Tuple[float, float]]] = None,
    norm_range: Tuple[float, float] = (0.0, 1.0),
) -> Union[np.ndarray, torch.Tensor]:
    """
    Multi-window extension point (spec Section 8): creates multi-channel volume
    by stacking multiple clinical windows (e.g. soft-tissue, bone, lung).

    Args:
        volume: 3D array (D, H, W).
        window_dict: Mapping of channel_name -> (min_hu, max_hu).
        norm_range: Target range per channel.

    Returns:
        4D array (C, D, H, W) where C = len(window_dict).
    """
    if window_dict is None:
        window_dict = {
            "soft_tissue": STANDARD_WINDOWS["abdominal_soft_tissue"],
            "bone": STANDARD_WINDOWS["bone"],
            "lung": STANDARD_WINDOWS["lung"],
        }

    channels = []
    for name, (min_hu, max_hu) in window_dict.items():
        ch = apply_hu_window(volume, min_hu=min_hu, max_hu=max_hu, normalize=True, norm_range=norm_range)
        channels.append(ch)

    if isinstance(volume, torch.Tensor):
        return torch.stack(channels, dim=0)
    else:
        return np.stack(channels, axis=0)
