"""
models/local_mamba.py

Local Windowed 3D Mamba module for Stage 4.
Partitions volumetric patch tokens into spatially contiguous 3D cubic neighborhoods
(e.g., w=2 -> 2x2x2=8 tokens per window across 5x5x5=125 windows), runs selective SSM
recurrence within each local window, and reconstructs the 3D volume grid.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

from models.mamba_block import MambaBlock


def window_partition3d(
    x: torch.Tensor,
    window_size: int = 2,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partitions a 3D token grid into spatially contiguous non-overlapping 3D cubic windows.

    Args:
        x: Input tensor of shape (B, Gz, Gy, Gx, C).
        window_size: Cubic window edge length w (must divide Gz, Gy, Gx).
    Returns:
        windows: Tensor of shape (B * num_windows, w^3, C).
        grid_windows: Tuple of (Sz, Sy, Sx) indicating window grid dimensions.
    """
    b, gz, gy, gx, c = x.shape
    w = window_size
    assert gz % w == 0 and gy % w == 0 and gx % w == 0, (
        f"Grid dimensions ({gz}, {gy}, {gx}) must be divisible by window_size {w}"
    )

    sz, sy, sx = gz // w, gy // w, gx // w

    # Reshape: (B, Sz, w, Sy, w, Sx, w, C)
    x = x.view(b, sz, w, sy, w, sx, w, c)

    # Permute to group window coordinates: (B, Sz, Sy, Sx, w, w, w, C)
    # This guarantees that tokens within a window are contiguous 3D neighborhoods.
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()

    # Fold window dimension into batch: (B * Sz * Sy * Sx, w^3, C)
    num_windows = sz * sy * sx
    window_tokens = w * w * w
    windows = x.view(b * num_windows, window_tokens, c)

    return windows, (sz, sy, sx)


def window_reverse3d(
    windows: torch.Tensor,
    window_size: int,
    grid_windows: Tuple[int, int, int],
    b: int,
) -> torch.Tensor:
    """
    Reconstructs the original 3D token grid from folded 3D window partitions.

    Args:
        windows: Folded window tensor of shape (B * num_windows, w^3, C).
        window_size: Cubic window edge length w.
        grid_windows: Tuple of (Sz, Sy, Sx) window grid dimensions.
        b: Original batch size B.
    Returns:
        x: Reconstructed tensor of shape (B, Gz, Gy, Gx, C).
    """
    w = window_size
    sz, sy, sx = grid_windows
    c = windows.shape[-1]

    # Reshape: (B, Sz, Sy, Sx, w, w, w, C)
    x = windows.view(b, sz, sy, sx, w, w, w, c)

    # Invert permutation: (B, Sz, w, Sy, w, Sx, w, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()

    # Reshape back to full 3D grid: (B, Sz*w, Sy*w, Sx*w, C)
    gz, gy, gx = sz * w, sy * w, sx * w
    x = x.view(b, gz, gy, gx, c)

    return x


class LocalWindowedMamba(nn.Module):
    """
    Local Windowed 3D Mamba Layer.
    Processes volumetric tokens in parallel across local 3D cubic windows of size w^3.
    """

    def __init__(
        self,
        d_model: int = 128,
        window_size: int = 2,
        n_layers: int = 2,
        d_state: int = 16,
        expand: float = 1.5,
        d_conv: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.n_layers = n_layers

        # Stack of Mamba blocks processing window sequences of length w^3
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D grid tensor of shape (B, Gz, Gy, Gx, C).
        Returns:
            out: Processed 3D grid tensor of shape (B, Gz, Gy, Gx, C).
        """
        b, gz, gy, gx, c = x.shape

        # 1. Partition into 3D cubic windows: (B * Nw, w^3, C)
        windows, grid_windows = window_partition3d(x, self.window_size)

        # 2. Process in parallel through local Mamba blocks
        for layer in self.layers:
            windows = layer(windows)

        # 3. Reconstruct native 3D spatial grid: (B, Gz, Gy, Gx, C)
        out = window_reverse3d(windows, self.window_size, grid_windows, b)
        return out
