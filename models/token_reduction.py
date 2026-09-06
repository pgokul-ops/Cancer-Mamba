"""
models/token_reduction.py

Token Reduction module for Stage 4 Hierarchical 3D Mamba.
Downsamples local 3D window tokens (w^3 tokens per window) into 1 regional token per window,
reducing the global sequence length from 1000 to 125 regional tokens.

Supported methods:
  - 'learned_proj' (default): Flattens each window's tokens and applies a linear projection + LayerNorm + GELU.
  - 'mean_pool': Computes the spatial mean across each window's tokens.
"""

from typing import Optional
import torch
import torch.nn as nn

from models.local_mamba import window_partition3d


class TokenReduction3D(nn.Module):
    """
    Downsamples 3D volumetric token windows into regional tokens for the global stage.

    Args:
        d_model: Input token feature dimension (default: 128).
        d_out: Output regional token dimension (default: d_model).
        window_size: Cubic window edge length w (default: 2 -> w^3 = 8 tokens).
        method: Reduction strategy, either 'learned_proj' (default) or 'mean_pool'.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_out: Optional[int] = None,
        window_size: int = 2,
        method: str = "learned_proj",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out or d_model
        self.window_size = window_size
        self.window_tokens = window_size ** 3
        self.method = method.lower()

        assert self.method in ["learned_proj", "mean_pool"], (
            f"Unsupported token reduction method: {method}. Choose 'learned_proj' or 'mean_pool'."
        )

        if self.method == "learned_proj":
            self.proj = nn.Sequential(
                nn.Linear(self.window_tokens * d_model, self.d_out),
                nn.LayerNorm(self.d_out),
                nn.GELU(),
            )
        else:
            if self.d_out != self.d_model:
                self.proj = nn.Linear(self.d_model, self.d_out)
            else:
                self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: 3D grid tensor (B, Gz, Gy, Gx, d_model) or window tensor (B, Nw, w^3, d_model).
        Returns:
            regional_tokens: (B, Nw, d_out) where Nw = (Gz/w) * (Gy/w) * (Gx/w) (e.g. 125).
        """
        if x.dim() == 5:
            # (B, Gz, Gy, Gx, d_model)
            b = x.shape[0]
            windows, grid_windows = window_partition3d(x, self.window_size)
            # windows: (B * Nw, w^3, d_model)
            num_windows = grid_windows[0] * grid_windows[1] * grid_windows[2]
            windows = windows.view(b, num_windows, self.window_tokens, self.d_model)
        elif x.dim() == 4:
            # Already in window format: (B, Nw, w^3, d_model)
            b = x.shape[0]
            windows = x
        else:
            raise ValueError(f"Expected 4D or 5D input to TokenReduction3D, got {x.dim()}D")

        b, nw, lw, c = windows.shape

        if self.method == "learned_proj":
            # Flatten window tokens: (B, Nw, lw * c) -> (B, Nw, d_out)
            flat_windows = windows.reshape(b, nw, lw * c)
            regional_tokens = self.proj(flat_windows)
        else:
            # Spatial average pool over window dimension: (B, Nw, c)
            mean_tokens = windows.mean(dim=2)
            regional_tokens = self.proj(mean_tokens)

        return regional_tokens
