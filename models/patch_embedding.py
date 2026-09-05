"""
models/patch_embedding.py

3D Patch Embedding module for volumetric CT representations in Cancer-Mamba.
Converts a 3D volume (B, 1, D_vol, H_vol, W_vol) into a token sequence (B, N, d_model).

Raster Scan Order Note:
  Spatial tokens are serialized into a 1D sequence in z-major, then y, then x order.
  This linear raster ordering is an explicit modeling choice: Mamba's 1D selective
  recurrence does not inherently understand 3D adjacency across scan-line boundaries.
  Alternative scan trajectories (e.g. multi-directional, Hilbert/Peano curves, or 3D
  hierarchical windows) represent candidate ablations reserved for Stage 4+.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class PatchEmbedding3D(nn.Module):
    """
    Projects 3D volumetric images into a sequence of 1D token embeddings.

    Args:
        in_channels: Number of input volume channels (default: 1 for HU CT).
        d_model: Token embedding dimension (default: 128).
        patch_size: 3D patch dimensions, either int or (Pz, Py, Px). Default: 8.
        volume_shape: Expected input volume dimensions (D, H, W). Default: (80, 80, 80).
        add_pos_embed: Whether to add a learnable 1D positional embedding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        patch_size: Union[int, Tuple[int, int, int]] = 8,
        volume_shape: Tuple[int, int, int] = (80, 80, 80),
        add_pos_embed: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            self.patch_size = patch_size

        self.d_model = d_model
        self.volume_shape = volume_shape

        # Patch projection via strided 3D convolution
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Compute grid dimensions: (Gz, Gy, Gx)
        self.grid_size = (
            volume_shape[0] // self.patch_size[0],
            volume_shape[1] // self.patch_size[1],
            volume_shape[2] // self.patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Learnable 1D positional embedding
        if add_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, 1, D, H, W) e.g. (B, 1, 80, 80, 80).
        Returns:
            tokens: (B, N, d_model) where N = Gz * Gy * Gx (e.g. 1000 for patch_size=8).
        """
        b, c, d, h, w = x.shape

        # Conv3d projection: (B, d_model, Gz, Gy, Gx)
        feat = self.proj(x)

        # Permute to (B, Gz, Gy, Gx, d_model) and flatten spatial dimensions in z-major order
        # (dim 1: Gz, dim 2: Gy, dim 3: Gx) -> raster scan ordering
        feat = feat.permute(0, 2, 3, 4, 1)  # (B, Gz, Gy, Gx, d_model)
        tokens = feat.flatten(1, 3)          # (B, Gz * Gy * Gx, d_model)

        if self.pos_embed is not None:
            # Handle possible dynamic sequence length if input shape changes
            if tokens.shape[1] == self.pos_embed.shape[1]:
                tokens = tokens + self.pos_embed
            else:
                # Interpolate pos_embed if sequence length differs
                pos_interp = torch.nn.functional.interpolate(
                    self.pos_embed.transpose(1, 2),
                    size=tokens.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
                tokens = tokens + pos_interp

        tokens = self.norm(tokens)
        return tokens
