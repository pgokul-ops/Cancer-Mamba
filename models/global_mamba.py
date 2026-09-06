"""
models/global_mamba.py

Global Mamba module for Stage 4 Hierarchical 3D Mamba.
Applies standard (non-windowed) S6 selective SSM recurrence over the regional token sequence
(length 125 tokens for w=2), capturing long-range volumetric interactions across the patient's anatomy.
"""

from typing import Optional
import torch
import torch.nn as nn

from models.mamba_block import MambaBlock


class GlobalMamba(nn.Module):
    """
    Global 3D Mamba stage processing the reduced regional token sequence.

    Args:
        d_model: Feature dimension (default: 128).
        num_tokens: Number of regional tokens (default: 125 for 10x10x10 grid with w=2).
        n_layers: Number of stacked global Mamba blocks (default: 2).
        d_state: SSM latent state dimension (default: 16).
        expand: SSM expansion factor (default: 1.5).
        d_conv: 1D causal convolution kernel size (default: 4).
        dropout: Dropout rate (default: 0.2).
        add_pos_embed: Whether to add learnable positional embeddings (default: True).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_tokens: int = 125,
        n_layers: int = 2,
        d_state: int = 16,
        expand: float = 1.5,
        d_conv: int = 4,
        dropout: float = 0.2,
        add_pos_embed: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.n_layers = n_layers

        if add_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

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
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Regional tokens of shape (B, num_tokens, d_model).
        Returns:
            out: Contextualized regional tokens of shape (B, num_tokens, d_model).
        """
        b, l, d = x.shape

        if self.pos_embed is not None:
            if l == self.pos_embed.shape[1]:
                x = x + self.pos_embed
            else:
                pos_interp = torch.nn.functional.interpolate(
                    self.pos_embed.transpose(1, 2),
                    size=l,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
                x = x + pos_interp

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x
