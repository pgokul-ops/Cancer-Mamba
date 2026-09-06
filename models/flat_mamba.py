"""
models/flat_mamba.py

Minimal Flat (Non-Hierarchical) 3D Mamba architecture for Stage 3.
Architecture:
  3D Patch Embedding (patch_size=8 -> 1000 tokens of dim d_model=128)
  -> N x MambaBlock (Pure PyTorch S6 Selective SSM layers)
  -> LayerNorm
  -> Global Mean Pooling over sequence dimension
  -> Linear Classifier Head -> 1 logit
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from models.mamba_block import MambaBlock
from models.patch_embedding import PatchEmbedding3D


class FlatMamba3D(nn.Module):
    """
    Flat 3D Mamba model processing a full volume as a single flat token sequence.
    No hierarchical downsampling or multi-resolution pooling is used in this stage.

    Args:
        in_channels: Input volume channels (default: 1).
        num_classes: Classification output dimension (default: 1).
        d_model: Hidden token dimension (default: 128).
        n_layers: Number of stacked Mamba blocks (default: 4).
        patch_size: 3D patch resolution (default: 8, giving 10x10x10=1000 tokens).
        volume_shape: Input volume dimensions (D, H, W). Default: (80, 80, 80).
        d_state: SSM latent state dimension (default: 16).
        expand: SSM block expansion factor (default: 1.5).
        d_conv: 1D causal convolution kernel size (default: 4).
        dropout: Dropout rate (default: 0.2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        patch_size: Union[int, Tuple[int, int, int]] = 8,
        volume_shape: Tuple[int, int, int] = (80, 80, 80),
        d_state: int = 16,
        expand: float = 1.5,
        d_conv: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.volume_shape = volume_shape

        # Patchify 3D volume into token sequence
        self.patch_embed = PatchEmbedding3D(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            volume_shape=volume_shape,
            add_pos_embed=True,
        )
        self.num_patches = self.patch_embed.num_patches

        # Stack of N flat Mamba blocks
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

        # Final normalization and classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input volume of shape (B, 1, D, H, W).
        Returns:
            logits: Output logits of shape (B, 1).
        """
        # Patchify: (B, 1000, d_model)
        tokens = self.patch_embed(x)

        # Process sequentially through flat Mamba blocks
        for layer in self.layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)

        # Sequence pooling: concatenate global average (volume background)
        # and feature maxima (salient dense/tumor activations) across 1000 tokens
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)  # (B, 2 * d_model)

        # Predict logit
        logits = self.classifier(pooled)  # (B, 1)
        return logits


def get_model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
