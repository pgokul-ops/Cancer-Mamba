"""
models/hierarchical_mamba.py

Stage 4 Hierarchical 3D Mamba Architecture.
Architecture:
  3D Patch Embedding (patch_size=8 -> 10x10x10 native grid, d_model=128)
  -> Local Windowed Mamba (w=2 cubic windows, 125 parallel windows of 8 tokens each)
  -> Token Reduction (learned_proj or mean_pool -> 125 regional tokens)
  -> Global Mamba (long-range regional recurrence across 125 tokens)
  -> LayerNorm
  -> [mean, max] Sequence Pooling -> (B, 256)
  -> 2-layer MLP Classifier Head -> (B, 1)
"""

from typing import Tuple, Union
import torch
import torch.nn as nn

from models.global_mamba import GlobalMamba
from models.local_mamba import LocalWindowedMamba
from models.patch_embedding import PatchEmbedding3D
from models.token_reduction import TokenReduction3D


class HierarchicalMamba3D(nn.Module):
    """
    Hierarchical 3D Mamba combining local 3D windowed recurrence and global regional recurrence.

    Args:
        in_channels: Input volume channels (default: 1).
        num_classes: Classification output dimension (default: 1).
        d_model: Hidden token feature dimension (default: 128).
        patch_size: 3D patch dimensions (default: 8).
        volume_shape: Input volume dimensions (default: (80, 80, 80)).
        window_size: 3D local window edge length w (default: 2 -> 5x5x5=125 windows).
        local_layers: Number of local windowed Mamba layers (default: 2).
        global_layers: Number of global regional Mamba layers (default: 2).
        token_reduction: Method to downsample windows ('learned_proj' or 'mean_pool'). Default: 'learned_proj'.
        d_state: SSM latent state dimension (default: 16).
        expand: SSM expansion factor (default: 1.5).
        d_conv: 1D causal convolution kernel size (default: 4).
        dropout: Dropout rate (default: 0.2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        d_model: int = 128,
        patch_size: Union[int, Tuple[int, int, int]] = 8,
        volume_shape: Tuple[int, int, int] = (80, 80, 80),
        window_size: int = 2,
        local_layers: int = 2,
        global_layers: int = 2,
        token_reduction: str = "learned_proj",
        d_state: int = 16,
        expand: float = 1.5,
        d_conv: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.volume_shape = volume_shape
        self.window_size = window_size
        self.local_layers = local_layers
        self.global_layers = global_layers
        self.token_reduction_method = token_reduction

        # 1. 3D Patch Embedding
        self.patch_embed = PatchEmbedding3D(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            volume_shape=volume_shape,
            add_pos_embed=True,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        # Compute number of regional windows
        self.num_windows = (
            (self.grid_size[0] // window_size)
            * (self.grid_size[1] // window_size)
            * (self.grid_size[2] // window_size)
        )

        # 2. Local Windowed Mamba (processes w^3 = 8 tokens per window in parallel)
        self.local_stage = LocalWindowedMamba(
            d_model=d_model,
            window_size=window_size,
            n_layers=local_layers,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
        )

        # 3. Token Reduction (downsamples 1000 tokens -> 125 regional tokens)
        self.token_reduction = TokenReduction3D(
            d_model=d_model,
            d_out=d_model,
            window_size=window_size,
            method=token_reduction,
        )

        # 4. Global Mamba (processes 125 regional tokens)
        self.global_stage = GlobalMamba(
            d_model=d_model,
            num_tokens=self.num_windows,
            n_layers=global_layers,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
            add_pos_embed=True,
        )

        # 5. Normalization and Salient Pooling Head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input volume of shape (B, 1, D, H, W).
        Returns:
            logits: Output classification logits of shape (B, 1).
        """
        # 1. Patch embedding returned as native 3D grid: (B, 10, 10, 10, d_model)
        grid_tokens = self.patch_embed(x, return_grid=True)

        # 2. Local 3D windowed recurrence: (B, 10, 10, 10, d_model)
        local_tokens = self.local_stage(grid_tokens)

        # 3. Token reduction to regional tokens: (B, 125, d_model)
        regional_tokens = self.token_reduction(local_tokens)

        # 4. Global long-range recurrence: (B, 125, d_model)
        global_tokens = self.global_stage(regional_tokens)

        # 5. Normalization
        global_tokens = self.norm(global_tokens)

        # 6. Salient sequence pooling: [mean, max] -> (B, 2 * d_model)
        pooled = torch.cat(
            [global_tokens.mean(dim=1), global_tokens.max(dim=1).values], dim=-1
        )

        # 7. Predict logit
        logits = self.classifier(pooled)
        return logits


def get_model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
