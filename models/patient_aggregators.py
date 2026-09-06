#!/usr/bin/env python3
"""
models/patient_aggregators.py

Patient-level volume aggregation architectures for multi-phase / longitudinal imaging:
  1. MeanPoolingAggregator: Masked average over patient's volume embeddings.
  2. MaxPoolingAggregator: Masked max over patient's volume embeddings.
  3. AttentionPoolingAggregator: Learnable query cross-attention over volume embeddings.
  4. PatientMambaAggregator: Selective SSM sequence modeling over ordered volume embeddings.

Each aggregator produces:
  - patient_embedding: (B, d_model)
  - logits: (B, 1) for 5-year survival binary classification
  - risk_scores: (B, 1) log hazard for Cox continuous survival modeling
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_block import MambaBlock


class BasePatientAggregator(nn.Module):
    """Base class with shared dual heads for classification and survival."""

    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_hidden, 1),
        )
        # Cox hazard head (bias is typically absorbed by baseline hazard)
        self.survival_head = nn.Linear(d_model, 1, bias=False)

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Abstract pooling method to be implemented by child classes."""
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Patient volume sequences of shape (B, L, d_model).
            mask: Boolean tensor of shape (B, L) (True for valid volumes, False for padded).
        Returns:
            Dict containing 'patient_embedding', 'logits', and 'risk_scores'.
        """
        pooled = self.pool(x, mask)
        logits = self.classifier(pooled)
        risk_scores = self.survival_head(pooled)
        return {
            "patient_embedding": pooled,
            "logits": logits,
            "risk_scores": risk_scores,
        }


class MeanPoolingAggregator(BasePatientAggregator):
    """Aggregates volume embeddings via masked mean pooling."""

    def __init__(self, d_model: int = 256, d_hidden: int = 64, dropout: float = 0.2):
        super().__init__(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
        emb_sum = (x * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        return emb_sum / counts


class MaxPoolingAggregator(BasePatientAggregator):
    """Aggregates volume embeddings via masked max pooling."""

    def __init__(self, d_model: int = 256, d_hidden: int = 64, dropout: float = 0.2):
        super().__init__(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        x_masked = x.masked_fill(~mask_expanded, -1e9)
        return x_masked.max(dim=1).values


class AttentionPoolingAggregator(BasePatientAggregator):
    """Aggregates volume embeddings via learnable query attention pooling."""

    def __init__(self, d_model: int = 256, d_hidden: int = 64, dropout: float = 0.2):
        super().__init__(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        self.query = nn.Linear(d_model, 1, bias=False)

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.query(x).squeeze(-1) / (self.d_model ** 0.5)  # (B, L)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (x * weights).sum(dim=1)


class PatientMambaAggregator(BasePatientAggregator):
    """
    Patient-level Selective State Space Model aggregator.
    Processes ordered volume sequences through Mamba blocks, followed by masked aggregation.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 1,
        d_state: int = 16,
        expand: float = 1.5,
        d_conv: int = 4,
        d_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
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

    def pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)

        # Zero-out padded positions before Mamba recurrence
        h = x * mask_f

        for layer in self.layers:
            h = layer(h)
            h = h * mask_f  # Re-zero padded positions after residual connection

        h = self.norm(h)
        h = h * mask_f

        # Masked mean of the causally updated representations
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = (h * mask_f).sum(dim=1) / counts
        return pooled
