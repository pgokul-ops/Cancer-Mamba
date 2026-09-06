#!/usr/bin/env python3
"""
tests/test_patient_mamba.py

Unit tests for Stage 5:
  1. Length-1 sequence edge case (passthrough vs padded).
  2. Padding mask invariance (padding tokens do not distort valid tokens).
  3. All 4 aggregators (Mean, Max, Attention, Mamba) gradient flow and outputs.
  4. Harrell C-index and Cox loss numerical correctness.
"""

import numpy as np
import pytest
import torch

from evaluation.survival_metrics import CoxLoss, harrell_c_index
from models.patient_aggregators import (
    AttentionPoolingAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
    PatientMambaAggregator,
)


def test_length_one_sequence_edge_case():
    """Verifies that length-1 sequences (33 single-volume patients) execute cleanly without NaNs."""
    model = PatientMambaAggregator(d_model=256, n_layers=1)
    model.eval()

    # B=2, L=1
    x = torch.randn(2, 1, 256)
    mask = torch.ones(2, 1, dtype=torch.bool)

    out = model(x, mask)
    assert out["patient_embedding"].shape == (2, 256)
    assert out["logits"].shape == (2, 1)
    assert out["risk_scores"].shape == (2, 1)
    assert not torch.isnan(out["patient_embedding"]).any()
    assert not torch.isnan(out["logits"]).any()
    assert not torch.isnan(out["risk_scores"]).any()


def test_padding_mask_invariance():
    """
    Verifies that adding padded positions to a patient sequence does not alter
    the pooled representation of Mean and Max aggregators.
    """
    mean_agg = MeanPoolingAggregator(d_model=256)
    max_agg = MaxPoolingAggregator(d_model=256)

    # 1 patient with L=2 valid volumes
    x_valid = torch.randn(1, 2, 256)
    mask_valid = torch.ones(1, 2, dtype=torch.bool)

    # Same patient in a batch padded to L=5
    x_padded = torch.zeros(1, 5, 256)
    x_padded[:, :2, :] = x_valid
    x_padded[:, 2:, :] = torch.randn(1, 3, 256)  # Arbitrary noise in padded positions
    mask_padded = torch.tensor([[True, True, False, False, False]], dtype=torch.bool)

    # Mean pooling
    pool_v = mean_agg.pool(x_valid, mask_valid)
    pool_p = mean_agg.pool(x_padded, mask_padded)
    assert torch.allclose(pool_v, pool_p, atol=1e-5), "Mean pooling affected by padding!"

    # Max pooling
    pool_v_max = max_agg.pool(x_valid, mask_valid)
    pool_p_max = max_agg.pool(x_padded, mask_padded)
    assert torch.allclose(pool_v_max, pool_p_max, atol=1e-5), "Max pooling affected by padding!"


@pytest.mark.parametrize("agg_cls", [
    MeanPoolingAggregator,
    MaxPoolingAggregator,
    AttentionPoolingAggregator,
    PatientMambaAggregator,
])
def test_all_aggregators_forward_backward(agg_cls):
    """Tests forward and backward gradient propagation across all 4 aggregators."""
    model = agg_cls(d_model=64, d_hidden=32)
    model.train()

    # B=3, L=4 with variable valid lengths [1, 2, 4]
    x = torch.randn(3, 4, 64, requires_grad=True)
    mask = torch.tensor([
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, True],
    ], dtype=torch.bool)

    out = model(x, mask)
    assert out["patient_embedding"].shape == (3, 64)
    assert out["logits"].shape == (3, 1)
    assert out["risk_scores"].shape == (3, 1)

    loss = out["logits"].sum() + out["risk_scores"].sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_harrell_c_index_correctness():
    """Verifies Harrell C-index on synthetic known survival profiles."""
    # Perfect concordance: risk [3, 2, 1] for survival times [10, 20, 30], all events=1
    c_perf = harrell_c_index(np.array([3.0, 2.0, 1.0]), np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1]))
    assert c_perf == 1.0

    # Perfectly inverted concordance
    c_inv = harrell_c_index(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1]))
    assert c_inv == 0.0

    # Tied predictions: should give 0.5
    c_tied = harrell_c_index(np.array([2.0, 2.0]), np.array([10.0, 20.0]), np.array([1, 1]))
    assert c_tied == 0.5


def test_cox_loss_gradient():
    """Verifies that CoxLoss computes finite loss and gradients."""
    loss_fn = CoxLoss()
    risk = torch.tensor([1.5, 0.5, -0.5], requires_grad=True)
    durations = torch.tensor([50.0, 150.0, 300.0])
    events = torch.tensor([1.0, 1.0, 0.0])

    loss = loss_fn(risk, durations, events)
    assert loss.item() > 0.0
    loss.backward()
    assert risk.grad is not None
    assert not torch.isnan(risk.grad).any()
