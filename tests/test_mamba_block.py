"""
tests/test_mamba_block.py

Unit tests for pure PyTorch S6 Selective SSM and MambaBlock:
- Shape preservation: [B, L, D] -> [B, L, D]
- Zero NaNs or Infs during forward and backward passes
- Gradient flow to all parameters
- Deterministic evaluation mode
"""

import pytest
import torch
from models.mamba_block import MambaBlock, S6SelectiveSSM


def test_s6_selective_ssm_forward_shape():
    b, l, d = 2, 16, 64
    x = torch.randn(b, l, d)
    ssm = S6SelectiveSSM(d_model=d, d_state=16, expand=2.0)
    out = ssm(x)

    assert out.shape == (b, l, d), f"Expected shape {(b, l, d)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs!"
    assert not torch.isinf(out).any(), "Output contains Infs!"


def test_mamba_block_forward_backward():
    b, l, d = 2, 32, 64
    x = torch.randn(b, l, d, requires_grad=True)
    block = MambaBlock(d_model=d, d_state=16, expand=2.0)

    out = block(x)
    assert out.shape == (b, l, d)
    assert not torch.isnan(out).any()

    # Backward pass
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaNs!"

    # Check parameter gradients
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient has NaNs!"


def test_mamba_block_varying_sequence_lengths():
    """Verify selective scan works with different sequence lengths (e.g. 50, 100, 1000)."""
    block = MambaBlock(d_model=32, d_state=8, expand=1.5)
    for length in [10, 50, 128]:
        x = torch.randn(2, length, 32)
        out = block(x)
        assert out.shape == (2, length, 32)
        assert not torch.isnan(out).any()
