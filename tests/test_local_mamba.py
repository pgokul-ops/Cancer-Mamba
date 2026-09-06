"""
tests/test_local_mamba.py

Unit tests for Stage 4 Local Windowed Mamba:
1. Mathematical invertibility: window_reverse3d(window_partition3d(x)) == x.
2. 3D Spatial Correctness: verifies that token window assignments correspond strictly
   to 3D coordinate neighborhoods (z//w, y//w, x//w), with zero cross-boundary leakage.
3. Shape preservation and gradient flow through LocalWindowedMamba.
"""

import pytest
import torch
from models.local_mamba import LocalWindowedMamba, window_partition3d, window_reverse3d


def test_window_partition_reverse_invertibility():
    """Verify that partitioning and reversing perfectly restores the original 3D tensor."""
    b, c = 2, 64
    for w in [1, 2, 5, 10]:
        x = torch.randn(b, 10, 10, 10, c)
        windows, grid_windows = window_partition3d(x, window_size=w)
        sz, sy, sx = grid_windows
        expected_windows = (10 // w) ** 3
        expected_tokens_per_window = w ** 3

        assert windows.shape == (b * expected_windows, expected_tokens_per_window, c)

        x_rec = window_reverse3d(windows, window_size=w, grid_windows=grid_windows, b=b)
        assert x_rec.shape == x.shape
        diff = (x - x_rec).abs().max().item()
        assert diff == 0.0, f"Inversion failed for window_size {w}, max diff: {diff}"


def test_spatial_coordinate_correctness():
    """
    Verify that every token in a window belongs to the true 3D spatial neighborhood
    and spot-check hand-calculated (z, y, x) -> window_id mappings.
    """
    w = 2
    gz, gy, gx = 10, 10, 10
    sz, sy, sx = gz // w, gy // w, gx // w  # 5, 5, 5

    # Fill tensor with explicit (z, y, x) integer coordinates in feature channels
    coords = torch.zeros(1, gz, gy, gx, 3, dtype=torch.long)
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                coords[0, z, y, x] = torch.tensor([z, y, x])

    windows, grid_windows = window_partition3d(coords, window_size=w)
    # Shape: (125, 8, 3)
    assert windows.shape == (125, 8, 3)

    # 1. Spot-check hand-calculated mappings:
    # (z=0, y=0, x=0) must be in window 0
    # (z=0, y=0, x=3) -> wz=0, wy=0, wx=1 -> window_id = 0*25 + 0*5 + 1 = 1
    # (z=3, y=5, x=7) -> wz=1, wy=2, wx=3 -> window_id = 1*25 + 2*5 + 3 = 38
    # (z=9, y=9, x=9) -> wz=4, wy=4, wx=4 -> window_id = 4*25 + 4*5 + 4 = 124
    test_cases = [
        ((0, 0, 0), 0),
        ((0, 0, 1), 0),
        ((0, 0, 2), 1),
        ((0, 0, 3), 1),
        ((3, 5, 7), 38),
        ((9, 9, 9), 124),
    ]

    for (tz, ty, tx), expected_wid in test_cases:
        target_token = torch.tensor([tz, ty, tx])
        # Find which window contains this exact token
        matches = [(wid, t) for wid in range(125) for t in range(8) if torch.equal(windows[wid, t], target_token)]
        assert len(matches) == 1, f"Token {(tz, ty, tx)} found in {len(matches)} windows!"
        found_wid = matches[0][0]
        assert found_wid == expected_wid, (
            f"Token {(tz, ty, tx)} mapped to window {found_wid}, expected {expected_wid}"
        )

    # 2. Exhaustive neighborhood boundary check across all 125 windows:
    for wid in range(125):
        wz = wid // (sy * sx)
        wy = (wid % (sy * sx)) // sx
        wx = wid % sx

        window_tokens = windows[wid]  # (8, 3)
        for t in range(8):
            pz, py, px = window_tokens[t].tolist()
            assert pz // w == wz, f"Window {wid} contains z={pz}, expected z in [{wz*w}, {wz*w+w-1}]"
            assert py // w == wy, f"Window {wid} contains y={py}, expected y in [{wy*w}, {wy*w+w-1}]"
            assert px // w == wx, f"Window {wid} contains x={px}, expected x in [{wx*w}, {wx*w+w-1}]"


def test_local_mamba_forward_backward():
    """Verify LocalWindowedMamba layer forward shape preservation and backward gradient flow."""
    b, gz, gy, gx, c = 2, 10, 10, 10, 64
    x = torch.randn(b, gz, gy, gx, c, requires_grad=True)
    layer = LocalWindowedMamba(d_model=c, window_size=2, n_layers=2, d_state=16, expand=1.5)

    out = layer(x)
    assert out.shape == (b, gz, gy, gx, c)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    for name, param in layer.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} received no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaNs!"
