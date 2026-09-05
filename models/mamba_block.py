"""
models/mamba_block.py

Pure PyTorch implementation of the S6 Selective State-Space Model (SSM) block.
Features:
  - Input-dependent parameter discretization (A, B, C, Delta).
  - Causal 1D depthwise convolution pre-mixing.
  - State-space recurrence with SiLU multiplicative gating.
  - Pre-LayerNorm and residual connection.
  - Pure PyTorch execution compatible with any GPU architecture (including Turing sm_75).
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan_sequential(
    u: torch.Tensor,       # (B, L, D_in)
    delta: torch.Tensor,   # (B, L, D_in)
    A: torch.Tensor,       # (D_in, N)
    B: torch.Tensor,       # (B, L, N)
    C: torch.Tensor,       # (B, L, N)
    D: Optional[torch.Tensor] = None,  # (D_in,)
) -> torch.Tensor:
    """
    Selective scan recurrence in pure PyTorch:
      dA = exp(delta * A)
      dB = delta * B
      h_t = dA_t * h_{t-1} + dB_t * u_t
      y_t = h_t @ C_t + D * u_t
    """
    b, l, d_in = u.shape
    n = A.shape[1]

    # Compute discretized A_bar: (B, L, D_in, N)
    # delta: (B, L, D_in, 1), A: (1, 1, D_in, N) -> delta * A: (B, L, D_in, N)
    delta_expanded = delta.unsqueeze(-1)
    A_expanded = A.view(1, 1, d_in, n)
    A_bar = torch.exp(delta_expanded * A_expanded)

    # Compute discretized B_bar * u:
    # B: (B, L, 1, N), u: (B, L, D_in, 1) -> (B, L, D_in, N)
    B_expanded = B.unsqueeze(2)
    u_expanded = u.unsqueeze(-1)
    Bu_bar = delta_expanded * B_expanded * u_expanded

    # C: (B, L, 1, N)
    C_expanded = C.unsqueeze(2)

    # Sequential scan loop over sequence dimension
    h = torch.zeros(b, d_in, n, device=u.device, dtype=u.dtype)
    ys = []

    for t in range(l):
        h = A_bar[:, t] * h + Bu_bar[:, t]
        y_t = (h * C_expanded[:, t]).sum(dim=-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (B, L, D_in)

    if D is not None:
        y = y + u * D.view(1, 1, d_in)

    return y


class S6SelectiveSSM(nn.Module):
    """
    Core S6 Selective State Space Model.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        expand: float = 2.0,
        d_conv: int = 4,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_conv = d_conv

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)

        # In-projection to [x_proj, z]
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 1D Depthwise Causal Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Input-dependent parameter projections from x_conv
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )

        # Delta projection from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt projection bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: inv_softplus(x) = log(exp(x) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Initialize S4D A parameter as negative diagonal (HiPPO style)
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            repeats=self.d_inner,
            dim=0,
        )
        self.A_log = nn.Parameter(torch.log(A))

        # Direct feedthrough parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) token sequence.
        Returns:
            out: (B, L, D) processed sequence.
        """
        b, l, d = x.shape

        # 1. Project input to x_proj and gating signal z: (B, L, 2 * D_in)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, L, D_in)

        # 2. 1D Causal Convolution: permute to (B, D_in, L)
        x_conv_in = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv_in)[:, :, :l]  # causal truncation
        x_conv = x_conv.transpose(1, 2)            # back to (B, L, D_in)
        x_conv = F.silu(x_conv)

        # 3. Input-dependent projection to delta, B, C
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2 * d_state)
        delta, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Delta activation via softplus: (B, L, D_in)
        delta = F.softplus(self.dt_proj(delta))

        # Negative exponent for stable state matrix A
        A = -torch.exp(self.A_log.float())  # (D_in, N)

        # 4. Selective Scan
        y = selective_scan_sequential(
            u=x_conv,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
        )

        # 5. Multiplicative gating with SiLU(z)
        y = y * F.silu(z)

        # 6. Output projection
        out = self.out_proj(y)
        return out


class MambaBlock(nn.Module):
    """
    Standard Mamba Layer with Pre-LayerNorm and Residual Connection:
      out = x + S6(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        expand: float = 2.0,
        d_conv: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S6SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        residual = x
        x_norm = self.norm(x)
        out = self.ssm(x_norm)
        out = self.dropout(out)
        return residual + out
