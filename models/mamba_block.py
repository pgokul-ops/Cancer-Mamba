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


class SelectiveScanFn(torch.autograd.Function):
    """
    Pure PyTorch autograd function for S6 selective scan recurrence.
    Provides:
      - Explicit FP32 accumulation to prevent FP16 overflow under AMP.
      - Exact analytical adjoint recurrence for the backward pass, eliminating
        autograd graph tracking overhead across 1000 sequence steps.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None):
        # u: (B, L, D_in)
        # delta: (B, L, D_in)
        # A: (D_in, N)
        # B: (B, L, N)
        # C: (B, L, N)
        # D: Optional (D_in,)
        b, l, d_in = u.shape
        n = A.shape[1]

        u_f = u.float()
        delta_f = delta.float()
        A_f = A.float()
        B_f = B.float()
        C_f = C.float()

        delta_expanded = delta_f.unsqueeze(-1)          # (B, L, D_in, 1)
        A_expanded = A_f.view(1, 1, d_in, n)             # (1, 1, D_in, N)
        A_bar = torch.exp(delta_expanded * A_expanded)   # (B, L, D_in, N)

        B_expanded = B_f.unsqueeze(2)                    # (B, L, 1, N)
        u_expanded = u_f.unsqueeze(-1)                   # (B, L, D_in, 1)
        Bu_bar = delta_expanded * B_expanded * u_expanded # (B, L, D_in, N)
        C_expanded = C_f.unsqueeze(2)                    # (B, L, 1, N)

        h_all = torch.empty(b, l, d_in, n, device=u.device, dtype=torch.float32)
        y = torch.empty(b, l, d_in, device=u.device, dtype=torch.float32)

        h = torch.zeros(b, d_in, n, device=u.device, dtype=torch.float32)
        for t in range(l):
            h = A_bar[:, t] * h + Bu_bar[:, t]
            h_all[:, t] = h
            y[:, t] = (h * C_expanded[:, t]).sum(dim=-1)

        if D is not None:
            y = y + u_f * D.float().view(1, 1, d_in)

        # Clamp y to prevent float16 overflow (max ~65504) under AMP
        y = y.clamp(min=-500.0, max=500.0)

        ctx.save_for_backward(u_f, delta_f, A_f, B_f, C_f, D, A_bar, h_all)
        ctx.has_D = D is not None
        return y.to(u.dtype)

    @staticmethod
    def backward(ctx, dy):
        u_f, delta_f, A_f, B_f, C_f, D, A_bar, h_all = ctx.saved_tensors
        b, l, d_in = dy.shape
        n = A_f.shape[1]
        dy_f = dy.float()

        C_expanded = C_f.unsqueeze(2)
        B_expanded = B_f.unsqueeze(2)
        u_expanded = u_f.unsqueeze(-1)
        delta_expanded = delta_f.unsqueeze(-1)
        A_expanded = A_f.view(1, 1, d_in, n)

        dh = torch.zeros(b, d_in, n, device=dy.device, dtype=torch.float32)
        du_f = torch.zeros_like(u_f)
        ddelta_f = torch.zeros_like(delta_f)
        dA_f = torch.zeros_like(A_f)
        dB_f = torch.zeros_like(B_f)
        dC_f = torch.zeros_like(C_f)
        dD = (dy_f * u_f).sum(dim=(0, 1)) if ctx.has_D else None

        if ctx.has_D:
            du_f = du_f + dy_f * D.float().view(1, 1, d_in)

        for t in reversed(range(l)):
            dh = dh + dy_f[:, t].unsqueeze(-1) * C_expanded[:, t]
            dC_f[:, t] = (dy_f[:, t].unsqueeze(-1) * h_all[:, t]).sum(dim=1)
            dBu_bar_t = dh

            du_f[:, t] = du_f[:, t] + (dBu_bar_t * delta_expanded[:, t] * B_expanded[:, t]).sum(dim=-1)
            dB_f[:, t] = dB_f[:, t] + (dBu_bar_t * delta_expanded[:, t] * u_expanded[:, t]).sum(dim=1)
            ddelta_Bu = (dBu_bar_t * B_expanded[:, t] * u_expanded[:, t]).sum(dim=-1)

            h_prev = h_all[:, t - 1] if t > 0 else torch.zeros(b, d_in, n, device=dy.device, dtype=torch.float32)
            dA_bar_t = dh * h_prev
            ddelta_A_term = dA_bar_t * A_bar[:, t]
            dA_f = dA_f + (ddelta_A_term * delta_expanded[:, t]).sum(dim=0)
            ddelta_A = (ddelta_A_term * A_expanded).sum(dim=-1)

            ddelta_f[:, t] = ddelta_f[:, t] + ddelta_Bu + ddelta_A
            dh = dh * A_bar[:, t]

        return (
            du_f.clamp(min=-500.0, max=500.0).to(ctx.saved_tensors[0].dtype),
            ddelta_f.clamp(min=-500.0, max=500.0).to(ctx.saved_tensors[1].dtype),
            dA_f.clamp(min=-500.0, max=500.0).to(ctx.saved_tensors[2].dtype),
            dB_f.clamp(min=-500.0, max=500.0).to(ctx.saved_tensors[3].dtype),
            dC_f.clamp(min=-500.0, max=500.0).to(ctx.saved_tensors[4].dtype),
            dD.clamp(min=-500.0, max=500.0) if dD is not None else None,
        )


def selective_scan_sequential(
    u: torch.Tensor,       # (B, L, D_in)
    delta: torch.Tensor,   # (B, L, D_in)
    A: torch.Tensor,       # (D_in, N)
    B: torch.Tensor,       # (B, L, N)
    C: torch.Tensor,       # (B, L, N)
    D: Optional[torch.Tensor] = None,  # (D_in,)
) -> torch.Tensor:
    """
    Selective scan recurrence in pure PyTorch.
    Delegates to SelectiveScanFn for high-speed analytical backward and FP32 numerical stability.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D)


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

        # Delta activation via softplus: (B, L, D_in) with continuous discretization clamp
        delta = F.softplus(self.dt_proj(delta)).clamp(min=1e-4, max=0.5)

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
        y = (y.clamp(min=-500.0, max=500.0) * F.silu(z)).clamp(min=-500.0, max=500.0)

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
