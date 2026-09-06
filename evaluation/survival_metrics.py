#!/usr/bin/env python3
"""
evaluation/survival_metrics.py

Survival analysis metrics and loss functions for patient-level modeling.
Provides:
  - harrell_c_index: Standard Harrell's Concordance Index in NumPy.
  - CoxLoss: Negative Log Partial Likelihood loss for continuous survival time in PyTorch.
"""

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def harrell_c_index(
    risk_scores: Union[np.ndarray, torch.Tensor],
    durations: Union[np.ndarray, torch.Tensor],
    events: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Computes Harrell's Concordance Index (C-index).
    Higher risk score indicates shorter predicted survival time.

    Args:
        risk_scores: Predicted risk scores / log hazards of shape (N,).
        durations: Observed survival times of shape (N,).
        events: Event indicators of shape (N,) (1 = event/death, 0 = right-censored).

    Returns:
        c_index: Concordance index between 0.0 and 1.0 (0.5 is random chance).
    """
    if isinstance(risk_scores, torch.Tensor):
        risk_scores = risk_scores.detach().cpu().numpy().flatten()
    if isinstance(durations, torch.Tensor):
        durations = durations.detach().cpu().numpy().flatten()
    if isinstance(events, torch.Tensor):
        events = events.detach().cpu().numpy().flatten()

    n = len(durations)
    if n == 0:
        return 0.5

    concordant = 0.0
    total_pairs = 0.0

    for i in range(n):
        # Patient i must have experienced the event
        if events[i] == 1:
            for j in range(n):
                if i == j:
                    continue
                # Patient j must have survived longer than patient i
                if durations[j] > durations[i]:
                    total_pairs += 1.0
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1.0
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5

    if total_pairs == 0:
        return 0.5

    return float(concordant / total_pairs)


class CoxLoss(nn.Module):
    """
    Negative Log Partial Likelihood loss for Cox proportional hazards model.
    Assumes Breslow approximation for tied event times.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        risk_scores: torch.Tensor,
        durations: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            risk_scores: Predicted log hazards of shape (B, 1) or (B,).
            durations: Survival times of shape (B,).
            events: Binary event indicators of shape (B,) (1.0 = death, 0.0 = censored).
        Returns:
            loss: Scalar negative log partial likelihood.
        """
        r = risk_scores.view(-1)
        t = durations.view(-1)
        e = events.view(-1)

        if e.sum() == 0:
            return torch.tensor(0.0, device=r.device, requires_grad=True)

        # Sort by duration in descending order
        sorted_indices = torch.argsort(t, descending=True)
        r_sorted = r[sorted_indices]
        e_sorted = e[sorted_indices]

        # Log cumulative sum of exp(r) along the sorted order
        # log_risk[i] represents log(sum_{j <= i} exp(r_sorted[j]))
        log_risk = torch.logcumsumexp(r_sorted, dim=0)

        # Negative log-likelihood contribution for uncensored patients
        uncensored_loss = r_sorted - log_risk
        event_loss = uncensored_loss * e_sorted

        num_events = e_sorted.sum()
        return -event_loss.sum() / (num_events + self.eps)
