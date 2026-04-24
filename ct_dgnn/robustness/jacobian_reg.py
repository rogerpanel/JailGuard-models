"""Jacobian-Frobenius regularizer for the type-specific ODE dynamics.

Implements Eq. (9) of the manuscript:

    L_Jacobian = λ_J · E_{t ∈ U[t_0, T]}[ ‖ ∂f_τ / ∂h_v |_t ‖_F² ]

with λ_J = 0.01. Hutchinson's estimator is used so that the regulariser
costs just one additional backward pass per step.
"""
from __future__ import annotations

import torch
from torch import nn


def jacobian_frobenius(
    f: nn.Module,
    h: torch.Tensor,
    message: torch.Tensor,
    node_type_ids: torch.Tensor,
    num_samples: int = 1,
) -> torch.Tensor:
    """Hutchinson estimator of ‖∂f/∂h‖_F² at the given operating point.

    Parameters
    ----------
    f : a module implementing ``f(h, message, node_type_ids) -> dh``.
    h : (N, d) hidden state (will be made `requires_grad`).
    message, node_type_ids : passed through to ``f`` unchanged.
    num_samples : number of Rademacher probes. Default 1 (minimal cost).
    """
    h = h.detach().clone().requires_grad_(True)
    total = torch.zeros((), device=h.device)
    for _ in range(num_samples):
        v = torch.randint_like(h, low=0, high=2).float().mul_(2).sub_(1)  # ±1
        out = f(h, message, node_type_ids)
        grad = torch.autograd.grad(
            outputs=out,
            inputs=h,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
        )[0]
        total = total + (grad * v).pow(2).sum() / h.size(0)
    return total / num_samples
