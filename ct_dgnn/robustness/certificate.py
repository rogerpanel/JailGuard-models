"""Grönwall-based certified detection-stability theorem (Theorem 1).

Given two interaction graphs differing in k query features with
``‖φ(q₁) − φ(q₂)‖ ≤ ε`` for each modified query, the change in the
campaign decision is bounded by:

    ‖ŷ₁ − ŷ₂‖ ≤ L_MLP · L_pool · k · L_g · exp(L_f · T) · ε     (Thm 1)

Inverting this relation, the certified radius is:

    ε* = Δ_margin / (L_MLP · L_pool · k · L_g · exp(L_f · T))    (Eq. 10)

where Δ_margin = τ₊ − τ₋ is the classification margin.
"""
from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class CertificateResult:
    radius: float
    lipschitz_total: float
    margin: float
    bound: float


def gronwall_bound(
    L_f: float,
    L_g: float,
    L_pool: float,
    L_mlp: float,
    k: int,
    T: float,
    epsilon: float,
) -> float:
    """Right-hand side of Theorem 1."""
    return L_mlp * L_pool * k * L_g * math.exp(L_f * T) * epsilon


def certified_radius(
    L_f: float,
    L_g: float,
    L_pool: float,
    L_mlp: float,
    k: int,
    T: float,
    margin: float,
) -> CertificateResult:
    """Compute ε* and the aggregate Lipschitz pre-factor."""
    if k <= 0:
        raise ValueError("k must be positive")
    if margin <= 0:
        raise ValueError("margin must be positive (Δ_margin = τ_+ − τ_-)")
    L_total = L_mlp * L_pool * k * L_g * math.exp(L_f * T)
    eps_star = margin / L_total
    return CertificateResult(
        radius=eps_star,
        lipschitz_total=L_total,
        margin=margin,
        bound=L_total,   # empirical bound evaluated at ε=1
    )


def certificate_from_model(model, k: int, T_seconds: float,
                           margin: float) -> CertificateResult:
    """Harvest all L_* from a trained CTDGNNJailGuard instance and
    evaluate Theorem 1. Assumes spectral normalization is active."""
    import torch

    with torch.no_grad():
        L_f = float(model.dynamics.lipschitz_constant().item())
        L_g = float(model.messages.lipschitz_constant().item())
        L_pool = float(model.pool.lipschitz_constant().item())
        L_mlp = float(model.campaign_head.lipschitz_constant().item())
    # The integration is expressed in a unitless time in the ODE solver;
    # we scale T by the maximum learned temporal scale to stay within
    # the 24-hour integration window assumed in Sec. IV-D.
    T_eff = T_seconds / 86_400.0
    return certified_radius(L_f, L_g, L_pool, L_mlp, k, T_eff, margin)
