"""Utility helpers for harvesting per-module Lipschitz constants.

The spectral-normalization parametrisation keeps ``σ(W) ≤ 1`` for every
weight matrix; the helpers below walk the module tree to aggregate
constants per logical sub-component (dynamics, messages, attention,
pool, classifier)."""
from __future__ import annotations

from typing import Dict

import torch

from ct_dgnn.models.ct_dgnn import CTDGNNJailGuard


@torch.no_grad()
def model_lipschitz_constants(model: CTDGNNJailGuard) -> Dict[str, float]:
    return {
        "L_f":    float(model.dynamics.lipschitz_constant().item()),
        "L_g":    float(model.messages.lipschitz_constant().item()),
        "L_attn": float(model.attention.lipschitz_constant().item()),
        "L_pool": float(model.pool.lipschitz_constant().item()),
        "L_mlp":  float(model.campaign_head.lipschitz_constant().item()),
    }


@torch.no_grad()
def projection_product(linears: list) -> float:
    """Multiply the spectral norms of a list of SpectralLinear layers."""
    prod = 1.0
    for layer in linears:
        prod *= float(layer.spectral_norm().item())
    return prod
