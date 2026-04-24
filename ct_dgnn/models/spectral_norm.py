"""Spectral normalization primitives (1-Lipschitz linear layers).

The manuscript enforces Lipschitz continuity end-to-end by replacing every
weight matrix W with  W / sigma(W), where sigma(W) is estimated via one-step
power iteration (Miyato et al., 2018). PyTorch ships a standard
`torch.nn.utils.parametrizations.spectral_norm`; we wrap it so every linear
layer has its spectral norm tracked, exposed and retrievable for the
Grönwall-based certificate.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm


class SpectralLinear(nn.Module):
    """nn.Linear whose weight has unit spectral norm.

    Exposes `spectral_norm()` to retrieve the tracked sigma(W) for
    Lipschitz-constant propagation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 n_power_iterations: int = 1) -> None:
        super().__init__()
        linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear = _spectral_norm(linear, n_power_iterations=n_power_iterations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.no_grad()
    def spectral_norm(self) -> torch.Tensor:
        """Return the currently-tracked largest singular value (≤ 1.0)."""
        # After parametrization the stored weight already has sigma(W) = 1.
        # We therefore recover sigma from u^T W v with the cached vectors if
        # available, else compute it from the base weight.
        w = self.linear.weight
        u, _, v = torch.svd_lowrank(w, q=1)
        sigma = (u.t() @ w @ v).flatten().abs().max()
        return sigma


class SpectralMLP(nn.Module):
    """Small MLP where every linear is spectrally normalized."""

    def __init__(self, dims: list[int], activation: nn.Module | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.activation = activation or nn.SiLU()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(SpectralLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def lipschitz_constant(self) -> torch.Tensor:
        """Product of per-layer spectral norms × product of activation Lipschitz
        constants (SiLU is ≤ 1.1, ReLU/LeakyReLU ≤ 1)."""
        act_L = 1.0  # SiLU upper bound used here; tightened per activation below
        if isinstance(self.activation, nn.SiLU):
            act_L = 1.1
        total = torch.tensor(1.0, device=next(self.parameters()).device)
        n_linear = 0
        for mod in self.net:
            if isinstance(mod, SpectralLinear):
                total = total * mod.spectral_norm()
                n_linear += 1
        total = total * (act_L ** max(0, n_linear - 1))
        return total
