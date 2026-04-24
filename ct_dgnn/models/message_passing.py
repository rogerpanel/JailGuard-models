"""Relation-specific message transformations g_r : R^d → R^d.

Eq. (4) of the manuscript aggregates neighbour contributions as
    Σ_{(u, r) ∈ N(v, t)} α_{v,u,r}(t) · g_r(h_u(t)).

Each g_r is a 1-Lipschitz linear transformation (plus bias), implemented
as a dictionary keyed by relation name. Spectral normalization keeps the
per-relation Lipschitz constant L_g ≤ 1.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ct_dgnn.models.spectral_norm import SpectralLinear


class RelationMessage(nn.Module):
    def __init__(self, relations: list[str], embed_dim: int) -> None:
        super().__init__()
        self.relations = relations
        self.embed_dim = embed_dim
        self.g: Dict[str, SpectralLinear] = nn.ModuleDict(
            {r: SpectralLinear(embed_dim, embed_dim) for r in relations}
        )

    def forward(self, h_u: torch.Tensor, relation: str) -> torch.Tensor:
        return self.g[relation](h_u)

    @torch.no_grad()
    def lipschitz_constant(self) -> torch.Tensor:
        sigmas = [layer.spectral_norm() for layer in self.g.values()]
        return torch.stack(sigmas).max()
