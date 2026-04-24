"""Temporal heterogeneous attention (Eq. 6 of the manuscript).

    α_{v,u,r}(t) = softmax_{(u',r')} LReLU( a_r^T [ W_r h_v(t) ‖
                                                    W_r h_u(t) ‖
                                                    φ_time(t − t_uv) ] )

Each (W_r, a_r) is a per-relation parameter set. The projections W_r are
spectrally normalized to keep the whole block Lipschitz-bounded.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from ct_dgnn.models.spectral_norm import SpectralLinear
from ct_dgnn.models.temporal_encoding import MultiScaleTimeEncoding


class TemporalHeterogeneousAttention(nn.Module):
    def __init__(
        self,
        relations: list[str],
        embed_dim: int,
        num_heads: int = 4,
        time_encoding: MultiScaleTimeEncoding | None = None,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.relations = relations
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.time_encoding = time_encoding or MultiScaleTimeEncoding()
        self.leaky = nn.LeakyReLU(leaky_slope)

        self.W_r: Dict[str, SpectralLinear] = nn.ModuleDict(
            {r: SpectralLinear(embed_dim, embed_dim) for r in relations}
        )
        a_dim = 2 * embed_dim + self.time_encoding.output_dim
        self.a_r: Dict[str, nn.Parameter] = nn.ParameterDict(
            {r: nn.Parameter(torch.randn(num_heads, a_dim // num_heads + (a_dim % num_heads > 0)))
             for r in relations}
        )
        # Simplify the parameterisation: use one a_r of size (num_heads, a_dim)
        for r in relations:
            self.a_r[r] = nn.Parameter(torch.randn(num_heads, a_dim))
        nn.init.xavier_uniform_(self.W_r[relations[0]].linear.weight)
        for r in relations:
            nn.init.xavier_uniform_(self.a_r[r])

    def score(
        self,
        h_v: torch.Tensor,          # (N_edges, embed_dim)
        h_u: torch.Tensor,          # (N_edges, embed_dim)
        relation: str,
        delta_t: torch.Tensor,      # (N_edges,)
    ) -> torch.Tensor:
        """Unnormalised attention logits per head, shape (N_edges, num_heads)."""
        W = self.W_r[relation]
        Wv = W(h_v)
        Wu = W(h_u)
        phi_t = self.time_encoding(delta_t)                       # (N_edges, 2K)
        concat = torch.cat([Wv, Wu, phi_t], dim=-1)               # (N_edges, 2d + 2K)
        a = self.a_r[relation]                                    # (H, 2d + 2K)
        logits = self.leaky(concat @ a.t())                       # (N_edges, H)
        return logits

    def forward(
        self,
        h_v: torch.Tensor,                  # (N_edges, embed_dim)
        h_u: torch.Tensor,                  # (N_edges, embed_dim)
        edge_relation: list[str],           # per-edge relation name
        delta_t: torch.Tensor,              # (N_edges,)
        target_index: torch.Tensor,         # (N_edges,) — index of destination node
        num_nodes: int,
    ) -> torch.Tensor:
        """Aggregate incoming messages into per-node hidden states."""
        device = h_v.device
        H = self.num_heads
        out = torch.zeros(num_nodes, self.embed_dim, device=device)

        for relation in set(edge_relation):
            mask = torch.tensor(
                [1 if r == relation else 0 for r in edge_relation],
                dtype=torch.bool, device=device,
            )
            if not mask.any():
                continue
            idx = target_index[mask]
            hv = h_v[mask]
            hu = h_u[mask]
            dt = delta_t[mask]
            logits = self.score(hv, hu, relation, dt)             # (E_r, H)

            # softmax over incoming neighbours per destination node.
            # Implemented through scatter-logsumexp.
            max_per_dst = torch.full((num_nodes, H), float("-inf"), device=device)
            max_per_dst.scatter_reduce_(0, idx.unsqueeze(-1).expand_as(logits), logits,
                                        reduce="amax", include_self=True)
            shifted = logits - max_per_dst[idx]
            expd = shifted.exp()
            denom = torch.zeros(num_nodes, H, device=device)
            denom.scatter_add_(0, idx.unsqueeze(-1).expand_as(expd), expd)
            alpha = expd / (denom[idx] + 1e-9)                    # (E_r, H)

            # message: per-head projection of neighbour's relation-mapped hidden state
            msg = self.W_r[relation](hu)                          # (E_r, embed_dim)
            msg = msg.view(-1, H, self.head_dim) * alpha.unsqueeze(-1)
            msg = msg.view(-1, self.embed_dim)
            out.scatter_add_(0, idx.unsqueeze(-1).expand_as(msg), msg)

        return out

    @torch.no_grad()
    def lipschitz_constant(self) -> torch.Tensor:
        sigmas = [w.spectral_norm() for w in self.W_r.values()]
        return torch.stack(sigmas).max()
