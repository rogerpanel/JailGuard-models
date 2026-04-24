"""Hierarchical Set2Set pooling over the session sub-graph.

Eq. (7) of the manuscript:

    ŷ_campaign(t) = σ(MLP(Set2Set({h_s(t) : s ∈ S_conn})))

where Set2Set (Vinyals et al., 2016) provides permutation-invariant
aggregation over a variable-size set of session embeddings. Both the
Set2Set LSTM and the classifier MLP are spectrally-normalized so their
Lipschitz constants compose into the end-to-end Grönwall certificate.
"""
from __future__ import annotations

import torch
from torch import nn

from ct_dgnn.models.spectral_norm import SpectralLinear, SpectralMLP


class HierarchicalSet2Set(nn.Module):
    """Set2Set pooler with spectral-normalized LSTM gates.

    For a set X = {x_i}_{i=1..n} with x_i ∈ R^d, Set2Set performs
    `processing_steps` iterations of:
        q_t          = LSTM(q_{t-1}, r_{t-1})
        e_{t,i}      = f(q_t, x_i)
        a_{t,i}      = softmax_i(e_{t,i})
        r_t          = Σ_i a_{t,i} x_i
    Output concat(q_T, r_T) ∈ R^{2d}.
    """

    def __init__(self, embed_dim: int, processing_steps: int = 3) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.processing_steps = processing_steps
        self.lstm = nn.LSTMCell(2 * embed_dim, embed_dim)
        # One-shot spectral normalization of the LSTM weight parameters.
        nn.utils.parametrizations.spectral_norm(self.lstm, name="weight_ih")
        nn.utils.parametrizations.spectral_norm(self.lstm, name="weight_hh")
        self.query_proj = SpectralLinear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Pool a batched collection of session embeddings.

        Parameters
        ----------
        x : (N, d) concatenated session embeddings across the batch.
        batch_index : (N,) assigning each row to a graph in the batch.

        Returns
        -------
        out : (B, 2d) pooled graph embeddings.
        """
        B = int(batch_index.max().item()) + 1
        d = self.embed_dim
        h = x.new_zeros(B, d)
        c = x.new_zeros(B, d)
        q = x.new_zeros(B, 2 * d)
        r = x.new_zeros(B, d)

        for _ in range(self.processing_steps):
            h, c = self.lstm(q, (h, c))                        # (B, d)
            q_proj = self.query_proj(h)                        # (B, d)
            # Attention scores over all elements.
            scores = (x * q_proj[batch_index]).sum(dim=-1)     # (N,)
            # softmax within each graph
            max_per_b = torch.full((B,), float("-inf"), device=x.device)
            max_per_b.scatter_reduce_(0, batch_index, scores, reduce="amax",
                                      include_self=True)
            shifted = (scores - max_per_b[batch_index]).exp()
            denom = torch.zeros(B, device=x.device)
            denom.scatter_add_(0, batch_index, shifted)
            alpha = shifted / (denom[batch_index] + 1e-9)      # (N,)
            r = torch.zeros(B, d, device=x.device)
            r.scatter_add_(0, batch_index.unsqueeze(-1).expand(-1, d),
                           alpha.unsqueeze(-1) * x)
            q = torch.cat([h, r], dim=-1)
        return q

    @torch.no_grad()
    def lipschitz_constant(self) -> torch.Tensor:
        """Worst-case multiplicative Lipschitz over processing_steps.

        A single Set2Set step is bounded by
            L_step = σ(W_ih) + σ(W_hh) + σ(Q),
        and the full pool is at most `processing_steps` repeated
        applications (conservative bound)."""
        # Pull sigmas from the parametrized weights.
        sigma_ih = self._sigma(self.lstm.weight_ih)
        sigma_hh = self._sigma(self.lstm.weight_hh)
        sigma_q = self.query_proj.spectral_norm()
        step = sigma_ih + sigma_hh + sigma_q
        return step ** self.processing_steps

    @staticmethod
    def _sigma(w: torch.Tensor) -> torch.Tensor:
        u, _, v = torch.svd_lowrank(w, q=1)
        return (u.t() @ w @ v).flatten().abs().max()


class CampaignClassifier(nn.Module):
    """MLP head that maps the Set2Set pooled embedding to a campaign score."""

    def __init__(self, embed_dim: int, num_classes: int = 2,
                 hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = SpectralMLP(
            dims=[2 * embed_dim, hidden, num_classes],
            activation=nn.SiLU(),
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)

    @torch.no_grad()
    def lipschitz_constant(self) -> torch.Tensor:
        return self.mlp.lipschitz_constant()
