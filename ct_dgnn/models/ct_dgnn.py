"""CT-DGNN-JailGuard — the top-level model composing graph construction,
continuous-time ODE dynamics, Lipschitz-certified classification, and
the LLM zero-shot analyzer.

Forward signature consumes an `InteractionBatch` (see
`ct_dgnn/data/graph_builder.py`) representing a rolling event window
and produces:

    - query_logits    : per-query benign/malicious softmax,
    - campaign_logits : per connected-component campaign decision,
    - node_embeddings : post-integration h_v(T),
    - lipschitz       : dict of end-to-end Lipschitz constants (for ε*).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from ct_dgnn.models.attention import TemporalHeterogeneousAttention
from ct_dgnn.models.message_passing import RelationMessage
from ct_dgnn.models.ode_dynamics import (
    HeterogeneousODEDynamics,
    NeuralODEBlock,
)
from ct_dgnn.models.pooling import CampaignClassifier, HierarchicalSet2Set
from ct_dgnn.models.spectral_norm import SpectralLinear
from ct_dgnn.models.temporal_encoding import MultiScaleTimeEncoding

NODE_TYPES: tuple[str, ...] = ("user", "session", "query", "model")
RELATIONS: tuple[str, ...] = (
    "initiates", "contains", "targets", "responds", "follows", "shares_pattern",
)


@dataclass
class CTDGNNOutput:
    query_logits: torch.Tensor
    campaign_logits: torch.Tensor
    node_embeddings: torch.Tensor
    lipschitz: Dict[str, torch.Tensor]


class CTDGNNJailGuard(nn.Module):
    def __init__(
        self,
        node_dims: dict[str, int],
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        ode_solver: str = "dopri5",
        ode_rtol: float = 1e-3,
        ode_atol: float = 1e-4,
        use_adjoint: bool = True,
        num_classes_query: int = 2,
        num_classes_campaign: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Per-type input projections (raw feature → shared d-dim space).
        self.input_proj: Dict[str, SpectralLinear] = nn.ModuleDict(
            {t: SpectralLinear(node_dims[t], embed_dim) for t in NODE_TYPES}
        )

        # 2. Shared temporal encoding, attention, and relation messages.
        self.time_encoding = MultiScaleTimeEncoding()
        self.attention = TemporalHeterogeneousAttention(
            relations=list(RELATIONS),
            embed_dim=embed_dim,
            num_heads=num_heads,
            time_encoding=self.time_encoding,
        )
        self.messages = RelationMessage(list(RELATIONS), embed_dim)

        # 3. Type-specific ODE dynamics.
        self.dynamics = HeterogeneousODEDynamics(
            node_types=list(NODE_TYPES),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.ode = NeuralODEBlock(
            self.dynamics,
            solver=ode_solver,
            rtol=ode_rtol,
            atol=ode_atol,
            use_adjoint=use_adjoint,
        )

        # 4. Readouts.
        self.query_head = SpectralLinear(embed_dim, num_classes_query)
        self.pool = HierarchicalSet2Set(embed_dim)
        self.campaign_head = CampaignClassifier(
            embed_dim, num_classes=num_classes_campaign, dropout=dropout
        )

    # ---------------------------------------------------------------
    def _embed_initial(self, batch) -> torch.Tensor:
        """Project raw per-type features into the shared d-dim space."""
        h = torch.zeros(batch.num_nodes, self.embed_dim, device=batch.x.device)
        for i, t in enumerate(NODE_TYPES):
            mask = batch.node_type_ids == i
            if mask.any():
                h[mask] = self.input_proj[t](batch.x[mask])
        return h

    def _aggregate(self, h: torch.Tensor, batch) -> torch.Tensor:
        """One pass of attention-weighted neighbourhood aggregation."""
        return self.attention(
            h_v=h[batch.edge_index[1]],
            h_u=h[batch.edge_index[0]],
            edge_relation=batch.edge_relation,
            delta_t=batch.edge_delta_t,
            target_index=batch.edge_index[1],
            num_nodes=batch.num_nodes,
        )

    # ---------------------------------------------------------------
    def forward(self, batch, *, integrate_steps: int = 3) -> CTDGNNOutput:
        """Run the full pipeline on an InteractionBatch.

        Within each inter-event window we:
          (i)  compute aggregated messages at the current topology,
          (ii) integrate f_τ over that window with dopri5,
          (iii)update per-node h.
        """
        h = self._embed_initial(batch)
        t_events = batch.event_times                    # (L+1,)
        for i in range(min(integrate_steps, t_events.numel() - 1)):
            message = self._aggregate(h, batch)
            self.ode.set_step_inputs(message, batch.node_type_ids)
            h = self.ode.integrate(h, t_events[i:i + 2])

        # Query-level classification.
        query_mask = batch.node_type_ids == NODE_TYPES.index("query")
        query_logits = self.query_head(h[query_mask])

        # Campaign-level classification over connected session components.
        session_mask = batch.node_type_ids == NODE_TYPES.index("session")
        session_h = h[session_mask]
        session_component = batch.session_component_index
        pooled = self.pool(session_h, session_component)
        campaign_logits = self.campaign_head(pooled)

        lipschitz = {
            "f_tau": self.dynamics.lipschitz_constant(),
            "g_r":   self.messages.lipschitz_constant(),
            "attn":  self.attention.lipschitz_constant(),
            "pool":  self.pool.lipschitz_constant(),
            "mlp":   self.campaign_head.lipschitz_constant(),
        }
        return CTDGNNOutput(
            query_logits=query_logits,
            campaign_logits=campaign_logits,
            node_embeddings=h,
            lipschitz=lipschitz,
        )
