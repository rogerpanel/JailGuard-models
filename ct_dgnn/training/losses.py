"""Composite training loss.

L = λ_clf · L_ce(campaign) + λ_q · L_ce(query) + λ_J · L_Jacobian
                                               + λ_c · L_contrastive
The Jacobian term and contrastive term implement Eq. (9) of the
manuscript and the feedback-loop contrastive update described in
Sec. IV-E respectively.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ct_dgnn.robustness.jacobian_reg import jacobian_frobenius


@dataclass
class LossWeights:
    classification: float = 1.0
    query: float = 0.5
    jacobian: float = 0.01
    contrastive: float = 0.1


class CampaignLoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        out,
        batch,
        dynamics: nn.Module | None = None,
        contrastive_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        parts: dict[str, torch.Tensor] = {}

        parts["campaign"] = self.ce(out.campaign_logits, batch.campaign_labels)
        parts["query"] = self.ce(out.query_logits, batch.query_labels)

        if dynamics is not None:
            # Jacobian regulariser on f_τ at the current operating point.
            query_mask = batch.node_type_ids == 2
            if query_mask.any():
                parts["jacobian"] = jacobian_frobenius(
                    f=dynamics,
                    h=out.node_embeddings[query_mask],
                    message=torch.zeros_like(out.node_embeddings[query_mask]),
                    node_type_ids=batch.node_type_ids[query_mask],
                )
            else:
                parts["jacobian"] = torch.zeros((), device=out.campaign_logits.device)
        else:
            parts["jacobian"] = torch.zeros((), device=out.campaign_logits.device)

        parts["contrastive"] = contrastive_loss if contrastive_loss is not None \
            else torch.zeros((), device=out.campaign_logits.device)

        total = (self.w.classification * parts["campaign"]
                 + self.w.query * parts["query"]
                 + self.w.jacobian * parts["jacobian"]
                 + self.w.contrastive * parts["contrastive"])

        return total, {k: float(v.item()) for k, v in parts.items()}
