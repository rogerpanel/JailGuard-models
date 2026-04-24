"""Projected Gradient Descent (PGD) adversarial evaluation.

Matches Fig. 5 of the manuscript: sweeps perturbation radius
ε ∈ {0.05, ..., 0.35}, reports empirical detection accuracy and
compares against the certified lower bound from Theorem 1.

Perturbations are applied only to query node features (matching the
threat model in Section III-D, k query features perturbed with
‖·‖₂ ≤ ε).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ct_dgnn.models.ct_dgnn import CTDGNNJailGuard


@dataclass
class PGDResult:
    epsilon: float
    clean_accuracy: float
    robust_accuracy: float


def _project_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    """Project a perturbation onto the ℓ₂ ball of radius ε (per row)."""
    norms = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    factor = torch.where(norms > eps, eps / norms, torch.ones_like(norms))
    return delta * factor


def pgd_attack(
    model: CTDGNNJailGuard,
    batch,
    epsilon: float,
    step_size: float = 0.01,
    n_steps: int = 40,
    target_type: str = "query",
) -> torch.Tensor:
    """Returns the adversarially perturbed feature tensor x_adv."""
    loss_fn = torch.nn.CrossEntropyLoss()
    x = batch.x.detach().clone()
    query_mask = batch.node_type_ids == 2   # NODE_TYPES.index("query") == 2
    target = batch.campaign_labels

    delta = torch.zeros_like(x[query_mask], requires_grad=True)
    for _ in range(n_steps):
        batch.x = x.clone()
        batch.x[query_mask] = batch.x[query_mask] + delta
        out = model(batch)
        loss = loss_fn(out.campaign_logits, target)
        loss.backward()
        with torch.no_grad():
            step = step_size * delta.grad.sign()
            delta = delta + step
            delta = _project_l2(delta, epsilon)
        delta.requires_grad_(True)

    batch.x = x.clone()
    batch.x[query_mask] = batch.x[query_mask] + delta.detach()
    return batch.x


@torch.no_grad()
def _accuracy(model: CTDGNNJailGuard, batch) -> float:
    out = model(batch)
    preds = out.campaign_logits.argmax(dim=-1)
    return float((preds == batch.campaign_labels).float().mean().item())


def pgd_evaluate(
    model: CTDGNNJailGuard,
    batch,
    epsilons: list[float],
    step_size: float = 0.01,
    n_steps: int = 40,
) -> list[PGDResult]:
    """Run the robustness sweep used in Fig. 5."""
    clean = _accuracy(model, batch)
    results: list[PGDResult] = []
    for eps in epsilons:
        x_orig = batch.x.clone()
        _ = pgd_attack(model, batch, eps, step_size, n_steps)
        robust = _accuracy(model, batch)
        batch.x = x_orig
        results.append(PGDResult(eps, clean, robust))
    return results
