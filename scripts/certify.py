"""Evaluate the Grönwall-based certified robustness (Theorem 1).

Harvests per-component Lipschitz constants from a trained CTDGNNJailGuard
instance and computes:

    ε* = Δ_margin / (L_MLP · L_pool · k · L_g · exp(L_f · T))

The paper target is ε* ≥ 0.15 at k = 50 perturbed queries and T = 24h."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ct_dgnn.models.ct_dgnn import CTDGNNJailGuard
from ct_dgnn.robustness.certificate import certificate_from_model
from ct_dgnn.robustness.lipschitz import model_lipschitz_constants
from ct_dgnn.utils.config import load_config
from ct_dgnn.utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--margin", type=float, default=0.5,
                    help="Δ_margin = τ_+ − τ_- (default 0.5 for two-class)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("certify")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CTDGNNJailGuard(
        node_dims=cfg.graph.node_dims,
        embed_dim=cfg.graph.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        use_adjoint=cfg.model.adjoint,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    Ls = model_lipschitz_constants(model)
    log.info(f"Lipschitz constants: {Ls}")

    res = certificate_from_model(model, k=args.k, T_seconds=args.hours * 3600,
                                 margin=args.margin)
    log.info(f"ε* = {res.radius:.4f}")
    log.info(f"aggregate Lipschitz (k={args.k}, T={args.hours}h): {res.lipschitz_total:.3f}")
    log.info(f"target (paper): ε* ≥ 0.15 → "
             f"{'PASS' if res.radius >= 0.15 else 'FAIL (consider tighter spectral norm)'}")


if __name__ == "__main__":
    main()
