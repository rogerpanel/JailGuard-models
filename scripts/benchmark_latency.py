"""Reproduce the P50 / P95 / P99 latency numbers from Table IV."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from ct_dgnn.evaluation.latency import measure_latency
from ct_dgnn.models.ct_dgnn import CTDGNNJailGuard
from ct_dgnn.utils.config import load_config
from ct_dgnn.utils.logging import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger("latency")
    device = cfg.device if torch.cuda.is_available() else "cpu"

    model = CTDGNNJailGuard(
        node_dims=cfg.graph.node_dims,
        embed_dim=cfg.graph.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        use_adjoint=cfg.model.adjoint,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    data_root = Path(args.data) / cfg.data.dataset
    batches: list = []
    for shard in sorted(data_root.glob("shard_*.pkl")):
        with shard.open("rb") as fh:
            batches.extend(pickle.load(fh))
        if len(batches) >= args.samples:
            break

    rep = measure_latency(model, batches[:args.samples], device=device)
    log.info(f"P50 / P95 / P99  :  {rep.p50:.1f} / {rep.p95:.1f} / {rep.p99:.1f}  ms")
    log.info(f"throughput       :  {rep.qps:,.0f}  queries/sec")


if __name__ == "__main__":
    main()
