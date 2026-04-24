"""Evaluate a trained checkpoint on a dataset and report the quantities
used in Tables III-IV of the manuscript: query-level AUC-ROC,
campaign-level F1, and PGD robustness sweep."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from ct_dgnn.evaluation.latency import measure_latency
from ct_dgnn.evaluation.metrics import campaign_f1, query_auc_roc
from ct_dgnn.models.ct_dgnn import CTDGNNJailGuard
from ct_dgnn.robustness.pgd_attack import pgd_evaluate
from ct_dgnn.utils.config import load_config
from ct_dgnn.utils.logging import get_logger
from ct_dgnn.utils.seed import set_seed


def _load_shards(path: Path) -> list:
    batches: list = []
    for shard in sorted(path.glob("shard_*.pkl")):
        with shard.open("rb") as fh:
            batches.extend(pickle.load(fh))
    return batches


def _predict_campaigns(model, batches, device):
    y_true: list = []
    y_score: list = []
    predicted_sets: list = []
    truth_sets: list = []
    model.eval()
    with torch.no_grad():
        for b in batches:
            b.x = b.x.to(device)
            b.node_type_ids = b.node_type_ids.to(device)
            b.edge_index = b.edge_index.to(device)
            b.edge_delta_t = b.edge_delta_t.to(device)
            b.event_times = b.event_times.to(device)
            b.session_component_index = b.session_component_index.to(device)
            b.query_labels = b.query_labels.to(device)
            b.campaign_labels = b.campaign_labels.to(device)

            out = model(b)
            probs = out.query_logits.softmax(dim=-1)[:, 1]
            y_true.extend(b.query_labels.cpu().tolist())
            y_score.extend(probs.cpu().tolist())

            camp_pred = out.campaign_logits.argmax(dim=-1).cpu().tolist()
            comp = b.session_component_index.cpu().tolist()
            for cid, p in enumerate(camp_pred):
                if p:
                    predicted_sets.append({i for i, c in enumerate(comp) if c == cid})
            camp_true = b.campaign_labels.cpu().tolist()
            for cid, g in enumerate(camp_true):
                if g:
                    truth_sets.append({i for i, c in enumerate(comp) if c == cid})
    return np.asarray(y_true), np.asarray(y_score), predicted_sets, truth_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    log = get_logger("evaluate")
    device = cfg.device if torch.cuda.is_available() else "cpu"

    batches = _load_shards(Path(args.data) / cfg.data.dataset)
    model = CTDGNNJailGuard(
        node_dims=cfg.graph.node_dims,
        embed_dim=cfg.graph.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        use_adjoint=cfg.model.adjoint,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    y_true, y_score, preds, truths = _predict_campaigns(model, batches, device)
    auc = query_auc_roc(y_true, y_score)
    f1 = campaign_f1(preds, truths, iou_threshold=cfg.eval.campaign_iou)
    log.info(f"query AUC-ROC: {auc:.4f}")
    log.info(f"campaign F1  : {f1:.4f}")

    pgd = pgd_evaluate(
        model, batches[0],
        epsilons=list(cfg.robustness.pgd_epsilons),
        step_size=cfg.robustness.pgd_step_size,
        n_steps=cfg.robustness.pgd_steps,
    )
    for r in pgd:
        log.info(f"PGD ε={r.epsilon:.2f}  clean={r.clean_accuracy:.3f} "
                 f"robust={r.robust_accuracy:.3f}")

    lat = measure_latency(model, batches[:cfg.eval.latency_samples],
                          device=device)
    log.info(f"latency ms — P50={lat.p50:.1f} P95={lat.p95:.1f} P99={lat.p99:.1f} "
             f"qps={lat.qps:,.0f}")


if __name__ == "__main__":
    main()
