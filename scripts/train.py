"""Train CT-DGNN-JailGuard on a processed shard directory."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from ct_dgnn.models.ct_dgnn import CTDGNNJailGuard
from ct_dgnn.training.losses import CampaignLoss, LossWeights
from ct_dgnn.training.trainer import CTDGNNTrainer
from ct_dgnn.utils.config import load_config
from ct_dgnn.utils.logging import get_logger
from ct_dgnn.utils.seed import set_seed


def _load_shards(path: Path) -> list:
    batches: list = []
    for shard in sorted(path.glob("shard_*.pkl")):
        with shard.open("rb") as fh:
            batches.extend(pickle.load(fh))
    return batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/latest")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    log = get_logger("train")
    device = cfg.device if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data) / cfg.data.dataset
    batches = _load_shards(data_root)
    if not batches:
        raise RuntimeError(f"no shards found under {data_root}. "
                           f"run scripts/preprocess.py first.")
    log.info(f"loaded {len(batches)} graph batches from {data_root}")

    split = int(0.9 * len(batches))
    train, val = batches[:split], batches[split:]

    model = CTDGNNJailGuard(
        node_dims=cfg.graph.node_dims,
        embed_dim=cfg.graph.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        ode_solver=cfg.model.ode_solver,
        ode_rtol=cfg.model.ode_rtol,
        ode_atol=cfg.model.ode_atol,
        use_adjoint=cfg.model.adjoint,
        dropout=cfg.model.dropout,
    )

    trainer = CTDGNNTrainer(
        model=model,
        train_loader=train,
        val_loader=val,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        grad_clip=cfg.training.grad_clip,
        device=device,
        loss=CampaignLoss(LossWeights(
            classification=cfg.training.loss_weights.classification,
            jacobian=cfg.training.loss_weights.jacobian,
            contrastive=cfg.training.loss_weights.contrastive,
        )),
        out_dir=args.out,
        patience=cfg.training.early_stop_patience,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
