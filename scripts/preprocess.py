"""Convert a raw dataset stream into pre-built InteractionBatch pickles
suitable for training without touching the embedding model each epoch.

Each pickle holds a rolling window of up to ``--window`` events. Saving
in shards of ``--shard-size`` keeps random-access I/O fast during
training (the trainer just iterates through the shards)."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch

from ct_dgnn.data.datasets import DATASET_REGISTRY
from ct_dgnn.data.graph_builder import APIInteractionGraphBuilder
from ct_dgnn.utils.config import load_config
from ct_dgnn.utils.logging import get_logger
from ct_dgnn.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None,
                    help="override config's data.dataset")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--window", type=int, default=512,
                    help="number of events per InteractionBatch")
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    log = get_logger("preprocess")
    dataset_name = args.dataset or cfg.data.dataset
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(Path(cfg.data.root) / dataset_name)
    log.info(f"loaded {dataset} with {len(dataset):,} events")

    builder = APIInteractionGraphBuilder(
        node_dims=cfg.graph.node_dims,
        similarity_threshold=cfg.data.similarity_threshold,
        max_history_hours=cfg.data.max_history_hours,
        sentence_model=cfg.data.sentence_transformer,
    )

    out = Path(args.out) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    buffer: list = []
    batches: list = []
    shard_idx = 0
    batch_idx = 0
    for ev in dataset:
        buffer.append(ev)
        if len(buffer) >= args.window:
            batch = builder.ingest(buffer)
            batches.append(batch)
            buffer.clear()
            batch_idx += 1
            if args.max_batches and batch_idx >= args.max_batches:
                break
            if len(batches) >= args.shard_size:
                with (out / f"shard_{shard_idx:05d}.pkl").open("wb") as fh:
                    pickle.dump(batches, fh)
                log.info(f"wrote shard {shard_idx} with {len(batches)} batches")
                batches.clear()
                shard_idx += 1

    if batches:
        with (out / f"shard_{shard_idx:05d}.pkl").open("wb") as fh:
            pickle.dump(batches, fh)
        log.info(f"wrote shard {shard_idx} with {len(batches)} batches")


if __name__ == "__main__":
    main()
