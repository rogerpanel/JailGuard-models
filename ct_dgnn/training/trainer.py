"""Trainer driving the AdamW + cosine schedule described in
Section V-E of the manuscript (100 epochs, lr=5e-4, wd=1e-4)."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from ct_dgnn.training.losses import CampaignLoss
from ct_dgnn.utils.logging import get_logger


class CTDGNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable,
        val_loader: Iterable | None = None,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        device: str = "cuda",
        loss: CampaignLoss | None = None,
        out_dir: str | Path = "runs/latest",
        patience: int = 15,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.device = device
        self.loss = loss or CampaignLoss()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.log = get_logger("ct_dgnn.trainer")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_lambda
        )

    def _lr_lambda(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return epoch / max(1, self.warmup_epochs)
        progress = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # ------------------------------------------------------------------
    def _step(self, batch) -> dict[str, float]:
        batch.x = batch.x.to(self.device)
        batch.node_type_ids = batch.node_type_ids.to(self.device)
        batch.edge_index = batch.edge_index.to(self.device)
        batch.edge_delta_t = batch.edge_delta_t.to(self.device)
        batch.event_times = batch.event_times.to(self.device)
        batch.query_labels = batch.query_labels.to(self.device)
        batch.campaign_labels = batch.campaign_labels.to(self.device)
        batch.session_component_index = batch.session_component_index.to(self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = self.model(batch)
        total, parts = self.loss(out, batch, dynamics=self.model.dynamics)
        total.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        parts["total"] = float(total.item())
        return parts

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        stats = {"loss": 0.0, "acc_campaign": 0.0, "acc_query": 0.0, "n": 0}
        for batch in self.val_loader:
            batch.x = batch.x.to(self.device)
            batch.node_type_ids = batch.node_type_ids.to(self.device)
            batch.edge_index = batch.edge_index.to(self.device)
            batch.edge_delta_t = batch.edge_delta_t.to(self.device)
            batch.event_times = batch.event_times.to(self.device)
            batch.query_labels = batch.query_labels.to(self.device)
            batch.campaign_labels = batch.campaign_labels.to(self.device)
            batch.session_component_index = batch.session_component_index.to(self.device)

            out = self.model(batch)
            loss, _ = self.loss(out, batch)
            stats["loss"] += float(loss.item())
            stats["acc_campaign"] += float(
                (out.campaign_logits.argmax(-1) == batch.campaign_labels).float().mean().item()
            )
            stats["acc_query"] += float(
                (out.query_logits.argmax(-1) == batch.query_labels).float().mean().item()
            )
            stats["n"] += 1
        n = max(stats["n"], 1)
        return {k: v / n for k, v in stats.items() if k != "n"}

    # ------------------------------------------------------------------
    def fit(self) -> None:
        best_val = float("inf")
        since_best = 0
        for epoch in range(self.epochs):
            t0 = time.time()
            running = {"total": 0.0, "n": 0}
            for batch in self.train_loader:
                parts = self._step(batch)
                running["total"] += parts["total"]
                running["n"] += 1
            self.scheduler.step()

            val = self._validate()
            dt = time.time() - t0
            n = max(running["n"], 1)
            avg = running["total"] / n
            self.log.info(
                f"epoch {epoch:03d} | loss={avg:.4f} | val={val} | {dt:.1f}s"
            )

            val_loss = val.get("loss", avg)
            if val_loss < best_val:
                best_val = val_loss
                since_best = 0
                torch.save(self.model.state_dict(), self.out_dir / "best.pt")
            else:
                since_best += 1
                if since_best >= self.patience:
                    self.log.info(f"early stop at epoch {epoch}")
                    break
        torch.save(self.model.state_dict(), self.out_dir / "final.pt")
