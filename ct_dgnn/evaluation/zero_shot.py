"""Leave-one-strategy-out zero-shot generalisation evaluation.

Strategy families are specified in ``configs/default.yaml``. For each
strategy S_i:
   1. Train on all events whose attack_strategy ≠ S_i.
   2. Test on events whose attack_strategy == S_i.
   3. Report campaign-level F1 and query-level AUC-ROC.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch

from ct_dgnn.evaluation.metrics import campaign_f1, query_auc_roc


def leave_one_strategy_out(
    events: Iterable,
    strategies: list[str],
    train_fn: Callable[[list], torch.nn.Module],
    eval_fn: Callable[[torch.nn.Module, list], dict],
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    all_events = list(events)
    for held in strategies:
        train = [e for e in all_events if getattr(e, "attack_strategy", None) != held]
        test = [e for e in all_events if getattr(e, "attack_strategy", None) == held]
        if not test:
            continue
        model = train_fn(train)
        results[held] = eval_fn(model, test)
    return results
