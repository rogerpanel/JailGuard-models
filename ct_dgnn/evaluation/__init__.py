"""Evaluation metrics. Metrics are pure numpy/sklearn; latency and
zero-shot helpers need torch and are imported lazily."""

from ct_dgnn.evaluation.metrics import (
    campaign_f1,
    campaign_iou,
    query_auc_roc,
)

_LAZY = {
    "measure_latency":         ("ct_dgnn.evaluation.latency",   "measure_latency"),
    "leave_one_strategy_out":  ("ct_dgnn.evaluation.zero_shot", "leave_one_strategy_out"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(name)


__all__ = [
    "campaign_f1",
    "campaign_iou",
    "query_auc_roc",
    "measure_latency",
    "leave_one_strategy_out",
]
