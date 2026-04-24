"""Evaluation metrics used in Table III of the manuscript.

- Query-level AUC-ROC (sklearn).
- Campaign-level F1 computed over connected-session subgraphs with
  ``IoU(predicted, ground-truth) ≥ 0.5`` (Section V-D).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def query_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def campaign_iou(pred: Iterable[int], truth: Iterable[int]) -> float:
    p = set(pred)
    t = set(truth)
    if not (p or t):
        return 1.0
    inter = len(p & t)
    union = len(p | t)
    return inter / max(union, 1)


def campaign_f1(
    predicted: list[set[int]],
    ground_truth: list[set[int]],
    iou_threshold: float = 0.5,
) -> float:
    """Match each predicted campaign to its best-IoU ground-truth and
    return the F1 over thresholded matches."""
    matched_gt: set[int] = set()
    tp = 0
    for p in predicted:
        best, best_j = -1.0, -1
        for j, g in enumerate(ground_truth):
            if j in matched_gt:
                continue
            iou = campaign_iou(p, g)
            if iou > best:
                best, best_j = iou, j
        if best >= iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            tp += 1
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    if tp == 0 and (fp + fn) == 0:
        return 1.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def query_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="binary", zero_division=0))
