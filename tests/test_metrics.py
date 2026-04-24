"""Tests for evaluation metrics."""
from __future__ import annotations

import numpy as np

from ct_dgnn.evaluation.metrics import campaign_f1, campaign_iou, query_auc_roc


def test_campaign_iou_empty_sets_are_equal():
    assert campaign_iou(set(), set()) == 1.0


def test_campaign_iou_full_overlap():
    assert campaign_iou({1, 2, 3}, {1, 2, 3}) == 1.0


def test_campaign_iou_partial():
    assert campaign_iou({1, 2, 3}, {2, 3, 4}) == 0.5


def test_campaign_f1_perfect():
    preds = [{1, 2}, {3, 4}]
    truth = [{1, 2}, {3, 4}]
    assert campaign_f1(preds, truth) == 1.0


def test_campaign_f1_mismatch():
    preds = [{1, 2}]
    truth = [{3, 4}]
    assert campaign_f1(preds, truth) == 0.0


def test_query_auc_random_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(size=200)
    auc = query_auc_roc(y, s)
    assert 0.3 < auc < 0.7
