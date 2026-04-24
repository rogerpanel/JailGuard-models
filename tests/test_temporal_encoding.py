"""Tests for the multi-scale temporal encoding."""
from __future__ import annotations

import torch

from ct_dgnn.models.temporal_encoding import DEFAULT_SCALES_SEC, MultiScaleTimeEncoding


def test_output_dim_is_2K():
    enc = MultiScaleTimeEncoding()
    out = enc(torch.tensor([0.0, 1.0, 86400.0]))
    assert out.shape == (3, 2 * len(DEFAULT_SCALES_SEC))


def test_zero_delta_gives_sin0_cos1():
    enc = MultiScaleTimeEncoding()
    out = enc(torch.tensor([0.0]))
    K = len(DEFAULT_SCALES_SEC)
    assert torch.allclose(out[0, :K], torch.zeros(K), atol=1e-6)
    assert torch.allclose(out[0, K:], torch.ones(K), atol=1e-6)


def test_scales_remain_positive_after_optimisation():
    enc = MultiScaleTimeEncoding(learnable=True)
    # Force a large negative gradient step on the raw scales.
    enc.raw_scales.data -= 100.0
    scales = enc.scales()
    assert (scales > 0).all()
