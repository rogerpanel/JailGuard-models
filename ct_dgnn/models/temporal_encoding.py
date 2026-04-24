"""Multi-scale sinusoidal temporal encoding.

Implements Eq. (5) of the manuscript:

    phi_time(Δt) = [ sin(2π Δt / T_k), cos(2π Δt / T_k) ]_{k=1..K}

with K = 8 learnable time scales initialised to
{1s, 10s, 1min, 10min, 1h, 6h, 24h, 7d}. The scales are kept positive
through a softplus reparametrisation so that gradient descent cannot drive
them to zero (which would collapse the encoding).
"""
from __future__ import annotations

import math

import torch
from torch import nn


DEFAULT_SCALES_SEC: tuple[float, ...] = (
    1.0,
    10.0,
    60.0,
    600.0,
    3_600.0,
    21_600.0,
    86_400.0,
    604_800.0,
)


class MultiScaleTimeEncoding(nn.Module):
    def __init__(self, scales_sec: tuple[float, ...] = DEFAULT_SCALES_SEC,
                 learnable: bool = True) -> None:
        super().__init__()
        scales = torch.tensor(scales_sec, dtype=torch.float32)
        # Store raw = log(exp(scales) - 1) so that softplus(raw) = scales.
        raw = torch.log(torch.expm1(scales.clamp(min=1e-6)))
        if learnable:
            self.raw_scales = nn.Parameter(raw)
        else:
            self.register_buffer("raw_scales", raw)
        self.K = len(scales_sec)
        self.output_dim = 2 * self.K

    def scales(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_scales) + 1e-6

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """delta_t: (..., ) tensor of time differences in seconds."""
        T = self.scales()                                    # (K,)
        angles = 2.0 * math.pi * delta_t.unsqueeze(-1) / T   # (..., K)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
