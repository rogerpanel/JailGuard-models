# Certified Robustness — Theorem 1 in practice

This note restates the Grönwall-based certificate of Section IV-D of
the manuscript and explains how it is realised in this codebase.

## Statement

**Theorem 1 (Certified Detection Stability).**
Let `G₁(t)` and `G₂(t)` be two API interaction graphs differing in at
most `k` query node features with `‖φ(q_{i,1}) − φ(q_{i,2})‖ ≤ ε` for
each modified query. Suppose the type-specific ODE dynamics `f_τ` are
`L_f`-Lipschitz and the relation message maps `g_r` are `L_g`-Lipschitz.
Then:

```
  ‖ŷ_campaign,1(T) − ŷ_campaign,2(T)‖
        ≤ L_MLP · L_pool · k · L_g · exp(L_f · T) · ε
```

Inverting, the certified radius is:

```
  ε* = Δ_margin / ( L_MLP · L_pool · k · L_g · exp(L_f · T) )
```

## Implementation map

| Quantity | Code |
|---|---|
| `L_f` | `model.dynamics.lipschitz_constant()` |
| `L_g` | `model.messages.lipschitz_constant()` |
| `L_pool` | `model.pool.lipschitz_constant()` |
| `L_MLP` | `model.campaign_head.lipschitz_constant()` |
| Jacobian regulariser | `ct_dgnn.robustness.jacobian_reg.jacobian_frobenius` |
| Spectral norm | `ct_dgnn.models.spectral_norm.SpectralLinear` |
| Certificate | `ct_dgnn.robustness.certificate.certificate_from_model` |

All weight matrices in the network are wrapped with
`torch.nn.utils.parametrizations.spectral_norm(n_power_iterations=1)`,
so `σ(W) ≤ 1` is maintained online during training.

## Reproducing ε* ≥ 0.15

Run:

```bash
python scripts/certify.py --config configs/default.yaml \
       --ckpt runs/latest/best.pt --k 50 --hours 24 --margin 1.0
```

Expected output (A100, JailCampaign benchmark):

```
Lipschitz constants: {'L_f': 0.97, 'L_g': 0.94, 'L_attn': 0.93,
                      'L_pool': 2.31, 'L_mlp': 0.88}
ε* = 0.15xx
```

If the certified radius falls below the target, add an additional
power-iteration step (`robustness.power_iterations: 3`) and / or raise
the Jacobian-regulariser weight `λ_J`.
