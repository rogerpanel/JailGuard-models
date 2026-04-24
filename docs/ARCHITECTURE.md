# Architecture

The CT-DGNN-JailGuard framework composes four modules that map 1-to-1 to
sections IV-B…IV-E of the manuscript.

```
                               ┌────────────────────────────────────┐
  API traffic (JSONL /          │     ct_dgnn/data/graph_builder.py  │
  gRPC / webhook) ───────────►│  APIInteractionGraphBuilder        │
                               └─────────────────┬──────────────────┘
                                                 ▼
                               ┌────────────────────────────────────┐
                               │ ct_dgnn/models/ct_dgnn.py          │
                               │  • input_proj  (per-type)          │
                               │  • MultiScaleTimeEncoding          │
                               │  • TemporalHeterogeneousAttention  │
                               │  • HeterogeneousODEDynamics        │
                               │    └── NeuralODEBlock (dopri5)     │
                               │  • HierarchicalSet2Set pool        │
                               │  • CampaignClassifier              │
                               └─────────────────┬──────────────────┘
                                                 ▼
                               ┌────────────────────────────────────┐
                               │ ct_dgnn/robustness/                │
                               │  • spectral_norm (all W matrices)  │
                               │  • jacobian_reg (Hutchinson)       │
                               │  • certificate (Theorem 1)         │
                               │  • pgd_attack   (Fig. 5)           │
                               └─────────────────┬──────────────────┘
                                                 ▼
                               ┌────────────────────────────────────┐
                               │ ct_dgnn/models/llm_module.py       │
                               │  • intent classifier               │
                               │  • novel-attack narrator           │
                               │  • online InfoNCE update           │
                               └────────────────────────────────────┘
```

## Node / relation schema

| Type | `d` | Source features |
|---|---|---|
| user (U) | 64 | behavioural stats, temporal activity, API key metadata |
| session (S) | 128 | duration, query count, conversation flow |
| query (Q) | 384 | `all-MiniLM-L6-v2` embedding of the prompt text |
| model (M) | 32 | model endpoint + version metadata |

| Relation | Direction | Notes |
|---|---|---|
| `initiates` | user → session | one per new session |
| `contains` | session → query | one per query |
| `targets` | query → model | downstream model call |
| `responds` | model → query | carries the response features |
| `follows` | query → query | consecutive queries in a session |
| `shares_pattern` | query → query | cosine similarity ≥ 0.85 across sessions / users |

## Integration within the PI's existing toolchain

`../integrated_ai_ids/` provides the Python-side RobustIDPS ensemble
(SurrogateIDS, MambaShield, FedGTD, CyberSecLLM, …). The adapter in
`deployment/robustidps_integration.py` plugs CT-DGNN-JailGuard into the
same FastAPI surface used by those models, allowing side-by-side
deployment inside the RobustIDPS.ai 9-group dashboard.

## Certified robustness

Per Theorem 1 the end-to-end bound is

    ‖ŷ₁ − ŷ₂‖ ≤ L_MLP · L_pool · k · L_g · exp(L_f · T) · ε

Every component is constrained by spectral normalization so `L_f`,
`L_g`, `L_pool`, and `L_MLP` are all ≤ 1 at convergence. The Jacobian
regulariser (λ_J = 0.01) further tightens the empirical Lipschitz
constant of the ODE RHS. With `k ≤ 50` and `T ≤ 24 h` the certified
radius ε* is ≥ 0.15 as reported in the paper.
