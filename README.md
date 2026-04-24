# CT-DGNN-JailGuard

**Continuous-Time Dynamic Graph Neural Networks with Certified Robustness for Real-Time Detection of LLM Jailbreak Campaigns via API Interaction Graphs**

Reference implementation for the IEEE submission of the same title
(manuscript & appendix: https://github.com/rogerpanel/JailGuard-models).

This repository provides the full training, evaluation, certification,
and benchmark-generation pipeline used in the paper. It is intended
as the reproducibility artefact for the IEEE publication and is
linked directly from the submitted manuscript.

---

## 1. At a glance

| Property | Value |
|---|---|
| Campaign-level F1 (avg over 5 eval datasets) | **96.8%** |
| Query-level AUC-ROC (avg) | **98.4%** |
| P99 inference latency (A100) | **38 ms** |
| Certified perturbation radius ε* | **≥ 0.15** (k=50, T=24 h) |
| Embedding dimension `d` | 128 |
| ODE integration steps `L` | 3 |
| Attention heads | 4 |
| Optimizer | AdamW (lr=5e-4, wd=1e-4, cosine over 100 epochs) |
| ODE solver | Dormand–Prince (`dopri5`), rtol=1e-3, atol=1e-4 |

## 2. Architecture

The system has four modules (`ct_dgnn/models/`):

1. **Heterogeneous graph construction** (`data/graph_builder.py`) —
   transforms raw API traffic into a continuous-time heterogeneous
   graph `G(t) = (V, E, R, φ, ψ)` with node types
   `{U, S, Q, M}` (user / session / query / model) and six
   relation types
   `{initiates, contains, targets, responds, follows, shares_pattern}`.
2. **Continuous-time Neural ODE dynamics** (`models/ode_dynamics.py`,
   `models/attention.py`) — type-specific ODE
   `dh_v/dt = f_τ(h_v, Σ α g_r(h_u))` solved with adaptive-step
   `dopri5` via `torchdiffeq`, with multi-scale sinusoidal temporal
   encoding `{1s, 10s, 1min, 10min, 1h, 6h, 24h, 7d}`.
3. **Lipschitz-certified classifier** (`robustness/`) — spectral
   normalization on every weight matrix, Jacobian-Frobenius
   regularizer on `f_τ`, and a Grönwall-based certificate giving
   `‖ŷ₁ − ŷ₂‖ ≤ L_MLP · L_pool · k · L_g · exp(L_f T) · ε`.
4. **LLM-augmented zero-shot analyzer** (`models/llm_module.py`) —
   Mistral-7B-Instruct for intent labels + natural-language narration
   of novel attack subgraphs, with a Bayesian online-update feedback
   loop for analyst-confirmed novel attacks.

The exact flow:

```
API traffic ─► Graph Construction ─► CT-DGNN Neural ODE ─► Lipschitz-Certified Classifier ─► Campaign Decision
                                           ▲                          ▲
                                           │                          │
                                           └─ LLM intent / narration ─┘
```

## 3. Datasets

Six datasets are used (see `docs/DATASETS.md`). Five are public; one
(`JailCampaign`) is constructed by this repo.

| Dataset | Interactions | Users | Campaigns/Classes | Source |
|---|---|---|---|---|
| JailbreakBench | 18.2K | 1,405 | 131 communities | [github.com/JailbreakBench/jailbreakbench](https://github.com/JailbreakBench/jailbreakbench) |
| HarmBench | 51.0K | synthetic | 510 categories | [github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) |
| WildJailbreak | 262K | in-the-wild | 5,704 tactic clusters | [huggingface.co/datasets/allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) |
| LMSYS-Chat-1M | 1.0M | 210K IPs | 25 LLM targets | [huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) |
| CIC-IoT-2023 | 46.0M | 105 devices | 33 attack types | [unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| JailCampaign *(ours)* | 523K | 12.4K | 5,247 campaigns | `scripts/build_jail_campaign.py` |

The `General Network Traffic` (DOI: 10.34740/kaggle/dsv/12483891)
and `Cloud, Microservices & Edge` (DOI: 10.34740/KAGGLE/DSV/12479689)
Kaggle bundles are ingested by `ct_dgnn/data/datasets/cic_iot_2023.py`
for the cross-domain network evaluation reported in Tab. II of the
manuscript.

Run `bash scripts/download_datasets.sh` to fetch everything.

## 4. Quick start

```bash
# 1. Install
pip install -e .

# 2. Download or synthesize datasets
bash scripts/download_datasets.sh
python scripts/build_jail_campaign.py --out data/jail_campaign --n_campaigns 5247

# 3. Preprocess into graph form
python scripts/preprocess.py --config configs/default.yaml

# 4. Train
python scripts/train.py --config configs/default.yaml

# 5. Evaluate (clean + PGD + certified radius)
python scripts/evaluate.py   --config configs/default.yaml --ckpt runs/latest/best.pt
python scripts/certify.py    --config configs/default.yaml --ckpt runs/latest/best.pt
python scripts/benchmark_latency.py --ckpt runs/latest/best.pt
```

## 5. Relation to prior work in this repository

This implementation is the direct continuation of the PhD codebase in
the parent `CV/` repository (see `../integrated_ai_ids/` and
`../robustidps_web_app/`). It reuses:

- The **CT-TGNN** Neural-ODE graph scaffolding from
  `../neural-ode-model.ipynb` and `../fedgtd-v2.ipynb`.
- The **Lipschitz / spectral-normalization certification** from
  `../stochastic-games-defense-models.ipynb`.
- The **CyberSecLLM** classifier plumbing from
  `../hybrid-stochastic-llm-transformer-model.ipynb`.
- The **RobustIDPS.ai** deployment shell from
  `../robustidps_web_app/` (integration adapter in
  `deployment/robustidps_integration.py`).

## 6. Citation

```bibtex
@article{anaedevha2026ctdgnn,
  title   = {{CT-DGNN-JailGuard}: Continuous-Time Dynamic Graph Neural
             Networks with Certified Robustness for Real-Time Detection
             of {LLM} Jailbreak Campaigns via {API} Interaction Graphs},
  author  = {Anaedevha, Roger Nick},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2026},
  note    = {Code: https://github.com/rogerpanel/CV/tree/main/ct\_dgnn\_jailguard}
}
```

## 7. License

MIT. Copyright 2026 Roger Nick Anaedevha.
