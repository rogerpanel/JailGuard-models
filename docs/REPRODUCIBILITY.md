# Reproducibility

This document lists the exact commands to reproduce every figure and
table in the manuscript.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Hardware used in the paper: 4× NVIDIA A100 (80 GB), 512 GB RAM, dual
AMD EPYC 7763. Single-GPU runs are fully supported at lower batch
sizes.

## Data

```bash
# Public datasets
bash scripts/download_datasets.sh data

# JailCampaign benchmark (523K events, 5,247 campaigns)
python scripts/build_jail_campaign.py --out data/jail_campaign
```

## Preprocessing

```bash
for ds in jailbreak_bench wild_jailbreak lmsys_chat cic_iot_2023 jail_campaign; do
    python scripts/preprocess.py --config configs/default.yaml --dataset $ds \
                                 --out data/processed
done
```

## Training (Table III main results)

```bash
for ds in jailbreak_bench wild_jailbreak lmsys_chat cic_iot_2023 jail_campaign; do
    python scripts/train.py --config configs/default.yaml \
                            --data data/processed \
                            --out runs/$ds
done
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml \
                           --ckpt runs/jail_campaign/best.pt     # Tab. III + Fig. 5
python scripts/certify.py  --ckpt runs/jail_campaign/best.pt \
                           --k 50 --hours 24 --margin 1.0        # ε* (Sec. IV-D)
python scripts/benchmark_latency.py --ckpt runs/jail_campaign/best.pt \
                           --samples 1000                        # Tab. IV
```

## Zero-shot leave-one-strategy-out (Sec. V-D)

```bash
python scripts/train.py --config configs/default.yaml   # edit cfg.eval.zero_shot_leave_out
```

## Deployment on RobustIDPS.ai

```bash
docker compose -f deployment/docker-compose.yml up --build
```

This starts the detector, a Redis graph-store, and a PostgreSQL
persistence layer matching the three-tier architecture described in
Section VI-A of the manuscript.
