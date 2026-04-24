# Datasets

Six datasets are used in the manuscript (Table II). Five are public;
one (`JailCampaign`) is constructed locally.

## 1. JailbreakBench (Chao et al., NeurIPS 2024)

- **Source:** https://github.com/JailbreakBench/jailbreakbench
- **Size:** 18,200 prompts
- **Role:** primary training signal for known-strategy attack artefacts
- **Temporal extension:** the DAN community metadata from Shen et al.,
  CCS 2024 (15,140 prompts in 131 communities) is overlaid to recover
  cross-account coordination.

Expected layout after `scripts/download_datasets.sh`:

```
data/jailbreak_bench/
  jbb_behaviors.jsonl
  jbb_artifacts/
  dan_communities.json
```

## 2. HarmBench (Mazeika et al., 2024)

- **Source:** https://github.com/centerforaisafety/HarmBench
- **Size:** 51,000 behaviours × 18 attack methods on 33 targets
- **Role:** *zero-shot evaluation only* — see
  `ct_dgnn/evaluation/zero_shot.py` and `configs/default.yaml::eval.zero_shot_leave_out`.

## 3. WildJailbreak (Jiang et al., NeurIPS 2024)

- **Source:** https://huggingface.co/datasets/allenai/wildjailbreak
- **Size:** 262K (vanilla, adversarial) pairs over 5,704 tactic clusters
- **Role:** in-the-wild behavioural diversity during training

## 4. LMSYS-Chat-1M (Zheng et al., 2024)

- **Source:** https://huggingface.co/datasets/lmsys/lmsys-chat-1m
- **Size:** 1M real conversations, ~210K unique IPs, 25 LLM targets
- **Role:** realistic benign traffic baseline. OpenAI moderation flags
  provide weak per-message labels.

## 5. CIC-IoT-2023 + Kaggle network bundles

- **Primary source:** https://www.unb.ca/cic/datasets/iotdataset-2023.html
- **Kaggle bundles (user-supplied DOIs):**
  - General Network Traffic — DOI 10.34740/kaggle/dsv/12483891
  - Cloud, Microservices & Edge — DOI 10.34740/KAGGLE/DSV/12479689
- **Size:** 46M flows (primary) + 84.2M records (bundled) across 83+
  unique attack classes
- **Role:** cross-domain validation. Flow records are re-interpreted as
  API-like events (src_ip → user, 5-tuple → session, feature vector →
  query, dst port → model endpoint).

Loader: `ct_dgnn/data/datasets/cic_iot_2023.py`.

## 6. JailCampaign *(ours)*

- **Source:** constructed by `scripts/build_jail_campaign.py`
- **Size:** 523K annotated interactions, 5,247 labelled campaigns,
  12,400 users, 6-month synthetic temporal span
- **Strategies mixed:** GCG, PAIR, AutoDAN, AmpleGCG, ArrAttack,
  Tree-of-Attack, Crescendo, Many-shot, Actor-Attack, Mixed.

Reproduce with:

```bash
python scripts/build_jail_campaign.py --out data/jail_campaign \
    --n_campaigns 5247 --total_events 523000
```

## Directory layout

```
data/
  jailbreak_bench/     # public, git-cloned
  harm_bench/          # public, git-cloned
  wild_jailbreak/      # HuggingFace cache
  lmsys_chat/          # HuggingFace cache
  cic_iot_2023/        # CSV flows — UNB portal + Kaggle bundles
  jail_campaign/       # locally synthesised
  processed/           # preprocessed InteractionBatch shards
    <dataset_name>/
      shard_00000.pkl
      ...
```
