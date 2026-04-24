#!/usr/bin/env bash
# Fetch every public dataset required for Table II of the manuscript.
# Usage:
#   bash scripts/download_datasets.sh [DATA_ROOT]
#
# Environment variables:
#   HF_TOKEN         HuggingFace access token (required for WildJailbreak + LMSYS).
#   KAGGLE_USERNAME  Kaggle creds (for CIC-IoT-2023 + Kaggle network bundles).
#   KAGGLE_KEY
#
# The JailCampaign benchmark is NOT downloaded — synthesize it locally via
#   python scripts/build_jail_campaign.py --out data/jail_campaign

set -euo pipefail
DATA_ROOT=${1:-data}
mkdir -p "${DATA_ROOT}"

echo ">>> [1/5] JailbreakBench (github, 18.2K)"
if [[ ! -d "${DATA_ROOT}/jailbreak_bench" ]]; then
  git clone --depth 1 https://github.com/JailbreakBench/jailbreakbench \
      "${DATA_ROOT}/jailbreak_bench"
fi

echo ">>> [2/5] HarmBench (github, 51K)"
if [[ ! -d "${DATA_ROOT}/harm_bench" ]]; then
  git clone --depth 1 https://github.com/centerforaisafety/HarmBench \
      "${DATA_ROOT}/harm_bench"
fi

echo ">>> [3/5] WildJailbreak (HuggingFace, 262K)"
python - <<'PY'
import os
from datasets import load_dataset
root = os.environ.get("DATA_ROOT", "data") + "/wild_jailbreak"
os.makedirs(root, exist_ok=True)
load_dataset("allenai/wildjailbreak", "train", cache_dir=root)
load_dataset("allenai/wildjailbreak", "eval", cache_dir=root)
PY

echo ">>> [4/5] LMSYS-Chat-1M (HuggingFace, 1M)"
python - <<'PY'
import os
from datasets import load_dataset
root = os.environ.get("DATA_ROOT", "data") + "/lmsys_chat"
os.makedirs(root, exist_ok=True)
load_dataset("lmsys/lmsys-chat-1m", split="train", cache_dir=root)
PY

echo ">>> [5/5] CIC-IoT-2023 + Kaggle network bundles (46M)"
mkdir -p "${DATA_ROOT}/cic_iot_2023"
# Paper primary: official UNB CIC portal.
echo "Download CIC-IoT-2023 from https://www.unb.ca/cic/datasets/iotdataset-2023.html"
echo "and extract CSVs to ${DATA_ROOT}/cic_iot_2023/"
# Plus the two Kaggle bundles the user requested.
if command -v kaggle >/dev/null 2>&1; then
  kaggle datasets download -d "rogeranaedevha/general-network-traffic" \
         -p "${DATA_ROOT}/cic_iot_2023" --unzip || true
  kaggle datasets download -d "rogeranaedevha/cloud-microservices-edge" \
         -p "${DATA_ROOT}/cic_iot_2023" --unzip || true
else
  cat <<'MSG'
 !! kaggle CLI not installed. Install with:
       pip install kaggle
       export KAGGLE_USERNAME=... KAGGLE_KEY=...
 Then rerun this script, or manually fetch:
    DOI 10.34740/kaggle/dsv/12483891   (General Network Traffic)
    DOI 10.34740/KAGGLE/DSV/12479689   (Cloud, Microservices & Edge)
MSG
fi

echo ">>> JailCampaign benchmark (local synthesis)"
python scripts/build_jail_campaign.py --out "${DATA_ROOT}/jail_campaign"

echo ">>> done. Dataset root: ${DATA_ROOT}"
