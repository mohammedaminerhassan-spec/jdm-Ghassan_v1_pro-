#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"

export CONFIG_PT="${CONFIG_PT:-configs/flash_109m_compact_2xt4.yaml}"
export CONFIG_SFT="${CONFIG_SFT:-configs/sft_flash_109m_compact_2xt4.yaml}"

KAGGLE_CAP_MIN=720
SPENT="${SESSION_SPENT_MIN:-0}"
LEFT=$(( KAGGLE_CAP_MIN - SPENT ))

export SESSION_LIMIT_MIN="${SESSION_LIMIT_MIN:-$(( LEFT - 30 ))}"
export TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-$(( SESSION_LIMIT_MIN - 60 ))}"

export PILOT_STEPS="${PILOT_STEPS:-20}"

echo "============================================================"
echo "  Ghassan English — 2xT4 Kaggle run"
echo "  recipe   : ${CONFIG_PT} + ${CONFIG_SFT}"
echo "  session  : ${SPENT}m already used, ${SESSION_LIMIT_MIN}m cap for this"
echo "             script, ${TIME_BUDGET_MIN}m of training, ${PILOT_STEPS} pilot steps"
echo "============================================================"

bash configs/kaggle/train_2xt4.sh --pt-fraction 60 "$@"
