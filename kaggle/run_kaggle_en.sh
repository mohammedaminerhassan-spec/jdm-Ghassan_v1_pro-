#!/usr/bin/env bash
# kaggle/run_kaggle_en.sh — ONE command for the whole English run on Kaggle
# 2xT4 (16 GB each, 12 h hard session cap, 20 GB saved-output cap).
#
# This is the only file you need to run. It picks the recipe that actually
# fits the session, wires the launcher to it, and accounts for the wall-clock
# the setup + shard-build cells already burned.
#
#   bash kaggle/run_kaggle_en.sh
#
# Optional environment:
#   SESSION_SPENT_MIN=150   minutes the session has ALREADY used (setup.sh +
#                           build_english_data.sh). Default 0.
#   KEEP_PT_CKPTS=1         keep the stage-1 checkpoints (costs output quota)
#   NO_EXPORT=1             skip the GGUF export
#
# Why this recipe and not the 1B (pro_1b_2xt4, the launcher default):
#   params/tok   1B  = 1037M total / 587M active   -> D/N ~0.02 in one session
#   params/tok   this =  109M total /  36M active  -> D/N ~2   (usable)
#   VRAM (dry-run, per GPU): 1B = 14.35 GiB of the 16 GiB gate (89.7% used,
#   10.3% margin — one step from the preflight FAIL threshold); this = 4.05
#   GiB (25%). The 1B cannot be trained well in 12 h at all, so the safe
#   recipe is also the better one.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

export CONFIG_PT="${CONFIG_PT:-configs/en_compact_2xt4.yaml}"
export CONFIG_SFT="${CONFIG_SFT:-configs/sft_en_compact_2xt4.yaml}"

# Kaggle kills the whole notebook at 720 min regardless of which cell is
# running. train_2xt4.sh starts its own clock at launch, so hand it the
# budget that is actually LEFT, otherwise a 2 h data build buys you a run
# that gets SIGKILLed mid-stage-B.
KAGGLE_CAP_MIN=720
SPENT="${SESSION_SPENT_MIN:-0}"
LEFT=$(( KAGGLE_CAP_MIN - SPENT ))
# 60 min of that stays as head-room for the final checkpoint + GGUF export
# (EXPORT_MARGIN_SEC) and for the Save Version itself.
export SESSION_LIMIT_MIN="${SESSION_LIMIT_MIN:-$(( LEFT - 30 ))}"
export TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-$(( SESSION_LIMIT_MIN - 60 ))}"

# Pilot steps: 20 at T=2048 is ~2-3 min of real stepping, and the pilot's
# tok/s is what the whole step plan is built from. Too few steps and CUDA
# init dominates the measurement and the plan comes out short.
export PILOT_STEPS="${PILOT_STEPS:-20}"

echo "============================================================"
echo "  Ghassan English — 2xT4 Kaggle run"
echo "  recipe   : ${CONFIG_PT} + ${CONFIG_SFT}"
echo "  session  : ${SPENT}m already used, ${SESSION_LIMIT_MIN}m cap for this"
echo "             script, ${TIME_BUDGET_MIN}m of training, ${PILOT_STEPS} pilot steps"
echo "============================================================"

bash kaggle/train_2xt4.sh --pt-fraction 60 "$@"
