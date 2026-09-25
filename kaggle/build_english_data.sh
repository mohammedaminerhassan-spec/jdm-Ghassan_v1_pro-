#!/usr/bin/env bash
# kaggle/build_english_data.sh — English-Pro shards from chat_if parquet
# (Ghassan v1 English-Pro: 350k deduped multi-turn EN convos).
#
# Inputs (Kaggle dataset, attach it to the notebook/session):
#   english_parquet/english_chat_part*.parquet        (~3.0GB, 298k rows)
#   english_parquet/english_instruction_part*.parquet (~126MB, 51k rows)
#   english_parquet/manifest.json
#   => mount under /kaggle/input/<dataset>/ or pass EN_PARQUET_DIR=<path>
#
# Tokenizer: artifacts/tokenizer/english32k.gtok SHIPS in the repo zip
# (trained --keep-case on chat_if; vocab 32000). Never rebuild it on Kaggle.
#
# Output: artifacts/shards_en/train_english_{chat,instruction}_*.gbin
#   (matches configs/en_pro.yaml + sft_en_pro.yaml mix — preflight verified)
#
# Shard recipe is seq_len 1024 (long convos are WINDOW-SPLIT, never truncated).
# All filtering is English-tuned: --style-mode en --keep-case.
#
# Requirements: build with Arrow (bash kaggle/setup.sh --with-parquet).
# Usage: bash kaggle/build_english_data.sh [EN_PARQUET_DIR]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_DIR}/build/bin/data_pipeline"
TOK="${REPO_DIR}/artifacts/tokenizer/english32k.gtok"
SHARD_EN="${REPO_DIR}/artifacts/shards_en"
SHARD_SEQ_LEN=1024

EN_DIR="${1:-${EN_PARQUET_DIR:-}}"
if [[ -z "${EN_DIR}" && -d "/kaggle/input" ]]; then
    for d in /kaggle/input/*/; do
        [[ -d "$d" ]] || continue
        # maxdepth 4: the all-in-one zip lays out
        #   <ds>/ghassan-en-pro/english_parquet/*.parquet
        if [[ -n "$(find "$d" -maxdepth 4 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)" ]]; then
            EN_DIR="$(dirname "$(find "$d" -maxdepth 4 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)")"
            break
        fi
        if [[ -n "$(find "$d" -maxdepth 4 -name '*.parquet' 2>/dev/null | head -n 1)" ]]; then
            EN_DIR="$d"
        fi
    done
fi
# Local fallbacks (in order):
#   1. dataset/english_parquet (vendored in THIS checkout, 522k chat + 478k inst)
#   2. ./english_parquet (prebuilt lake)
#   3. kaggle_upload/english_parquet (vendored inside the project zip)
# FIX P2: the script never searched dataset/english_parquet although the lake
# ships there in this checkout, forcing a manual EN_PARQUET_DIR every time.
if [[ -z "${EN_DIR}" || ! -d "${EN_DIR}" ]]; then
    if [[ -n "$(find "${REPO_DIR}/dataset/english_parquet" -maxdepth 1 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)" ]]; then
        EN_DIR="${REPO_DIR}/dataset/english_parquet"
    elif [[ -n "$(find "${REPO_DIR}/english_parquet" -maxdepth 1 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)" ]]; then
        EN_DIR="${REPO_DIR}/english_parquet"
    elif [[ -n "$(find "${REPO_DIR}/kaggle_upload/english_parquet" -maxdepth 1 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)" ]]; then
        EN_DIR="${REPO_DIR}/kaggle_upload/english_parquet"
    fi
fi

[[ -f "${BIN}" ]] || { echo "[ERROR] missing ${BIN}. Run setup.sh --with-parquet first."; exit 1; }
[[ -f "${TOK}" ]] || { echo "[ERROR] missing tokenizer ${TOK} (ships in the repo zip)."; exit 1; }
[[ -n "${EN_DIR}" && -d "${EN_DIR}" ]] || {
    echo "[ERROR] English parquet dir not found. Attach the dataset and/or pass:"
    echo "  EN_PARQUET_DIR=/kaggle/input/<ds> bash kaggle/build_english_data.sh"
    exit 1
}
"${BIN}" parquet --probe >/dev/null 2>&1 || {
    echo "[ERROR] no Arrow backend. Rebuild: bash kaggle/setup.sh --with-parquet"
    exit 1
}
mkdir -p "${SHARD_EN}"

echo "============================================================"
echo "  Ghassan English-Pro — parquet -> shards"
echo "  lake: ${EN_DIR}"
echo "============================================================"

# ---- tokenizer gate (32k keep-case, never silent 16k/darija fallback)
TOK_VOCAB=$("${BIN}" tok-info --tokenizer "${TOK}" 2>/dev/null | grep -oE 'vocab_size=[0-9]+' | cut -d= -f2 || true)
[[ "${TOK_VOCAB}" == "32000" ]] || { echo "[ERROR] tokenizer vocab=${TOK_VOCAB:-?}, need 32000."; exit 1; }
echo "[tok] ok: ${TOK} (vocab 32000, keep-case)"

echo ""
echo "[1/2] english_chat -> ${SHARD_EN} ..."
"${BIN}" parquet --mode chat --lake "${EN_DIR}" --match english_chat \
    --tokenizer "${TOK}" --expect-vocab 32000 \
    --out "${SHARD_EN}" --domain english_chat \
    --style-mode en --keep-case \
    --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" \
    --report "${SHARD_EN}/report_chat.txt"

echo ""
echo "[2/2] english_instruction -> ${SHARD_EN} ..."
"${BIN}" parquet --mode chat --lake "${EN_DIR}" --match english_instruction \
    --tokenizer "${TOK}" --expect-vocab 32000 \
    --out "${SHARD_EN}" --domain english_instruction \
    --style-mode en --keep-case \
    --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" \
    --report "${SHARD_EN}/report_instruction.txt"

# ---- [3/3] behavior seasoning: authored persona/discipline dialogues --------
# The Hermes lake teaches knowledge, not persona. These ~15k authored
# conversations (greetings, honesty, refusal, neutrality, dialogue flow) are
# what make the model behave like Ghassan instead of generic Hermes output.
# Small on purpose (~2M tokens): seasoning, not the meal. SFT samples it at
# 0.15 (see sft_en_pro.yaml mix); pretrain ignores it (lake dominates there).
# Set SKIP_BEHAVIOR=1 to skip (not recommended for the English-Pro model).
BEHAVIOR_N="${BEHAVIOR_N:-15000}"
if [[ "${SKIP_BEHAVIOR:-0}" == "1" ]]; then
    echo ""
    echo "[3/3] behavior SKIPPED (SKIP_BEHAVIOR=1) — SFT mix expects english_behavior/* !"
else
    echo ""
    echo "[3/3] english_behavior (${BEHAVIOR_N} authored conversations) -> ${SHARD_EN} ..."
    BEHAVIOR_JSON="${SHARD_EN}/synth_en_behavior.jsonl"
    "${BIN}" synth --lang en --n "${BEHAVIOR_N}" --seed 4321 \
        --max-template-uses 400 --out "${BEHAVIOR_JSON}"
    "${BIN}" build --chat "${BEHAVIOR_JSON}" \
        --tokenizer "${TOK}" --expect-vocab 32000 \
        --out "${SHARD_EN}" --domain english_behavior \
        --style-mode en --keep-case \
        --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" \
        --report "${SHARD_EN}/report_behavior.txt"
    rm -f "${BEHAVIOR_JSON}"
fi

echo ""
echo "---- inspect (MEASURED totals are the source of truth) ----"
"${BIN}" inspect --shards "${SHARD_EN}" --tokenizer "${TOK}" || true

echo ""
echo "---- domain inventory vs configs/en_pro.yaml + sft_en_pro.yaml mix ----"
MIX_MISSING=0
for domain in english_chat english_instruction english_behavior; do
    n=$(find "${SHARD_EN}" -name "train_${domain}_*.gbin" 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo "  [MISSING] train_${domain}_*.gbin"; MIX_MISSING=1
    else
        echo "  [ok] ${domain}: $n shard(s)"
    fi
done
[[ "${MIX_MISSING}" -eq 0 ]] || { echo "[ERROR] mix mismatch — aborting before GPU hours burn."; exit 1; }

echo ""
echo "============================================================"
echo "  Done. Train with (single T4, resume-safe):"
echo "  CONFIG_PT=configs/en_pro.yaml CONFIG_SFT=configs/sft_en_pro.yaml \\"
echo "  PT_DIR=artifacts/shards_en SFT_DIR=artifacts/shards_en \\"
echo "  CKPT_PT=artifacts/checkpoints/en_pro CKPT_SFT=artifacts/checkpoints/en_pro_sft \\"
echo "  GGUF_OUT=artifacts/ghassan-en-pro_q4_0.gguf \\"
echo "  bash kaggle/train_1b.sh --time-budget-min 540 --export-profile q4_0"
echo "  Chat with: ghassan-ai chat --model artifacts/ghassan-en-pro_q4_0.gguf --persona en"
echo "============================================================"
