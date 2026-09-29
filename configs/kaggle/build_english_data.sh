#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${REPO_DIR}/build/bin/data_pipeline"
TOK="${REPO_DIR}/artifacts/tokenizer/english32k.gtok"
SHARD_EN="${REPO_DIR}/artifacts/shards_en"
SHARD_SEQ_LEN=1024

EN_DIR="${1:-${EN_PARQUET_DIR:-}}"
if [[ -z "${EN_DIR}" && -d "/kaggle/input" ]]; then

    hit=""
    hit="$(find /kaggle/input -maxdepth 8 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)"
    if [[ -n "${hit}" ]]; then
        EN_DIR="$(dirname "${hit}")"
    else
        any=""
        any="$(find /kaggle/input -maxdepth 8 -name '*.parquet' 2>/dev/null | head -n 1)"
        if [[ -n "${any}" ]]; then
            EN_DIR="$(dirname "${any}")"
        fi
    fi
fi

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
    echo "  EN_PARQUET_DIR=/kaggle/input/<ds> bash configs/kaggle/build_english_data.sh"
    exit 1
}
"${BIN}" parquet --probe >/dev/null 2>&1 || {
    echo "[ERROR] no Arrow backend. Rebuild: bash configs/kaggle/setup.sh --with-parquet"
    exit 1
}
mkdir -p "${SHARD_EN}"

echo "============================================================"
echo "  Ghassan English-Pro — parquet -> shards"
echo "  lake: ${EN_DIR}"
echo "============================================================"

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
echo "---- domain inventory vs configs/flash_480m_single.yaml + sft_flash_480m_single.yaml mix ----"
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
echo "  CONFIG_PT=configs/flash_480m_single.yaml CONFIG_SFT=configs/sft_flash_480m_single.yaml \\"
echo "  PT_DIR=artifacts/shards_en SFT_DIR=artifacts/shards_en \\"
echo "  CKPT_PT=artifacts/checkpoints/en_pro CKPT_SFT=artifacts/checkpoints/en_pro_sft \\"
echo "  GGUF_OUT=artifacts/ghassan-en-pro_q4_0.gguf \\"
echo "  bash configs/kaggle/train_1b.sh --time-budget-min 540 --export-profile q4_0"
echo "  Chat with: ghassan-ai chat --model artifacts/ghassan-en-pro_q4_0.gguf --persona en"
echo "============================================================"
