#!/usr/bin/env bash
# kaggle/build_billion_data.sh — Build ~1B-token shards for Ultra 1B
#
# What it does:
#   1. synth pretrain  : 15 batches x 100k = 1.5M conversations (~570M tokens)
#   2. synth sft       : 2 batches x 100k = 200k conversations (~80M tokens)
#   3. json data       : your "Ai dariga datasets/" JSON/JSONL (~200-400M tokens)
#   4. build shards    : artifacts/shards_1b + artifacts/shards_1b_sft
#
# Usage:
#   bash kaggle/build_billion_data.sh [--json-dir "Ai dariga datasets/"]
#
# Requirements:
#   - build/bin/data_pipeline built (bash kaggle/setup.sh)
#   - artifacts/tokenizer/darija.gtok exists
#   - ~30GB free disk (synth jsonl ~8GB + shards ~4GB)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_DIR}/build/bin/data_pipeline"
TOK="${REPO_DIR}/artifacts/tokenizer/darija32k.gtok"
if [[ ! -f "${TOK}" ]]; then TOK="${REPO_DIR}/artifacts/tokenizer/darija.gtok"; fi
SYNTH_DIR="${REPO_DIR}/artifacts/synth"
SHARD_1B="${REPO_DIR}/artifacts/shards_1b"
SHARD_SFT="${REPO_DIR}/artifacts/shards_1b_sft"
JSON_DIR="${1:-Ai dariga datasets/}"
if [[ $# -gt 0 && "$1" == "--json-dir" ]]; then JSON_DIR="$2"; fi

[[ -f "${BIN}" ]] || { echo "[ERROR] missing ${BIN}. Run setup.sh first."; exit 1; }
[[ -f "${TOK}" ]] || { echo "[ERROR] missing tokenizer ${TOK}."; exit 1; }
mkdir -p "${SYNTH_DIR}" "${SHARD_1B}" "${SHARD_SFT}"

echo "============================================================"
echo "  Ghassan Ultra 1B — Billion-scale data build"
echo "  target: ~1B tokens (570M synth + json + SFT)"
echo "============================================================"

# ---- 1. pretrain synth in batches (crash-safe, resumable) ----
echo ""
echo "[1/4] Pretrain synth: 15 x 100k (1.5M total)..."
for i in $(seq 0 14); do
  OUT="${SYNTH_DIR}/synth_1b_part$(printf %02d $i).jsonl"
  if [[ -f "$OUT" ]]; then echo "  skip part $i (exists)"; continue; fi
  SEED=$((1234 + i * 7919))
  echo "  batch $i/14 seed=${SEED} -> $OUT"
  "${BIN}" synth --out "$OUT" --n 100000 --seed "$SEED" \
    --max-template-uses 20000 || { echo "[WARN] batch $i failed, retry with fresh seed"; continue; }
done
echo "[1/4] merging..."
cat "${SYNTH_DIR}"/synth_1b_part*.jsonl > "${SYNTH_DIR}/synthetic_1b.jsonl"
wc -l "${SYNTH_DIR}/synthetic_1b.jsonl"

# ---- 2. sft synth ----
echo ""
echo "[2/4] SFT synth: 2 x 100k (200k total)..."
for i in 0 1; do
  OUT="${SYNTH_DIR}/synth_1b_sft_part0${i}.jsonl"
  if [[ -f "$OUT" ]]; then echo "  skip sft part $i (exists)"; continue; fi
  SEED=$((9999 + i * 104729))
  "${BIN}" synth --out "$OUT" --n 100000 --seed "$SEED" --max-template-uses 2000
done
cat "${SYNTH_DIR}"/synth_1b_sft_part*.jsonl > "${SYNTH_DIR}/synthetic_1b_sft.jsonl"

# ---- 3+4. build shards ----
echo ""
echo "[3/4] Building pretrain shards -> ${SHARD_1B} ..."
# domain split so data.mix in ultra_1b.yaml works:
#   - synth -> darija_conversational + darija_reasoning_logic (split by script inside pipeline)
#   - json  -> educational_science + general_culture_law
"${BIN}" build --tokenizer "$TOK" --out "$SHARD_1B" \
  --chat "${SYNTH_DIR}/synthetic_1b.jsonl" \
  --domain darija_conversational \
  --shard-tokens 50000000 --seq-len 1024 \
  --report "${SHARD_1B}/report.txt"

if [[ -e "$JSON_DIR" ]]; then
  echo ""
  echo "[json] adding $JSON_DIR ..."
  "${BIN}" json --dir "$JSON_DIR" --tokenizer "$TOK" \
    --out "$SHARD_1B" --domain educational_science \
    --shard-tokens 50000000 --seq-len 1024 || echo "[WARN] json step failed, continuing with synth only"
else
  echo "[WARN] no json path $JSON_DIR, synth-only shards"
fi

echo ""
echo "[4/4] Building SFT shards -> ${SHARD_SFT} ..."
"${BIN}" build --tokenizer "$TOK" --out "$SHARD_SFT" \
  --chat "${SYNTH_DIR}/synthetic_1b_sft.jsonl" \
  --domain darija_chat \
  --shard-tokens 50000000 --seq-len 1024 \
  --report "${SHARD_SFT}/report.txt"

echo ""
echo "---- inspect ----"
"${BIN}" inspect --shards "$SHARD_1B" --tokenizer "$TOK" || true
"${BIN}" inspect --shards "$SHARD_SFT" --tokenizer "$TOK" || true
echo ""
echo "============================================================"
echo "  Done. Check totals above: want train ~= 900M-1.1B tokens."
echo "  Next: bash kaggle/train_1b.sh"
echo "============================================================"
