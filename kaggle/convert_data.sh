#!/usr/bin/env bash
# kaggle/convert_data.sh
# ----------------------------------------------------------------
# Converts the JSON/JSONL datasets into .gbin training shards
# using the built C++ data_pipeline tool.
# Run this AFTER setup.sh has completed successfully.
#
# Usage:
#   bash kaggle/convert_data.sh [--json-dir <dir>] [--tokenizer <path>]
# ----------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
BINARY="${BUILD_DIR}/bin/data_pipeline"

# ---- default paths (can be overridden via CLI args)
JSON_DIR="${REPO_DIR}/Ai dariga datasets"
TOKENIZER="${REPO_DIR}/artifacts/tokenizer/darija32k.gtok"
if [[ ! -f "${TOKENIZER}" ]]; then TOKENIZER="${REPO_DIR}/artifacts/tokenizer/darija.gtok"; fi
OUT_DIR="${REPO_DIR}/artifacts/shards"
VAL_RATIO="0.005"
SHARD_TOKENS="50000000"
SEQ_LEN="4096"

# ---- parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json-dir)      JSON_DIR="$2";    shift 2 ;;
        --tokenizer)     TOKENIZER="$2";   shift 2 ;;
        --out)           OUT_DIR="$2";     shift 2 ;;
        --val-ratio)     VAL_RATIO="$2";   shift 2 ;;
        --shard-tokens)  SHARD_TOKENS="$2"; shift 2 ;;
        --seq-len)       SEQ_LEN="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Ghassan AI — JSON/JSONL → .gbin Conversion"
echo "============================================================"
echo ""
echo "  json dir     : ${JSON_DIR}"
echo "  tokenizer    : ${TOKENIZER}"
echo "  output dir   : ${OUT_DIR}"
echo "  val ratio    : ${VAL_RATIO}"
echo "  shard tokens : ${SHARD_TOKENS}"
echo "  seq len      : ${SEQ_LEN}"
echo ""

# ---- verify binary
if [[ ! -f "${BINARY}" ]]; then
    echo "[ERROR] data_pipeline binary not found: ${BINARY}"
    echo "        Run kaggle/setup.sh first."
    exit 1
fi

# ---- verify tokenizer
if [[ ! -f "${TOKENIZER}" ]]; then
    echo "[ERROR] Tokenizer not found: ${TOKENIZER}"
    echo "        The tokenizer must be trained first."
    echo "        Run: ${BUILD_DIR}/bin/train_tokenizer --help"
    exit 1
fi

# ---- verify json directory
if [[ ! -e "${JSON_DIR}" ]]; then
    echo "[ERROR] JSON path not found: ${JSON_DIR}"
    exit 1
fi

JSON_COUNT=$(find "${JSON_DIR}" -iname "*.json" -o -iname "*.jsonl" 2>/dev/null | wc -l)
CSV_COUNT=$(find "${JSON_DIR}" -name "*.csv" 2>/dev/null | wc -l)
if [[ "${JSON_COUNT}" -eq 0 && "${CSV_COUNT}" -gt 0 ]]; then
    echo "[data] No JSON, but ${CSV_COUNT} CSV(s) found -> CSV path (csvs + build)."
    echo "[data] Step 1/2: CSV tables -> corpus text..."
    "${BINARY}" csvs --dir "${JSON_DIR}" --out "${OUT_DIR}/../corpus/corpus_csv_full.txt" || exit 1
    echo "[data] Step 2/2: corpus text -> .gbin shards..."
    "${BINARY}" build --tokenizer "${TOKENIZER}" \
        --text "${OUT_DIR}/../corpus/corpus_csv_full.txt" \
        --out "${OUT_DIR}" --val-ratio "${VAL_RATIO}" \
        --shard-tokens "${SHARD_TOKENS}" --seq-len 1024 || exit 1
    echo "[data] CSV conversion done."
    exit 0
fi
if [[ "${JSON_COUNT}" -eq 0 ]]; then
    echo "[ERROR] No .json/.jsonl nor .csv files found in: ${JSON_DIR}"
    exit 1
fi
echo "[data] Found ${JSON_COUNT} JSON file(s):"
find "${JSON_DIR}" -iname "*.json" -o -iname "*.jsonl" | sort | while read -r f; do
    SIZE=$(du -h "$f" 2>/dev/null | cut -f1)
    echo "       ${SIZE}  $(basename "$f")"
done
echo ""

# ---- inspect first file to show schema mix
echo "[data] Inspecting first JSON file..."
FIRST_JSON=$(find "${JSON_DIR}" -iname "*.json" -o -iname "*.jsonl" | sort | head -n 1)
"${BINARY}" json-inspect --file "${FIRST_JSON}" || true
echo ""

# ---- create output directory
mkdir -p "${OUT_DIR}"

# ---- check if shards already exist
EXISTING=$(find "${OUT_DIR}" -name "*.gbin" 2>/dev/null | wc -l)
if [[ "${EXISTING}" -gt 0 ]]; then
    echo "[data] Warning: ${EXISTING} existing .gbin shard(s) found in ${OUT_DIR}"
    echo "[data] They will be overwritten / supplemented."
    echo ""
fi

# ---- run conversion
echo "[data] Starting JSON → .gbin conversion..."
echo "[data] Chat/instruction docs get SFT masks (assistant-only loss)."
echo ""

START_TIME=$(date +%s)

"${BINARY}" json \
    --dir "${JSON_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --out "${OUT_DIR}" \
    --val-ratio "${VAL_RATIO}" \
    --shard-tokens "${SHARD_TOKENS}" \
    --seq-len "${SEQ_LEN}"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
ELAPSED_SEC=$(( ELAPSED % 60 ))

echo ""
echo "[data] Conversion completed in ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""

# ---- verify output
TRAIN_SHARDS=$(find "${OUT_DIR}" -name "train_*.gbin" 2>/dev/null | wc -l)
VAL_SHARDS=$(find "${OUT_DIR}" -name "val_*.gbin" 2>/dev/null | wc -l)
echo "[data] Output shards:"
echo "       train: ${TRAIN_SHARDS} shard(s)"
echo "       val  : ${VAL_SHARDS} shard(s)"
if [[ "${TRAIN_SHARDS}" -eq 0 ]]; then
    echo "[ERROR] No training shards produced. Check the log above."
    exit 1
fi

# ---- inspect shards with the built tool
echo ""
echo "[data] Inspecting produced shards..."
"${BUILD_DIR}/bin/data_pipeline" inspect \
    --shards "${OUT_DIR}" \
    --tokenizer "${TOKENIZER}" || true

echo ""
echo "============================================================"
echo "  Conversion DONE"
echo "  Shards are in: ${OUT_DIR}"
echo "  Next step: bash kaggle/train.sh"
echo "============================================================"
