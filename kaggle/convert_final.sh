#!/usr/bin/env bash
# kaggle/convert_final.sh — (Re)build training shards from the shipped data.
# Primary path uses the PREBUILT shards in the data zip (verified locally).
# This script only rebuilds what is missing.
#
# Layout expected (after unzipping the data zip into the repo root):
#   Ai dariga datasets/final_dataset/darija_chat_format.jsonl
#   artifacts/tokenizer/darija.gtok
#   artifacts/shards_pt/train_*.gbin   (optional: rebuilt thin if missing)
#   artifacts/shards_sft/train_*.gbin  (optional: rebuilt if missing)
# ----------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_DIR}/build/bin/data_pipeline"
TOK="${REPO_DIR}/artifacts/tokenizer/darija32k.gtok"
if [[ ! -f "${TOK}" ]]; then TOK="${REPO_DIR}/artifacts/tokenizer/darija.gtok"; fi
CHAT="${REPO_DIR}/Ai dariga datasets/final_dataset/darija_chat_format.jsonl"

[[ -f "${BIN}" ]] || { echo "[ERROR] data_pipeline missing. Run setup.sh first."; exit 1; }
[[ -f "${TOK}" ]] || { echo "[ERROR] Tokenizer missing: ${TOK}"; exit 1; }

# ---- SFT shards (chat format, assistant-masked) ----
if [[ $(find "${REPO_DIR}/artifacts/shards_sft" -name "train_*.gbin" 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "[sft] Prebuilt SFT shards found — keeping them."
else
    [[ -f "${CHAT}" ]] || { echo "[ERROR] Chat file missing: ${CHAT}"; exit 1; }
    echo "[sft] Building SFT shards from ${CHAT} ..."
    mkdir -p "${REPO_DIR}/artifacts/shards_sft"
    "${BIN}" build --tokenizer "${TOK}" --chat "${CHAT}" \
        --out "${REPO_DIR}/artifacts/shards_sft" \
        --val-ratio 0.05 --shard-tokens 50000000 --seq-len 1024
fi

# ---- pretrain shards (raw LM) ----
if [[ $(find "${REPO_DIR}/artifacts/shards_pt" -name "train_*.gbin" 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "[pt] Prebuilt pretrain shards found — keeping them."
else
    echo "[pt] WARNING: prebuilt pretrain shards missing."
    echo "[pt] Building a THIN fallback from Q/A text (slower to converge)..."
    [[ -f "${CHAT}" ]] || { echo "[ERROR] Chat file missing: ${CHAT}"; exit 1; }
    THIN="${REPO_DIR}/artifacts/corpus_thin.txt"
    python3 -c "
import json,sys
n=0
out=open('${THIN}','w',encoding='utf-8')
for line in open('${CHAT}',encoding='utf-8'):
    x=json.loads(line)
    for m in x.get('messages',[]):
        t=m.get('content','').replace(chr(10),' ').strip()
        if len(t)>5:
            out.write(t+chr(10)); n+=1
print('thin lines:',n)"
    mkdir -p "${REPO_DIR}/artifacts/shards_pt"
    "${BIN}" build --tokenizer "${TOK}" --text "${THIN}" \
        --out "${REPO_DIR}/artifacts/shards_pt" \
        --val-ratio 0.01 --shard-tokens 50000000 --seq-len 1024
fi

echo ""
echo "[verify] Shard inventory:"
"${BIN}" inspect --shards "${REPO_DIR}/artifacts/shards_pt" --tokenizer "${TOK}" || true
"${BIN}" inspect --shards "${REPO_DIR}/artifacts/shards_sft" --tokenizer "${TOK}" || true
echo "[convert_final] DONE"
