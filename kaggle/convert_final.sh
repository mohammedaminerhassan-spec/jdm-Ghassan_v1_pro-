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
# PRO-HARDEN: لا fallback صامت 16k + CHAT قابل للتجاوز + /kaggle/input fallback.
if [[ -n "${TOKENIZER_OVERRIDE:-}" ]]; then TOK="${TOKENIZER_OVERRIDE}"; fi
if [[ ! -f "${TOK}" ]]; then
    echo "[ERROR] 32k tokenizer missing: ${TOK} (16k fallback DISABLED)."
    exit 1
fi
CHAT="${CHAT_FILE:-${REPO_DIR}/Ai dariga datasets/final_dataset/darija_chat_format.jsonl}"
if [[ ! -f "${CHAT}" && -d "/kaggle/input" ]]; then
    found="$(find /kaggle/input -name 'darija_chat_format.jsonl' 2>/dev/null | head -n 1 || true)"
    if [[ -n "${found}" ]]; then CHAT="${found}"; fi
fi

[[ -f "${BIN}" ]] || { echo "[ERROR] data_pipeline missing. Run setup.sh first."; exit 1; }
[[ -f "${TOK}" ]] || { echo "[ERROR] Tokenizer missing: ${TOK}"; exit 1; }

# ---- SFT shards (chat format, assistant-masked) ----
if [[ $(find "${REPO_DIR}/artifacts/shards_sft" -name "train_*.gbin" 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "[sft] Prebuilt SFT shards found — keeping them."
else
    [[ -f "${CHAT}" ]] || { echo "[ERROR] Chat file missing: ${CHAT}"; exit 1; }
    echo "[sft] Building SFT shards from ${CHAT} ..."
    mkdir -p "${REPO_DIR}/artifacts/shards_sft"
    "${BIN}" build --tokenizer "${TOK}" --expect-vocab 32000 --chat "${CHAT}" \
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
    # PRO-HARDEN: تمرير المسارات عبر argv (آمن مع مسافات "Ai dariga datasets")
    # بدل string interpolation الذي كان ينكسر على المسارات ذات المسافات.
    CHAT_IN="${CHAT}" THIN_OUT="${THIN}" python3 -c "
import json,os
chat=os.environ['CHAT_IN']; thin=os.environ['THIN_OUT']
n=0
out=open(thin,'w',encoding='utf-8')
for line in open(chat,encoding='utf-8'):
    x=json.loads(line)
    for m in x.get('messages',[]):
        t=m.get('content','').replace(chr(10),' ').strip()
        if len(t)>5:
            out.write(t+chr(10)); n+=1
print('thin lines:',n)"
    mkdir -p "${REPO_DIR}/artifacts/shards_pt"
    "${BIN}" build --tokenizer "${TOK}" --expect-vocab 32000 --text "${THIN}" \
        --out "${REPO_DIR}/artifacts/shards_pt" \
        --val-ratio 0.01 --shard-tokens 50000000 --seq-len 1024
fi

echo ""
echo "[verify] Shard inventory:"
"${BIN}" inspect --shards "${REPO_DIR}/artifacts/shards_pt" --tokenizer "${TOK}" || true
"${BIN}" inspect --shards "${REPO_DIR}/artifacts/shards_sft" --tokenizer "${TOK}" || true
echo "[convert_final] DONE"
