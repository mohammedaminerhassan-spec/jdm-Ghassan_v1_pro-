#!/usr/bin/env bash
# T4-FAST 10h training: English 120M dense, single GPU.
# Usage: bash configs/kaggle/train_t4_fast.sh [--time-budget-min 540] [--no-export]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"
CONFIG="${REPO_DIR}/configs/t4_en_120m_fast.yaml"
TIME_BUDGET_MIN=540
EXPORT_GGUF=1
EXPORT_PROFILE="q8_0"
GGUF_OUT="${REPO_DIR}/artifacts/ghassan-en-120m_${EXPORT_PROFILE}.gguf"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --time-budget-min) TIME_BUDGET_MIN="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --no-export) EXPORT_GGUF=0; shift ;;
    --export-profile) EXPORT_PROFILE="$2"; shift 2 ;;
    *) shift ;;
  esac
done
BINARY="${BUILD_DIR}/bin/gai_train"
GEN_BIN="${BUILD_DIR}/bin/ghassan-ai"
TOK="${REPO_DIR}/artifacts/tokenizer/english32k.gtok"
if [[ ! -f "${TOK}" ]]; then echo "[ERROR] missing ${TOK}; run setup.sh --with-parquet first"; exit 1; fi
if ! command -v nvidia-smi &>/dev/null; then echo "[ERROR] no GPU"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[cpu] cores: $(nproc) threads for dataloader+OpenMP"
TRAIN_SHARDS=$(find "${REPO_DIR}/artifacts/shards_en" -name "train_*.gbin" 2>/dev/null | wc -l)
if [[ "${TRAIN_SHARDS}" -eq 0 ]]; then echo "[ERROR] no shards; run build_english_data.sh"; exit 1; fi
echo "[data] shards: ${TRAIN_SHARDS}"
# 30-step pilot on the REAL fast config (not a different smoke config)
echo "[pilot] 30 steps to measure steady tok/s..."
P_START=$(date +%s)
"${BINARY}" --config "${CONFIG}" --device cuda --tokenizer "${TOK}" \
  --max-steps 30 --warmup 10 --resume none \
  --eval-every 1000 --save-every 1000 --log-every 5 \
  --checkpoint-dir /tmp/ghassan_t4_pilot --output-budget-mb 14000 2>&1 | tail -5 || true
P_END=$(date +%s); P_ELAPSED=$((P_END-P_START)); [[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1
echo "[pilot] 30 steps in ${P_ELAPSED}s"
BUDGET_SEC=$((TIME_BUDGET_MIN*60)); USED_SEC=$((P_END-P_START)); REMAIN_SEC=$((BUDGET_SEC-USED_SEC-900))
yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
B=$(yget batch_size "${CONFIG}"); T=$(yget seq_len "${CONFIG}"); A=$(yget grad_accum "${CONFIG}")
TOK_PER_STEP=$((B*T*A))
# conservative steady estimate from pilot (drop first 5 steps = CUDA warmup)
TPS_EST=$((TOK_PER_STEP*25/P_ELAPSED)); [[ "${TPS_EST}" -le 100 ]] && TPS_EST=5000
MAX_STEPS=$((REMAIN_SEC*TPS_EST/TOK_PER_STEP)); [[ "${MAX_STEPS}" -lt 50 ]] && MAX_STEPS=50
WARMUP_STEPS=$((MAX_STEPS/12)); [[ "${WARMUP_STEPS}" -lt 50 ]] && WARMUP_STEPS=50
echo "[plan] ${TOK_PER_STEP} tok/step, ~${TPS_EST} tok/s -> max_steps=${MAX_STEPS} warmup=${WARMUP_STEPS}"
T_START=$(date +%s)
if [[ "${EXPORT_GGUF}" -eq 1 ]]; then
  "${BINARY}" --config "${CONFIG}" --device cuda --tokenizer "${TOK}" \
    --max-steps "${MAX_STEPS}" --warmup "${WARMUP_STEPS}" \
    --export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}" --threads "$(nproc)"
else
  "${BINARY}" --config "${CONFIG}" --device cuda --tokenizer "${TOK}" \
    --max-steps "${MAX_STEPS}" --warmup "${WARMUP_STEPS}" --threads "$(nproc)"
fi
T_END=$(date +%s); echo "[train] took $(((T_END-T_START)/60))m"
if [[ "${EXPORT_GGUF}" -eq 1 && -f "${GGUF_OUT}" ]]; then
  "${GEN_BIN}" generate --model "${GGUF_OUT}" --persona en \
    --prompt "Hello! Tell me about yourself and what you can help with." --max-tokens 80 || echo "[smoke] WARN failed"
fi
echo "Done: ${GGUF_OUT}"
