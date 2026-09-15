#!/usr/bin/env bash
# kaggle/train_1b.sh — Ghassan v1 Ultra 1B two-stage training (Kaggle GPU, multi-session safe)
#
#   Stage A (pretrain): configs/ultra_1b.yaml on artifacts/shards_1b (~1B tokens)
#   Stage B (sft)     : configs/sft_ultra_1b.yaml on artifacts/shards_1b_sft
#   -> GGUF export -> Darija smoke test. WSD scheduler + resume=auto so you can
#   stop/resume across Kaggle sessions until loss converges.
#
# Usage:
#   bash kaggle/train_1b.sh                                   # 9h budget default
#   bash kaggle/train_1b.sh --time-budget-min 500
#   bash kaggle/train_1b.sh --pilot-only
#   bash kaggle/train_1b.sh --no-export
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="full"
CONFIG_PT="${REPO_DIR}/configs/ultra_1b.yaml"
CONFIG_SFT="${REPO_DIR}/configs/sft_ultra_1b.yaml"
TIME_BUDGET_MIN=540
EXPORT_GGUF=1
EXPORT_PROFILE="q4_k"
PILOT_STEPS=20
EXPORT_MARGIN_SEC=900
PT_FRACTION=60

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot-only)      MODE="pilot";   shift ;;
        --time-budget-min) TIME_BUDGET_MIN="$2"; shift 2 ;;
        --no-export)       EXPORT_GGUF=0;  shift ;;
        --export-profile)  EXPORT_PROFILE="$2"; shift 2 ;;
        --pt-fraction)     PT_FRACTION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

BINARY="${REPO_DIR}/build/bin/gai_train"
GEN_BIN="${REPO_DIR}/build/bin/ghassan-ai"
TOK="${REPO_DIR}/artifacts/tokenizer/darija32k.gtok"
if [[ ! -f "${TOK}" ]]; then TOK="${REPO_DIR}/artifacts/tokenizer/darija.gtok"; fi
PT_DIR="${REPO_DIR}/artifacts/shards_1b"
SFT_DIR="${REPO_DIR}/artifacts/shards_1b_sft"
CKPT_PT="${REPO_DIR}/artifacts/checkpoints/ultra_1b"
GGUF_OUT="${REPO_DIR}/artifacts/ghassan-v1-ultra-1b_${EXPORT_PROFILE}.gguf"

echo "============================================================"
echo "  Ghassan v1 Ultra 1B — Two-Stage Training [${MODE}]"
echo "  1.00B total / 324M active MoE | Lion + WSD + resume=auto"
echo "============================================================"

[[ -f "${BINARY}" ]] || { echo "[ERROR] gai_train missing. Run setup.sh first."; exit 1; }
command -v nvidia-smi &>/dev/null || { echo "[ERROR] No GPU. Kaggle: Settings -> Accelerator -> GPU."; exit 1; }
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2>/dev/null | \
    awk -F, '{printf "  %s | free: %s | total: %s\n", $1,$2,$3}'

[[ -f "${TOK}" ]] || { echo "[ERROR] Tokenizer missing: ${TOK}"; exit 1; }
PT_SHARDS=$(find "${PT_DIR}" -name "train_*.gbin" 2>/dev/null | wc -l)
SFT_SHARDS=$(find "${SFT_DIR}" -name "train_*.gbin" 2>/dev/null | wc -l)
[[ "${PT_SHARDS}" -gt 0 ]] || { echo "[ERROR] No pretrain shards in ${PT_DIR}. Run build_billion_data.sh first."; exit 1; }
[[ "${SFT_SHARDS}" -gt 0 ]] || { echo "[ERROR] No SFT shards in ${SFT_DIR}. Run build_billion_data.sh first."; exit 1; }
echo "[data] pretrain shards: ${PT_SHARDS} | sft shards: ${SFT_SHARDS}"

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
mkdir -p "${CKPT_PT}" "${REPO_DIR}/artifacts/checkpoints/ultra_1b_sft"

echo ""
echo "[pilot] Measuring 1B speed (${PILOT_STEPS} steps)..."
P_START=$(date +%s)
# 10/10: do NOT force --gemm-fp16 0. Pilot must measure the REAL recipe (fp16 ON
# for T4). Old script forced fp32, measured 3-5x slower speed, then planned
# steps from that wrong number AND trained the full run in slow fp32.
"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PILOT_STEPS}"
P_END=$(date +%s)
P_ELAPSED=$(( P_END - P_START )); [[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1
P_TPS=$(( $(yget batch_size "${CONFIG_PT}") * $(yget seq_len "${CONFIG_PT}") * $(yget grad_accum "${CONFIG_PT}") * PILOT_STEPS / P_ELAPSED ))
echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${P_TPS} tok/s"
if [[ "${MODE}" == "pilot" ]]; then echo "[pilot] Done."; exit 0; fi

FULL_TPS=$(( $(yget batch_size "${CONFIG_PT}") * $(yget seq_len "${CONFIG_PT}") * $(yget grad_accum "${CONFIG_PT}") ))
echo "[plan] Full config: ${FULL_TPS} tokens/step (1B model, B=1 Lion)"
BUDGET_SEC=$(( TIME_BUDGET_MIN * 60 ))
USED_SEC=$(($(date +%s) - P_START))
REMAIN_SEC=$(( BUDGET_SEC - USED_SEC - EXPORT_MARGIN_SEC ))
[[ "${REMAIN_SEC}" -lt 600 ]] && { echo "[ERROR] <10min left. Aborting."; exit 1; }
TOTAL_STEPS=$(awk "BEGIN {printf \"%d\", (${REMAIN_SEC} * ${P_TPS}) / ${FULL_TPS}}")
[[ "${TOTAL_STEPS}" -lt 100 ]] && TOTAL_STEPS=100
PT_STEPS=$(( TOTAL_STEPS * PT_FRACTION / 100 ))
SFT_STEPS=$(( TOTAL_STEPS - PT_STEPS ))
[[ "${PT_STEPS}" -lt 50 ]] && PT_STEPS=50
[[ "${SFT_STEPS}" -lt 50 ]] && SFT_STEPS=50
PT_WARM=$(( PT_STEPS / 10 ));  [[ "${PT_WARM}" -gt 800 ]] && PT_WARM=800
SFT_WARM=$(( SFT_STEPS / 10 )); [[ "${SFT_WARM}" -gt 200 ]] && SFT_WARM=200
EVAL_CAD=$(( TOTAL_STEPS / 10 )); [[ "${EVAL_CAD}" -lt 50 ]] && EVAL_CAD=50
echo "[plan] budget=${TIME_BUDGET_MIN}min remain~=${REMAIN_SEC}s"
echo "[plan] total_steps=${TOTAL_STEPS} (pretrain=${PT_STEPS}, sft=${SFT_STEPS})"
echo "[plan] ~$(( TOTAL_STEPS * FULL_TPS / 1000000 ))M tokens this session"

echo ""
echo "[stage-A] Pretraining Ultra 1B (${PT_STEPS} steps)..."
T0=$(date +%s)
"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PT_STEPS}" --warmup "${PT_WARM}" \
    --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
    --checkpoint-dir "${CKPT_PT}" --resume auto
echo "[stage-A] took $(( ($(date +%s) - T0) / 60 ))m"

echo ""
echo "[stage-B] SFT Ultra 1B (${SFT_STEPS} steps)..."
T0=$(date +%s)
if [[ "${EXPORT_GGUF}" -eq 1 ]]; then
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --resume auto \
        --export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}"
else
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --resume auto
fi
echo "[stage-B] took $(( ($(date +%s) - T0) / 60 ))m"

if [[ "${EXPORT_GGUF}" -eq 1 ]] && [[ -f "${GGUF_OUT}" ]]; then
    echo ""
    echo "[smoke] Darija generation test..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --tokenizer "${TOK}" \
        --prompt "labas, kidayer? chno smitk?" --max-tokens 40 || echo "[smoke] WARNING: failed (file may still be usable)"
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --tokenizer "${TOK}" \
        --prompt "شنو هي العاصمة ديال المغرب؟" --max-tokens 60 || true
fi

echo ""
echo "============================================================"
echo "  Done! GGUF: ${GGUF_OUT}"
echo "  Resume anytime: same command continues from last.ckpt (WSD safe)"
echo "============================================================"
