#!/usr/bin/env bash
# kaggle/train_full.sh — Ghassan v1 Flash two-stage training (Kaggle GPU, one session).
#
#   Stage A (pretrain, ~55% of budget): raw Darija LM on artifacts/shards_pt
#   Stage B (SFT,      ~45% of budget): chat tuning on artifacts/shards_sft
#   -> GGUF export -> Darija smoke test. Fits inside ONE Kaggle session.
#
# Usage:
#   bash kaggle/train_full.sh                                   # 6h budget (default)
#   bash kaggle/train_full.sh --time-budget-min 420             # custom budget
#   bash kaggle/train_full.sh --export-profile q4_k             # default fp16
#   bash kaggle/train_full.sh --no-export                       # skip GGUF export
#   bash kaggle/train_full.sh --pilot-only                      # measure speed only
# ----------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="full"
CONFIG_PT="${REPO_DIR}/configs/flash_moe.yaml"
CONFIG_SFT="${REPO_DIR}/configs/sft.yaml"
CONFIG_PILOT="${REPO_DIR}/kaggle/configs/pilot_moe.yaml"
TIME_BUDGET_MIN=360
EXPORT_GGUF=1
EXPORT_PROFILE="fp16"
PILOT_STEPS=100
EXPORT_MARGIN_SEC=900
PT_FRACTION=55   # % of training time for pretrain (rest goes to SFT)

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
PT_DIR="${REPO_DIR}/artifacts/shards_pt"
SFT_DIR="${REPO_DIR}/artifacts/shards_sft"
CKPT_PT="${REPO_DIR}/artifacts/checkpoints/flash"
GGUF_OUT="${REPO_DIR}/artifacts/ghassan-v1-flash_${EXPORT_PROFILE}.gguf"

echo "============================================================"
echo "  Ghassan v1 Flash — Two-Stage Kaggle Training [${MODE}]"
echo "  A: pretrain  B: sft  -> GGUF export -> Darija smoke test"
echo "============================================================"

[[ -f "${BINARY}" ]] || { echo "[ERROR] gai_train missing. Run setup.sh first."; exit 1; }
command -v nvidia-smi &>/dev/null || {
    echo "[ERROR] No NVIDIA GPU. Kaggle: Settings -> Accelerator -> GPU."; exit 1; }
echo ""
echo "[GPU]"
nvidia-smi --query-gpu=name,memory.free,memory.total \
    --format=csv,noheader 2>/dev/null | \
    awk -F, '{printf "  %s | free: %s | total: %s\n", $1,$2,$3}'

[[ -f "${TOK}" ]] || { echo "[ERROR] Tokenizer missing: ${TOK}"; exit 1; }
PT_SHARDS=$(find "${PT_DIR}" -name "train_*.gbin" 2>/dev/null | wc -l)
SFT_SHARDS=$(find "${SFT_DIR}" -name "train_*.gbin" 2>/dev/null | wc -l)
[[ "${PT_SHARDS}" -gt 0 ]] || { echo "[ERROR] No pretrain shards in ${PT_DIR}"; exit 1; }
[[ "${SFT_SHARDS}" -gt 0 ]] || { echo "[ERROR] No SFT shards in ${SFT_DIR}"; exit 1; }
echo "[data] pretrain shards: ${PT_SHARDS} | sft shards: ${SFT_SHARDS}"

CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | cut -d. -f1 | tr -d ' ')
FP16_OVERRIDE=()
if [[ -n "${CC_MAJOR}" ]] && [[ "${CC_MAJOR}" -lt 7 ]]; then
    echo "[prec] CC ${CC_MAJOR}.x has no FP16 tensor cores -> forcing fp32 GEMMs"
    FP16_OVERRIDE=(--gemm-fp16 0)
else
    echo "[prec] CC ${CC_MAJOR:-?}.x -> fp16 tensor GEMMs"
fi

# TEMPORARY STABILITY OVERRIDE (NaN incident): the fp16 GEMM path was never
# validated (parity test disables it) and NaNs the loss immediately.
# Force fp32 GEMMs for pilot + both stages until kernels are audited.
# To re-enable fp16 later, delete the two lines below.
FP16_OVERRIDE+=(--gemm-fp16 0)
echo "[prec] fp16 GEMMs DISABLED by stability override (fp32 everywhere)"

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

mkdir -p "${CKPT_PT}" "${REPO_DIR}/artifacts/checkpoints/flash_sft"

# ---------------------------------------------------------------- pilot: measure tok/s
echo ""
echo "[pilot] Measuring real speed (${PILOT_STEPS} steps on pretrain shards)..."
P_START=$(date +%s)
"${BINARY}" --config "${CONFIG_PILOT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PILOT_STEPS}" "${FP16_OVERRIDE[@]}"
P_END=$(date +%s)
P_ELAPSED=$(( P_END - P_START ))
[[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1
P_TPS=$(( $(yget batch_size "${CONFIG_PILOT}") * $(yget seq_len "${CONFIG_PILOT}") \
           * $(yget grad_accum "${CONFIG_PILOT}") * PILOT_STEPS / P_ELAPSED ))
echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${P_TPS} tok/s"
if [[ "${MODE}" == "pilot" ]]; then echo "[pilot] Done."; exit 0; fi

# ---------------------------------------------------------------- budget -> steps
FULL_TPS=$(( $(yget batch_size "${CONFIG_PT}") * $(yget seq_len "${CONFIG_PT}") \
              * $(yget grad_accum "${CONFIG_PT}") ))
echo "[plan] Full config: ${FULL_TPS} tokens/step"
BUDGET_SEC=$(( TIME_BUDGET_MIN * 60 ))
USED_SEC=$(($(date +%s) - P_START))
REMAIN_SEC=$(( BUDGET_SEC - USED_SEC - EXPORT_MARGIN_SEC ))
[[ "${REMAIN_SEC}" -lt 600 ]] && { echo "[ERROR] <10min left of budget. Aborting."; exit 1; }
# seconds per full step ≈ FULL_TPS / P_TPS (pilot tokens are representative)
TOTAL_STEPS=$(awk "BEGIN {printf \"%d\", (${REMAIN_SEC} * ${P_TPS}) / ${FULL_TPS}}")
[[ "${TOTAL_STEPS}" -lt 100 ]] && TOTAL_STEPS=100
PT_STEPS=$(( TOTAL_STEPS * PT_FRACTION / 100 ))
SFT_STEPS=$(( TOTAL_STEPS - PT_STEPS ))
[[ "${PT_STEPS}" -lt 50 ]] && PT_STEPS=50
[[ "${SFT_STEPS}" -lt 50 ]] && SFT_STEPS=50
PT_WARM=$(( PT_STEPS / 10 ));  [[ "${PT_WARM}" -gt 500 ]] && PT_WARM=500
SFT_WARM=$(( SFT_STEPS / 10 )); [[ "${SFT_WARM}" -gt 100 ]] && SFT_WARM=100
EVAL_CAD=$(( TOTAL_STEPS / 10 )); [[ "${EVAL_CAD}" -lt 50 ]] && EVAL_CAD=50
echo "[plan] budget=${TIME_BUDGET_MIN}min used=${USED_SEC}s remain~=${REMAIN_SEC}s"
echo "[plan] total_steps=${TOTAL_STEPS} (pretrain=${PT_STEPS}, sft=${SFT_STEPS})"
echo "[plan] ~$(( TOTAL_STEPS * FULL_TPS / 1000000 ))M tokens this session"

# ---------------------------------------------------------------- Stage A: pretrain
echo ""
echo "[stage-A] Pretraining on raw Darija (${PT_STEPS} steps)..."
T0=$(date +%s)
"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PT_STEPS}" --warmup "${PT_WARM}" \
    --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
    --checkpoint-dir "${CKPT_PT}" --resume auto "${FP16_OVERRIDE[@]}"
echo "[stage-A] took $(( ($(date +%s) - T0) / 60 ))m"

# ---------------------------------------------------------------- Stage B: SFT
echo ""
echo "[stage-B] Instruction tuning on Darija chat (${SFT_STEPS} steps)..."
T0=$(date +%s)
if [[ "${EXPORT_GGUF}" -eq 1 ]]; then
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --resume auto "${FP16_OVERRIDE[@]}" \
        --export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}"
else
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --resume auto "${FP16_OVERRIDE[@]}"
fi
echo "[stage-B] took $(( ($(date +%s) - T0) / 60 ))m"

# ---------------------------------------------------------------- smoke test (Darija!)
if [[ "${EXPORT_GGUF}" -eq 1 ]] && [[ -f "${GGUF_OUT}" ]]; then
    echo ""
    echo "[smoke] Darija generation test..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --tokenizer "${TOK}" \
        --prompt "labas, kidayer? chno smitk?" --max-tokens 40 || \
        echo "[smoke] WARNING: smoke test failed (model file may still be usable)"
    echo ""
    echo "[smoke] Second prompt (Arabic script)..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --tokenizer "${TOK}" \
        --prompt "شنو هي العاصمة ديال المغرب؟" --max-tokens 60 || true
fi

echo ""
echo "[GPU] After training:"
nvidia-smi --query-gpu=name,memory.free,memory.total \
    --format=csv,noheader 2>/dev/null | \
    awk -F, '{printf "  %s | free: %s | total: %s\n", $1,$2,$3}'
echo ""
echo "============================================================"
echo "  Done! GGUF: ${GGUF_OUT}"
echo "  Chat: ./build/bin/ghassan-ai chat --model ${GGUF_OUT} --tokenizer ${TOK}"
echo "============================================================"
