#!/usr/bin/env bash
# kaggle/train_full.sh — Ghassan v1 Pro English two-stage training (Kaggle GPU, one session).
#
#   Stage A (pretrain, ~55% of budget): English LM on artifacts/shards_en
#   Stage B (SFT,      ~45% of budget): chat tuning on artifacts/shards_en
#   -> GGUF export -> English smoke test. Fits inside ONE Kaggle session.
#
# Usage:
#   bash kaggle/train_full.sh                                   # 6h budget (default)
#   bash kaggle/train_full.sh --time-budget-min 420             # custom budget
#   bash kaggle/train_full.sh --export-profile q4_0             # default fp16
#   bash kaggle/train_full.sh --no-export                       # skip GGUF export
#   bash kaggle/train_full.sh --pilot-only                      # measure speed only
# ----------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="full"
CONFIG_PT="${REPO_DIR}/configs/en_pro.yaml"
CONFIG_SFT="${REPO_DIR}/configs/sft_en_pro.yaml"
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
TOK="${REPO_DIR}/artifacts/tokenizer/english32k.gtok"
PT_DIR="${REPO_DIR}/artifacts/shards_en"
SFT_DIR="${REPO_DIR}/artifacts/shards_en"
CKPT_PT="${REPO_DIR}/artifacts/checkpoints/en_pro"
CKPT_SFT="${REPO_DIR}/artifacts/checkpoints/en_pro_sft"
GGUF_OUT="${REPO_DIR}/artifacts/ghassan-v1-pro_${EXPORT_PROFILE}.gguf"

# FIX P2 (6h run lost on preemption): train_full.sh had no persist trap while
# train_1b.sh does. Copy the same snapshot-on-exit so Kaggle preemption keeps
# checkpoints + GGUF (mirrors train_1b.sh:74-83).
persist_output() {
    if [[ -d "/kaggle/working" ]]; then
        mkdir -p /kaggle/working/output 2>/dev/null || true
        cp -r "${CKPT_PT}" /kaggle/working/output/ 2>/dev/null || true
        cp -r "${CKPT_SFT}" /kaggle/working/output/ 2>/dev/null || true
        cp -f "${GGUF_OUT}" /kaggle/working/output/ 2>/dev/null || true
        echo "[persist] snapshot copied to /kaggle/working/output"
    fi
}
trap persist_output EXIT INT TERM

echo "============================================================"
echo "  Ghassan v1 Pro English — Two-Stage Kaggle Training [${MODE}]"
echo "  A: pretrain  B: sft  -> GGUF export -> English smoke test"
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

# (Old fp32 override REMOVED 2026-09-16: it forced 3-5x-slower pure-fp32 SGEMM
# on pilot + both stages, so the pilot measured the WRONG recipe. The fp16
# path is validated by the setup.sh CUDA parity gate + loss-scaler machinery;
# pilot and stages now run the yaml recipe. Pre-Volta GPUs still fall back
# to fp32 via the CC<7 check above.)

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

mkdir -p "${CKPT_PT}" "${REPO_DIR}/artifacts/checkpoints/en_pro_sft"

# ---------------------------------------------------------------- pilot: measure tok/s
echo ""
echo "[pilot] Measuring real speed (${PILOT_STEPS} steps on pretrain shards)..."
P_START=$(date +%s)
# --resume none: a stale pilot checkpoint would make this do ~zero work
# and corrupt the session budget with an absurd tok/s (see train_1b.sh).
"${BINARY}" --config "${CONFIG_PILOT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PILOT_STEPS}" --resume none "${FP16_OVERRIDE[@]}"
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
echo "[stage-A] Pretraining Pro English (${PT_STEPS} steps)..."
T0=$(date +%s)
"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PT_STEPS}" --warmup "${PT_WARM}" \
    --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
    --checkpoint-dir "${CKPT_PT}" --resume auto "${FP16_OVERRIDE[@]}"
echo "[stage-A] took $(( ($(date +%s) - T0) / 60 ))m"

# ---------------------------------------------------------------- Stage B: SFT
echo ""
echo "[stage-B] Instruction tuning on English chat (${SFT_STEPS} steps)..."
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

# ---------------------------------------------------------------- smoke test (English!)
# SELF-CONTAINED + FATAL (v2 audit P0-35): no --tokenizer sidecar (the GGUF
# must carry its own tokenizer) and no swallowed failure — a broken export
# must fail the run, never print "Done!" over it.
if [[ "${EXPORT_GGUF}" -eq 1 ]] && [[ -f "${GGUF_OUT}" ]]; then
    echo ""
    echo "[smoke] English generation test (no sidecar)..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --persona en \
        --prompt "Hello, who are you? What can you help me with?" --max-tokens 60 || \
        { echo "[smoke] FAIL: self-contained GGUF generation failed"; exit 1; }
    echo ""
    echo "[smoke] Second prompt (factual, no sidecar)..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --persona en \
        --prompt "What is the capital of Morocco?" --max-tokens 60 || \
        { echo "[smoke] FAIL: self-contained GGUF generation failed (factual)"; exit 1; }
    echo "[smoke] OK: GGUF runs standalone, no sidecar needed"
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
