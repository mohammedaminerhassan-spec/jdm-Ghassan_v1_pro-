#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODE="full"
CONFIG_PT="${REPO_DIR}/configs/flash_480m_single.yaml"
CONFIG_SFT="${REPO_DIR}/configs/sft_flash_480m_single.yaml"
TIME_BUDGET_MIN=360
EXPORT_GGUF=1
EXPORT_PROFILE="fp16"
PILOT_STEPS=100
EXPORT_MARGIN_SEC=900
PT_FRACTION=55

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
TOK="${TOK:-${REPO_DIR}/artifacts/tokenizer/english32k.gtok}"
PT_DIR="${PT_DIR:-${REPO_DIR}/artifacts/shards_en}"
SFT_DIR="${SFT_DIR:-${REPO_DIR}/artifacts/shards_en}"
CKPT_PT="${CKPT_PT:-${REPO_DIR}/artifacts/checkpoints/en_pro}"
CKPT_SFT="${CKPT_SFT:-${REPO_DIR}/artifacts/checkpoints/en_pro_sft}"
GGUF_OUT="${GGUF_OUT:-${REPO_DIR}/artifacts/ghassan-v1-pro_${EXPORT_PROFILE}.gguf}"

OUTPUT_BUDGET_MB="${OUTPUT_BUDGET_MB:-17408}"
SESSION_SPENT_MIN="${SESSION_SPENT_MIN:-0}"
if [[ "${SESSION_SPENT_MIN}" -gt 0 ]]; then
    TIME_BUDGET_MIN=$(( TIME_BUDGET_MIN - SESSION_SPENT_MIN ))
    [[ "${TIME_BUDGET_MIN}" -lt 60 ]] && {
        echo "[ERROR] only ${TIME_BUDGET_MIN}m left after ${SESSION_SPENT_MIN}m spent; refusing a doomed run."
        exit 1
    }
    echo "[time] session already spent ~${SESSION_SPENT_MIN}m; training budget -> ${TIME_BUDGET_MIN}m"
fi

persist_output() {
    [[ -d "/kaggle/working" ]] || return 0
    local dest="/kaggle/working/output"
    mkdir -p "${dest}" 2>/dev/null || true
    local src
    for src in "${CKPT_PT}" "${CKPT_SFT}" "${GGUF_OUT}"; do
        case "${src}" in
            /kaggle/working/*)
                echo "[persist] already persisted by the platform, skip copy: ${src}" ;;
            *)
                echo "[persist] copying ${src} -> ${dest}/"
                cp -r "${src}" "${dest}/" 2>/dev/null || true ;;
        esac
    done
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

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

mkdir -p "${CKPT_PT}" "${CKPT_SFT}"

run_preflight() {
    echo ""
    echo "==================== preflight ===================="
    [[ -f "${CONFIG_PT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_PT}"; return 1; }
    [[ -f "${CONFIG_SFT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_SFT}"; return 1; }
    SFT_PT_CKPT="$(yget pretrained_checkpoint "${CONFIG_SFT}" || true)"
    if [[ -n "${SFT_PT_CKPT}" ]]; then
        case "${SFT_PT_CKPT}" in
            /*) SFT_PT_CKPT_ABS="${SFT_PT_CKPT}" ;;
            *)  SFT_PT_CKPT_ABS="${REPO_DIR}/${SFT_PT_CKPT}" ;;
        esac
        if [[ "${SFT_PT_CKPT_ABS}" != "${CKPT_PT}/last.ckpt" ]]; then
            echo "[preflight FAIL] SFT->PT chain mismatch: yaml wants ${SFT_PT_CKPT_ABS}, stage A writes ${CKPT_PT}/last.ckpt"
            echo "              fix: export CKPT_PT=$(dirname "${SFT_PT_CKPT_ABS}") or edit training.pretrained_checkpoint"
            return 1
        fi
        echo "[preflight] SFT->PT checkpoint chain OK"
    fi
    [[ -f "${TOK}" ]] || { echo "[preflight FAIL] tokenizer missing: ${TOK}"; return 1; }
    echo "[preflight] tokenizer present"
    for d in "${PT_DIR}" "${SFT_DIR}"; do
        n=$(find "$d" -name "train_*.gbin" 2>/dev/null | wc -l)
        [[ "$n" -gt 0 ]] || { echo "[preflight FAIL] no train shards in $d"; return 1; }
    done
    echo "[preflight] shard dirs non-empty"
    "${BINARY}" --config "${CONFIG_PT}" --dry-run --device cpu --strict-config --output-budget-mb "${OUTPUT_BUDGET_MB}" || return 1
    "${BINARY}" --config "${CONFIG_SFT}" --dry-run --device cpu --strict-config --output-budget-mb "${OUTPUT_BUDGET_MB}" || return 1
    echo "==================== preflight: ALL GATES PASSED ===================="
    return 0
}
run_preflight || { echo "[ERROR] preflight failed — fix the gates above."; exit 1; }

echo ""
echo "[pilot] Measuring real speed (${PILOT_STEPS} steps on the REAL PT recipe)..."
P_START=$(date +%s)

PILOT_CKPT="${PILOT_CKPT:-/tmp/gai_pilot_full}"
rm -rf "${PILOT_CKPT}"; mkdir -p "${PILOT_CKPT}"
trap 'rm -rf "${PILOT_CKPT}"; persist_output' EXIT INT TERM

"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PILOT_STEPS}" --warmup 0 --resume none \
    --checkpoint-dir "${PILOT_CKPT}" --output-budget-mb "${OUTPUT_BUDGET_MB}" "${FP16_OVERRIDE[@]}"
rm -rf "${PILOT_CKPT}"
P_END=$(date +%s)
P_ELAPSED=$(( P_END - P_START ))
[[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1
P_TPS=$(( $(yget batch_size "${CONFIG_PILOT}") * $(yget seq_len "${CONFIG_PILOT}") \
           * $(yget grad_accum "${CONFIG_PILOT}") * PILOT_STEPS / P_ELAPSED ))
echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${P_TPS} tok/s"
if [[ "${MODE}" == "pilot" ]]; then echo "[pilot] Done."; exit 0; fi

FULL_TPS=$(( $(yget batch_size "${CONFIG_PT}") * $(yget seq_len "${CONFIG_PT}") \
              * $(yget grad_accum "${CONFIG_PT}") ))
echo "[plan] Full config: ${FULL_TPS} tokens/step"
BUDGET_SEC=$(( TIME_BUDGET_MIN * 60 ))
USED_SEC=$(($(date +%s) - P_START))
REMAIN_SEC=$(( BUDGET_SEC - USED_SEC - EXPORT_MARGIN_SEC ))
[[ "${REMAIN_SEC}" -lt 600 ]] && { echo "[ERROR] <10min left of budget. Aborting."; exit 1; }

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

echo ""
echo "[stage-A] Pretraining Pro English (${PT_STEPS} steps)..."
T0=$(date +%s)
"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PT_STEPS}" --warmup "${PT_WARM}" \
    --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
    --checkpoint-dir "${CKPT_PT}" --resume auto "${FP16_OVERRIDE[@]}" \
    --output-budget-mb "${OUTPUT_BUDGET_MB}"
echo "[stage-A] took $(( ($(date +%s) - T0) / 60 ))m"

echo ""
echo "[stage-B] Instruction tuning on English chat (${SFT_STEPS} steps)..."
T0=$(date +%s)

if [[ "${EXPORT_GGUF}" -eq 1 ]]; then
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --checkpoint-dir "${CKPT_SFT}" \
        --resume auto "${FP16_OVERRIDE[@]}" \
        --output-budget-mb "${OUTPUT_BUDGET_MB}" \
        --export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}"
else
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --checkpoint-dir "${CKPT_SFT}" \
        --resume auto "${FP16_OVERRIDE[@]}" \
        --output-budget-mb "${OUTPUT_BUDGET_MB}"
fi
echo "[stage-B] took $(( ($(date +%s) - T0) / 60 ))m"

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
