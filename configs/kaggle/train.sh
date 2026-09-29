#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"

MODE="oneshot"
CONFIG_FULL="${REPO_DIR}/configs/flash_480m_single.yaml"
CONFIG_PILOT="${REPO_DIR}/configs/pilot_moe_smoke.yaml"
TIME_BUDGET_MIN=360
EXPORT_GGUF=1
EXPORT_PROFILE="fp16"
PILOT_STEPS=100
EXPORT_MARGIN_SEC=900

GGUF_OUT="${REPO_DIR}/artifacts/ghassan-v1-pro_${EXPORT_PROFILE}.gguf"

while [[ $
    case "$1" in
        --pro)               CONFIG_FULL="${REPO_DIR}/configs/pro_1b_single.yaml"; GGUF_OUT="${REPO_DIR}/artifacts/ghassan-v1-pro-1b_${EXPORT_PROFILE}.gguf"; shift ;;
        --pilot-only)        MODE="pilot";   shift ;;
        --full)              MODE="full";    shift ;;
        --dry-run)           MODE="dryrun";  shift ;;
        --config)            CONFIG_FULL="$2"; shift 2 ;;
        --time-budget-min)   TIME_BUDGET_MIN="$2"; shift 2 ;;
        --no-export)         EXPORT_GGUF=0;  shift ;;
        --export-profile)    EXPORT_PROFILE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ "${CONFIG_FULL}" == *pro_1b_single.yaml ]]; then
    GGUF_OUT="${REPO_DIR}/artifacts/ghassan-v1-pro-1b_${EXPORT_PROFILE}.gguf"
else
    GGUF_OUT="${REPO_DIR}/artifacts/ghassan-v1-pro_${EXPORT_PROFILE}.gguf"
fi

BINARY="${BUILD_DIR}/bin/gai_train"
GEN_BIN="${BUILD_DIR}/bin/ghassan-ai"

TOK="${REPO_DIR}/artifacts/tokenizer/english32k.gtok"
if [[ ! -f "${TOK}" ]]; then
    echo "[ERROR] 32k tokenizer missing: ${TOK} (legacy 16k fallback DISABLED)."
    echo "[ERROR] Run: bash configs/kaggle/setup.sh --with-parquet  (trains english32k.gtok automatically)"
    exit 1
fi

persist_output() {
    if [[ -d "/kaggle/working" ]]; then
        mkdir -p /kaggle/working/output 2>/dev/null || true
        cp -r "${REPO_DIR}/artifacts/checkpoints" /kaggle/working/output/ 2>/dev/null || true
        cp -f "${GGUF_OUT}" /kaggle/working/output/ 2>/dev/null || true
        echo "[persist] snapshot copied to /kaggle/working/output"
    fi
}
trap persist_output EXIT INT TERM

echo "============================================================"
echo "  Ghassan v1 Pro English — Kaggle Training [${MODE}] (${CONFIG_FULL})"
echo "  Pro 480M recipe. For the 1B flagship use train_1b.sh (or --pro here)"
echo "============================================================"

if [[ ! -f "${BINARY}" ]]; then
    echo "[ERROR] gai_train not found. Run setup.sh --require-gpu first."
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] No NVIDIA GPU detected (nvidia-smi missing)."
    echo "[ERROR] Enable Kaggle Accelerator -> GPU, then re-run."
    exit 1
fi
echo ""
echo "[GPU]"
nvidia-smi --query-gpu=name,memory.free,memory.total \
           --format=csv,noheader 2>/dev/null | \
    awk -F, '{printf "  %s | free: %s | total: %s\n", $1,$2,$3}'

if [[ ! -f "${TOK}" ]]; then
    echo "[ERROR] Tokenizer not found: ${TOK}"
    echo "        Run setup.sh --with-parquet first (it trains the tokenizer automatically)."
    exit 1
fi
TRAIN_SHARDS=$(find "${REPO_DIR}/artifacts/shards_en" -name "train_*.gbin" 2>/dev/null | wc -l)
if [[ "${TRAIN_SHARDS}" -eq 0 ]]; then
    echo "[ERROR] No training shards. Run: bash configs/kaggle/build_english_data.sh"
    exit 1
fi
echo "[data] Training shards: ${TRAIN_SHARDS}"

CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | cut -d. -f1 | tr -d ' ')
FP16_OVERRIDE=()
if [[ -n "${CC_MAJOR}" ]] && [[ "${CC_MAJOR}" -lt 7 ]]; then
    echo "[prec] GPU CC ${CC_MAJOR}.x has no FP16 tensor cores -> forcing fp32 GEMMs"
    FP16_OVERRIDE=(--gemm-fp16 0)
else
    echo "[prec] GPU CC ${CC_MAJOR:-?}.x -> fp16 tensor GEMMs per config"
fi

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

mkdir -p "${REPO_DIR}/artifacts/checkpoints/pilot" "${REPO_DIR}/artifacts/checkpoints/en_pro"

run_train() {
    local cfg="$1"; local dev="$2"; shift 2
    "${BINARY}" --config "${cfg}" --device "${dev}" --tokenizer "${TOK}" "$@"
}

if [[ "${MODE}" == "dryrun" ]]; then
    run_train "${CONFIG_FULL}" auto --dry-run
    exit 0
fi

PILOT_TPS=0
if [[ "${MODE}" == "oneshot" ]] || [[ "${MODE}" == "pilot" ]]; then
    echo ""
    echo "[pilot] Measuring real speed (${PILOT_STEPS} steps)..."
    P_START=$(date +%s)

    run_train "${CONFIG_PILOT}" cuda --max-steps "${PILOT_STEPS}" --resume none "${FP16_OVERRIDE[@]}"
    P_END=$(date +%s)
    P_ELAPSED=$(( P_END - P_START ))
    if [[ "${P_ELAPSED}" -le 0 ]]; then P_ELAPSED=1; fi

    P_TPS=$(( $(yget batch_size "${CONFIG_PILOT}") * $(yget seq_len "${CONFIG_PILOT}") * $(yget grad_accum "${CONFIG_PILOT}") * PILOT_STEPS / P_ELAPSED ))
    PILOT_TPS=${P_TPS}
    echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${PILOT_TPS} tok/s"
    if [[ "${MODE}" == "pilot" ]]; then
        echo "[pilot] Done (pilot-only mode)."
        exit 0
    fi
fi

if [[ "${MODE}" == "oneshot" ]]; then
    F_TPS=$(( $(yget batch_size "${CONFIG_FULL}") * $(yget seq_len "${CONFIG_FULL}") * $(yget grad_accum "${CONFIG_FULL}") ))
    echo "[plan] Full config: ${F_TPS} tokens/step"

    FULL_SPS=$(awk "BEGIN {printf \"%.2f\", ${F_TPS} / ${PILOT_TPS}}")
    BUDGET_SEC=$(( TIME_BUDGET_MIN * 60 ))
    USED_SEC=$(($(date +%s) - P_START))
    REMAIN_SEC=$(( BUDGET_SEC - USED_SEC - EXPORT_MARGIN_SEC ))
    MAX_STEPS=$(awk "BEGIN {printf \"%d\", ${REMAIN_SEC} / ${FULL_SPS}}")
    if [[ "${MAX_STEPS}" -lt 50 ]]; then MAX_STEPS=50; fi
    echo "[plan] Budget ${TIME_BUDGET_MIN}min, used ${USED_SEC}s -> max_steps=${MAX_STEPS}"
    echo "[plan] Expected initial loss: ~ln(32000) ≈ 10.37 (+ tiny MoE aux term)"

    WARMUP_STEPS=$(( MAX_STEPS / 10 ))
    if [[ "${WARMUP_STEPS}" -lt 20 ]]; then WARMUP_STEPS=20; fi
    EXTRA_ARGS=(--max-steps "${MAX_STEPS}" --warmup "${WARMUP_STEPS}")
    ACTIVE_CONFIG="${CONFIG_FULL}"
else
    echo "[plan] FULL schedule mode (no budget cap, runs the yaml schedule)"
    EXTRA_ARGS=()
    ACTIVE_CONFIG="${CONFIG_FULL}"
fi

echo ""
echo "[train] Starting full run..."
T_START=$(date +%s)
if [[ "${EXPORT_GGUF}" -eq 1 ]]; then
    run_train "${ACTIVE_CONFIG}" cuda "${EXTRA_ARGS[@]}" "${FP16_OVERRIDE[@]}" \
        --export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}"
else
    run_train "${ACTIVE_CONFIG}" cuda "${EXTRA_ARGS[@]}" "${FP16_OVERRIDE[@]}"
fi
T_END=$(date +%s)
echo "[train] Full run took $(( (T_END - T_START) / 60 ))m $(( (T_END - T_START) % 60 ))s"

if [[ "${EXPORT_GGUF}" -eq 1 ]] && [[ -f "${GGUF_OUT}" ]]; then
    echo ""
    echo "[smoke] Testing the exported model (English prompt, no sidecar)..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --persona en \
        --prompt "Hello, who are you? What can you help me with?" --max-tokens 60 || \
        { echo "[smoke] FAIL: self-contained GGUF generation failed"; exit 1; }
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
echo "  Chat: ${GEN_BIN} chat --model ${GGUF_OUT} --tokenizer ${TOK}"
echo "============================================================"
