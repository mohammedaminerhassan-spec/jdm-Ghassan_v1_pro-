#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_DIR}"
BIN="${REPO_DIR}/build/bin"
TOK="artifacts/tokenizer/english32k.gtok"
SHARDS="artifacts/shards_en"
CFG="configs/flash_480m_single.yaml"

case "${CFG}" in *1b*) OUTPUT_BUDGET_MB=18432;; *) OUTPUT_BUDGET_MB=17408;; esac

echo "=============================================================="
echo "  CELL 8 — pilot: measure the real speed of ${CFG}"
echo "=============================================================="

echo ""
echo "----- [1/5] commit under test -----"
git log --oneline -1 2>/dev/null || echo "  (no git repo — Kaggle package)"
echo "  gpu       : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tr '\n' ' ')"
echo "  ram       : $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
echo "  disk      : $(df -h . | awk 'NR==2{print $4" free of "$2" ("$5" used)"}')"
echo "  cpu cores : $(nproc)"

echo ""
echo "----- [2/5] recipe under test -----"
grep -E "vocab_size|hidden_size|num_layers|num_experts|moe_top_k|moe_expert_dim" "${CFG}" | sed 's/^/  /'
grep -E "batch_size|seq_len|grad_accum|learning_rate|warmup_steps|optimizer|max_steps|epochs" "${CFG}" | sed 's/^/  /'

echo ""
echo "----- [3/5] pilot: 20 steps of the real recipe -----"
P_START=$(date +%s)
"${BIN}/gai_train" --config "${CFG}" --device cuda --tokenizer "${TOK}" \
    --data "${SHARDS}" --max-steps 20 --warmup 5 \
    --eval-every 20 --eval-batches 2 --save-every 20 --log-every 1 \
    --checkpoint-dir /tmp/ghassan_pilot_pilot --resume none \
    --output-budget-mb "${OUTPUT_BUDGET_MB}" 2>&1 | tee /tmp/ghassan_pilot_pilot.log \
    | grep -E "^\[info \] (step|  >> val)|host RAM|\[disk\]|\[quota\]|pretrain done|\[scaler\]|ckpt\] (saved|published)|OOM|guard"
P_END=$(date +%s)
ELAPSED=$(( P_END - P_START )); [[ "${ELAPSED}" -le 0 ]] && ELAPSED=1

echo ""
echo "----- [4/5] the new resource guards, as this machine reports them -----"
grep -E "host RAM|\[disk\]|\[quota\]" /tmp/ghassan_pilot_pilot.log | sed 's/^/  /' || true
grep -E "OOM guard|disk guard|host RAM guard|host RAM cannot|output quota guard" /tmp/ghassan_pilot_pilot.log \
    && { echo "  [FAIL] a resource guard fired on the flagship recipe"; exit 1; } \
    || echo "  [ok] no resource guard fired (VRAM/RAM/disk/quota all fit)"

echo ""
echo "----- [5/5] measured throughput -> honest plans -----"

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
B=$(yget batch_size "${CFG}"); T=$(yget seq_len "${CFG}"); A=$(yget grad_accum "${CFG}")
TOK_PER_STEP=$(( B * T * A ))
TPS=$(( TOK_PER_STEP * 20 / ELAPSED ))
echo "  pilot     : 20 steps in ${ELAPSED}s (includes model load + arena + first CUDA init)"
LAST_TPS=$(grep -oE "[0-9.]+[KM]? tok/s" /tmp/ghassan_pilot_pilot.log | tail -1)
echo "  steady    : ${LAST_TPS} (from the last step line, excludes startup)"
echo "  budget    : ${TOK_PER_STEP} tokens/step (B=${B} T=${T} accum=${A})"

CORPUS_TOKENS=$("${BIN}/data_pipeline" inspect --shards "${SHARDS}" 2>/dev/null \
    | awk '/^  total: /{v=$2;
           if (v ~ /K$/) m=1000; else if (v ~ /M$/) m=1000000;
           else if (v ~ /B$/) m=1000000000; else if (v ~ /T$/) m=1000000000000;
           gsub(/[^0-9.]/,"",v); t+=v*m} END{printf "%.0f", t+0}')
[[ "${CORPUS_TOKENS}" -gt 0 ]] || CORPUS_TOKENS=0
if [[ "${CORPUS_TOKENS}" -gt 0 ]]; then
    echo "  corpus    : ${CORPUS_TOKENS} tokens (from data_pipeline inspect)"
else
    echo "  corpus    : unknown (inspect failed) — the epoch column is blank"
fi
echo ""
printf "  %-12s %10s %12s %14s\n" "budget" "steps" "tokens" "epochs"
for H in 2 4 6 8 10; do
    STEPS=$(( H * 3600 * TPS / TOK_PER_STEP ))
    if [[ "${CORPUS_TOKENS}" -gt 0 ]]; then
        EP=$(awk -v s="${STEPS}" -v t="${TOK_PER_STEP}" -v c="${CORPUS_TOKENS}" \
             'BEGIN{printf "%.2f", s*t/c}')
    else
        EP="-"
    fi
    printf "  %-12s %10s %12s %14s\n" "${H}h" "${STEPS}" \
        "$(( STEPS * TOK_PER_STEP / 1000000 ))M" "${EP}"
done
echo ""
echo "  NOTE: the step count above is a FIRST session. The trainer now keeps a"
echo "        checkpoint's original plan, so later sessions continue it instead"
echo "        of reshaping the LR curve."
echo ""
echo "=============================================================="
echo "  CELL 8 DONE — no training yet; the numbers above are the plan input"
echo "=============================================================="
