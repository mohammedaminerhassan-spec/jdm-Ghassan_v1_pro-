#!/usr/bin/env bash
# kaggle/cell8_pilot.sh — measure the REAL speed of the flagship recipe, then
# print the plan for every plausible session budget. Trains nothing.
#
# Why this is a cell of its own: every step-count estimate before this was
# arithmetic on assumed T4 throughput. The pilot runs 20 steps of the ACTUAL
# recipe (fp16 GEMMs on, activation checkpointing on, real 65,536-token steps,
# real shards) and measures tok/s, which is the only number the budget maths is
# allowed to trust. It also prints the host-RAM/disk guards so the numbers come
# from the real 30 GB box, not from a formula.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
BIN="${REPO_DIR}/build/bin"
TOK="artifacts/tokenizer/english32k.gtok"
SHARDS="artifacts/shards_en"
CFG="configs/en_pro.yaml"

echo "=============================================================="
echo "  CELL 8 — pilot: measure the real speed of ${CFG}"
echo "=============================================================="

echo ""
echo "----- [1/5] commit under test -----"
git log --oneline -1
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
    --checkpoint-dir /tmp/cell8_pilot --resume none 2>&1 | tee /tmp/cell8_pilot.log \
    | grep -E "^\[info \] (step|  >> val)|host RAM|\[disk\]|pretrain done|\[scaler\]|ckpt\] (saved|published)|OOM|guard"
P_END=$(date +%s)
ELAPSED=$(( P_END - P_START )); [[ "${ELAPSED}" -le 0 ]] && ELAPSED=1

echo ""
echo "----- [4/5] the new resource guards, as this machine reports them -----"
grep -E "host RAM|\[disk\]" /tmp/cell8_pilot.log | sed 's/^/  /' || true
grep -E "OOM guard|disk guard|host RAM guard|host RAM cannot" /tmp/cell8_pilot.log \
    && { echo "  [FAIL] a resource guard fired on the flagship recipe"; exit 1; } \
    || echo "  [ok] no resource guard fired (RAM/disk/VRAM all fit)"

echo ""
echo "----- [5/5] measured throughput -> honest plans -----"
TOK_PER_STEP=$(( $(grep -m1 'batch_size:' "${CFG}" | tr -dc '0-9') * $(grep -m1 'seq_len:' "${CFG}" | tr -dc '0-9') * $(grep -m1 'grad_accum:' "${CFG}" | tr -dc '0-9') ))
TPS=$(( TOK_PER_STEP * 20 / ELAPSED ))
echo "  pilot     : 20 steps in ${ELAPSED}s (includes model load + arena + first CUDA init)"
LAST_TPS=$(grep -oE "[0-9.]+[KM]? tok/s" /tmp/cell8_pilot.log | tail -1)
echo "  steady    : ${LAST_TPS} (from the last step line, excludes startup)"
echo "  budget    : ${TOK_PER_STEP} tokens/step"
echo ""
printf "  %-12s %10s %12s %14s\n" "budget" "steps" "tokens" "epochs(323.65M)"
for H in 2 4 6 8 10; do
    STEPS=$(( H * 3600 * TPS / TOK_PER_STEP ))
    printf "  %-12s %10s %12s %14s\n" "${H}h" "${STEPS}" \
        "$(( STEPS * TOK_PER_STEP / 1000000 ))M" \
        "$(awk -v s="${STEPS}" -v t="${TOK_PER_STEP}" 'BEGIN{printf "%.2f", s*t/323650000}')"
done
echo ""
echo "  NOTE: the step count above is a FIRST session. The trainer now keeps a"
echo "        checkpoint's original plan, so later sessions continue it instead"
echo "        of reshaping the LR curve."
echo ""
echo "=============================================================="
echo "  CELL 8 DONE — no training yet; the numbers above are the plan input"
echo "=============================================================="
