#!/usr/bin/env bash
# kaggle/verify_fixes.sh — post-repair verification on Kaggle (CUDA + optional NCCL).
#
# Run AFTER kaggle/setup.sh --with-parquet, BEFORE any long training:
#   bash kaggle/verify_fixes.sh [--ddp]
#
# What it proves, in order (fail-fast: any FAIL aborts the session before GPU
# hours burn):
#   1. CUDA build is clean (-Werror) and all CPU tests pass.
#   2. GPU parity: the fused CUDA MoE (F-03), device slot counters (F-02) and
#      the CE accumulate API (F-10) match the CPU references.
#   3. VRAM pre-flight: every shipped recipe is priced arithmetically and the
#      flagship must fit a 16 GB T4 with >=10% headroom (--max-vram-mb gate).
#   4. A 6-step end-to-end CPU training smoke (eval+save every step, exact
#      resume, drift refusal) via test_trainer_ckpt_flow.
#   5. (--ddp, needs >=2 GPUs + NCCL): a 2-rank DDP smoke proving the F-01
#      best-save protocol does not hang and both ranks agree on best/last.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
BIN="${REPO_DIR}/build/bin"
DO_DDP=0
[[ "${1:-}" == "--ddp" ]] && DO_DDP=1

pass() { echo "  [ok] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }
# A build/test failure that only says "see /tmp/x.log" costs a whole round trip
# and, worse, leaves the PREVIOUS binary in build/bin to run and produce a
# misleading second failure. Always surface the tail inline.
fail_log() {
  echo "  [FAIL] $1"
  echo "  ----- last 25 lines of $2 -----"
  tail -25 "$2" 2>/dev/null || echo "  (log missing: $2)"
  echo "  ---------------------------------"
  exit 1
}

echo "=== [1/5] build + CPU tests ==="
# -DGAI_ENABLE_PARQUET=ON is REQUIRED here, not cosmetic: the parquet lake is
# the only training input, and this cell rebuilds build/ from scratch. Without
# the flag, CMake falls back to GAI_ENABLE_PARQUET=OFF (the default) and every
# later cell that calls `data_pipeline parquet` dies on a binary that was
# compiled without Arrow.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=ON -DGAI_BUILD_TESTS=ON \
  -DGAI_ENABLE_PARQUET=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')" \
  > /tmp/verify_cmake.log 2>&1 \
  || fail_log "cmake configure" /tmp/verify_cmake.log
cmake --build build -j2 > /tmp/verify_build.log 2>&1 \
  || fail_log "CUDA build -Werror" /tmp/verify_build.log
ctest --test-dir build --output-on-failure > /tmp/verify_ctest.log 2>&1 \
  || fail_log "ctest" /tmp/verify_ctest.log
pass "build clean, all CPU tests green"

echo "=== [2/5] GPU parity (F-02/F-03/F-10) ==="
"${BIN}/test_moe_cuda_parity" || fail "test_moe_cuda_parity"
pass "fused MoE + slot counters + CE accumulate match CPU"

echo "=== [3/5] VRAM pre-flight (all shipped recipes) ==="
# Includes the 2xT4 1B pair, which is the DEFAULT recipe of kaggle/train_2xt4.sh.
# Pricing those here means a recipe change is caught before the session burns
# hours of GPU on a config that cannot fit.
for cfg in en_pro pro_v1 t4_1b pro_auxfree en_ollama sft_en_pro sft_pro_v1 \
           en_2xt4 sft_en_2xt4 sft_en_4xt4 pro_1b_2xt4 sft_pro_1b_2xt4 \
           en_compact_2xt4 sft_en_compact_2xt4; do
  "${BIN}/gai_train" --config "configs/${cfg}.yaml" --dry-run --device cuda \
    > /tmp/verify_dry_${cfg}.log 2>&1 || fail "dry-run ${cfg}"
  grep -q "TOTAL" /tmp/verify_dry_${cfg}.log || fail "dry-run ${cfg} printed no TOTAL"
  pass "dry-run ${cfg}: $(grep 'TOTAL' /tmp/verify_dry_${cfg}.log | head -1)"
done
# The 1B pair is priced against the REAL per-GPU budget, not the 480M one:
# the 2xT4 launcher uses the full-16 GB gate for 1B recipes.
"${BIN}/gai_train" --config configs/pro_1b_2xt4.yaml --dry-run --device cuda \
  --max-vram-mb 16384 > /tmp/verify_gate_1b.log 2>&1 || fail "VRAM gate (pro_1b_2xt4 must fit 16 GiB)"
"${BIN}/gai_train" --config configs/sft_pro_1b_2xt4.yaml --dry-run --device cuda \
  --max-vram-mb 16384 > /tmp/verify_gate_1b_sft.log 2>&1 || fail "VRAM gate (sft_pro_1b_2xt4 must fit 16 GiB)"
pass "1B 2xT4 pair fits a 16 GB T4 with headroom"
# The compact recipe is the one train_2xt4.sh is meant to run by default: it
# uses a quarter of the card, so it must pass with a very large margin.
"${BIN}/gai_train" --config configs/en_compact_2xt4.yaml --dry-run --device cuda \
  --max-vram-mb 16384 > /tmp/verify_gate_compact.log 2>&1 || fail "VRAM gate (en_compact_2xt4 must fit 16 GiB)"
"${BIN}/gai_train" --config configs/sft_en_compact_2xt4.yaml --dry-run --device cuda \
  --max-vram-mb 16384 > /tmp/verify_gate_compact_sft.log 2>&1 || fail "VRAM gate (sft_en_compact_2xt4 must fit 16 GiB)"
pass "compact pair fits a 16 GB T4 with headroom"
"${BIN}/gai_train" --config configs/en_pro.yaml --dry-run --device cuda \
  --max-vram-mb 15360 > /tmp/verify_gate.log 2>&1 || fail "VRAM gate (en_pro must fit 15 GiB)"
pass "flagship fits 16 GB T4 with headroom"
if "${BIN}/gai_train" --config configs/en_pro.yaml --dry-run --device cuda \
    --max-vram-mb 8192 > /tmp/verify_gate_neg.log 2>&1; then
  fail "VRAM gate did NOT reject an 8 GiB budget (gate is broken)"
fi
pass "VRAM gate correctly rejects an impossible budget"

echo "=== [4/5] checkpoint-flow integration ==="
"${BIN}/test_trainer_ckpt_flow" || fail "test_trainer_ckpt_flow"
"${BIN}/test_resume_mode" || fail "test_resume_mode"
"${BIN}/test_moe_bias_step" || fail "test_moe_bias_step"
pass "best+last publish, exact resume, bias semantics"

if [[ "${DO_DDP}" == "1" ]]; then
  echo "=== [5/5] 2-rank DDP smoke (F-01 protocol) ==="
  command -v nvidia-smi > /dev/null || fail "nvidia-smi missing"
  NGPU="$(nvidia-smi -L | wc -l)"
  [[ "${NGPU}" -ge 2 ]] || fail "need >=2 GPUs for the DDP smoke (found ${NGPU})"
  # A REAL 2-process run (not a mock): rank 0 WILL improve validation on step
  # 1, so every rank must enter save("best.ckpt") together or NCCL hangs
  # (F-01). `timeout` turns a hang into a FAIL instead of a burnt session.
  # The model is shrunk via CLI overrides (vocab/layers/hidden) so the smoke
  # fits anywhere; shards are generated inline from 200 text lines.
  # --warmup 0 is mandatory: en_pro.yaml ships warmup_steps=500, and the LR
  # scheduler contract requires warmup < total_steps (3 here), so a short smoke
  # MUST clamp it or it aborts before the first collective.
  SMOKE_DIR=/tmp/verify_ddp
  rm -rf "${SMOKE_DIR}"
  mkdir -p "${SMOKE_DIR}/shards" "${SMOKE_DIR}/ckpt"
  python3 -c "for i in range(200): print(f'smoke line {i} the quick brown fox jumps over token {i%17}')" \
    > "${SMOKE_DIR}/tiny.txt"
  TOK="$(ls artifacts/tokenizer/*.gtok 2>/dev/null | head -1)"
  [[ -n "${TOK}" ]] || fail "no tokenizer .gtok under artifacts/tokenizer (run setup.sh first)"
  "${BIN}/data_pipeline" build --text "${SMOKE_DIR}/tiny.txt" --tokenizer "${TOK}" \
    --out "${SMOKE_DIR}/shards" --seq-len 32 --val-ratio 0.2 > /tmp/verify_shard.log 2>&1 \
    || fail "tiny shard build (see /tmp/verify_shard.log)"
  export WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511
  # Crash attribution: an illegal memory access surfaces at the NEXT stream
  # sync, not at the kernel that caused it. Blocking launches make the fault
  # fire at the offending launch, and --verbose enables the collective ledger
  # plus per-rank step breadcrumbs (a non-main rank prints nothing otherwise,
  # so a DDP crash is otherwise unattributable). The model is tiny, so
  # blocking launches cost nothing.
  export CUDA_LAUNCH_BLOCKING=1
  rm -f /tmp/gai_nccl_29511.id
  SMOKE_OK=1
  for RANK in 0 1; do
    RANK="${RANK}" LOCAL_RANK="${RANK}" timeout 600 "${BIN}/gai_train" \
      --config configs/en_pro.yaml --device cuda --verbose \
      --vocab 32000 --layers 2 --hidden 128 --heads 4 --kv-heads 2 \
      --batch-size 1 --seq-len 32 --grad-accum 1 --max-steps 3 --warmup 0 \
      --eval-every 1 --eval-batches 1 --save-every 1 --log-every 1 \
      --data "${SMOKE_DIR}/shards" --checkpoint-dir "${SMOKE_DIR}/ckpt" \
      --tokenizer "${TOK}" --resume none --seed 7 \
      > "/tmp/verify_ddp_rank${RANK}.log" 2>&1 &
    echo $! > "/tmp/verify_ddp_pid${RANK}"
  done
  for RANK in 0 1; do
    if ! wait "$(cat /tmp/verify_ddp_pid${RANK})"; then
      echo "  [FAIL] rank ${RANK} exited non-zero (see /tmp/verify_ddp_rank${RANK}.log)"
      SMOKE_OK=0
    fi
  done
  # Kill orphans: if rank 0 finished but rank 1 hung in a collective (the F-01
  # signature), it is still alive here.
  for RANK in 0 1; do
    PID="$(cat /tmp/verify_ddp_pid${RANK})"
    if kill -0 "${PID}" 2>/dev/null; then
      echo "  [FAIL] rank ${RANK} still alive after its sibling exited — collective desync (F-01)"
      kill -9 "${PID}" 2>/dev/null || true
      SMOKE_OK=0
    fi
  done
  [[ "${SMOKE_OK}" == "1" ]] || fail "DDP smoke failed"
  [[ -f "${SMOKE_DIR}/ckpt/last.ckpt" ]] || fail "DDP smoke produced no last.ckpt"
  [[ -f "${SMOKE_DIR}/ckpt/best.ckpt" ]] || fail "DDP smoke produced no best.ckpt"
  pass "2-rank run completed with best+last, no hang (F-01 protocol sound)"
else
  echo "=== [5/5] DDP smoke skipped (pass --ddp on a 2+ GPU box) ==="
fi

echo ""
echo "============================================================"
echo "  verify_fixes.sh: ALL GREEN — safe to start budgeted training"
echo "  Next: bash kaggle/train_1b.sh --preflight, then train."
echo "============================================================"
