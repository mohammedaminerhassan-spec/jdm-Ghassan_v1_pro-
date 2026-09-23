#!/bin/bash
# Launch script for 4x T4 training with NCCL DDP.
# Usage: ./train_4xt4.sh [config.yaml]
# Default: configs/t4_1b.yaml (LEGACY 4xT4-DDP experiment, NOT the single-T4
# flagship — that is configs/ultra_1b.yaml via kaggle/train_1b.sh).
# Pass an SFT config for stage 2.
set -euo pipefail

CONFIG=${1:-configs/t4_1b.yaml}

echo "=========================================="
echo "Ghassan AI - 4x T4 FP16-tensor-core DDP"
echo "=========================================="
echo "Config: $CONFIG"
echo ""

# PRO-HARDEN: هذا السكربت كان بلا بوابة GPU (كل train*.sh الأخرى فيها).
# على جلسة بدون 4xT4 كان يطلق 4 ranks على GPU واحد -> OOM/Hang غامض.
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi missing: 4xT4 DDP needs 4 NVIDIA GPUs."
    exit 1
fi
GPU_N=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${GPU_N}" -lt 4 ]]; then
    echo "[ERROR] found ${GPU_N} GPU(s), need 4 for train_4xt4.sh."
    echo "[ERROR] Single-T4: bash kaggle/train_1b.sh  (flagship recipe)"
    exit 1
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${REPO_DIR}/build/bin/gai_train"
[[ -x "${BIN}" ]] || { echo "[ERROR] ${BIN} missing. Run kaggle/setup.sh first."; exit 1; }

# NCCL tuning for Kaggle 4xT4 (single node, no InfiniBand).
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
# PRO-HARDEN: WORLD_SIZE ثابت 4 كان يمنع single-GPU debug. قابل للتجاوز.
export WORLD_SIZE=${WORLD_SIZE:-4}
# FIX: stale NCCL rendezvous file from a crashed run makes ranks 1..3 read an
# old ID before rank 0 rewrites it -> ncclCommInitRank mismatch / hang.
# Remove it before spawn so rank 0 always publishes a fresh ID (Kaggle 4xT4).
rm -f "/tmp/gai_nccl_${MASTER_PORT}.id"
# NOTE: do NOT set CUDA_VISIBLE_DEVICES per rank here. All 4 GPUs stay
# visible and each rank picks its own via LOCAL_RANK (see cuda_utils probe
# + Trainer::init_distributed). Hiding GPUs breaks rank->device mapping.
unset CUDA_VISIBLE_DEVICES

PIDS=()
# PRO-HARDEN: trap يقتل الرتب اليتيمة عند SIGTERM (كانت تبقى معلقة على T4).
cleanup_ranks() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
# FIX P2: loop was hardcoded 0..3 while WORLD_SIZE is overridable above,
# so WORLD_SIZE=2 still spawned 4 ranks -> OOM. Honor WORLD_SIZE.
trap cleanup_ranks INT TERM
for i in $(seq 0 $((WORLD_SIZE - 1))); do
    LOCAL_RANK=$i RANK=$i "${BIN}" --config "$CONFIG" &
    PIDS[$i]=$!
    echo "Started rank $i/$WORLD_SIZE (pid ${PIDS[$i]})"
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=1; fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "!!! a training rank failed, killing the rest"
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    exit 1
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
