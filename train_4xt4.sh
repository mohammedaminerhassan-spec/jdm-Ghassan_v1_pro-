#!/bin/bash
# Launch script for 4x T4 training with NCCL DDP.
# Usage: ./train_4xt4.sh [config.yaml]
# Default: the 1B flagship (pretrain). Pass an SFT config for stage 2.
set -e

CONFIG=${1:-configs/t4_1b.yaml}

echo "=========================================="
echo "Ghassan AI - 4x T4 FP16-tensor-core DDP"
echo "=========================================="
echo "Config: $CONFIG"
echo ""

# NCCL tuning for Kaggle 4xT4 (single node, no InfiniBand).
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
# NOTE: do NOT set CUDA_VISIBLE_DEVICES per rank here. All 4 GPUs stay
# visible and each rank picks its own via LOCAL_RANK (see cuda_utils probe
# + Trainer::init_distributed). Hiding GPUs breaks rank->device mapping.
unset CUDA_VISIBLE_DEVICES

PIDS=()
for i in 0 1 2 3; do
    LOCAL_RANK=$i RANK=$i ./build/bin/gai_train --config "$CONFIG" &
    PIDS[$i]=$!
    echo "Started rank $i (pid ${PIDS[$i]})"
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
