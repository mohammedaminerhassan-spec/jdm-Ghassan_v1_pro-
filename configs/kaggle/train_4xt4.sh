#!/bin/bash

set -euo pipefail

CONFIG=${1:-configs/pro_1b_4xt4_legacy.yaml}

echo "=========================================="
echo "Ghassan AI - 4x T4 FP16-tensor-core DDP"
echo "=========================================="
echo "Config: $CONFIG"
echo ""

if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi missing: 4xT4 DDP needs 4 NVIDIA GPUs."
    exit 1
fi
GPU_N=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${GPU_N}" -lt 4 ]]; then
    echo "[ERROR] found ${GPU_N} GPU(s), need 4 for train_4xt4.sh."
    echo "[ERROR] Single-T4: bash configs/kaggle/train_1b.sh  (flagship recipe)"
    exit 1
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${REPO_DIR}" || exit 1
BIN="${REPO_DIR}/build/bin/gai_train"
[[ -x "${BIN}" ]] || { echo "[ERROR] ${BIN} missing. Run kaggle/setup.sh first."; exit 1; }

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

export WORLD_SIZE=${WORLD_SIZE:-4}

rm -f "/tmp/gai_nccl_${MASTER_PORT}.id"

unset CUDA_VISIBLE_DEVICES

PIDS=()

cleanup_ranks() {
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
}

EXPORT_ARGS=()
if [[ -n "${EXPORT_GGUF_OUT:-}" ]]; then
    EXPORT_ARGS=(--export "${EXPORT_GGUF_OUT}" --export-profile "${EXPORT_PROFILE:-q4_0}")
    echo "Export: ${EXPORT_GGUF_OUT} [${EXPORT_PROFILE:-q4_0}] (rank 0 only)"
fi

trap cleanup_ranks INT TERM
for i in $(seq 0 $((WORLD_SIZE - 1))); do
    LOCAL_RANK=$i RANK=$i "${BIN}" --config "$CONFIG" --device cuda "${EXPORT_ARGS[@]}" &
    PIDS[$i]=$!
    echo "Started rank $i/$WORLD_SIZE (pid ${PIDS[$i]})"
done

FAIL=0
ALIVE=${WORLD_SIZE}
while [[ "${ALIVE}" -gt 0 ]]; do
    for pid in "${PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null || continue

        state="$(awk '{print $3}' "/proc/${pid}/stat" 2>/dev/null || echo R)"
        [[ "${state}" == "Z" ]] && continue
        if wait "${pid}" 2>/dev/null; then :; else FAIL=1; fi
        ALIVE=$(( ALIVE - 1 ))
    done
    [[ "${ALIVE}" -gt 0 ]] && sleep 5
done

if [ "$FAIL" -ne 0 ]; then
    echo "!!! a training rank failed, killing the rest"
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
    exit 1
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
