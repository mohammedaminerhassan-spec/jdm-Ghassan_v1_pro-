#!/usr/bin/env bash
# kaggle/train_2xt4.sh — Ghassan v1 Pro 2xT4 DDP production launcher (Kaggle T4x2).
#
# Target: 2x Tesla T4 (16GB each), ~29GB host RAM, 4 CPU cores, 12h session,
# 20GB auto-saved /kaggle/working. Each rank owns a FULL model replica (DDP
# data-parallel, NOT sharded): 16+16GB is NOT 32GB shared VRAM.
#
# Flow: PRECHECK -> NCCL/CUDA CHECK -> CONFIG PARITY -> DATA CHECK ->
# TOKENIZER CHECK -> VRAM PLAN -> OUTPUT QUOTA -> 2-RANK SMOKE/PILOT ->
# TRAIN (PT+SFT) -> CHECKPOINT -> OPTIONAL EXPORT -> FINAL SMOKE.
#
# Usage:
#   bash kaggle/train_2xt4.sh                                   # full PT+SFT, ~640min budget
#   bash kaggle/train_2xt4.sh --preflight                       # gates only, no training
#   bash kaggle/train_2xt4.sh --pilot-only                       # 2-rank pilot, measures global tok/s
#   bash kaggle/train_2xt4.sh --time-budget-min 500 --pt-fraction 60
#   CONFIG_PT=configs/pro_v1.yaml CONFIG_SFT=configs/sft_pro_v1.yaml bash kaggle/train_2xt4.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="full"
SESSION_LIMIT_MIN="${SESSION_LIMIT_MIN:-700}"   # 12h = 720, keep margin
DEFAULT_BUDGET=$(( SESSION_LIMIT_MIN - 60 ))
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-$DEFAULT_BUDGET}"
[[ "${TIME_BUDGET_MIN}" -gt "$(( SESSION_LIMIT_MIN - 20 ))" ]] && {
    echo "[ERROR] --time-budget-min ${TIME_BUDGET_MIN} exceeds session limit ${SESSION_LIMIT_MIN}."
    exit 1
}
CONFIG_PT="${CONFIG_PT:-${REPO_DIR}/configs/en_2xt4.yaml}"
CONFIG_SFT="${CONFIG_SFT:-${REPO_DIR}/configs/sft_en_2xt4.yaml}"
PT_DIR="${PT_DIR:-${REPO_DIR}/artifacts/shards_en}"
SFT_DIR="${SFT_DIR:-${REPO_DIR}/artifacts/shards_en}"
CKPT_PT="${CKPT_PT:-${REPO_DIR}/artifacts/checkpoints/en_2xt4}"
CKPT_SFT="${CKPT_SFT:-${REPO_DIR}/artifacts/checkpoints/en_2xt4_sft}"
GGUF_OUT="${GGUF_OUT:-${REPO_DIR}/artifacts/ghassan-2xt4_q4_0.gguf}"
TOK="${TOK:-${REPO_DIR}/artifacts/tokenizer/english32k.gtok}"
OUTPUT_BUDGET_MB="${OUTPUT_BUDGET_MB:-19456}"
EXPORT_GGUF=1
EXPORT_PROFILE="q4_0"
PILOT_STEPS=20
EXPORT_MARGIN_SEC=900
PT_FRACTION=60
SKIP_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot-only) MODE="pilot"; shift ;;
        --preflight) MODE="preflight"; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --time-budget-min) TIME_BUDGET_MIN="$2"; shift 2 ;;
        --no-export) EXPORT_GGUF=0; shift ;;
        --export-profile) EXPORT_PROFILE="$2"; shift 2 ;;
        --pt-fraction) PT_FRACTION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

BINARY="${REPO_DIR}/build/bin/gai_train"
GEN_BIN="${REPO_DIR}/build/bin/ghassan-ai"
PIPE_BIN="${REPO_DIR}/build/bin/data_pipeline"
TESTCFG_BIN="${REPO_DIR}/build/bin/test_configs"

echo "============================================================"
echo "  Ghassan v1 Pro — 2xT4 DDP Production [${MODE}]"
echo "  DDP data-parallel: 2 replicas, NCCL sync, per-GPU <=16GB"
echo "============================================================"

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

# ---------------- PRECHECK: binary + GPUs ----------------
[[ -x "${BINARY}" ]] || { echo "[ERROR] ${BINARY} missing. Run: bash kaggle/setup.sh --with-parquet --require-nccl"; exit 1; }
command -v nvidia-smi &>/dev/null || { echo "[ERROR] nvidia-smi missing: 2xT4 needs 2 NVIDIA GPUs."; exit 1; }
GPU_N=$(nvidia-smi -L 2>/dev/null | wc -l)
[[ "${GPU_N}" -ge 2 ]] || { echo "[ERROR] found ${GPU_N} GPU(s), need >=2 for train_2xt4.sh."; exit 1; }
echo "[gpu] found ${GPU_N} GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | awk -F, '{printf "  GPU%s:%s total:%s\n",$1,$2,$3}'

# ---------------- NCCL/CUDA CHECK ----------------
echo "[nccl] verifying NCCL build + rendezvous hygiene..."
if grep -q "GAI_HAVE_NCCL:BOOL=ON" "${REPO_DIR}/build/CMakeCache.txt" 2>/dev/null; then
    echo "[nccl] OK: GAI_HAVE_NCCL=ON"
else
    echo "[ERROR] binary lacks NCCL (single-GPU build). Rebuild: bash kaggle/setup.sh --with-parquet --require-nccl"
    exit 1
fi
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=2
rm -f "/tmp/gai_nccl_${MASTER_PORT}.id"
unset CUDA_VISIBLE_DEVICES
echo "[nccl] WORLD_SIZE=2 MASTER=${MASTER_ADDR}:${MASTER_PORT} (all GPUs visible, rank picks via LOCAL_RANK)"

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

run_preflight() {
    echo ""
    echo "==================== preflight ===================="
    [[ -f "${CONFIG_PT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_PT}"; return 1; }
    [[ -f "${CONFIG_SFT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_SFT}"; return 1; }
    echo "[preflight] configs present"
    # Arch parity: SFT function must equal pretrain function (canonical identity).
    if [[ -x "${TESTCFG_BIN}" ]]; then
        "${TESTCFG_BIN}" > /tmp/preflight_arch.log 2>&1 || { echo "[preflight FAIL] arch parity (see /tmp/preflight_arch.log)"; tail -20 /tmp/preflight_arch.log; return 1; }
        echo "[preflight] arch parity OK (test_configs incl 2xt4 pairs)"
    else
        echo "[preflight WARN] ${TESTCFG_BIN} missing, skipping arch unit gate (runtime SFT gate still active)"
    fi
    # Tokenizer 32k == model vocab.
    [[ -f "${TOK}" ]] || { echo "[preflight FAIL] tokenizer missing: ${TOK}"; return 1; }
    TOK_VOCAB=$("${PIPE_BIN}" tok-info --tokenizer "${TOK}" 2>/dev/null | grep -oE 'vocab_size=[0-9]+' | cut -d= -f2 || true)
    MODEL_VOCAB=$(yget vocab_size "${CONFIG_PT}")
    [[ "${TOK_VOCAB}" == "${MODEL_VOCAB}" && -n "${TOK_VOCAB}" ]] || { echo "[preflight FAIL] tokenizer vocab=${TOK_VOCAB:-?} != model ${MODEL_VOCAB}"; return 1; }
    echo "[preflight] tokenizer vocab ok: ${TOK_VOCAB}"
    # Shards + declared mix domains (declared weight, shard count, effective weight).
    for pair in "${CONFIG_PT}:${PT_DIR}" "${CONFIG_SFT}:${SFT_DIR}"; do
        yaml="${pair%%:*}"; dir="${pair##*:}"
        python3 - "$yaml" "$dir" <<'EOF' || return 1
import glob, os, re, sys
yaml_path, shard_dir = sys.argv[1], sys.argv[2]
mix, in_mix = {}, False
for line in open(yaml_path):
    s = line.rstrip("\n")
    if s == "  mix:":
        in_mix = True
        continue
    if in_mix:
        if re.match(r"^[A-Za-z]", s):
            break
        m = re.match(r"^    ([A-Za-z_][A-Za-z0-9_]*):\s*([0-9.eE+-]+)", s)
        if m:
            mix[m.group(1)] = float(m.group(2))
if not mix:
    print("  [preflight FAIL] no mix block in " + yaml_path)
    sys.exit(1)
found = {d: sorted(glob.glob(os.path.join(shard_dir, "train_" + d + "_*.gbin"))) for d in mix}
missing = [d for d, fs in found.items() if not fs]
wsum = sum(w for d, w in mix.items() if d not in missing)
print("  mix table for " + os.path.basename(yaml_path) + ":")
for d, w in mix.items():
    eff = (w / wsum) if d not in missing and wsum > 0 else 0.0
    print("    %-22s declared=%5.2f shards=%3d effective=%5.2f %s" % (d, w, len(found[d]), eff, "MISSING" if d in missing else ""))
if missing:
    print("  [preflight FAIL] declared mix domains with zero shards: " + ", ".join(missing))
    sys.exit(1)
EOF
    done
    "${PIPE_BIN}" inspect --shards "${PT_DIR}" --tokenizer "${TOK}" || return 1
    # VRAM plan per-GPU (arithmetic) + hard gate 15GiB (16GB minus headroom).
    # --strict-config: unknown/dead keys fail here, never mid-run.
    "${BINARY}" --config "${CONFIG_PT}" --dry-run --device cuda --strict-config || return 1
    "${BINARY}" --config "${CONFIG_SFT}" --dry-run --device cuda --strict-config || return 1
    "${BINARY}" --config "${CONFIG_PT}" --dry-run --device cuda --strict-config --max-vram-mb 15360 || { echo "[preflight FAIL] PT recipe exceeds 15GiB per-GPU"; return 1; }
    "${BINARY}" --config "${CONFIG_SFT}" --dry-run --device cuda --strict-config --max-vram-mb 15360 || { echo "[preflight FAIL] SFT recipe exceeds 15GiB per-GPU"; return 1; }
    echo "[preflight] VRAM per-GPU OK (<=15GiB, preserves B/T envelope)"
    echo "==================== preflight: ALL GATES PASSED ===================="
    return 0
}

if [[ "${MODE}" == "preflight" ]]; then run_preflight; exit $?; fi
if [[ "${SKIP_PREFLIGHT}" -ne 1 ]]; then
    run_preflight || { echo "[ERROR] preflight failed."; exit 1; }
else
    echo "[preflight] SKIPPED via --skip-preflight (NOT advised)"
fi

# ---------------- 2-RANK PILOT (measures GLOBAL throughput) ----------------
echo ""
echo "[pilot] 2-rank DDP pilot (${PILOT_STEPS} steps, global tok/s from BOTH ranks)..."
P_START=$(date +%s)
PIDS=()
cleanup_pilot() { for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap 'cleanup_pilot; persist_output' INT TERM
for i in 0 1; do
    RANK=$i LOCAL_RANK=$i "${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
        --data "${PT_DIR}" --max-steps "${PILOT_STEPS}" --resume none \
        --checkpoint-dir "${CKPT_PT}" --output-budget-mb "${OUTPUT_BUDGET_MB}" \
        > "/tmp/pilot_2xt4_rank${i}.log" 2>&1 &
    PIDS[$i]=$!
    echo "  started rank $i (pid ${PIDS[$i]})"
done
FAIL=0
for pid in "${PIDS[@]}"; do if ! wait "$pid"; then FAIL=1; fi; done
[[ "${FAIL}" -eq 0 ]] || { echo "[pilot FAIL] a rank failed:"; tail -30 /tmp/pilot_2xt4_rank*.log; exit 1; }
# Kill orphans (collective desync signature: sibling still alive).
for pid in "${PIDS[@]}"; do if kill -0 "$pid" 2>/dev/null; then echo "[pilot FAIL] orphan rank ${pid} (collective hang)"; kill -9 "$pid" 2>/dev/null || true; FAIL=1; fi; done
[[ "${FAIL}" -eq 0 ]] || exit 1
P_END=$(date +%s)
P_ELAPSED=$(( P_END - P_START )); [[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1
# Global tokens/step INCLUDES world_size (DDP contract, tested in code).
B=$(yget batch_size "${CONFIG_PT}"); T=$(yget seq_len "${CONFIG_PT}"); A=$(yget grad_accum "${CONFIG_PT}")
GLOBAL_TPS=$(( B * T * A * 2 ))
P_TPS=$(( GLOBAL_TPS * PILOT_STEPS / P_ELAPSED ))
echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${P_TPS} GLOBAL tok/s (per-GPU B=${B} T=${T} accum=${A} x2 ranks)"
if [[ "${MODE}" == "pilot" ]]; then echo "[pilot] Done."; exit 0; fi

FULL_TPS=${GLOBAL_TPS}
BUDGET_SEC=$(( TIME_BUDGET_MIN * 60 ))
USED_SEC=$(($(date +%s) - P_START))
REMAIN_SEC=$(( BUDGET_SEC - USED_SEC - EXPORT_MARGIN_SEC ))
[[ "${REMAIN_SEC}" -lt 600 ]] && { echo "[ERROR] <10min left. Aborting."; exit 1; }
TOTAL_STEPS=$(awk "BEGIN {printf \"%d\", (${REMAIN_SEC} * ${P_TPS}) / ${FULL_TPS}}")
[[ "${TOTAL_STEPS}" -lt 100 ]] && TOTAL_STEPS=100
PT_STEPS=$(( TOTAL_STEPS * PT_FRACTION / 100 ))
SFT_STEPS=$(( TOTAL_STEPS - PT_STEPS ))
[[ "${PT_STEPS}" -lt 50 ]] && PT_STEPS=50
[[ "${SFT_STEPS}" -lt 50 ]] && SFT_STEPS=50
PT_WARM=$(( PT_STEPS / 10 ));  [[ "${PT_WARM}" -gt 800 ]] && PT_WARM=800
SFT_WARM=$(( SFT_STEPS / 10 )); [[ "${SFT_WARM}" -gt 200 ]] && SFT_WARM=200
EVAL_CAD=$(( TOTAL_STEPS / 10 )); [[ "${EVAL_CAD}" -lt 50 ]] && EVAL_CAD=50
echo "[plan] budget=${TIME_BUDGET_MIN}min remain~=${REMAIN_SEC}s total=${TOTAL_STEPS} (PT=${PT_STEPS} SFT=${SFT_STEPS}) ~$(( TOTAL_STEPS * FULL_TPS / 1000000 ))M tokens"

launch_ddp() {
    local cfg="$1" ckpt="$2" data="$3" steps="$4" warm="$5" extra=("${@:6}")
    local pids=()
    for i in 0 1; do
        RANK=$i LOCAL_RANK=$i "${BINARY}" --config "$cfg" --device cuda --tokenizer "${TOK}" \
            --data "$data" --max-steps "$steps" --warmup "$warm" \
            --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
            --checkpoint-dir "$ckpt" --resume auto \
            --output-budget-mb "${OUTPUT_BUDGET_MB}" "${extra[@]}" &
        pids[$i]=$!
        echo "  rank $i pid ${pids[$i]}"
    done
    local fail=0
    for pid in "${pids[@]}"; do if ! wait "$pid"; then fail=1; fi; done
    for pid in "${pids[@]}"; do if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fail=1; fi; done
    [[ "$fail" -eq 0 ]] || { echo "[ERROR] a training rank failed, siblings killed"; return 1; }
    return 0
}

echo ""
echo "[stage-A] Pretrain 2xT4 (${PT_STEPS} steps)..."
T0=$(date +%s)
launch_ddp "${CONFIG_PT}" "${CKPT_PT}" "${PT_DIR}" "${PT_STEPS}" "${PT_WARM}"
echo "[stage-A] took $(( ($(date +%s) - T0) / 60 ))m"

echo ""
echo "[stage-B] SFT 2xT4 (${SFT_STEPS} steps)..."
T0=$(date +%s)
if [[ "${EXPORT_GGUF}" -eq 1 ]]; then
    launch_ddp "${CONFIG_SFT}" "${CKPT_SFT}" "${SFT_DIR}" "${SFT_STEPS}" "${SFT_WARM}" --export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}"
else
    launch_ddp "${CONFIG_SFT}" "${CKPT_SFT}" "${SFT_DIR}" "${SFT_STEPS}" "${SFT_WARM}"
fi
echo "[stage-B] took $(( ($(date +%s) - T0) / 60 ))m"

if [[ "${EXPORT_GGUF}" -eq 1 ]] && [[ -f "${GGUF_OUT}" ]]; then
    echo ""
    echo "[smoke] final GGUF self-contained generation..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --persona en --prompt "Hello, who are you? What can you help me with?" --max-tokens 60 || { echo "[smoke] FAIL"; exit 1; }
    echo "[smoke] OK"
fi

echo ""
echo "============================================================"
echo "  Done 2xT4! GGUF: ${GGUF_OUT}"
echo "  Both T4s used via NCCL DDP, per-GPU <=16GB, global tok/s measured."
echo "============================================================"
