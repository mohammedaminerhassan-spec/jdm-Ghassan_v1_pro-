#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SESSION_START=$(date +%s)
session_elapsed() { echo $(( $(date +%s) - SESSION_START )); }
MODE="full"

SESSION_LIMIT_MIN="${SESSION_LIMIT_MIN:-540}"
DEFAULT_BUDGET=$(( SESSION_LIMIT_MIN - 60 ))
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-$DEFAULT_BUDGET}"
[[ "${TIME_BUDGET_MIN}" -gt "$(( SESSION_LIMIT_MIN - 20 ))" ]] && {
    echo "[ERROR] --time-budget-min ${TIME_BUDGET_MIN} exceeds session limit ${SESSION_LIMIT_MIN}."
    exit 1
}

CONFIG_PT="${CONFIG_PT:-${REPO_DIR}/configs/pro_1b_2xt4.yaml}"
CONFIG_SFT="${CONFIG_SFT:-${REPO_DIR}/configs/sft_pro_1b_2xt4.yaml}"

RECIPE_TAG="$(basename "${CONFIG_PT}" .yaml)"
RECIPE_TAG="${RECIPE_TAG#sft_}"
PT_DIR="${PT_DIR:-${REPO_DIR}/artifacts/shards_en}"
SFT_DIR="${SFT_DIR:-${PT_DIR}}"
CKPT_PT="${CKPT_PT:-${REPO_DIR}/artifacts/checkpoints/${RECIPE_TAG}}"
CKPT_SFT="${CKPT_SFT:-${CKPT_PT}_sft}"
GGUF_OUT="${GGUF_OUT:-${REPO_DIR}/artifacts/ghassan-${RECIPE_TAG}_q4_0.gguf}"
TOK="${TOK:-${REPO_DIR}/artifacts/tokenizer/english32k.gtok}"

if [[ -z "${OUTPUT_BUDGET_MB:-}" ]]; then
    case "${CONFIG_PT}" in *1b*) OUTPUT_BUDGET_MB=18432;; *) OUTPUT_BUDGET_MB=17408;; esac
fi
EXPORT_GGUF=1
EXPORT_PROFILE="q4_0"

PILOT_STEPS="${PILOT_STEPS:-}"
if [[ -z "${PILOT_STEPS}" ]]; then
    case "${CONFIG_PT}" in *1b*) PILOT_STEPS=8;; *) PILOT_STEPS=20;; esac
fi

EXPORT_MARGIN_SEC="${EXPORT_MARGIN_SEC:-}"
if [[ -z "${EXPORT_MARGIN_SEC}" ]]; then
    case "${CONFIG_PT}" in *1b*) EXPORT_MARGIN_SEC=1200;; *) EXPORT_MARGIN_SEC=900;; esac
fi
PT_FRACTION=60
SKIP_PREFLIGHT=0

KEEP_PT_CKPTS="${KEEP_PT_CKPTS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot-only) MODE="pilot"; shift ;;
        --preflight) MODE="preflight"; shift ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --time-budget-min) TIME_BUDGET_MIN="$2"; shift 2 ;;
        --no-export) EXPORT_GGUF=0; shift ;;
        --export-profile) EXPORT_PROFILE="$2"; shift 2 ;;
        --pt-fraction) PT_FRACTION="$2"; shift 2 ;;
        --pilot-steps) PILOT_STEPS="$2"; shift 2 ;;
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
echo "  recipe    : ${RECIPE_TAG}"
echo "  PT        : ${CONFIG_PT}"
echo "  SFT       : ${CONFIG_SFT}"
echo "  ckpt PT   : ${CKPT_PT}"
echo "  ckpt SFT  : ${CKPT_SFT}"
echo "  gguf      : ${GGUF_OUT}  (rank 0 only)"
echo "  out budget: ${OUTPUT_BUDGET_MB} MB"

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

[[ -x "${BINARY}" ]] || { echo "[ERROR] ${BINARY} missing. Run: bash configs/kaggle/setup.sh --with-parquet --require-nccl"; exit 1; }
command -v nvidia-smi &>/dev/null || { echo "[ERROR] nvidia-smi missing: 2xT4 needs 2 NVIDIA GPUs."; exit 1; }
GPU_N=$(nvidia-smi -L 2>/dev/null | wc -l)
[[ "${GPU_N}" -ge 2 ]] || { echo "[ERROR] found ${GPU_N} GPU(s), need >=2 for train_2xt4.sh."; exit 1; }
echo "[gpu] found ${GPU_N} GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | awk -F, '{printf "  GPU%s:%s total:%s\n",$1,$2,$3}'

echo "[nccl] verifying NCCL build + rendezvous hygiene..."
if grep -q "GAI_HAVE_NCCL:BOOL=ON" "${REPO_DIR}/build/CMakeCache.txt" 2>/dev/null; then
    echo "[nccl] OK: GAI_HAVE_NCCL=ON"
else
    echo "[ERROR] binary lacks NCCL (single-GPU build). Rebuild: bash configs/kaggle/setup.sh --with-parquet --require-nccl"
    exit 1
fi
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1

export NCCL_COMM_WATCHDOG_TIMEOUT=${NCCL_COMM_WATCHDOG_TIMEOUT:-600}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

export WORLD_SIZE=2
rm -f "/tmp/gai_nccl_${MASTER_PORT}.id"

if [[ -n "${GAI_NCCL_ID_FILE:-}" ]]; then
    rm -f "${GAI_NCCL_ID_FILE}"
    echo "[nccl] custom rendezvous file: ${GAI_NCCL_ID_FILE}"
fi
unset CUDA_VISIBLE_DEVICES
echo "[nccl] WORLD_SIZE=2 MASTER=${MASTER_ADDR}:${MASTER_PORT} (all GPUs visible, rank picks via LOCAL_RANK)"

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

_GAI_RANK_EXITED() {
    local p="$1" st
    [[ -n "${p}" ]] || return 0
    [[ -d "/proc/${p}" ]] || return 0
    st=$(awk '{print $3}' "/proc/${p}/stat" 2>/dev/null) || return 0
    [[ "${st}" == "Z" ]]
}

wait_ddp_ranks() {
    local label="$1" pid0="$2" pid1="$3" grace="${4:-180}"
    local poll=5 first=-1 waited=0 st=0

    while [[ "${first}" -lt 0 ]]; do
        if _GAI_RANK_EXITED "${pid0}"; then first=0; break; fi
        if _GAI_RANK_EXITED "${pid1}"; then first=1; break; fi
        sleep "${poll}"; waited=$(( waited + poll ))
        if [[ "${waited}" -ge 46000 ]]; then
            echo "[${label}] no rank exited after ${waited}s — killing both"
            kill -9 "${pid0}" "${pid1}" 2>/dev/null || true
            return 1
        fi
    done
    local pids=("${pid0}" "${pid1}")
    local exited="${pids[$first]}" survivor="${pids[$((1 - first))]}"
    if wait "${exited}"; then st=0; else st=$?; fi
    if [[ "${st}" -ne 0 ]]; then
        echo "[${label}] rank ${first} (pid ${exited}) exited ${st} — killing sibling ${survivor}"
        kill -9 "${survivor}" 2>/dev/null || true
        wait "${survivor}" 2>/dev/null || true
        return 1
    fi

    waited=0
    while ! _GAI_RANK_EXITED "${survivor}" && [[ "${waited}" -lt "${grace}" ]]; do
        sleep "${poll}"; waited=$(( waited + poll ))
    done
    if ! _GAI_RANK_EXITED "${survivor}"; then
        echo "[${label}] rank ${first} exited but ${survivor} survived ${grace}s"
        echo "[${label}] — collective desync (NCCL hang). Killing it."
        kill -9 "${survivor}" 2>/dev/null || true
        wait "${survivor}" 2>/dev/null || true
        return 1
    fi
    if wait "${survivor}"; then :; else
        echo "[${label}] surviving rank ${survivor} exited non-zero"
        return 1
    fi
    return 0
}

run_preflight() {
    echo ""
    echo "==================== preflight ===================="
    [[ -f "${CONFIG_PT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_PT}"; return 1; }
    [[ -f "${CONFIG_SFT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_SFT}"; return 1; }
    echo "[preflight] configs present"

    SFT_PT_CKPT="$(yget pretrained_checkpoint "${CONFIG_SFT}" || true)"
    if [[ -n "${SFT_PT_CKPT}" ]]; then

        case "${SFT_PT_CKPT}" in
            /*) SFT_PT_CKPT_ABS="${SFT_PT_CKPT}" ;;
            *)  SFT_PT_CKPT_ABS="${REPO_DIR}/${SFT_PT_CKPT}" ;;
        esac
        EXPECT_PT_LCKPT="${CKPT_PT}/last.ckpt"
        if [[ "${SFT_PT_CKPT_ABS}" != "${EXPECT_PT_LCKPT}" ]]; then
            SFT_PT_DIR_ABS="$(dirname "${SFT_PT_CKPT_ABS}")"
            DERIVED_PT_DIR="${REPO_DIR}/artifacts/checkpoints/${RECIPE_TAG}"
            echo "[preflight FAIL] SFT pretrain checkpoint chain mismatch."
            echo "              SFT yaml wants to load : ${SFT_PT_CKPT_ABS}"
            echo "              stage A would write   : ${EXPECT_PT_LCKPT}"
            echo "              stage B would then abort (fail-fast) after the whole pretrain."
            echo "              fix (pick one):"
            echo "                1) drop the CKPT_PT override  -> derived: ${DERIVED_PT_DIR}"
            echo "                2) export CKPT_PT=${SFT_PT_DIR_ABS}"
            echo "                3) set training.pretrained_checkpoint=${SFT_PT_CKPT} in ${CONFIG_SFT}"
            return 1
        fi
        echo "[preflight] SFT->PT checkpoint chain OK (${EXPECT_PT_LCKPT})"
    else
        echo "[preflight WARN] ${CONFIG_SFT} has no training.pretrained_checkpoint"
    fi

    if [[ -x "${TESTCFG_BIN}" ]]; then
        "${TESTCFG_BIN}" > /tmp/preflight_arch.log 2>&1 || { echo "[preflight FAIL] arch parity (see /tmp/preflight_arch.log)"; tail -20 /tmp/preflight_arch.log; return 1; }
        echo "[preflight] arch parity OK (test_configs incl 2xt4 pairs)"
    else
        echo "[preflight WARN] ${TESTCFG_BIN} missing, skipping arch unit gate (runtime SFT gate still active)"
    fi

    [[ -f "${TOK}" ]] || { echo "[preflight FAIL] tokenizer missing: ${TOK}"; return 1; }
    TOK_VOCAB=$("${PIPE_BIN}" tok-info --tokenizer "${TOK}" 2>/dev/null | grep -oE 'vocab_size=[0-9]+' | cut -d= -f2 || true)
    MODEL_VOCAB=$(yget vocab_size "${CONFIG_PT}")
    [[ "${TOK_VOCAB}" == "${MODEL_VOCAB}" && -n "${TOK_VOCAB}" ]] || { echo "[preflight FAIL] tokenizer vocab=${TOK_VOCAB:-?} != model ${MODEL_VOCAB}"; return 1; }
    echo "[preflight] tokenizer vocab ok: ${TOK_VOCAB}"

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

    VRAM_MB=15360
    case "${CONFIG_PT}" in *1b*) VRAM_MB=16384;; esac
    "${BINARY}" --config "${CONFIG_PT}" --dry-run --device cuda --strict-config --output-budget-mb "${OUTPUT_BUDGET_MB}" || return 1
    "${BINARY}" --config "${CONFIG_SFT}" --dry-run --device cuda --strict-config --output-budget-mb "${OUTPUT_BUDGET_MB}" || return 1
    "${BINARY}" --config "${CONFIG_PT}" --dry-run --device cuda --strict-config --max-vram-mb "${VRAM_MB}" --output-budget-mb "${OUTPUT_BUDGET_MB}" || { echo "[preflight FAIL] PT recipe exceeds per-GPU budget (${VRAM_MB}MiB)"; return 1; }
    "${BINARY}" --config "${CONFIG_SFT}" --dry-run --device cuda --strict-config --max-vram-mb "${VRAM_MB}" --output-budget-mb "${OUTPUT_BUDGET_MB}" || { echo "[preflight FAIL] SFT recipe exceeds per-GPU budget (${VRAM_MB}MiB)"; return 1; }
    echo "[preflight] VRAM per-GPU OK (<=$(( VRAM_MB / 1024 ))GiB, preserves B/T envelope)"
    echo "==================== preflight: ALL GATES PASSED ===================="
    return 0
}

if [[ "${MODE}" == "preflight" ]]; then run_preflight; exit $?; fi
if [[ "${SKIP_PREFLIGHT}" -ne 1 ]]; then
    run_preflight || { echo "[ERROR] preflight failed."; exit 1; }
else
    echo "[preflight] SKIPPED via --skip-preflight (NOT advised)"
fi

echo ""
echo "[pilot] 2-rank DDP pilot (${PILOT_STEPS} steps, global tok/s from BOTH ranks)..."
P_START=$(date +%s)
PIDS=()
cleanup_pilot() {
    for pid in "${PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue

        kill -9 "${pid}" 2>/dev/null || true
    done
    PIDS=()
}

trap 'cleanup_pilot; persist_output' EXIT INT TERM

PILOT_CKPT="${PILOT_CKPT:-/tmp/gai_pilot_2xt4}"
rm -rf "${PILOT_CKPT}"; mkdir -p "${PILOT_CKPT}"
for i in 0 1; do
    RANK=$i LOCAL_RANK=$i "${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
        --data "${PT_DIR}" --max-steps "${PILOT_STEPS}" --warmup 0 --resume none \
        --checkpoint-dir "${PILOT_CKPT}" --output-budget-mb "${OUTPUT_BUDGET_MB}" \
        > "/tmp/pilot_2xt4_rank${i}.log" 2>&1 &
    PIDS[$i]=$!
    echo "  started rank $i (pid ${PIDS[$i]}, ckpt -> ${PILOT_CKPT})"
done
FAIL=0

if ! wait_ddp_ranks "pilot" "${PIDS[0]}" "${PIDS[1]}" 120; then
    FAIL=1
fi
[[ "${FAIL}" -eq 0 ]] || { echo "[pilot FAIL] a rank failed:"; tail -30 /tmp/pilot_2xt4_rank*.log; exit 1; }
P_END=$(date +%s)
P_ELAPSED=$(( P_END - P_START )); [[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1

B=$(yget batch_size "${CONFIG_PT}"); T=$(yget seq_len "${CONFIG_PT}"); A=$(yget grad_accum "${CONFIG_PT}")
TOK_PER_STEP_GLOBAL=$(( B * T * A * 2 ))
P_TPS=$(( TOK_PER_STEP_GLOBAL * PILOT_STEPS / P_ELAPSED ))
echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${P_TPS} GLOBAL tok/s (per-GPU B=${B} T=${T} accum=${A} x2 ranks)"
if [[ "${MODE}" == "pilot" ]]; then echo "[pilot] Done."; exit 0; fi

FULL_TPS=${TOK_PER_STEP_GLOBAL}
BUDGET_SEC=$(( TIME_BUDGET_MIN * 60 ))
USED_SEC=$(($(date +%s) - P_START))
REMAIN_SEC=$(( BUDGET_SEC - USED_SEC - EXPORT_MARGIN_SEC ))

SESSION_CAP_SEC=$(( SESSION_LIMIT_MIN * 60 ))
SESSION_USED_SEC=$(session_elapsed)
SESSION_LEFT_SEC=$(( SESSION_CAP_SEC - SESSION_USED_SEC - EXPORT_MARGIN_SEC ))
echo "[time] session used ~$(( SESSION_USED_SEC / 60 ))m, left ~$(( SESSION_LEFT_SEC / 60 ))m of ${SESSION_LIMIT_MIN}m cap"
if [[ "${SESSION_LEFT_SEC}" -lt 1800 ]]; then
    echo "[ERROR] <30min left in the session cap; refusing to start training that cannot finish."
    exit 1
fi
if [[ "${REMAIN_SEC}" -gt "${SESSION_LEFT_SEC}" ]]; then
    echo "[time] training budget capped by session remainder: ${REMAIN_SEC}s -> ${SESSION_LEFT_SEC}s"
    REMAIN_SEC="${SESSION_LEFT_SEC}"
fi
[[ "${REMAIN_SEC}" -lt 600 ]] && { echo "[ERROR] <10min left. Aborting."; exit 1; }
TOTAL_STEPS=$(awk "BEGIN {printf \"%d\", (${REMAIN_SEC} * ${P_TPS}) / ${FULL_TPS}}")
[[ "${TOTAL_STEPS}" -lt 100 ]] && TOTAL_STEPS=100
PT_STEPS=$(( TOTAL_STEPS * PT_FRACTION / 100 ))
SFT_STEPS=$(( TOTAL_STEPS - PT_STEPS ))
[[ "${PT_STEPS}" -lt 50 ]] && PT_STEPS=50
[[ "${SFT_STEPS}" -lt 50 ]] && SFT_STEPS=50

YAML_PT_WARM=$(yget warmup_steps "${CONFIG_PT}");  [[ "${YAML_PT_WARM}" =~ ^[0-9]+$ ]] || YAML_PT_WARM=800
YAML_SFT_WARM=$(yget warmup_steps "${CONFIG_SFT}"); [[ "${YAML_SFT_WARM}" =~ ^[0-9]+$ ]] || YAML_SFT_WARM=200
PT_WARM=$(( PT_STEPS / 10 ));  [[ "${PT_WARM}" -gt "${YAML_PT_WARM}" ]] && PT_WARM="${YAML_PT_WARM}"
SFT_WARM=$(( SFT_STEPS / 10 )); [[ "${SFT_WARM}" -gt "${YAML_SFT_WARM}" ]] && SFT_WARM="${YAML_SFT_WARM}"

PT_FLOOR=$(( PT_STEPS / 12 )); [[ "${PT_FLOOR}" -lt 20 ]] && PT_FLOOR=20
SFT_FLOOR=$(( SFT_STEPS / 12 )); [[ "${SFT_FLOOR}" -lt 15 ]] && SFT_FLOOR=15
[[ "${PT_WARM}" -lt "${PT_FLOOR}" ]] && PT_WARM="${PT_FLOOR}"
[[ "${SFT_WARM}" -lt "${SFT_FLOOR}" ]] && SFT_WARM="${SFT_FLOOR}"

[[ "${PT_WARM}" -ge "${PT_STEPS}" ]] && PT_WARM=$(( PT_STEPS > 1 ? PT_STEPS - 1 : 1 ))
[[ "${SFT_WARM}" -ge "${SFT_STEPS}" ]] && SFT_WARM=$(( SFT_STEPS > 1 ? SFT_STEPS - 1 : 1 ))
echo "[plan] warmup PT=${PT_WARM}/${PT_STEPS} (yaml ${YAML_PT_WARM}) SFT=${SFT_WARM}/${SFT_STEPS} (yaml ${YAML_SFT_WARM})"

EVAL_CAD=$(( TOTAL_STEPS / 8 )); [[ "${EVAL_CAD}" -lt 25 ]] && EVAL_CAD=25
[[ "${EVAL_CAD}" -ge "${TOTAL_STEPS}" ]] && EVAL_CAD=$(( TOTAL_STEPS > 1 ? TOTAL_STEPS - 1 : 1 ))
echo "[plan] budget=${TIME_BUDGET_MIN}min remain~=${REMAIN_SEC}s total=${TOTAL_STEPS} (PT=${PT_STEPS} SFT=${SFT_STEPS}) ~$(( TOTAL_STEPS * FULL_TPS / 1000000 ))M tokens"

DATASET_TOKENS=0
if [[ -d "${PT_DIR}" ]]; then
    DATASET_TOKENS=$("${PIPE_BIN}" inspect --shards "${PT_DIR}" 2>/dev/null \
        | awk '/^  total: /{v=$2; m=1;
               if (v ~ /K$/) m=1000; else if (v ~ /M$/) m=1000000;
               else if (v ~ /B$/) m=1000000000; else if (v ~ /T$/) m=1000000000000;
               gsub(/[^0-9.]/,"",v); t+=v*m} END{printf "%.0f", t+0}' || echo 0)
fi
if [[ "${DATASET_TOKENS}" -gt 0 ]]; then
    PT_TOKENS=$(( PT_STEPS * FULL_TPS ))
    EPOCHS=$(awk "BEGIN {printf \"%.2f\", ${PT_TOKENS}/${DATASET_TOKENS}}")
    echo "[data] corpus ~${DATASET_TOKENS} tokens; PT plan ${PT_TOKENS} tokens = ${EPOCHS} epoch(s)"
    if awk "BEGIN {exit !(${EPOCHS} > 4.0)}"; then
        echo "[plan] NOTE: >4 epochs over this corpus. More steps will overfit it;"
        echo "       raise --pt-fraction (less SFT) or accept a domain-specialised model."
    fi
else
    echo "[data] corpus token count unavailable (inspect failed) — epoch coverage unknown"
fi

launch_ddp() {
    local cfg="$1" ckpt="$2" data="$3" steps="$4" warm="$5" extra=("${@:6}")
    local pids=()
    for i in 0 1; do
        local rank_extra=()

        if [[ "${EXPORT_GGUF}" -eq 1 && "${i}" -eq 0 ]]; then
            rank_extra=(--export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}")
        fi
        RANK=$i LOCAL_RANK=$i "${BINARY}" --config "$cfg" --device cuda --tokenizer "${TOK}" \
            --data "$data" --max-steps "$steps" --warmup "$warm" \
            --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
            --checkpoint-dir "$ckpt" --resume auto \
            --output-budget-mb "${OUTPUT_BUDGET_MB}" \
            "${extra[@]}" "${rank_extra[@]}" &
        pids[$i]=$!
        echo "  rank $i pid ${pids[$i]}"
    done

    local grace=$(( EXPORT_MARGIN_SEC + 300 ))
    [[ "${EXPORT_GGUF}" -eq 0 ]] && grace=300

    if ! wait_ddp_ranks "train" "${pids[0]}" "${pids[1]}" "${grace}"; then
        for pid in "${pids[@]}"; do kill -9 "${pid}" 2>/dev/null || true; done
        echo "[ERROR] a training rank failed or hung; siblings killed"
        return 1
    fi
    return 0
}

echo ""
echo "[stage-A] Pretrain 2xT4 (${PT_STEPS} steps)..."
T0=$(date +%s)
launch_ddp "${CONFIG_PT}" "${CKPT_PT}" "${PT_DIR}" "${PT_STEPS}" "${PT_WARM}"
echo "[stage-A] took $(( ($(date +%s) - T0) / 60 ))m"

if [[ -f "${CKPT_PT}/best.ckpt" && "${KEEP_PT_CKPTS}" -eq 0 ]]; then
    PT_BEST_SZ=$(du -m "${CKPT_PT}/best.ckpt" 2>/dev/null | cut -f1)
    rm -f "${CKPT_PT}/best.ckpt" && echo "[clean] removed ${CKPT_PT}/best.ckpt (${PT_BEST_SZ} MB)"
fi

echo ""
echo "[stage-B] SFT 2xT4 (${SFT_STEPS} steps)..."
T0=$(date +%s)
launch_ddp "${CONFIG_SFT}" "${CKPT_SFT}" "${SFT_DIR}" "${SFT_STEPS}" "${SFT_WARM}"
echo "[stage-B] took $(( ($(date +%s) - T0) / 60 ))m"

if [[ -d "${CKPT_PT}" && "${KEEP_PT_CKPTS}" -eq 0 ]]; then
    PT_SZ=$(du -sm "${CKPT_PT}" 2>/dev/null | cut -f1)
    rm -rf "${CKPT_PT}" && echo "[clean] removed pretrain ckpt dir (${PT_SZ} MB) — SFT holds the weights"
fi

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
