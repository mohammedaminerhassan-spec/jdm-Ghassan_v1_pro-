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
#   bash kaggle/train_2xt4.sh                                   # full PT+SFT (1B default)
#   bash kaggle/train_2xt4.sh --preflight                       # gates only, no training
#   bash kaggle/train_2xt4.sh --pilot-only                       # 2-rank pilot, measures global tok/s
#   bash kaggle/train_2xt4.sh --time-budget-min 500 --pt-fraction 60
#   CONFIG_PT=configs/en_2xt4.yaml bash kaggle/train_2xt4.sh    # 480M family
#   bash kaggle/train_2xt4.sh --time-budget-min 240             # short quota (~4h train)
#
# Everything downstream of the recipe (checkpoint dirs, GGUF name, VRAM gate,
# output budget) is DERIVED from the config basename, so a recipe pair can
# never be wired to the wrong checkpoint dir.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# GLOBAL WALL CLOCK. Every budget below derives from time REMAINING in the
# Kaggle session, not from when a phase happens to start: setup + data +
# preflight + pilot all burn the same 12h limit as training.
SESSION_START=$(date +%s)
session_elapsed() { echo $(( $(date +%s) - SESSION_START )); }
MODE="full"
# SINGLE-SESSION BUDGET. Kaggle caps a GPU session at 12h, and the goal is to
# finish PT + SFT + export INSIDE ONE session, so the plan is built on 8h
# (480 min) and refuses anything above the session limit. Raise deliberately:
#   SESSION_LIMIT_MIN=700 TIME_BUDGET_MIN=620 bash kaggle/train_2xt4.sh
# The budget covers training only: setup/data/preflight/pilot run before it.
SESSION_LIMIT_MIN="${SESSION_LIMIT_MIN:-540}"   # 9h: hard ceiling incl. overhead
DEFAULT_BUDGET=$(( SESSION_LIMIT_MIN - 60 ))     # 480 min = 8h of training
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-$DEFAULT_BUDGET}"
[[ "${TIME_BUDGET_MIN}" -gt "$(( SESSION_LIMIT_MIN - 20 ))" ]] && {
    echo "[ERROR] --time-budget-min ${TIME_BUDGET_MIN} exceeds session limit ${SESSION_LIMIT_MIN}."
    exit 1
}
# DEFAULT RECIPE = the 1B family (pro_1b_2xt4: ~1.04B total / ~612M active).
# Override CONFIG_PT (+ CONFIG_SFT) to run another family; the derived paths
# below follow automatically.
CONFIG_PT="${CONFIG_PT:-${REPO_DIR}/configs/pro_1b_2xt4.yaml}"
CONFIG_SFT="${CONFIG_SFT:-${REPO_DIR}/configs/sft_pro_1b_2xt4.yaml}"
# Derive the recipe tag from the PT config basename (pro_1b_2xt4 / en_2xt4 / ...).
RECIPE_TAG="$(basename "${CONFIG_PT}" .yaml)"
RECIPE_TAG="${RECIPE_TAG#sft_}"   # sft_ prefix carries no extra information here
PT_DIR="${PT_DIR:-${REPO_DIR}/artifacts/shards_en}"
SFT_DIR="${SFT_DIR:-${PT_DIR}}"
CKPT_PT="${CKPT_PT:-${REPO_DIR}/artifacts/checkpoints/${RECIPE_TAG}}"
CKPT_SFT="${CKPT_SFT:-${CKPT_PT}_sft}"
GGUF_OUT="${GGUF_OUT:-${REPO_DIR}/artifacts/ghassan-${RECIPE_TAG}_q4_0.gguf}"
TOK="${TOK:-${REPO_DIR}/artifacts/tokenizer/english32k.gtok}"
# Kaggle /kaggle/working cap is 20GB (build+data+train). 480M recipes project
# ~11.3GB; 1B recipes ~16.0GB (2 snapshots + GGUF), so they get 18GB. Both stay
# under the panel cap. An explicit OUTPUT_BUDGET_MB always wins.
if [[ -z "${OUTPUT_BUDGET_MB:-}" ]]; then
    case "${CONFIG_PT}" in *1b*|*pro_v1*|*t4_1b*) OUTPUT_BUDGET_MB=18432;; *) OUTPUT_BUDGET_MB=17408;; esac
fi
EXPORT_GGUF=1
EXPORT_PROFILE="q4_0"
# PILOT cost scales with grad_accum (1B: 64 micros/step). 20 pilot steps on 1B
# would burn ~30-45min of quota just measuring; 8 steps (512 micros) measures
# the same steady-state tok/s. Explicit PILOT_STEPS (or --pilot-steps) wins.
PILOT_STEPS="${PILOT_STEPS:-}"
if [[ -z "${PILOT_STEPS}" ]]; then
    case "${CONFIG_PT}" in *1b*|*pro_v1*|*t4_1b*) PILOT_STEPS=8;; *) PILOT_STEPS=20;; esac
fi
# Export margin: q4_0 quant of 1B params + 8GB writes on 4 Kaggle cores needs
# more than the 480M's 15min. Recipe-aware so the plan never eats the export.
EXPORT_MARGIN_SEC="${EXPORT_MARGIN_SEC:-}"
if [[ -z "${EXPORT_MARGIN_SEC}" ]]; then
    case "${CONFIG_PT}" in *1b*|*pro_v1*|*t4_1b*) EXPORT_MARGIN_SEC=1200;; *) EXPORT_MARGIN_SEC=900;; esac
fi
PT_FRACTION=60
SKIP_PREFLIGHT=0
# Keep the pretrain stage's checkpoints after SFT. Default 0: SFT already holds
# the trained weights and the GGUF is the deliverable, and a 480M AdamW snapshot
# is 5.8 GB that Kaggle's 20 GB saved-output cap cannot afford to keep twice.
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
    # /kaggle/working itself is what "Save Version" persists. Sources already
    # inside it must NOT be copied into output/ — that stores every byte TWICE
    # against the 20GB quota (a successful train could then fail at Save).
    # Only sources from outside (rare; the repo lives under working on Kaggle)
    # are copied.
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
# Belt-and-braces on top of wait_ddp_ranks(): if a rank ever wedges INSIDE a
# collective, NCCL's own watchdog aborts the job instead of parking the
# survivor in cudaStreamSynchronize until Kaggle kills the whole session.
# NCCL_COMM_WATCHDOG_TIMEOUT is the real variable (seconds); there is no
# NCCL_TIMEOUT / NCCL_ASYNC_ERROR_HANDLING in modern NCCL.
export NCCL_COMM_WATCHDOG_TIMEOUT=${NCCL_COMM_WATCHDOG_TIMEOUT:-600}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
# Concurrent sessions on one host: override MASTER_PORT per run
# (e.g. MASTER_PORT=29501 bash kaggle/train_2xt4.sh) and optionally
# GAI_NCCL_ID_FILE=/tmp/gai_nccl_custom.id to avoid rendezvous collision.
export WORLD_SIZE=2
rm -f "/tmp/gai_nccl_${MASTER_PORT}.id"
# Also honor a custom rendezvous file if provided.
if [[ -n "${GAI_NCCL_ID_FILE:-}" ]]; then
    rm -f "${GAI_NCCL_ID_FILE}"
    echo "[nccl] custom rendezvous file: ${GAI_NCCL_ID_FILE}"
fi
unset CUDA_VISIBLE_DEVICES
echo "[nccl] WORLD_SIZE=2 MASTER=${MASTER_ADDR}:${MASTER_PORT} (all GPUs visible, rank picks via LOCAL_RANK)"

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }

# ---------------- 2-RANK WATCHDOG ------------------------------------------
# A dead rank leaves its sibling blocked FOREVER: training/distributed.cpp has
# no NCCL timeout, so the survivor spins in cudaStreamSynchronize until Kaggle
# kills the whole 12h session. A sequential `wait pid0; wait pid1` cannot save
# it — wait(pid1) never returns, so the kill-orphans loop after it is
# unreachable dead code. Instead reap the FIRST rank to exit, then give the
# survivor a bounded grace window and SIGKILL it if it outlives the window.
#
# wait_ddp_ranks <label> <pid0> <pid1> [grace_seconds]
#   0 = both ranks exited 0.  1 = a rank failed, or one had to be killed.
# Uses `wait -n -p PID` (bash >= 5.1, which is what Kaggle ships) so we learn
# WHICH rank finished. Without -p we cannot tell, so we conservatively treat
# the first exit as a failure and kill the survivor immediately.
# ---------------- 2-RANK WATCHDOG ------------------------------------------
# A dead rank leaves its sibling blocked FOREVER: training/distributed.cpp has
# no NCCL timeout, so the survivor spins in cudaStreamSynchronize until Kaggle
# kills the whole 12h session. A sequential `wait pid0; wait pid1` cannot save
# it — wait(pid1) never returns, so the kill-orphans loop after it is dead
# code. Reap the FIRST rank to exit, then bound how long the survivor may take.
#
# This deliberately does NOT use `wait -n -p PID`. Bash clears the -p variable
# when there is no child left to wait for, and under `set -u` reading it then
# aborts the script with "PID: unbound variable" — which is exactly what
# happened on the first pilot run: the ranks had already exited, `wait -n`
# reaped nothing, and the shell died leaving both ranks orphaned and burning
# the GPU for the rest of the session. Polling /proc for the zombie state is
# exact, has no bash-version caveats, and cannot abort the script.
#
# 0 = both ranks exited 0.  1 = a rank failed, or one had to be killed.
_GAI_RANK_EXITED() {
    local p="$1" st
    [[ -n "${p}" ]] || return 0
    [[ -d "/proc/${p}" ]] || return 0                 # process gone
    st=$(awk '{print $3}' "/proc/${p}/stat" 2>/dev/null) || return 0
    [[ "${st}" == "Z" ]]                             # zombie == exited, unreaped
}

wait_ddp_ranks() {
    local label="$1" pid0="$2" pid1="$3" grace="${4:-180}"
    local poll=5 first=-1 waited=0 st=0
    # Bound far above any session: the first rank only exits when the whole
    # stage finishes, which for a full budgeted run is hours.
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
    # The sibling runs the same collectives and should follow within seconds.
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
    # SFT->PT CHECKPOINT CHAIN GATE. The SFT yaml names the pretrain
    # checkpoint it must load; if that path is not the directory stage A
    # actually writes, stage B dies AFTER the whole pretrain. Fail here with
    # the exact fix instead of burning the session.
    SFT_PT_CKPT="$(yget pretrained_checkpoint "${CONFIG_SFT}" || true)"
    if [[ -n "${SFT_PT_CKPT}" ]]; then
        # The yaml carries a repo-relative path; CKPT_PT is absolute. Resolve
        # to the same form before comparing, else a correct pair looks broken.
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
    # VRAM plan per-GPU (arithmetic) + hard gate (16GB minus headroom).
    # --strict-config: unknown/dead keys fail here, never mid-run.
    # Forward the output quota so the preflight gate matches the live guard.
    # 1B recipes peak honestly at ~14.4GB, so they use the full-16GB gate;
    # the 480M recipes keep the stricter 15GiB gate (their envelope is smaller).
    VRAM_MB=15360
    case "${CONFIG_PT}" in *1b*|*pro_v1*|*t4_1b*) VRAM_MB=16384;; esac
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

# ---------------- 2-RANK PILOT (measures GLOBAL throughput) ----------------
echo ""
echo "[pilot] 2-rank DDP pilot (${PILOT_STEPS} steps, global tok/s from BOTH ranks)..."
P_START=$(date +%s)
PIDS=()
cleanup_pilot() {
    for pid in "${PIDS[@]:-}"; do
        [[ -n "${pid}" ]] || continue
        # -9, not TERM: a rank wedged inside a NCCL collective ignores SIGTERM
        # and would keep holding its 15 GB card until the session ends.
        kill -9 "${pid}" 2>/dev/null || true
    done
    PIDS=()
}
# EXIT matters as much as INT/TERM here. The pilot launched its ranks as
# background children, so any early exit of the script itself (a `set -u`
# abort, a failed gate, an unhandled error) used to leave both ranks running
# as orphans, burning the GPU for the rest of the session with nobody waiting
# for them. Trap every path and make the handler idempotent.
trap 'cleanup_pilot; persist_output' EXIT INT TERM
# The pilot writes to a THROWAWAY checkpoint dir, never ${CKPT_PT}. The trainer
# always publishes an unconditional last.ckpt when it finishes (trainer.cpp
# run_pretrain tail), so pointing the pilot at the real dir left a step-N
# last.ckpt behind; stage A then started with `--resume auto`, inherited those
# N steps (trained at --warmup 0, i.e. full peak LR on a random init) and
# logged a bogus "[sched] RESUME MISMATCH". It also spent real output quota on
# a throwaway snapshot. /tmp is outside the /kaggle/working quota root.
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
# The pilot is small and symmetric, so the survivor gets only a short grace:
# if it has not exited shortly after its sibling, it is wedged in NCCL.
if ! wait_ddp_ranks "pilot" "${PIDS[0]}" "${PIDS[1]}" 120; then
    FAIL=1
fi
[[ "${FAIL}" -eq 0 ]] || { echo "[pilot FAIL] a rank failed:"; tail -30 /tmp/pilot_2xt4_rank*.log; exit 1; }
P_END=$(date +%s)
P_ELAPSED=$(( P_END - P_START )); [[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1
# Global tokens/step INCLUDES world_size (DDP contract, tested in code).
B=$(yget batch_size "${CONFIG_PT}"); T=$(yget seq_len "${CONFIG_PT}"); A=$(yget grad_accum "${CONFIG_PT}")
TOK_PER_STEP_GLOBAL=$(( B * T * A * 2 ))
P_TPS=$(( TOK_PER_STEP_GLOBAL * PILOT_STEPS / P_ELAPSED ))
echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${P_TPS} GLOBAL tok/s (per-GPU B=${B} T=${T} accum=${A} x2 ranks)"
if [[ "${MODE}" == "pilot" ]]; then echo "[pilot] Done."; exit 0; fi

FULL_TPS=${TOK_PER_STEP_GLOBAL}
BUDGET_SEC=$(( TIME_BUDGET_MIN * 60 ))
USED_SEC=$(($(date +%s) - P_START))
REMAIN_SEC=$(( BUDGET_SEC - USED_SEC - EXPORT_MARGIN_SEC ))
# Cap by the ACTUAL session remainder: TIME_BUDGET is training-only, but setup
# + data + preflight + pilot already burned session wall-clock. Without this
# cap a slow setup phase pushes PT+SFT past the 12h Kaggle kill.
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
# Warmup follows the YAML (stability-critical: 1B+Lion was validated at 1500,
# the old 800 cap silently shortened it). Short runs scale down to steps/10.
YAML_PT_WARM=$(yget warmup_steps "${CONFIG_PT}");  [[ "${YAML_PT_WARM}" =~ ^[0-9]+$ ]] || YAML_PT_WARM=800
YAML_SFT_WARM=$(yget warmup_steps "${CONFIG_SFT}"); [[ "${YAML_SFT_WARM}" =~ ^[0-9]+$ ]] || YAML_SFT_WARM=200
PT_WARM=$(( PT_STEPS / 10 ));  [[ "${PT_WARM}" -gt "${YAML_PT_WARM}" ]] && PT_WARM="${YAML_PT_WARM}"
SFT_WARM=$(( SFT_STEPS / 10 )); [[ "${SFT_WARM}" -gt "${YAML_SFT_WARM}" ]] && SFT_WARM="${YAML_SFT_WARM}"
# A 50-step floor is a FLOOR, not a target: when the whole stage is only a few
# hundred steps, a 50-step warmup is 10-20% of it, but on a 200-step stage it
# would be a quarter of the run spent ramping the LR. Cap it at 8% of the
# stage so a slow pilot cannot turn the entire budget into warmup and leave the
# model barely trained (which is exactly what happened at 1000 tok/s).
PT_FLOOR=$(( PT_STEPS / 12 )); [[ "${PT_FLOOR}" -lt 20 ]] && PT_FLOOR=20
SFT_FLOOR=$(( SFT_STEPS / 12 )); [[ "${SFT_FLOOR}" -lt 15 ]] && SFT_FLOOR=15
[[ "${PT_WARM}" -lt "${PT_FLOOR}" ]] && PT_WARM="${PT_FLOOR}"
[[ "${SFT_WARM}" -lt "${SFT_FLOOR}" ]] && SFT_WARM="${SFT_FLOOR}"
# And never let warmup eat the stage: a warmup at or past the total would
# never reach peak LR at all.
[[ "${PT_WARM}" -ge "${PT_STEPS}" ]] && PT_WARM=$(( PT_STEPS > 1 ? PT_STEPS - 1 : 1 ))
[[ "${SFT_WARM}" -ge "${SFT_STEPS}" ]] && SFT_WARM=$(( SFT_STEPS > 1 ? SFT_STEPS - 1 : 1 ))
echo "[plan] warmup PT=${PT_WARM}/${PT_STEPS} (yaml ${YAML_PT_WARM}) SFT=${SFT_WARM}/${SFT_STEPS} (yaml ${YAML_SFT_WARM})"
# Same floor-vs-target problem as warmup: eval_every=50 on a 300-step stage
# evaluates 6 times, which is fine, but on a 150-step stage it would evaluate
# every third step and spend real time on validation instead of training. One
# eval per 12% of the run, with a small floor.
EVAL_CAD=$(( TOTAL_STEPS / 8 )); [[ "${EVAL_CAD}" -lt 25 ]] && EVAL_CAD=25
[[ "${EVAL_CAD}" -ge "${TOTAL_STEPS}" ]] && EVAL_CAD=$(( TOTAL_STEPS > 1 ? TOTAL_STEPS - 1 : 1 ))
echo "[plan] budget=${TIME_BUDGET_MIN}min remain~=${REMAIN_SEC}s total=${TOTAL_STEPS} (PT=${PT_STEPS} SFT=${SFT_STEPS}) ~$(( TOTAL_STEPS * FULL_TPS / 1000000 ))M tokens"

# How much of the corpus does that plan actually cover? The decision "is one
# 8h session enough" depends on it: more than ~2 epochs means we are fitting
# the data, not learning from it. Parsed from `data_pipeline inspect`
# ("  total: N tokens", human_count style: 850 / 1.20K / 427.76M / 1.04B).
# Best-effort only (never fails the run).
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
        # EXPORT ON RANK 0 ONLY. train_main.cpp exports unconditionally (no
        # rank gate), so passing --export to both ranks made two processes
        # write the SAME gguf path concurrently -> truncated/corrupt file.
        # Safe: the export runs after trainer.run(), i.e. after every
        # collective, so rank 1 simply finishes and exits.
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
    # Rank 0 also writes the GGUF AFTER the last collective, so it legitimately
    # outlives rank 1 by the export time. The survivor grace must cover that,
    # or the watchdog kills a healthy rank 0 mid-export and the run "fails"
    # after the checkpoint is already on disk.
    local grace=$(( EXPORT_MARGIN_SEC + 300 ))
    [[ "${EXPORT_GGUF}" -eq 0 ]] && grace=300
    # Reap on every path: if this function returns early (watchdog failure) or
    # the whole script aborts, a live rank would keep its 15 GB card and the
    # NEXT stage would then fail its own VRAM or quota gate. The pilot's
    # cleanup_pilot covers PIDS; training ranks are tracked separately.
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

# Drop the pretrain stage's best.ckpt before SFT starts. SFT only ever reads
# last.ckpt (training.pretrained_checkpoint), and keeping a second full
# snapshot doubled the bytes on disk for nothing — enough on its own to push
# stage B past the output quota on a single session.
if [[ -f "${CKPT_PT}/best.ckpt" && "${KEEP_PT_CKPTS}" -eq 0 ]]; then
    PT_BEST_SZ=$(du -m "${CKPT_PT}/best.ckpt" 2>/dev/null | cut -f1)
    rm -f "${CKPT_PT}/best.ckpt" && echo "[clean] removed ${CKPT_PT}/best.ckpt (${PT_BEST_SZ} MB)"
fi

echo ""
echo "[stage-B] SFT 2xT4 (${SFT_STEPS} steps)..."
T0=$(date +%s)
launch_ddp "${CONFIG_SFT}" "${CKPT_SFT}" "${SFT_DIR}" "${SFT_STEPS}" "${SFT_WARM}"
echo "[stage-B] took $(( ($(date +%s) - T0) / 60 ))m"

# The SFT ranks have loaded the pretrain weights into GPU memory by now, so
# the whole pretrain checkpoint directory is dead weight for the rest of the
# session. Freeing it here is what lets the final saved output hold the SFT
# checkpoints AND the GGUF inside Kaggle's cap.
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
