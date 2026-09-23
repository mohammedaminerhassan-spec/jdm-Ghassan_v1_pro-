#!/usr/bin/env bash
# kaggle/train_1b.sh — Ghassan v1 Pro English two-stage training (Kaggle GPU, multi-session safe)
#
#   Stage A (pretrain): configs/en_pro.yaml on artifacts/shards_en (English lake)
#   Stage B (sft)     : configs/sft_en_pro.yaml on artifacts/shards_en
#   -> GGUF export -> English smoke test. WSD scheduler + resume=auto so you can
#   stop/resume across Kaggle sessions until loss converges.
#
# Usage:
#   bash kaggle/train_1b.sh                                   # 9h budget default
#   bash kaggle/train_1b.sh --time-budget-min 500
#   bash kaggle/train_1b.sh --pilot-only
#   bash kaggle/train_1b.sh --no-export
#   bash kaggle/train_1b.sh --preflight      # gates only, no training
#   bash kaggle/train_1b.sh --skip-preflight # skip auto-gates (not advised)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="full"
# Overridable recipe (Pro 1B run shown below; defaults = Pro 480M English).
# Example:
#   CONFIG_PT=configs/pro_v1.yaml CONFIG_SFT=configs/sft_pro_v1.yaml \
#   PT_DIR=artifacts/shards_en SFT_DIR=artifacts/shards_en \
#   CKPT_PT=artifacts/checkpoints/pro_v1 CKPT_SFT=artifacts/checkpoints/pro_v1_sft \
#   GGUF_OUT=artifacts/ghassan-v1-pro-1b_q4_0.gguf \
#   bash kaggle/train_1b.sh --time-budget-min 420 --export-profile q4_0 --pt-fraction 70
CONFIG_PT="${CONFIG_PT:-${REPO_DIR}/configs/en_pro.yaml}"
CONFIG_SFT="${CONFIG_SFT:-${REPO_DIR}/configs/sft_en_pro.yaml}"
TIME_BUDGET_MIN=540
EXPORT_GGUF=1
EXPORT_PROFILE="q4_0"
PILOT_STEPS=20
EXPORT_MARGIN_SEC=900
PT_FRACTION=60

SKIP_PREFLIGHT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot-only)      MODE="pilot";   shift ;;
        --preflight)       MODE="preflight"; shift ;;
        --skip-preflight)  SKIP_PREFLIGHT=1; shift ;;
        --time-budget-min) TIME_BUDGET_MIN="$2"; shift 2 ;;
        --no-export)       EXPORT_GGUF=0;  shift ;;
        --export-profile)  EXPORT_PROFILE="$2"; shift 2 ;;
        --pt-fraction)     PT_FRACTION="$2"; shift 2 ;;
        *) shift ;;
    esac
done

BINARY="${REPO_DIR}/build/bin/gai_train"
GEN_BIN="${REPO_DIR}/build/bin/ghassan-ai"
TOK="${TOK:-${REPO_DIR}/artifacts/tokenizer/english32k.gtok}"
# PRO-HARDEN: fallback الصامت إلى 16k كان يضيع run كاملا ثم يفشل عند البوابة.
# نفشل فورا إن غاب 32k (setup.sh يدربه تلقائيا من English lake).
if [[ ! -f "${TOK}" ]]; then
    echo "[ERROR] 32k tokenizer missing: ${TOK} (legacy 16k fallback DISABLED: it would waste embeddings)."
    echo "[ERROR] Run: bash kaggle/setup.sh --with-parquet  (trains english32k.gtok automatically)"
    exit 1
fi
PT_DIR="${PT_DIR:-${REPO_DIR}/artifacts/shards_en}"
SFT_DIR="${SFT_DIR:-${REPO_DIR}/artifacts/shards_en}"
CKPT_PT="${CKPT_PT:-${REPO_DIR}/artifacts/checkpoints/en_pro}"
CKPT_SFT="${CKPT_SFT:-${REPO_DIR}/artifacts/checkpoints/en_pro_sft}"
GGUF_OUT="${GGUF_OUT:-${REPO_DIR}/artifacts/ghassan-v1-pro_${EXPORT_PROFILE}.gguf}"

echo "============================================================"
echo "  Ghassan v1 Pro English — Two-Stage Training [${MODE}]"
echo "  FLAGSHIP single-T4 recipe: ~480M total / ~204M active MoE | WSD + resume"
echo "============================================================"

# PRO-HARDEN: حفظ checkpoints/GGUF في /kaggle/working/output (دائم) لا في
# artifacts المؤقتة فقط. trap عند EXIT/INT/TERM ينسخ آخر حالة حتى لو
# أوقفت Kaggle الجلسة — بلا هذا يضيع 9h تدريب.
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

[[ -f "${BINARY}" ]] || { echo "[ERROR] gai_train missing. Run setup.sh first."; exit 1; }
command -v nvidia-smi &>/dev/null || { echo "[ERROR] No GPU. Kaggle: Settings -> Accelerator -> GPU."; exit 1; }
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2>/dev/null | \
    awk -F, '{printf "  %s | free: %s | total: %s\n", $1,$2,$3}'

[[ -f "${TOK}" ]] || { echo "[ERROR] Tokenizer missing: ${TOK}"; exit 1; }
PT_SHARDS=$(find "${PT_DIR}" -name "train_*.gbin" 2>/dev/null | wc -l)
SFT_SHARDS=$(find "${SFT_DIR}" -name "train_*.gbin" 2>/dev/null | wc -l)
[[ "${PT_SHARDS}" -gt 0 ]] || { echo "[ERROR] No pretrain shards in ${PT_DIR}. Run build_english_data.sh first."; exit 1; }
[[ "${SFT_SHARDS}" -gt 0 ]] || { echo "[ERROR] No SFT shards in ${SFT_DIR}. Run build_english_data.sh first."; exit 1; }
echo "[data] pretrain shards: ${PT_SHARDS} | sft shards: ${SFT_SHARDS}"
# DISK FIT (19.5GB Kaggle): Pro ckpt is ~4GB each (last+best=8GB per stage).
# Auto-clean intermediates that are never needed during training, then guard.
echo "[disk] before training:"; df -h "${REPO_DIR}" | tail -n 1
rm -rf "${REPO_DIR}/artifacts/corpus" "${REPO_DIR}/artifacts/synth" 2>/dev/null || true
# If shards_en still has a leftover merged jsonl, it is dead weight now.
find "${REPO_DIR}/artifacts" -maxdepth 2 -name "synthetic_*.jsonl" -delete 2>/dev/null || true
FREE_KB=$(df "${REPO_DIR}" | awk 'NR==2{print $4}')
# Need ~10GB free for 2x Pro ckpt + GGUF export temp. Fail fast, never mid-run OOM.
if [[ "${FREE_KB}" -lt 10485760 ]]; then
  echo "[ERROR] Disk too full for Pro training (free <10GB). 19.5GB Kaggle needs:"
  echo "  rm -rf artifacts/corpus artifacts/synth <merged jsonl> (done above)"
  echo "  Keep only shards + tokenizer. Parquet source can be detached after sharding."
  echo "  Current:"; df -h "${REPO_DIR}" | tail -n 1; du -sh "${REPO_DIR}/artifacts/"* 2>/dev/null || true
  exit 1
fi

yget() { grep -E "^[[:space:]]*$1:" "$2" | head -n 1 | sed -e 's/^[^:]*:[[:space:]]*//' -e 's/[[:space:]]*#.*$//' -e 's/^[[:space:]]*//;s/[[:space:]]*$//'; }
mkdir -p "${CKPT_PT}" "${CKPT_SFT}"
PIPE_BIN="${REPO_DIR}/build/bin/data_pipeline"

# ---------------------------------------------------------------- preflight
# Phase-3 gates that run WITHOUT a GPU: config/tokenizer/shard truth BEFORE
# any GPU hour burns. A declared mix domain with zero shards, a wrong-vocab
# tokenizer, or a memory estimate over budget FAILS here — never mid-run.
# (CUDA gates — parity test, nvidia-smi, real tok/s — stay in setup.sh and
# the pilot, which genuinely need the GPU.)
run_preflight() {
    echo ""
    echo "==================== preflight ===================="
    # P1. configs exist
    [[ -f "${CONFIG_PT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_PT}"; return 1; }
    [[ -f "${CONFIG_SFT}" ]] || { echo "[preflight FAIL] missing ${CONFIG_SFT}"; return 1; }
    echo "[preflight] configs present"
    # P2. tokenizer file is 32k AND matches the model vocab (never legacy 16k)
    [[ -f "${TOK}" ]] || { echo "[preflight FAIL] tokenizer missing: ${TOK}"; return 1; }
    TOK_VOCAB=$("${PIPE_BIN}" tok-info --tokenizer "${TOK}" 2>/dev/null \
        | grep -oE 'vocab_size=[0-9]+' | cut -d= -f2 || true)
    MODEL_VOCAB=$(yget vocab_size "${CONFIG_PT}")
    [[ "${TOK_VOCAB}" == "${MODEL_VOCAB}" && -n "${TOK_VOCAB}" ]] || {
        echo "[preflight FAIL] tokenizer vocab=${TOK_VOCAB:-unreadable} != model vocab=${MODEL_VOCAB}"
        return 1
    }
    echo "[preflight] tokenizer vocab ok: ${TOK_VOCAB}"
    # P3. shard dirs non-empty
    for d in "${PT_DIR}" "${SFT_DIR}"; do
        n=$(find "$d" -name "train_*.gbin" 2>/dev/null | wc -l)
        [[ "$n" -gt 0 ]] || { echo "[preflight FAIL] no train shards in $d"; return 1; }
    done
    echo "[preflight] shard dirs non-empty"
    # P4. declared mix vs built shards (effective distribution on record)
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
    print("  [preflight FAIL] no mix block parsed in " + yaml_path)
    sys.exit(1)
found = {d: sorted(glob.glob(os.path.join(shard_dir, "train_" + d + "_*.gbin"))) for d in mix}
missing = [d for d, fs in found.items() if not fs]
wsum = sum(w for d, w in mix.items() if d not in missing)
print("  mix table for " + os.path.basename(yaml_path) + ":")
for d, w in mix.items():
    eff = (w / wsum) if d not in missing and wsum > 0 else 0.0
    print("    %-22s declared=%5.2f shards=%3d effective=%5.2f %s"
          % (d, w, len(found[d]), eff, "MISSING" if d in missing else ""))
if missing:
    print("  [preflight FAIL] declared mix domains with zero shards: " + ", ".join(missing))
    sys.exit(1)
EOF
    done
    # P5. measured shard totals (source of truth, not comments)
    "${PIPE_BIN}" inspect --shards "${PT_DIR}" --tokenizer "${TOK}" || return 1
    "${PIPE_BIN}" inspect --shards "${SFT_DIR}" --tokenizer "${TOK}" || return 1
    # P6. memory truth on CPU (no GPU needed for the estimate)
    "${BINARY}" --config "${CONFIG_PT}" --dry-run --device cpu || return 1
    "${BINARY}" --config "${CONFIG_SFT}" --dry-run --device cpu || return 1
    echo "==================== preflight: ALL GATES PASSED ===================="
    return 0
}

if [[ "${MODE}" == "preflight" ]]; then run_preflight; exit $?; fi

if [[ "${SKIP_PREFLIGHT}" -ne 1 ]]; then
    run_preflight || { echo "[ERROR] preflight failed — fix the gates above (or --skip-preflight)."; exit 1; }
else
    echo "[preflight] SKIPPED via --skip-preflight (recipe NOT verified)"
fi

echo ""
echo "[pilot] Measuring Pro speed (${PILOT_STEPS} steps)..."
P_START=$(date +%s)
# 10/10: do NOT force --gemm-fp16 0. Pilot must measure the REAL recipe (fp16 ON
# for T4). Old script forced fp32, measured 3-5x slower speed, then planned
# steps from that wrong number AND trained the full run in slow fp32.
# --resume none is MANDATORY here: with resume:auto an old checkpoint at
# step >= PILOT_STEPS would make the pilot do ~zero work in ~zero seconds,
# and the absurd tok/s would corrupt the whole session budget below.
"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PILOT_STEPS}" --resume none
P_END=$(date +%s)
P_ELAPSED=$(( P_END - P_START )); [[ "${P_ELAPSED}" -le 0 ]] && P_ELAPSED=1
P_TPS=$(( $(yget batch_size "${CONFIG_PT}") * $(yget seq_len "${CONFIG_PT}") * $(yget grad_accum "${CONFIG_PT}") * PILOT_STEPS / P_ELAPSED ))
echo "[pilot] ${PILOT_STEPS} steps in ${P_ELAPSED}s -> ~${P_TPS} tok/s"
if [[ "${MODE}" == "pilot" ]]; then echo "[pilot] Done."; exit 0; fi

FULL_TPS=$(( $(yget batch_size "${CONFIG_PT}") * $(yget seq_len "${CONFIG_PT}") * $(yget grad_accum "${CONFIG_PT}") ))
echo "[plan] Full config: ${FULL_TPS} tokens/step ($(yget optimizer "${CONFIG_PT}") B=$(yget batch_size "${CONFIG_PT}") T=$(yget seq_len "${CONFIG_PT}") acc=$(yget grad_accum "${CONFIG_PT}"))"
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
echo "[plan] budget=${TIME_BUDGET_MIN}min remain~=${REMAIN_SEC}s"
echo "[plan] total_steps=${TOTAL_STEPS} (pretrain=${PT_STEPS}, sft=${SFT_STEPS})"
echo "[plan] ~$(( TOTAL_STEPS * FULL_TPS / 1000000 ))M tokens this session"

echo ""
echo "[stage-A] Pretraining Pro (${PT_STEPS} steps)..."
T0=$(date +%s)
"${BINARY}" --config "${CONFIG_PT}" --device cuda --tokenizer "${TOK}" \
    --data "${PT_DIR}" --max-steps "${PT_STEPS}" --warmup "${PT_WARM}" \
    --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
    --checkpoint-dir "${CKPT_PT}" --resume auto
echo "[stage-A] took $(( ($(date +%s) - T0) / 60 ))m"

echo ""
echo "[stage-B] SFT Pro (${SFT_STEPS} steps)..."
T0=$(date +%s)
if [[ "${EXPORT_GGUF}" -eq 1 ]]; then
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --resume auto \
        --export "${GGUF_OUT}" --export-profile "${EXPORT_PROFILE}"
else
    "${BINARY}" --config "${CONFIG_SFT}" --device cuda --tokenizer "${TOK}" \
        --data "${SFT_DIR}" --max-steps "${SFT_STEPS}" --warmup "${SFT_WARM}" \
        --eval-every "${EVAL_CAD}" --save-every "${EVAL_CAD}" \
        --resume auto
fi
echo "[stage-B] took $(( ($(date +%s) - T0) / 60 ))m"

if [[ "${EXPORT_GGUF}" -eq 1 ]] && [[ -f "${GGUF_OUT}" ]]; then
    echo ""
    echo "[smoke] English generation test (SELF-CONTAINED: no --tokenizer, the"
    echo "  GGUF must carry its own tokenizer; any failure below is FATAL)..."
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --persona en \
        --prompt "Hello, who are you? What can you help me with?" --max-tokens 60 || \
        { echo "[smoke] FAIL: self-contained GGUF generation failed"; exit 1; }
    "${GEN_BIN}" generate --model "${GGUF_OUT}" --persona en \
        --prompt "What is the capital of Morocco?" --max-tokens 60 || \
        { echo "[smoke] FAIL: self-contained GGUF generation failed (factual)"; exit 1; }
    echo "[smoke] OK: GGUF runs standalone, no sidecar needed"
fi

echo ""
echo "============================================================"
echo "  Done! GGUF: ${GGUF_OUT}"
echo "  Resume anytime: same command continues from last.ckpt (WSD safe)"
echo "============================================================"
