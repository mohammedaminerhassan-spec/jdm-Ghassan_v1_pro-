#!/usr/bin/env bash
# kaggle/build_billion_data.sh — Build ultra-scale shards for Ultra 1B
#
# What it does (counts come from configs/synth_billion.yaml — that file is
# the source of truth, NOT the numbers that used to be hardcoded here):
#   1. synth pretrain  : 14 batches x 50k = 700k conversations (~266M tokens,
#                        governor+reasoning turns included)
#   2. synth sft       : 2 batches x 75k = 150k conversations (~57M tokens)
#   3. json data       : your "Ai dariga datasets/" JSON/JSONL (~200-400M tokens)
#   4. csv vocab       : dataset/dataset-main/*.csv -> darija_vocab domain
#                        (YOUR words/conjugation/grammar tables — the model's
#                        Darija lexicon; was silently unused before)
#   5. speak data      : model_speak/ (single-script answers) -> darija_speak
#                        domain in BOTH pretrain and SFT shards (the style +
#                        script-matching contract; skipped with a warning if
#                        the dir is absent so pretrain never hard-fails)
#   5b. grounded Q&A   : dataset/qa_darija/ (275k extractive Q&A over the CSV
#                        tables, dataset/build_qa.py) -> darija_qa domain in
#                        BOTH pretrain and SFT shards (response supervision;
#                        skipped with a warning if absent)
#   6. build shards    : artifacts/shards_1b + artifacts/shards_1b_sft
# Planned total ~= 0.5-0.7B tokens. The MEASURED shard totals printed by
# `data_pipeline inspect` at the end are the source of truth — never scale
# training configs from comments. (The legacy "~1B" name is kept for paths.)
#
# Usage:
#   bash kaggle/build_billion_data.sh [--json-dir "Ai dariga datasets/"]
#
# Requirements:
#   - build/bin/data_pipeline built (bash kaggle/setup.sh)
#   - artifacts/tokenizer/darija32k.gtok exists (verified 32k, never legacy)
#   - 19.5GB Kaggle HDD fit (2GB user data): synth parts are deleted right
#     after merge, and the merged ~4GB jsonl is deleted right after sharding.
#     Peak ~= jsonl 4GB + shards 3GB = 7GB, final = shards only.
#     NEVER keep corpus txt + jsonl + shards + 2x ckpt at once on Kaggle.
set -euo pipefail

# RETIRED Darija route: its configs (ultra_1b/sft_ultra_1b) were removed
# (English-only pack). Without this guard the script would burn hours of
# synth + sharding and only die at the final mix check.
# English flow: EN_PARQUET_DIR=<lake> bash kaggle/build_english_data.sh
# Override (unsupported): LEGACY_DARIJA=1 bash kaggle/build_billion_data.sh
if [[ "${LEGACY_DARIJA:-0}" != "1" ]]; then
    echo "[ERROR] build_billion_data.sh is the retired Darija data route (configs removed)."
    echo "[ERROR] English flow: EN_PARQUET_DIR=<lake> bash kaggle/build_english_data.sh"
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_DIR}/build/bin/data_pipeline"
TOK="${REPO_DIR}/artifacts/tokenizer/darija32k.gtok"
# PRO-HARDEN: لا fallback صامت إلى 16k (البوابة بعده ترفض أصلا؛ نفشل هنا أوضح).
if [[ ! -f "${TOK}" ]]; then
    echo "[ERROR] 32k tokenizer missing: ${TOK}. Run setup.sh first."
    exit 1
fi
SYNTH_DIR="${REPO_DIR}/artifacts/synth"
SHARD_1B="${REPO_DIR}/artifacts/shards_1b"
SHARD_SFT="${REPO_DIR}/artifacts/shards_1b_sft"
# PRO-HARDEN: JSON_DIR كان CWD-relative ("Ai dariga datasets/") بلا REPO prefix
# ولا env override ولا /kaggle/input بحث — يفشل على Kaggle بصمت.
JSON_DIR="${JSON_DIR:-${REPO_DIR}/Ai dariga datasets/}"
if [[ $# -gt 0 && "$1" == "--json-dir" ]]; then JSON_DIR="$2"; fi
if [[ ! -e "${JSON_DIR}" && -d "/kaggle/input" ]]; then
    for d in /kaggle/input/*/; do
        [[ -d "$d" ]] || continue
        if [[ -n "$(find "$d" -maxdepth 4 \( -iname '*.json' -o -iname '*.jsonl' \) 2>/dev/null | head -n 1)" ]]; then
            JSON_DIR="$d"
            break
        fi
    done
fi
# Extra inputs (override with env). Missing dirs warn + skip, never fail:
# pretraining must survive a fresh clone that only has dataset/dataset-main.
CSV_DIR="${CSV_DIR:-${REPO_DIR}/dataset/dataset-main}"
MODEL_SPEAK_DIR="${MODEL_SPEAK_DIR:-${REPO_DIR}/dataset/model_speak}"
# Typed lake with the same rows (dataset/build_parquet_lake.py). Used as the
# darija_vocab source when the raw CSVs are absent (they now live outside the
# repo: ~/Desktop/Darija_CSV_backup). Needs the Arrow build (--with-parquet).
LAKE_DIR="${LAKE_DIR:-${REPO_DIR}/dataset/parquet/by_domain}"
# Grounded Q&A supervision built from the CSV tables themselves
# (dataset/build_qa.py: 275k question/answer pairs, every answer extractive
# from its source row). Regenerate locally after any CSV edit, or override:
# QA_DIR=/path/to/qa_json bash kaggle/build_billion_data.sh
QA_DIR="${QA_DIR:-${REPO_DIR}/dataset/qa_darija}"
# Training recipe is seq_len 512 (T4 16GB fit): shards are cut at 512 so no
# token is wasted on 1024-token docs the trainer would truncate anyway.
SHARD_SEQ_LEN=512

[[ -f "${BIN}" ]] || { echo "[ERROR] missing ${BIN}. Run setup.sh first."; exit 1; }
[[ -f "${TOK}" ]] || { echo "[ERROR] missing tokenizer ${TOK}."; exit 1; }
mkdir -p "${SYNTH_DIR}" "${SHARD_1B}" "${SHARD_SFT}"

echo "============================================================"
echo "  Ghassan Ultra 1B — ultra-scale data build"
echo "  target: ~0.5-0.7B planned (266M synth + json + SFT, MEASURED below)"
echo "============================================================"

# ---- 1. pretrain synth in batches (crash-safe, resumable) ----
# Counts honor configs/synth_billion.yaml (700k total, cap 500/template):
# 1.5M conversations at 57%+ of the corpus memorizes templates (collapse).
echo ""
echo "[1/4] Pretrain synth: 14 x 50k (700k total, cap 500/template)..."
for i in $(seq 0 13); do
  OUT="${SYNTH_DIR}/synth_1b_part$(printf %02d $i).jsonl"
  if [[ -f "$OUT" ]]; then echo "  skip part $i (exists)"; continue; fi
  SEED=$((1234 + i * 7919))
  echo "  batch $i/13 seed=${SEED} -> $OUT"
  # F-17: the old code printed "retry with fresh seed" and then did NOT retry
  # (bare `continue`), so a failed batch silently shrank the corpus and every
  # later stage trained anyway. Synth is MANDATORY for this route: try once
  # more with a genuinely fresh seed, then fail closed so the operator sees it.
  if ! "${BIN}" synth --out "$OUT" --n 50000 --seed "$SEED" \
       --max-template-uses 500; then
    SEED=$((SEED + 1000003))
    echo "  [retry] batch $i failed, retrying with fresh seed=${SEED}"
    if ! "${BIN}" synth --out "$OUT" --n 50000 --seed "$SEED" \
         --max-template-uses 500; then
      echo "[ERROR] synth batch $i failed twice (seed exhausted); aborting."
      echo "  Missing batches would silently shrink the corpus below the recipe."
      echo "  Fix the failure (disk? binary?) and re-run; existing parts resume."
      exit 1
    fi
  fi
done
echo "[1/4] merging..."
cat "${SYNTH_DIR}"/synth_1b_part*.jsonl > "${SYNTH_DIR}/synthetic_1b.jsonl"
wc -l "${SYNTH_DIR}/synthetic_1b.jsonl"
# DISK FIT (19.5GB Kaggle): parts are now redundant -> delete immediately.
# Saves ~4GB before the SFT synth stage even starts.
rm -f "${SYNTH_DIR}"/synth_1b_part*.jsonl
echo "[disk] after merge+cleanup:"; df -h "${REPO_DIR}" | tail -n 1; du -sh "${SYNTH_DIR}" 2>/dev/null || true

# ---- 2. sft synth (150k total per synth_billion.yaml, cap 300/template) ----
echo ""
echo "[2/4] SFT synth: 2 x 75k (150k total, cap 300/template)..."
for i in 0 1; do
  OUT="${SYNTH_DIR}/synth_1b_sft_part0${i}.jsonl"
  if [[ -f "$OUT" ]]; then echo "  skip sft part $i (exists)"; continue; fi
  SEED=$((9999 + i * 104729))
  # F-17: same fail-closed contract as the pretrain loop above.
  if ! "${BIN}" synth --out "$OUT" --n 75000 --seed "$SEED" --max-template-uses 300; then
    SEED=$((SEED + 1000003))
    echo "  [retry] sft part $i failed, retrying with fresh seed=${SEED}"
    "${BIN}" synth --out "$OUT" --n 75000 --seed "$SEED" --max-template-uses 300 || {
      echo "[ERROR] sft synth part $i failed twice; aborting (see above)."; exit 1; }
  fi
done
cat "${SYNTH_DIR}"/synth_1b_sft_part*.jsonl > "${SYNTH_DIR}/synthetic_1b_sft.jsonl"
rm -f "${SYNTH_DIR}"/synth_1b_sft_part*.jsonl
echo "[disk] after SFT merge+cleanup:"; df -h "${REPO_DIR}" | tail -n 1

# ---- tokenizer gate: shards for 32k models must be encoded with the 32k
# file. A silent 16k fallback here would waste the whole training run.
echo ""
echo "[tok] verifying tokenizer is 32k..."
TOK_VOCAB=$("${BIN}" tok-info --tokenizer "${TOK}" 2>/dev/null | grep -oE 'vocab_size=[0-9]+' | cut -d= -f2 || true)
if [[ "${TOK_VOCAB}" != "32000" ]]; then
  echo "[ERROR] tokenizer ${TOK} has vocab_size=${TOK_VOCAB:-unreadable}, need 32000."
  echo "[ERROR] Run setup.sh (trains darija32k.gtok) or pass a 32k file. Refusing to encode shards."
  exit 1
fi
echo "[tok] ok: ${TOK} (vocab 32000)"

# ---- 3-6. build shards ----
# Domain reality (must match data.mix in ultra_1b.yaml / sft_ultra_1b.yaml):
# each invocation below labels its WHOLE output with one --domain; the
# pipeline does NOT split one file across domains.
#   synth jsonl      -> darija_conversational (pretrain) / darija_chat (SFT)
#   dataset-main CSV -> darija_vocab (the repo's own lexicon)
#   model_speak/     -> darija_speak (single-script style contract)
#   qa_darija/       -> darija_qa (grounded Q&A supervision, pretrain + SFT)
#   JSON corpus      -> educational_science
# To widen the mix: add an invocation with --domain <name> for materialized
# shards, then append the weight to the yaml mix.
echo ""
echo "[3/6] Building pretrain shards -> ${SHARD_1B} ..."
"${BIN}" build --tokenizer "$TOK" --expect-vocab 32000 --out "$SHARD_1B" \
  --chat "${SYNTH_DIR}/synthetic_1b.jsonl" \
  --domain darija_conversational \
  --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" \
  --report "${SHARD_1B}/report.txt"

echo ""
echo "[4/6] Darija vocab (CSVs -> darija_vocab, else lake -> darija_vocab) ..."
# NOTE: the two sources are alternatives (elif), never both: two invocations
# with the same --domain would restart shard numbering and overwrite
# train_darija_vocab_0000.gbin.
if [[ -d "$CSV_DIR" ]]; then
  "${BIN}" csvs --dir "$CSV_DIR" --out "${SYNTH_DIR}/corpus_darija_vocab.txt" || \
    echo "[WARN] csvs step failed, continuing without darija_vocab"
  if [[ -f "${SYNTH_DIR}/corpus_darija_vocab.txt" ]]; then
    "${BIN}" build --tokenizer "$TOK" --expect-vocab 32000 --out "$SHARD_1B" \
      --text "${SYNTH_DIR}/corpus_darija_vocab.txt" \
      --domain darija_vocab \
      --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
      echo "[WARN] darija_vocab build failed, continuing"
    rm -f "${SYNTH_DIR}/corpus_darija_vocab.txt"
  fi
elif [[ -d "$LAKE_DIR" ]] && "${BIN}" parquet --probe >/dev/null 2>&1; then
  echo "  [route] no CSVs: reading the typed lake natively (same 158k rows)"
  "${BIN}" parquet --lake "$LAKE_DIR" --mode text --tokenizer "$TOK" --expect-vocab 32000 \
    --out "$SHARD_1B" --domain darija_vocab \
    --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
    echo "[WARN] lake darija_vocab build failed, continuing"
else
  echo "[WARN] no csv dir $CSV_DIR and no native-parquet lake $LAKE_DIR,"
  echo "  skipping darija_vocab (mix will renormalize; rebuild with CSVs"
  echo "  present or setup.sh --with-parquet for the lake route)"
fi

echo ""
echo "[5/6] Speak data (model_speak/ -> darija_speak, pretrain + SFT) ..."
if [[ -d "$MODEL_SPEAK_DIR" ]]; then
  "${BIN}" json --dir "$MODEL_SPEAK_DIR" --tokenizer "$TOK" --expect-vocab 32000 \
    --out "$SHARD_1B" --domain darija_speak \
    --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
    echo "[WARN] pretrain darija_speak failed, continuing"
  "${BIN}" json --dir "$MODEL_SPEAK_DIR" --tokenizer "$TOK" --expect-vocab 32000 \
    --out "$SHARD_SFT" --domain darija_speak \
    --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
    echo "[WARN] sft darija_speak failed, continuing"
else
  echo "[WARN] no model_speak dir $MODEL_SPEAK_DIR."
  echo "  Run tools/make_model_ready.py + tools/make_speak_data.py on your"
  echo "  curated JSONL first (see configs/sft_darija_1b.yaml header), or set"
  echo "  MODEL_SPEAK_DIR=<path>. Continuing without darija_speak."
fi

if [[ -e "$JSON_DIR" ]]; then
  echo ""
  echo "[json] adding $JSON_DIR ..."
  "${BIN}" json --dir "$JSON_DIR" --tokenizer "$TOK" --expect-vocab 32000 \
    --out "$SHARD_1B" --domain educational_science \
    --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || echo "[WARN] json step failed, continuing with synth only"
else
  echo "[WARN] no json path $JSON_DIR, synth-only shards"
fi

echo ""
echo "[5b/6] Grounded Q&A (qa_darija/ -> darija_qa, pretrain + SFT) ..."
echo "  275k extractive Q&A over the CSV lexicon/grammar (the supervision that"
echo "  teaches RESPONDING, not just reciting tables; assistant-only loss)."
# Route selection: native parquet (ONE file, no JSON parsing) when the binary
# was built with Arrow (setup.sh --with-parquet), else the JSON shards that
# ship in the same directory. Same docs, same masks, same domain either way.
QA_ROUTE="json"
if [[ -d "$QA_DIR" ]] && "${BIN}" parquet --probe >/dev/null 2>&1; then
  QA_ROUTE="parquet"
  echo "  [route] native parquet (Arrow build detected)"
else
  echo "  [route] JSON shards (portable; identical docs/masks)"
fi
if [[ -d "$QA_DIR" ]]; then
  if [[ "$QA_ROUTE" == "parquet" && -f "$QA_DIR/qa_all.parquet" ]]; then
    "${BIN}" parquet --lake "$QA_DIR/qa_all.parquet" --mode qa --tokenizer "$TOK" --expect-vocab 32000 \
      --out "$SHARD_1B" --domain darija_qa \
      --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
      echo "[WARN] pretrain darija_qa failed, continuing"
    "${BIN}" parquet --lake "$QA_DIR/qa_all.parquet" --mode qa --tokenizer "$TOK" --expect-vocab 32000 \
      --out "$SHARD_SFT" --domain darija_qa \
      --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
      echo "[WARN] sft darija_qa failed, continuing"
  else
    "${BIN}" json --dir "$QA_DIR" --tokenizer "$TOK" --expect-vocab 32000 \
      --out "$SHARD_1B" --domain darija_qa \
      --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
      echo "[WARN] pretrain darija_qa failed, continuing"
    "${BIN}" json --dir "$QA_DIR" --tokenizer "$TOK" --expect-vocab 32000 \
      --out "$SHARD_SFT" --domain darija_qa \
      --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" || \
      echo "[WARN] sft darija_qa failed, continuing"
  fi
else
  echo "[WARN] no qa dir $QA_DIR."
  echo "  Run python dataset/build_qa.py locally (needs dataset/parquet lake)"
  echo "  or set QA_DIR=<path>. Continuing without darija_qa (mix renormalizes)."
fi

echo ""
echo "[6/6] Building SFT shards -> ${SHARD_SFT} ..."
"${BIN}" build --tokenizer "$TOK" --expect-vocab 32000 --out "$SHARD_SFT" \
  --chat "${SYNTH_DIR}/synthetic_1b_sft.jsonl" \
  --domain darija_chat \
  --shard-tokens 50000000 --seq-len "${SHARD_SEQ_LEN}" \
  --report "${SHARD_SFT}/report.txt"

# DISK FIT: merged jsonl (8GB) is now fully encoded into .gbin shards.
# Delete it immediately so 3GB user data + 4GB shards + 8GB ckpt fits 19.5GB.
# Shards are the ONLY training input from here on.
echo "[cleanup] removing merged jsonl (shards are complete) to fit 19.5GB..."
rm -f "${SYNTH_DIR}/synthetic_1b.jsonl" "${SYNTH_DIR}/synthetic_1b_sft.jsonl"
rm -rf "${REPO_DIR}/artifacts/corpus" 2>/dev/null || true
echo "[disk] final:"; df -h "${REPO_DIR}" | tail -n 1; du -sh "${SHARD_1B}" "${SHARD_SFT}" 2>/dev/null || true

echo ""
echo "---- inspect (MEASURED totals are the source of truth) ----"
"${BIN}" inspect --shards "$SHARD_1B" --tokenizer "$TOK" || true
"${BIN}" inspect --shards "$SHARD_SFT" --tokenizer "$TOK" || true
echo ""
echo "---- domain inventory vs configs/ultra_1b.yaml + sft_ultra_1b.yaml mix ----"
echo "  (every declared mix domain must have train_<domain>_*.gbin;"
echo "   missing domains renormalize away SILENTLY at train time)"
MIX_MISSING=0
check_mix() {
  local yaml="$1" dir="$2"
  echo "  -- $(basename "$yaml") in $dir --"
  while read -r domain; do
    [ -z "$domain" ] && continue
    n=$(find "$dir" -name "train_${domain}_*.gbin" 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
      echo "  [MISSING] mix domain '$domain': no train_${domain}_*.gbin in $dir"
      MIX_MISSING=1
    else
      echo "  [ok] $domain: $n shard(s)"
    fi
  done < <(awk '/^  mix:/{f=1;next} f&&/^    [a-z_]+:/{print $1} f&&/^[^ #]/{exit}' \
    "$yaml" | tr -d ':')
}
check_mix "${REPO_DIR}/configs/ultra_1b.yaml" "${SHARD_1B}"
check_mix "${REPO_DIR}/configs/sft_ultra_1b.yaml" "${SHARD_SFT}"
# F-17: a declared mix domain with no shards used to only warn, so the run went
# ahead on a silently renormalized mixture (wrong data, full GPU cost). Fail
# closed: either materialize the missing domains (--domain) or remove them from
# the yaml and re-run this script. Optional domains must be removed from the
# mix explicitly — never treated as "probably fine".
if [ "${MIX_MISSING}" -ne 0 ]; then
  echo "  [ERROR] declared-vs-built mix mismatch: either materialize the missing"
  echo "  domains (--domain) or remove them from the yaml mix. Training would"
  echo "  renormalize over found domains only (see trainer startup mix log),"
  echo "  which silently changes the data recipe. Refusing to continue."
  exit 1
fi
echo "  effective mix == declared mix (all declared domains materialized above)"
echo ""
echo "============================================================"
echo "  Done. Record the MEASURED train totals above and scale training configs"
echo "  from them — not from comments. Planned: ~0.5-0.7B tokens."
echo "  Next: bash kaggle/train_1b.sh --preflight, then train."
echo "============================================================"
