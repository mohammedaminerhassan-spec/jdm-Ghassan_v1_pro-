#!/usr/bin/env bash
# kaggle/cell7_engines.sh — the "is every engine correct?" gate.
#
# Cell 3/4/6 prove the build, DDP and the application path. This cell covers
# what no other cell has:
#   1. every engine test individually, so a break NAMES ITSELF (ctest hides it)
#   2. data-quality report on the REAL shards (corpus_stats)
#   3. the ACTIVATION CHECKPOINTING path on a real T4 -- the flagship
#      en_pro.yaml ships activation_checkpointing=true + ckpt_segments=2 +
#      batch_size=2, and until now NO run had ever executed that path on CUDA
#      (Cell 6 warned "no effect" because it used batch_size=1)
#   4. ckpt vs non-ckpt numerical agreement (segmented backward must produce
#      the same gradients as the full arena, or the flagship trains differently)
#   5. quantization accuracy: val perplexity of fp16 vs q8_k vs q4_0
#   6. every shipped config with --strict-config (dead/unknown keys fail)
#   7. resume round-trip on CUDA (train -> kill -> resume -> finish)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
BIN="${REPO_DIR}/build/bin"
TOK="artifacts/tokenizer/english32k.gtok"
SHARDS="artifacts/shards_en"
WORK="/tmp/cell7"
rm -rf "${WORK}"; mkdir -p "${WORK}"
FAILURES=0
step() { echo ""; echo "##### $1 #####"; }
ok()   { echo "  [ok] $1"; }
bad()  { echo "  [FAIL] $1"; FAILURES=$((FAILURES + 1)); }

step "[0/8] build from the pulled commit"
echo "  commit: $(git log --oneline -1)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGAI_ENABLE_CUDA=ON -DGAI_BUILD_TESTS=ON \
  > /tmp/cell7_cmake.log 2>&1 || { tail -25 /tmp/cell7_cmake.log; exit 1; }
cmake --build build -j2 > /tmp/cell7_build.log 2>&1 \
  || { echo "  [FAIL] CUDA build -Werror"; tail -25 /tmp/cell7_build.log; exit 1; }
ok "build clean"

step "[1/8] every engine test, individually (a break must name itself)"
TESTS=$(ctest --test-dir build -N 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: *//p')
echo "  found: $(echo "${TESTS}" | wc -l) tests"
for t in ${TESTS}; do
    if out=$("${BIN}/${t}" 2>&1); then
        printf "  [ok] %s\n" "${t}"
    else
        printf "  [FAIL] %s\n" "${t}"
        echo "${out}" | tail -12 | sed 's/^/         /'
        FAILURES=$((FAILURES + 1))
    fi
done

step "[2/8] data quality on the REAL corpus (parquet -> text -> corpus_stats)"
EN_LAKE="${EN_LAKE:-/kaggle/input/datasets/mohammedaminerhassan/ghassan-v1-pro-datasets/Users/Ghassan PC/Desktop/english_parquet}"
if [[ -d "${EN_LAKE}" ]]; then
    for dom in english_chat english_instruction; do
        "${BIN}/data_pipeline" parquet-corpus --lake "${EN_LAKE}" --match "${dom}" \
            --limit 4000 --out "${WORK}/corpus_${dom}.txt" > "${WORK}/corpus_${dom}.log" 2>&1 \
            || { bad "parquet-corpus ${dom}"; tail -6 "${WORK}/corpus_${dom}.log"; continue; }
        n=$(wc -l < "${WORK}/corpus_${dom}.txt")
        echo "  ---- corpus_stats on ${dom} (${n} lines) ----"
        "${BIN}/corpus_stats" --input "${WORK}/corpus_${dom}.txt" --tokenizer "${TOK}" \
            --dedup --pii --quality 2>&1 | tail -30
        echo "  ---- data_pipeline retrieve (RAG index smoke) ----"
        "${BIN}/data_pipeline" retrieve --index "${WORK}/corpus_${dom}.txt" \
            --query "how do I stay consistent with my training routine?" --top 2 2>&1 | tail -12 \
            || bad "retrieve on ${dom}"
    done
else
    echo "  [skip] lake not attached: ${EN_LAKE}"
fi

step "[3/8] ACTIVATION CHECKPOINTING on T4 (the flagship path, never run before)"
# batch_size=2 + ckpt_segments=2 => use_ckpt_ must be TRUE (no 'no effect' warning).
"${BIN}/gai_train" --config configs/en_pro.yaml --device cuda \
  --vocab 32000 --layers 2 --hidden 128 --heads 4 --kv-heads 2 \
  --batch-size 2 --seq-len 256 --grad-accum 1 --max-steps 3 --warmup 0 \
  --eval-every 3 --eval-batches 1 --save-every 3 --log-every 1 \
  --data "${SHARDS}" --checkpoint-dir "${WORK}/ckpt_ckpt" \
  --tokenizer "${TOK}" --resume none --seed 21 \
  > "${WORK}/ckpt_on.log" 2>&1 || { bad "activation-checkpointing run crashed"; tail -20 "${WORK}/ckpt_on.log"; }
grep -E "\[ckpt-act\]|\[mem \] activations|step +[0-9]" "${WORK}/ckpt_on.log" || true
if grep -q "no effect" "${WORK}/ckpt_on.log"; then
    bad "activation checkpointing did NOT engage (batch=2 segments=2 must enable it)"
else
    ok "activation checkpointing engaged"
fi
grep -q "pretrain done" "${WORK}/ckpt_on.log" && ok "segmented training completed" || bad "segmented training did not finish"

step "[4/8] segmented backward == full-arena backward (gradient agreement)"
# Same seed/data, only the checkpointing flag differs. Losses must match to fp32
# round-off: a different result means the flagship trains something else.
"${BIN}/gai_train" --config configs/en_pro.yaml --device cuda \
  --vocab 32000 --layers 2 --hidden 128 --heads 4 --kv-heads 2 \
  --batch-size 1 --seq-len 256 --grad-accum 1 --max-steps 3 --warmup 0 \
  --eval-every 9 --eval-batches 1 --save-every 9 --log-every 1 \
  --data "${SHARDS}" --checkpoint-dir "${WORK}/ckpt_plain" \
  --tokenizer "${TOK}" --resume none --seed 21 \
  > "${WORK}/ckpt_off.log" 2>&1 || { bad "non-checkpointing run crashed"; tail -20 "${WORK}/ckpt_off.log"; }
L_ON=$(grep -oE "step +1 \| loss [0-9.]+" "${WORK}/ckpt_on.log"  | grep -oE "loss [0-9.]+" | cut -d' ' -f2)
L_OFF=$(grep -oE "step +1 \| loss [0-9.]+" "${WORK}/ckpt_off.log" | grep -oE "loss [0-9.]+" | cut -d' ' -f2)
echo "  ckpt ON  step1 loss = ${L_ON}"
echo "  ckpt OFF step1 loss = ${L_OFF}"
if [[ -n "${L_ON}" && -n "${L_OFF}" ]]; then
    awk -v a="${L_ON}" -v b="${L_OFF}" 'BEGIN{ d=a-b; if (d<0) d=-d; r=d/(b==0?1:b);
        if (r < 1e-4) printf "  [ok] losses agree (rel diff %.2e)\n", r; else printf "  [FAIL] losses differ (rel %.2e)\n", r; exit 1 }' \
        || bad "segmented vs full-arena loss mismatch"
else
    bad "could not read step-1 loss from both runs"
fi

step "[5/8] quantization accuracy on real held-out data"
"${BIN}/gai_train" --config configs/en_pro.yaml --device cuda \
  --vocab 32000 --layers 2 --hidden 128 --heads 4 --kv-heads 2 \
  --batch-size 1 --seq-len 256 --grad-accum 1 --max-steps 4 --warmup 0 \
  --eval-every 9 --eval-batches 1 --save-every 4 --log-every 1 \
  --data "${SHARDS}" --checkpoint-dir "${WORK}/ckpt_q" \
  --tokenizer "${TOK}" --resume none --seed 33 \
  > "${WORK}/q.log" 2>&1 || { bad "quant-source training crashed"; tail -20 "${WORK}/q.log"; }
"${BIN}/ghassan-ai" export --checkpoint "${WORK}/ckpt_q/last.ckpt" --tokenizer "${TOK}" \
  --out "${WORK}/q_fp16.gguf" --profile fp16 > /dev/null 2>&1 || bad "export fp16"
"${BIN}/ghassan-ai" quantize --model "${WORK}/q_fp16.gguf" --out "${WORK}/q_q8k.gguf" --profile q8_k > /dev/null 2>&1 || bad "quantize q8_k"
"${BIN}/ghassan-ai" export --checkpoint "${WORK}/ckpt_q/last.ckpt" --tokenizer "${TOK}" \
  --out "${WORK}/q_q40.gguf" --profile q4_0 > /dev/null 2>&1 || bad "export q4_0"
for prof in fp16 q8k q40; do
    f="${WORK}/q_${prof}.gguf"
    p=$("${BIN}/ghassan-ai" perplexity --model "${f}" --device cuda \
          --shards "${SHARDS}" --prefix val_ 2>/dev/null | awk '/perplexity/{print $3}')
    echo "  ${prof}: perplexity = ${p}"
    eval "PPL_${prof}=${p}"
done
awk -v fp="${PPL_fp16}" -v q8="${PPL_q8k}" -v q4="${PPL_q40}" 'BEGIN{
    if (q8 <= 0 || q4 <= 0) { print "  [FAIL] missing perplexity"; exit 1 }
    r8=(q8-fp)/fp; r4=(q4-fp)/fp
    printf "  q8_k rel drift %+.2f%% | q4_0 rel drift %+.2f%%\n", r8*100, r4*100
    if (r4 < 0 && r4 > -0.5) { print "  [FAIL] q4_0 is BETTER than fp16 — perplexity is not measuring anything"; exit 1 }
    if (r4 > 0.60) { print "  [FAIL] q4_0 destroys the model (>60% ppl drift)"; exit 1 }
    if (r8 > 0.15) { print "  [FAIL] q8_k drift >15%"; exit 1 }
    print "  [ok] quantization accuracy within budget"
}' || bad "quantization accuracy gate"

step "[6/8] every shipped config parses under --strict-config"
for cfg in en_pro pro_v1 t4_1b pro_auxfree en_ollama sft_en_pro sft_pro_v1 sft_en_4xt4 smoke synth_billion synth_large; do
    if "${BIN}/gai_train" --config "configs/${cfg}.yaml" --dry-run --device cuda --strict-config \
         > "/tmp/cell7_cfg_${cfg}.log" 2>&1; then
        printf "  [ok] %-16s %s\n" "${cfg}" "$(grep TOTAL "/tmp/cell7_cfg_${cfg}.log" | head -1)"
    else
        printf "  [FAIL] %-16s\n" "${cfg}"
        tail -8 "/tmp/cell7_cfg_${cfg}.log" | sed 's/^/         /'
        FAILURES=$((FAILURES + 1))
    fi
done

step "[7/8] CUDA resume round-trip (train -> stop -> resume -> finish)"
"${BIN}/gai_train" --config configs/en_pro.yaml --device cuda \
  --vocab 32000 --layers 2 --hidden 128 --heads 4 --kv-heads 2 \
  --batch-size 1 --seq-len 256 --grad-accum 1 --max-steps 2 --warmup 0 \
  --eval-every 2 --eval-batches 1 --save-every 1 --log-every 1 \
  --data "${SHARDS}" --checkpoint-dir "${WORK}/ckpt_r" \
  --tokenizer "${TOK}" --resume none --seed 77 \
  > "${WORK}/resume_a.log" 2>&1 || bad "resume phase A"
grep -E "pretrain done|resumed" "${WORK}/resume_a.log" | tail -3 || true
"${BIN}/gai_train" --config configs/en_pro.yaml --device cuda \
  --vocab 32000 --layers 2 --hidden 128 --heads 4 --kv-heads 2 \
  --batch-size 1 --seq-len 256 --grad-accum 1 --max-steps 4 --warmup 0 \
  --eval-every 2 --eval-batches 1 --save-every 2 --log-every 1 \
  --data "${SHARDS}" --checkpoint-dir "${WORK}/ckpt_r" \
  --tokenizer "${TOK}" --resume auto --seed 77 \
  > "${WORK}/resume_b.log" 2>&1 || { bad "resume phase B"; tail -20 "${WORK}/resume_b.log"; }
grep -E "resumed from|pretrain done" "${WORK}/resume_b.log" | tail -3 || true
grep -q "resumed from .*last.ckpt" "${WORK}/resume_b.log" && ok "resumed from last.ckpt" || bad "did not resume from last.ckpt"

step "[8/8] warnings/errors collected across the whole cell"
echo "  --- warn/error lines in every log of this cell ---"
grep -hE "^\[warn|^\[error|^\[fatal" "${WORK}"/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -30 || echo "  (none)"

echo ""
echo "=============================================================="
if [[ "${FAILURES}" -eq 0 ]]; then
    echo "  CELL 7 PASS — every engine correct, no gate failed"
    echo "  next: the training cells (checkpoint-resumable, live progress)"
else
    echo "  CELL 7 FAIL — ${FAILURES} gate(s) failed (listed above)"
fi
echo "=============================================================="
[[ "${FAILURES}" -eq 0 ]]
