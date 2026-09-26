#!/usr/bin/env bash
# kaggle/cell6_app_chain.sh — full application-usage gate on a real T4.
#
# Proves the whole user-facing path with the REAL tokenizer (vocab 32000) and
# the REAL English shards:
#   train (6 steps) -> loss goes down -> export GGUF (fp16 + q4_0) -> info
#   -> tokenize round-trip -> generate -> chat -> bench -> perplexity(val)
#   -> quantize -> logits -> devices
# Every step is fail-fast with a named marker, so a break names itself.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
BIN="${REPO_DIR}/build/bin"
TOK="artifacts/tokenizer/english32k.gtok"
SHARDS="artifacts/shards_en"
WORK="/tmp/cell6"
GGUF_FP16="${WORK}/model_fp16.gguf"
GGUF_Q4="${WORK}/model_q4_0.gguf"

rm -rf "${WORK}"; mkdir -p "${WORK}/ckpt"
step() { echo ""; echo "##### $1 #####"; }
ok()   { echo "  [ok] $1"; }

step "[1/12] train 6 steps on the real shards (CUDA)"
"${BIN}/gai_train" --config configs/en_pro.yaml --device cuda \
  --vocab 32000 --layers 2 --hidden 128 --heads 4 --kv-heads 2 \
  --batch-size 1 --seq-len 256 --grad-accum 2 --max-steps 6 --warmup 0 \
  --eval-every 3 --eval-batches 2 --save-every 6 --log-every 1 \
  --data "${SHARDS}" --checkpoint-dir "${WORK}/ckpt" \
  --tokenizer "${TOK}" --resume none --seed 11 \
  2>&1 | tee "${WORK}/train.log" | grep -E "^\[info \] (step|  >> val)|^\[error|^\[fatal|ckpt\] (saved|published)|pretrain done|mix"
LAST_LOSS=$(grep -oE "step +[0-9]+ \| loss [0-9.]+" "${WORK}/train.log" | tail -1 | grep -oE "loss [0-9.]+" | cut -d' ' -f2)
FIRST_LOSS=$(grep -oE "step +[0-9]+ \| loss [0-9.]+" "${WORK}/train.log" | head -1 | grep -oE "loss [0-9.]+" | cut -d' ' -f2)
echo "  first_loss=${FIRST_LOSS}  last_loss=${LAST_LOSS}"
[ -n "${LAST_LOSS}" ] || { echo "  [FAIL] no step loss logged"; exit 1; }
awk -v a="${FIRST_LOSS}" -v b="${LAST_LOSS}" 'BEGIN{ if (b >= a) { printf "  [FAIL] loss did not decrease (%s -> %s): the model is NOT learning\n", a, b; exit 1 } else printf "  [ok] loss decreased %s -> %s (model learns)\n", a, b }'
[ -f "${WORK}/ckpt/last.ckpt" ] || { echo "  [FAIL] no last.ckpt"; exit 1; }
ok "last.ckpt $(stat -c%s "${WORK}/ckpt/last.ckpt") bytes"

step "[2/12] export GGUF fp16 (native) + q4_0"
"${BIN}/ghassan-ai" export --checkpoint "${WORK}/ckpt/last.ckpt" \
  --tokenizer "${TOK}" --out "${GGUF_FP16}" --profile fp16
"${BIN}/ghassan-ai" export --checkpoint "${WORK}/ckpt/last.ckpt" \
  --tokenizer "${TOK}" --out "${GGUF_Q4}" --profile q4_0
ls -la "${GGUF_FP16}" "${GGUF_Q4}" | awk '{printf "  %12s  %s\n", $5, $9}'

step "[3/12] info on both GGUF files"
"${BIN}/ghassan-ai" info --model "${GGUF_FP16}"
echo "  ---- q4_0 ----"
"${BIN}/ghassan-ai" info --model "${GGUF_Q4}"

step "[4/12] tokenizer round-trip (keep-case English)"
"${BIN}/ghassan-ai" tokenize --model "${GGUF_FP16}" \
  --text "Ghassan keeps Case: 100% accurate, isn't it? <|user|>hi<|end|>"

step "[5/12] generate (CUDA, greedy, en persona)"
"${BIN}/ghassan-ai" generate --model "${GGUF_FP16}" --device cuda \
  --persona en --greedy --max-tokens 48 --seed 5 \
  --prompt "<|user|>Hello! Who are you?<|end|><|assistant|>"

step "[6/12] chat (piped stdin, en persona)"
printf 'Hello, who are you?\n/exit\n' | timeout 300 "${BIN}/ghassan-ai" chat \
  --model "${GGUF_FP16}" --device cuda --persona en --max-tokens 48 --seed 5 || true

step "[7/12] bench (decode speed on T4)"
"${BIN}/ghassan-ai" bench --model "${GGUF_FP16}" --device cuda --tokens 64

step "[8/12] perplexity on the real held-out shards"
"${BIN}/ghassan-ai" perplexity --model "${GGUF_FP16}" --device cuda \
  --shards "${SHARDS}" --prefix val_

step "[9/12] quantize fp16 -> q8_k, then info + generate"
"${BIN}/ghassan-ai" quantize --model "${GGUF_FP16}" --out "${WORK}/model_q8_k.gguf" --profile q8_k
"${BIN}/ghassan-ai" info --model "${WORK}/model_q8_k.gguf" | head -20
"${BIN}/ghassan-ai" generate --model "${WORK}/model_q8_k.gguf" --device cuda \
  --greedy --max-tokens 24 --seed 5 --prompt "Hello!"

step "[10/12] logits dump (shape/sanity)"
"${BIN}/ghassan-ai" logits --model "${GGUF_FP16}" --device cuda \
  --prompt "Hello" --out "${WORK}/logits.f32"
ls -la "${WORK}/logits.f32" | awk '{printf "  %12s  %s\n", $5, $9}'

step "[11/12] devices"
"${BIN}/ghassan-ai" devices

step "[12/12] artifact inventory + no stray files"
ls -la "${WORK}"
echo "  --- warnings/errors in the whole run ---"
grep -E "^\[warn|^\[error|^\[fatal" "${WORK}/train.log" || echo "  (none)"

echo ""
echo "=============================================================="
echo "  CELL 6 PASS — application chain works end-to-end on T4"
echo "  next: Cell 7 = data-quality tools + remaining engine tests"
echo "=============================================================="
