# Ghassan AI — Moroccan Darija MoE Language Model (GGUF-only)

Pure **C++20 + CUDA**, zero Python runtime. Decoder-only pre-norm Transformer with
DeepSeek-style Mixture-of-Experts, GQA, RoPE, RMSNorm, SwiGLU, QK-Norm, z-loss, YaRN.

> **Format policy: GGUF-only.** All models are single-file self-contained `.gguf`
> (config + weights + tokenizer embedded). Legacy `.gai` files still *load* for
> backward compat but are deprecated — all new exports, quantizes and docs use `.gguf`
> because Ollama / LM Studio / llama.cpp only speak GGUF.

## Models

| Config | Params | Active/token | Layers | Experts | Vocab | Target |
|---|---|---|---|---|---|---|
| `configs/flash_moe.yaml` | ~480M | ~204M | 26 | 8x top2 + shared | 32k | fast T4 / CPU |
| `configs/ultra_1b.yaml` FLAGSHIP | ~1.016B | ~336M | 36 | 10x top2 + shared | 32k | Kaggle T4 1x16GB |
| `configs/darija_1b.yaml` | ~1.016B | ~336M | 36 | 10x top2 | 32k | Darija-Arab |
| `configs/en_pro.yaml` | ~480M-1B | — | — | MoE | 32k | English track |
| `configs/pro_moe.yaml`, `pro_v1.yaml` | Pro | — | — | aux-free / SWA / NeoX ready | 32k | research |

Parameter math is verified in `configs/ultra_1b.yaml:1-24` and `model/model.h:16-72`.

## Quickstart (GGUF)

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\ghassan-ai info --model artifacts\ghassan-1b-q4_0.gguf
.\build\bin\ghassan-ai chat --model artifacts\ghassan-1b-q4_0.gguf --persona darija
.\build\bin\ghassan-ai generate --model artifacts\ghassan-1b-q4_0.gguf --prompt "شنو حالك؟" --temp 0.7 --top-p 0.92 --min-p 0.05 --repeat 1.12 --no-repeat-ngram 3
.\build\bin\ghassan-ai tokenize --model artifacts\ghassan-1b-q4_0.gguf --text "salam, chno smiytek?"
.\build\bin\ghassan-ai bench --model artifacts\ghassan-1b-q4_0.gguf --tokens 128
.\build\bin\ghassan-ai eval --model artifacts\ghassan-1b-q4_0.gguf --suite evaluation/datasets
```

Ollama / LM Studio:
- `native` GGUF (`general.architecture=ghassan`) loads in `ghassan-ai` on any machine.
- `dense` GGUF with `--compat llama` loads in llama.cpp / Ollama / LM Studio.
- `MoE` GGUF with `--compat llama_moe` (experimental Mixtral-style mapping) loads in
  recent llama.cpp for inference; training / fine-tune stays in `ghassan-ai`.

## Training (Kaggle T4)

```bash
bash kaggle/setup.sh            # CUDA autodetect, ccache, 32k BPE vocab gate
bash kaggle/build_billion_data.sh  # synth 700k + SFT 150k + qa_darija 275k -> shards_1b/*.gbin
bash kaggle/train_1b.sh         # flagship 2-stage PT60/SFT40, Lion, B1/T512/acc128, fp16 GEMM
bash kaggle/convert_data.sh     # inspect + inventory
# export is always GGUF:
build/bin/gai_train --config configs/ultra_1b.yaml --export artifacts/ghassan-1b.gguf --export-profile fp16 --tokenizer artifacts/tokenizer/darija32k.gtok
build/bin/ghassan-ai quantize --model artifacts/ghassan-1b.gguf --out artifacts/ghassan-1b-q4_0.gguf --profile q4_0 --measure
```

T4 recipe is strict: `optimizer lion, batch 1, seq 512, acc 128 (=65k tok/step),
gemm_fp16 true, loss_scale 16384`. AdamW/B2/T1024 OOMs (12.2GB vs 16GB+).
See `configs/ultra_1b.yaml:55-92`.

## Dataset pipeline

`tools/data_pipeline_main.cpp`: `clean -> normalise -> PII -> quality -> toxicity ->
langid -> style(darija/en) -> dedup -> tokenize -> shard`.

- Languages: Darija-Arab, Darija-Latin/Arabizi (3/7/9 preserved, شنو=chno=shnou),
  MSA, French code-switch 12-15%, English separate track.
- Shards: `train_<domain>_*.gbin` 50M tok (`GBIN` magic, u16/u32 + mask, EOS docs),
  `val 0.5%` canonical-hash split, long docs window-split, per-domain reports.
- Dedup `dataset/dedup.h:12`: exact canonical hash + MinHash LSH
  `128 hashes / 16 bands / shingle5 / J0.85` streaming + eval blocklist.
- Mix `configs/ultra_1b.yaml:110-117`: darija_conversational 0.40, vocab 0.20,
  speak 0.15, educational_science 0.15, qa 0.10. Weights renormalize — read startup log.
- Scale note (Chinchilla ~20 tok/param): 1B needs ~20B tokens. Shipped
  `~0.5-0.7B` is a pilot. For a strong model mix in FineWeb-Edu, StarCoder2,
  FineMath + your Darija QA, then `data_pipeline build --domain <name>`.

## Advanced techniques (DeepSeek-V3 / GLM-4 / LLaMA-3 class)

- MoE top-k + shared expert + aux-loss 0.01 + jitter 0.01 (`model/model.cpp:24`).
  `moe_aux_free=true` (Pro) = true aux-loss-free with EMA bias update, no aux grad
  (requires `moe_aux_scale: 0.0`).
- QK-Norm per-head (`use_qk_norm`), z-loss 1e-4, NTK rope_scale + YaRN mscale
  auto `0.1*ln(s)+1` + full ramp `rope_yarn_low/high` (`model/model.h:42-63`).
- RoPE `rope_type 0=interleaved` (stable) / `1=neox half-rotate` (Pro new ckpts).
- SWA `sliding_window>0` (Mistral-style local mask, CPU+CUDA, GGUF round-trip).
- Sampling `inference/sampler.h:9`: temp/top-k/top-p/min-p/repeat/freq/pres +
  `no_repeat_ngram` anti-loop for Darija + GPU fast top-k (~1KB D2H/token).
- Precision: fp32 masters + fp16 tensor GEMM + loss-scale 16384/2000, BF16 opt-in
  (Ampere+), Q8_0/Q4_0 block quant + F16 norms/embeddings.
- File types: `.gguf` deploy (self-contained), `.gbin` train shards,
  `.gtok` tokenizer image, `.ckpt` resume (weights+moments+loader+RNG),
  `.parquet` lake interchange (Arrow opt), `.json/.jsonl` raw.

## VRAM guide (GGUF)

`VRAM ~= params * bits/8 + 1-4GB ctx`. 1B: fp16 ~2.4GB, Q8 ~1.2GB, Q4_0 ~0.7GB.
T4-16GB fits PT with Lion; 4xT4 fits DDP via `train_4xt4.sh` (NCCL, stale-rendezvous cleanup).

## Eval

`evaluation/datasets/*.jsonl` (committed, no download): basic/darija/arabizi/
code-switch/context/multiturn/commonsense/culture/instruction/repetition/
hallucination/toxicity/naturalness. `ghassan-ai eval` prints score + robotic/toxic/
repetition/distinct/darija_ratio + human sheet.

## Layout

```
configs/ core/ cuda/ dataset/ docker/ evaluation/datasets/ format/
inference/ kaggle/ model/ quantization/ tests/ tokenizer/ tools/ training/
```

CPU always builds; CUDA `sm_75` default (T4 ref). `/WX -Werror` zero-warning.
See `CMakeLists.txt:1-60`.

## Gemini chat logger (distillation data)

`tools/gemini_chat.py` (stdlib only, no install) and `tools/gemini_chat.cpp`
(C++17 + libcurl, optional `gemini-chat` target) chat with the Gemini API and
auto-append every turn to `gemini_log.jsonl` — ready-made SFT/chat data for
`data_pipeline` (distillation: teacher answers -> train Ghassan).

```powershell
$env:GEMINI_API_KEY = "..."   # https://aistudio.google.com/apikey
python tools/gemini_chat.py --prompt "شنو حالك؟"          # single shot
python tools/gemini_chat.py                                 # REPL, 'exit' l khrouj
python tools/gemini_chat.py --model gemini-3.6-flash --system "jawb b Darija"
```

Default model `gemini-3.6-flash` (cheap quota-friendly; Google katbedel smiyat
models mra mra — `python tools/gemini_chat.py --list-models` tchoflik chno khdam
b key dyalek). Default system prompt = persona dyal Ghassan b Darija
(`--system "..."` tbedlo, `--no-system` tfiha). `--no-log` disables logging.
Errors (403/429/network) never logged — ghir ajwiba s7a7a.
Privacy: prompts go to Google — don't send private CSVs.

Backends khrin (ila Google blocka project dyalek — `--backend`):
```powershell
ollama create ghassan-qwen -f ollama/Modelfile   # GGUF local + persona Ghassan
python tools/gemini_chat.py --backend ollama --model ghassan-qwen --prompt "salam"
$env:OPENROUTER_API_KEY = "..."   # https://openrouter.ai/keys
python tools/gemini_chat.py --backend openrouter --model "<model>:free" --prompt "salam"
```
Log format (id/question/response, mkammel bin runs): `{"id":1,"ts":...,"backend":"ollama",
"model":"ghassan-qwen","prompt":"...","response":"..."}`. `--max-tokens` (ollama,
default 256) bound l jawab; `think:false` + `<think>` strip kay7eydo reasoning traces.
