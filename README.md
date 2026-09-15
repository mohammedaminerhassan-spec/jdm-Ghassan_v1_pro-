# Ghassan AI v1 — T4-ONLY Moroccan Darija LLM (C++20, zero-dependency)

> **T4-ONLY build:** CPU (OpenMP) + CUDA sm_75 (Tesla T4). No Metal/Vulkan/TPU code paths by design.
> **Zero-warning policy:** MSVC `/WX` + GCC `-Werror`. Any warning fails the build.
> **1B MoE:** ~1.02B total / ~408M active (top-2/8 + shared expert), GQA, RoPE+YaRN, QK-Norm, SwiGLU-MoE, z-loss.

## Quick start (local weak PC — CPU fast path)

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\bin\Release\ghassan-ai.exe chat --model model.gai --device cpu
.\build\bin\Release\ghassan-ai.exe generate --model model.gai --prompt "labas, kidayer?" --max-tokens 128 --device cpu
```

CPU speed tricks (already default):
- OpenMP pinned to all cores (`--threads N` to override)
- KV-cache contiguous per-layer, greedy decode for max tok/s: `--greedy`
- Quantized model for weak RAM: `ghassan-ai quantize --model fp16.gai --out int4.gai --profile int4`
- Bench your PC: `ghassan-ai bench --model model.gai --tokens 64`

## Kaggle T4 training (flagship 1B)

```bash
bash kaggle/setup.sh
bash kaggle/build_billion_data.sh
bash kaggle/train_1b.sh                 # pretrain -> SFT -> GGUF export -> Darija smoke test
# 4xT4 DDP:
./train_4xt4.sh configs/t4_1b.yaml
```

Memory budget (T4 16GB, Lion mandatory, B=1, T=512, grad_accum=128):
`4.09GB weights + 4.09GB grads + 4.09GB Lion + 0.45GB MoE caches + 0.13GB logits + 0.25GB eval + 1GB slack + 0.32GB NCCL ≈ 14.7GB`
Guards fail fast before OOM: attention T² >800MB = error, total >95% = error, >85% = warn.

## Layout

- `core/` Tensor/Device/Ops (CPU ref + CUDA mirror, no autograd graph — manual backward + `test_gradcheck`)
- `model/` pre-norm Transformer-MoE (DeepSeek-style routing + shared expert)
- `tokenizer/` byte-level BPE Darija-aware (32k, byte fallback, never UNK)
- `training/` Trainer + GBIN DataLoader (streaming, domain mix) + Lion/AdamW + WSD/cosine + Checkpoint + NCCL DDP
- `inference/` Generator (prefill + KV-cache decode) + Sampler (temp/top-k/top-p/min-p/repeat)
- `cuda/` kernels/attention/moe (FP16 tensor cores, fp32 masters + loss scaling)
- `quantization/` Q8_0/Q4_0/Q4_1/F16/BF16 + error metrics
- `format/` `.gai` self-contained (model+tok+cfg) + GGUF bridge
- `evaluation/` 14-category Darija benchmark + human sheet
- `tools/` `ghassan-ai` (chat/generate/info/quantize/bench/tokenize/eval/export/devices) + `gai_train`
- `configs/` `t4_1b.yaml` flagship, `ultra_1b.yaml`, `sft_*`, `flash/pro`
- `tests/` `test_gradcheck` (finite-diff safety net) + model/cuda/tiny_train/tokenizer/t4_fixes

## Tests

```bat
ctest --test-dir build -C Release --output-on-failure
.\build\bin\Release\test_gradcheck.exe
.\build\bin\Release\test_model.exe
```

## Why faster than big generic models on T4 / weak CPU?

- One arch (sm_75), no fatbins, no dead backends
- Compute-only FP16 GEMMs (large only) + fp32 masters + dynamic loss scaling
- Fused multi-tensor grad-norm (1 sync, not ~200), GPU MoE aux fractions (no host roundtrip)
- Activation recompute (probs per-layer, not per-model), streaming shards (0.4MB/50M tok, not GBs)
- Lion optimizer (50% opt memory vs AdamW), tied embeddings, GQA 4x smaller KV
