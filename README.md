
---

# Comprehensive Codebase Analysis: Ghassan AI Model

## 1. What the Model Is and Does

**Ghassan AI** is a **decoder-only pre-norm Transformer language model** with **Mixture-of-Experts (MoE)** in every FFN block, built from scratch in C++/CUDA. It is designed to train on **NVIDIA Tesla T4 GPUs** (16GB VRAM, sm_75) in Kaggle notebook environments.

The model comes in several sizes:
- **Flash (~467M total, ~191M active)**: 26 layers, 768 hidden, 12 heads, 4 KV heads, 8 experts top-2 + shared
- **Pro 480M**: 26 layers, 768 hidden, 12 heads, 4 KV heads, 8 experts top-2 + shared, QK-Norm
- **Pro 1B (~1.02B total, ~408M active)**: 26 layers, 1024 hidden, 16 heads, 4 KV heads, 8 experts top-2 + shared
- **Pro 1B (36-layer)**: 36 layers, 768 hidden, 12 heads, 4 KV heads, 10 experts top-2 + shared

Key architectural features:
- **DeepSeek-style MoE**: top-k routing with a shared always-on expert, load-balance aux loss, aux-loss-free bias option (V3 style)
- **Grouped-Query Attention (GQA)**: 4 KV heads shared across 12-16 query heads
- **RoPE** with NTK/YaRN scaling support, interleaved or NeoX half-rotate
- **QK-Norm** (per-head QK RMSNorm) for MoE stability
- **SwiGLU** activation in FFN
- **Tied embeddings** (input embedding = output lm_head)
- **Sliding Window Attention (SWA)** support
- **Z-loss** for router logit penalty

---

## 2. Full Directory Structure and File Purposes

```
Ghassan Ai model/
├── CMakeLists.txt                    # Build system (CUDA, OpenMP, Parquet, NCCL)
├── .gitignore
├── .gitattributes
│
├── configs/                          # YAML training configurations
│   ├── en_2xt4.yaml                  # 2xT4 DDP 480M pretrain config
│   ├── en_ollama.yaml                # Ollama deployment config
│   ├── en_pro.yaml                   # Single-T4 480M pretrain
│   ├── pro_auxfree.yaml              # Aux-loss-free MoE variant
│   ├── pro_v1.yaml                   # Flagship 1B single-T4 pretrain
│   ├── sft_en_2xt4.yaml              # 2xT4 SFT config
│   ├── sft_en_4xt4.yaml              # 4xT4 SFT config
│   ├── sft_en_pro.yaml               # Single-T4 SFT
│   ├── sft_pro_v1.yaml               # 1B SFT
│   ├── smoke.yaml                    # Smoke test config
│   ├── synth_billion.yaml            # Synthetic data 1B config
│   ├── synth_large.yaml              # Synthetic data large config
│   └── t4_1b.yaml                    # 4xT4 DDP 1B pretrain
│
├── core/                             # Foundation layer
│   ├── common.h / common.cpp         # Logging, error handling, utilities
│   ├── config.h / config.cpp         # YAML subset parser
│   ├── device.h / device.cpp         # Device abstraction (CPU/CUDA), memory queries
│   ├── dtype.h / dtype.cpp           # Data type definitions (F32, F16, BF16, I32)
│   ├── mmap.h / mmap.cpp             # Memory-mapped file I/O
│   ├── ops.h / ops.cpp               # CPU operation dispatch, GEMM, MoE CPU path
│   ├── ops_cpu.h / ops_cpu.cpp       # CPU reference implementations
│   ├── ops_moe.h / ops_moe.cpp       # CPU MoE reference
│   ├── rng.h                         # Deterministic RNG (xorshift-style)
│   ├── signals.h / signals.cpp       # SIGINT/SIGTERM handlers for graceful shutdown
│   ├── tensor.h / tensor.cpp         # Tensor class (refcounted storage, CPU/CUDA)
│   └── unicode.h / unicode.cpp       # Unicode utilities
│
├── cuda/                             # CUDA/C++ GPU kernels
│   ├── attention.cu                  # Flash-style causal GQA attention (fwd/bwd/decode)
│   ├── cuda_ops.h                    # CUDA operation declarations
│   ├── cuda_utils.h / cuda_utils.cu  # CUDA utilities (cublas handle, device memory check)
│   ├── kernels.cu                    # Elementwise, RMSNorm, RoPE, SwiGLU, CE, AdamW, Lion, sampling
│   └── moe.cu                        # MoE routing, grouped dispatch, fused elementwise, backward
│
├── model/                            # Model definition
│   ├── model.h / model.cpp           # Model class, parameters, activations, forward/backward
│
├── training/                         # Training infrastructure
│   ├── checkpoint.h / checkpoint.cpp # Save/load/resume (weights + optimizer + scheduler + RNG)
│   ├── dataloader.h / dataloader.cpp # Shard reader (.gbin format), streaming, batch sampling
│   ├── distributed.h / distributed.cpp # NCCL DDP context, all-reduce, broadcast
│   ├── optimizer.h / optimizer.cpp   # AdamW, Lion, Muon optimizers
│   ├── scheduler.h                   # LR scheduler (cosine, WSD)
│   ├── trainer.h / trainer.cpp       # Main training loop, DDP orchestration, checkpoint writer
│
├── tokenizer/                        # Tokenizer
│   ├── bpe_trainer.h / bpe_trainer.cpp   # BPE tokenizer training
│   ├── chat_template.h / chat_template.cpp # Chat template formatting
│   ├── normalizer.h / normalizer.cpp     # Text normalization
│   └── tokenizer.h / tokenizer.cpp       # Tokenizer main
│
├── dataset/                          # Data pipeline
│   ├── cleaner.h / cleaner.cpp       # Text cleaning
│   ├── corpus_stats.h / corpus_stats.cpp # Corpus statistics
│   ├── dedup.h / dedup.cpp           # Deduplication
│   ├── english_dialogue_data.h/.cpp  # English dialogue data processing
│   ├── english_logic.h / english_logic.cpp # English logic data
│   ├── english_synth.h / english_synth.cpp # Synthetic English data
│   ├── json_reader.h / json_reader.cpp # JSON data reader
│   ├── langid.h / langid.cpp         # Language identification
│   ├── parquet_reader.h / parquet_reader.cpp # Apache Arrow Parquet reader
│   ├── retrieval.h / retrieval.cpp   # Data retrieval
│   ├── synth.h / synth.cpp           # Synthetic data generation
│   ├── synth_data.h / synth_data.cpp # Synthetic data structures
│   └── english_parquet/              # Parquet data files (21 .parquet shards)
│
├── inference/                        # Inference engine
│   ├── chat.h / chat.cpp             # Chat interface
│   ├── generator.h / generator.cpp   # Text generation
│   ├── kv_cache.h / kv_cache.cpp     # KV cache with ring buffer + pinned prefix
│   └── sampler.h / sampler.cpp       # Token sampling (top-k, penalties, etc.)
│
├── evaluation/                       # Evaluation
│   ├── benchmark.h / benchmark.cpp   # Benchmarking
│   └── perplexity.h / perplexity.cpp # Perplexity evaluation
│
├── format/                           # Model export
│   ├── gai_format.h / gai_format.cpp # Native .gai format
│   └── gguf_format.h / gguf_format.cpp # GGUF format export
│
├── quantization/                     # Quantization
│   └── quantize.h / quantize.cpp     # INT4/INT8 quantization
│
├── tools/                            # CLI tools
│   ├── cli_common.h / cli_common.cpp # CLI argument parsing
│   ├── corpus_stats_main.cpp         # Corpus statistics tool
│   ├── data_pipeline_main.cpp        # Data pipeline tool (parquet -> shards)
│   ├── main.cpp                      # Main CLI (ghassan-ai)
│   ├── train_main.cpp                # Training entry point (gai_train)
│   └── train_tokenizer_main.cpp      # Tokenizer training entry point
│
├── kaggle/                           # Kaggle notebook scripts
│   ├── configs/pilot_moe.yaml        # Pilot MoE config
│   ├── build_billion_data.sh         # Build 1B token dataset
│   ├── build_english_data.sh         # Build English dataset from parquet
│   ├── cell6_app_chain.sh            # App chain cell
│   ├── cell7_engines.sh              # Engines cell
│   ├── cell8_pilot.sh                # Pilot cell
│   ├── convert_data.sh               # Data conversion
│   ├── convert_final.sh              # Final data conversion
│   ├── package.ps1 / package.sh      # Packaging scripts
│   ├── setup.sh                      # Build + setup script
│   ├── train.sh                      # Single-T4 training launcher
│   ├── train_1b.sh                   # 1B model training launcher
│   ├── train_2xt4.sh                 # 2xT4 DDP training launcher
│   ├── train_4xt4.sh                 # 4xT4 DDP training launcher
│   ├── train_full.sh                 # Full training launcher
│   └── verify_fixes.sh               # Verification script
│
└── tests/                            # Test suite (28 test files)
    ├── test_ckpt_backpressure.cpp    # Checkpoint backpressure
    ├── test_configs.cpp              # Config validation
    ├── test_contamination_blocklist.cpp
    ├── test_device_stage.cpp
    ├── test_dtype.cpp
    ├── test_ema_bias_persist.cpp
    ├── test_english_dialogue.cpp
    ├── test_fp16_cache_order.cpp
    ├── test_fp16_gemm_parity.cpp
    ├── test_gradcheck.cpp
    ├── test_host_resources.cpp
    ├── test_kv_ring.cpp
    ├── test_memory_plan.cpp
    ├── test_moe_bias_step.cpp
    ├── test_moe_cuda_parity.cpp
    ├── test_moe_fused.cpp
    ├── test_muon_gate.cpp
    ├── test_perplexity_harness.cpp
    ├── test_regressions.cpp
    ├── test_resume_mode.cpp
    ├── test_save_barrier.cpp
    ├── test_sce_accumulate.cpp
    ├── test_sequence_pack.cpp
    ├── test_sft_mask.cpp
    ├── test_snapshot_memory.cpp
    ├── test_telemetry.cpp
    ├── test_trainer_ckpt_flow.cpp
    └── test_vocab_gate.cpp
```

---

## 3. Training Architecture and Distributed Training (DDP)

### Training Loop Design

The training loop in `trainer.cpp` follows this pattern:

1. **Schedule resolution**: Either `max_steps` or `epochs` mode (mutually exclusive, fail-fast if ambiguous)
2. **Per-step loop**:
   - Zero gradients
   - Set MoE jitter seed (deterministic per step + rank)
   - Inner loop over `grad_accum` micro-batches:
     - Fetch batch (async prefetch from background thread)
     - `forward_backward_micro()` -- forward + backward, accumulate grads
   - **DDP gradient sync** (`sync_gradients()`)
   - **Global token count sync** (`sync_ntok_sum()`)
   - Unscale + average gradients by global supervised token count
   - Optimizer step (AdamW/Lion/Muon)
   - MoE aux-free bias sync (if enabled)
   - Logging, evaluation, checkpointing

### DDP Implementation

The DDP uses **NCCL** for multi-GPU communication:

- **Initialization** (`distributed.cpp`): File-based rendezvous (rank 0 publishes NCCL unique ID to `/tmp/gai_nccl_<port>.id`, other ranks read it). Supports SLURM, torchrun, and manual env vars.
- **Gradient synchronization** (`sync_gradients()`): Fused bucketed all-reduce SUM. Large tensors (>=1M elements) get individual NCCL calls; small tensors are packed into a persistent staging buffer and reduced once. **No division by world_size** -- the caller divides by exact global token count.
- **Model broadcast** (`sync_model()`): Broadcasts all weights from rank 0 to ensure identical initialization.
- **Token count sync** (`sync_ntok_sum()`): All-reduces supervised token count across ranks for proper loss scaling.
- **Eval best decision sync** (`sync_eval_best()`): Broadcasts the "is this the best model?" decision so all ranks enter `save()` together (prevents NCCL collective desync).
- **MoE bias sync** (`sync_moe_bias()`): All-reduces expert slot counts, applies EMA bias update once per optimizer step.

### DDP Data Handling

- **Rank-aware sampler seeds**: `train_seed = base_seed + rank * 1000003` (disjoint streams, deterministic)
- **Validation seed**: Same on all ranks (but eval runs on rank 0 only)
- **DDP resume**: Rank 0 restores via `set_state()`; other ranks rebuild via `reseed(rank_seed) + skip(saved_batches)`
- **World size mismatch on resume**: Fails fast if checkpoint was saved with different world size

### Potential DDP Issues Found

1. **No gradient averaging by world_size**: The code uses pure SUM all-reduce and divides by `ntok_global` (global supervised token count). This is **correct** for token-weighted gradients but means the effective learning rate scales with the number of tokens, not the number of GPUs. This is intentional (DeepSeek-style token-weighted training).

2. **NCCL collective ordering**: The code is very careful about collective ordering -- every rank must enter collectives the same number of times. The `sync_eval_best()` broadcast prevents rank 0 from entering `save()` alone. The `save()` function has barriers before and after the async write.

3. **No gradient bucketing by size threshold**: The threshold `kLarge = 1 << 20` (1M elements) determines individual vs. packed all-reduce. This is reasonable but could be tuned.

4. **Single-node only**: The DDP assumes all GPUs are on the same node (file-based rendezvous in `/tmp`). No multi-node support.

5. **No gradient compression**: Full fp32 all-reduce, no fp16/bf16 gradient compression.

---

## 4. CUDA/C++ Kernel Code

### `cuda/attention.cu` -- Flash-Style Causal GQA Attention

- **Forward**: Tiled flash attention with online softmax. One warp per query position. K/V tiles in shared memory. O(T) memory per query row (not O(T^2)). Supports segment IDs for sequence packing.
- **SWA (Sliding Window Attention)**: Separate kernel for windowed attention. Fixed a 32x overcount bug where warp reduction was incorrectly applied.
- **Backward**: One block per (batch, kv-head, query-tile). Recomputes attention probs per layer (activation checkpointing). Uses atomicAdd for dk/dv accumulation.
- **Decode**: One block per head, warp-parallel over KV length. Supports ring buffer cache with pinned prefix.

### `cuda/kernels.cu` -- Elementwise and Utility Kernels

- **GEMM**: cuBLAS wrapper with FP16 tensor-core fast path (compute-only mixed precision, fp32 masters). BF16 path for Ampere+.
- **RMSNorm**: Forward and backward with block-level reductions.
- **RoPE**: Interleaved and NeoX half-rotate variants, with YaRN NTK scaling.
- **SwiGLU**: Forward and backward.
- **Cross-Entropy**: Fused online softmax + loss + dlogits in one kernel. Supports z-loss. Chunked CE with device-side accumulation (zero host traffic during micro-batch loop).
- **AdamW/Lion**: Fused optimizer step kernels.
- **Sampling**: Repetition penalties (histogram-based, O(V+history) instead of O(V*history)), top-K (exact for K<=128), argmax.
- **Global norm**: Fused multi-tensor gradient norm (one sync instead of 200).

### `cuda/moe.cu` -- MoE GPU Implementation

- **Routing**: Warp-per-token router (32 threads cooperate per token). Stable softmax + top-K with jitter. Aux-loss-free bias variant.
- **Grouped dispatch**: Slots stay on GPU, only expert counts cross host. Fused elementwise kernels (pack, save, scatter-add, gather, scale).
- **Forward**: Router logits -> route -> shared expert -> routed experts (grouped GEMMs) -> scatter-add.
- **Backward**: Recompute shared expert, gather saved activations, per-expert GEMMs for weight grads, router backward with DL^T @ x GEMM (no atomics for router weight grad).
- **Aux fractions**: Device-side computation, only ne floats cross host.

---

## 5. Memory Management Patterns

### GPU VRAM Management

1. **Monotonic workspace pools**: Both GEMM (`g_ws`) and MoE (`g_moe_ws`) workspaces grow monotonically in 64MB chunks. Never freed until shutdown. Pre-sized from recipe before first step (`reserve_workspaces()`).

2. **FP16 weight cache**: Persistent fp16 copies of all 2D weights + fused QKV per layer. Lazy refresh on `forward_body()` when dirty. Saves fp32->fp16 conversion overhead.

3. **Activation checkpointing**: Optional batch-splitting (B=2 -> 2x [1,T] segments). Peak activation memory drops ~Sx. Math stays identical (grads sum linearly).

4. **Chunked cross-entropy**: `ce_chunks` splits the [N,V] logits/dlogits into row-block scratch. 4 chunks saves ~393MB at B=2,T=1024,V=32k.

5. **Attention probs checkpointing**: Attention probs are NOT stored per layer (would be O(T^2) x L = 2.6GB). Recomputed per layer in backward into a single transient buffer.

6. **OOM guards**: Pre-flight memory check with 95% fail / 85% warn thresholds. 1B model has specific guards (batch_size must be 1, seq_len must be 512).

### Host RAM Management

1. **Checkpoint writer**: Background thread with bounded queue. RAM budget = physical RAM - 8GB reserve, clamped to [8GB, 32GB]. Backpressure when budget exceeded.

2. **Snapshot dedup**: Same-step best+last share one snapshot via `shared_ptr`. Hard links for same-step different-name saves.

3. **Streaming dataloader**: Header-only shard loading (doc table in RAM, tokens streamed from disk). Per-thread cached file descriptors.

4. **Async prefetch**: Background thread loads next batch while GPU computes current step.

### Potential Memory Issues Found

1. **Workspace pools never freed during training**: The `g_ws` and `g_moe_ws` pools grow monotonically and are only freed at shutdown. If the recipe is misconfigured, these could grow larger than needed. However, pre-sizing from `workspace_plan()` mitigates this.

2. **Checkpoint snapshot retention**: The `snapshot_cache_` is reset after each step's saves are queued, but between reset and queue, there's a window where 2 full snapshots exist. The backpressure mechanism handles this correctly.

3. **No explicit `cudaDeviceSynchronize()` before memory checks**: The OOM guard uses arithmetic estimates, not actual `cudaMemGetInfo()`, so fragmentation could cause OOM despite the estimate passing.

4. **Thread-local staging buffers**: `t_u16_stage` in the dataloader and `toks`/`masks` vectors in `fill_from_shard` are thread-local and grow monotonically. These are bounded by `seq_len` so they're small.

5. **KV cache ring buffer**: The `KVCache` class uses a ring buffer with a pinned prefix. When the cache is full, it overwrites the oldest entries. The `allocate_slot()` method correctly handles wraparound.

---

## 6. Data Pipeline Design

### Shard Format (.gbin)

```
[magic "GBIN"][version][dtype (0=u16, 1=u32)][flags]
[n_tokens][n_docs]
[token stream]
[doc table: n_docs * u64 start offsets]
[loss mask (optional, flags&1): n_tokens bytes]
```

### Data Flow

1. **Parquet files** (21 shards in `english_parquet/`) -> `data_pipeline parquet-corpus` -> text corpus
2. **Text corpus** -> `train_tokenizer` -> BPE tokenizer (32k vocab, keep-case)
3. **Parquet + tokenizer** -> `data_pipeline` -> `.gbin` shards (train/val, with optional loss masks)
4. **Shards** -> `DataLoader` -> random window sampling -> batches

### DataLoader Features

- **Streaming mode**: Header-only load, tokens read from disk on demand (T4 RAM saver)
- **Uniform sampling**: Random window from random shard, weighted by token count
- **Domain mixture sampling**: Weighted sampling across `train_<domain>_*.gbin` groups
- **Sequence packing**: Multiple short docs packed end-to-end with block-causal mask
- **Deterministic**: Seed-based, fully checkpointable (RNG state + batch count)
- **Fast forward**: O(1) RNG replay for DDP resume (no disk reads)
- **Async prefetch**: Background thread loads next batch

### Potential Data Issues Found

1. **No data validation at load time**: The `Shard::load_header()` validates magic, version, and doc offsets, but doesn't check token ranges. An out-of-range token would read OOB embeddings silently.

2. **Thread-local file descriptor cache**: The `ShardFdCache` is thread-local, which is correct for the prefetch thread, but if multiple threads access the same shard, each gets its own fd. This is fine for the current single-prefetch-thread design.

3. **No data sharding across DDP ranks**: Each rank reads from the same shard set but with different RNG seeds. This means different ranks may read the same shard simultaneously, causing disk I/O contention. However, the random window sampling makes it unlikely they read the same region.

4. **Loss mask handling**: The loss mask is optional and per-token. The `fill_from_shard` function correctly applies masks to targets (setting -100 for masked positions). The `all_shards_have_mask()` check enforces SFT shards have masks.

---

## 7. Configuration System

The config parser is a minimal YAML subset parser supporting:
- Nested maps (2-space indent)
- Scalars, inline lists, comments, quoted strings
- Dotted key access (`model.hidden_size`)
- Strict mode (`--strict-config`) that fails on unknown keys
- Known-key validation to catch typos

Key configs:
- `en_2xt4.yaml`: 2xT4 DDP, 480M model, B=2 T=1024, AdamW, WSD scheduler, 2 epochs
- `t4_1b.yaml`: 4xT4 DDP, 1B model, B=1 T=512, Lion optimizer, 2 epochs
- `pro_v1.yaml`: Single-T4, 1B 36-layer, B=1 T=512, Lion, 2 epochs

---

## 8. Kaggle Notebooks

The `kaggle/` directory contains shell scripts for Kaggle GPU sessions:

- **`setup.sh`**: Builds from source, installs Apache Arrow, trains tokenizer, verifies data
- **`train_2xt4.sh`**: Full 2xT4 DDP pipeline (preflight -> pilot -> PT -> SFT -> export GGUF)
- **`train_4xt4.sh`**: 4xT4 DDP launcher (simpler, no preflight)
- **`train_1b.sh`**: Single-T4 1B training launcher
- **`build_english_data.sh`**: Builds .gbin shards from parquet lake

The Kaggle scripts include extensive preflight checks: GPU count, NCCL availability, config parity, tokenizer vocab match, shard presence, VRAM plan, output quota projection.

---

## 9. Notable Bugs and Issues Found

### Fixed Bugs (documented in comments)

1. **DDP startup deadlock**: `sync_model()` was only called on non-rank-0, causing ranks 1-3 to hang. Fixed by calling on all ranks.
2. **Stale NCCL ID file**: Crashed runs left stale ID files. Fixed by unlinking before spawn.
3. **Loss scaler in FP32 mode**: Scaler was applied even when `gemm_fp16` was off, suppressing gradients by 1/65536. Fixed by parking scaler when FP32-only.
4. **SWA 32x overcount**: Warp reduction was incorrectly applied when all lanes computed the same sum. Fixed by removing reduction.
5. **MoE memory storm**: Old kernels were slot-serial with scalar loads. Fixed with vectorized float4 access.
6. **Router gradient atomics**: Old version did ne*d atomicAdds per token. Fixed with DL^T @ x GEMM.
7. **Checkpoint double-save**: Final save could duplicate the last step's save. Fixed by checking if already saved.
8. **DDP resume stall**: Replaying GBs from disk to resume. Fixed with O(1) fast_forward.
9. **NCCL collective desync on save**: Rank 0 could enter save() alone. Fixed with broadcast of best decision.
10. **fp16 cache stale after resume**: Cache wasn't invalidated after weight restore. Fixed with `mark_weights_dirty()`.

### Potential Remaining Issues

1. **`k_route` kernel jitter hash**: The jitter uses a hash that includes the step seed, but the hash function `jitter_u()` may have collisions. The comment says "bit-reproducible per token" but the quality of the hash distribution isn't verified.

2. **`moe_build_groups` host sync**: The `cudaMemcpy` of `h_offsets` from device to host is a blocking sync per layer per microbatch. This is a performance bottleneck (36 layers x 128 microbatches = 4608 syncs per step). The comment acknowledges this but says "tiny ne-only host sync."

3. **`global_sq_norm_multi` partial buffer**: The workspace for gradient norm is sized as `grid * nparts` where `grid=16` and `nparts` can be ~200. This is 3200 floats = 12.8KB, which is fine, but the workspace is shared with other operations. If the workspace is too small, it could cause incorrect results.

4. **`fill_from_shard` retry logic**: The 4-attempt retry for finding a good window could theoretically loop forever if all attempts fail. The `best_avail == 0` check returns true (empty row), which could lead to silent padding.

5. **`DataLoader::next` batch allocation**: `out.ids.assign(B*T, 0)` allocates a new vector every batch. The prefetch thread helps, but the allocation itself could cause memory fragmentation over time.

6. **`KVCache` ring buffer**: When the cache is full and `allocate_slot()` is called, it overwrites the oldest entry. But the `start_` pointer wraps around, and if `pinned_` is set, the ring capacity is `max_len_ - pinned_`. If `pinned_ == max_len_`, the cache is full and has no rolling slots, which triggers a `GAI_FAIL`. This is correct but could be surprising.

7. **No `cudaMemGetInfo` check before workspace allocation**: The workspace pools use `cudaMalloc` which could fail silently if VRAM is exhausted. The `CU_CHECK` macro would catch this, but the error message might not be clear.

8. **`Shard::read_window` OOB check**: The check `start <= total && len <= total - start` is correct, but if `start > total`, the subtraction `total - start` would underflow (unsigned). The check order prevents this (short-circuit evaluation), but it's fragile.

9. **DDP world size mismatch**: If `WORLD_SIZE` env var is set but `cfg_.ddp` is false, the code forces DDP on. This could be surprising if the user intended single-GPU training but had a stale env var.

10. **`Trainer::save` snapshot capture**: The snapshot is captured on rank 0 only, then the `capture_ok` flag is broadcast. If capture fails on rank 0, all ranks are notified. But the `save_async` is only called on rank 0, so non-main ranks just hit the barrier. This is correct but means the snapshot is never captured on non-main ranks, which is fine since they don't write checkpoints.

---

## 10. Summary

This is a **well-engineered, production-grade LLM training framework** written in C++/CUDA with:

- **Solid architecture**: Clean separation of concerns (model, training, data, inference, evaluation)
- **T4-specific optimization**: Every design decision is made for the T4's 16GB VRAM constraint
- **Comprehensive DDP**: NCCL-based with careful collective ordering, rank-aware data sampling, and proper resume
- **Memory safety**: Monotonic workspace pools, pre-flight OOM guards, streaming dataloader, checkpoint backpressure
- **Rich feature set**: MoE, GQA, RoPE/YaRN, QK-Norm, SWA, sequence packing, activation checkpointing, chunked CE, 3 optimizers, 2 schedulers
- **Extensive testing**: 28 test files covering configs, parity, memory, resume, MoE, etc.
- **Kaggle integration**: Full pipeline from parquet data to trained GGUF model

The codebase shows signs of extensive debugging and production hardening, with detailed comments explaining every design decision and bug fix. The DDP implementation is particularly careful about collective ordering and data consistency across ranks.
