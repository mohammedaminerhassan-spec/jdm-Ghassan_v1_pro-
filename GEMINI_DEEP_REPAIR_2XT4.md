# Ghassan v1 Pro — Gemini Deep Repair Task: 2xT4 Kaggle Production

Repository: mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-
Base HEAD: d3322a8a0940594b6f5ae8445979a0ce195b5e74

## Mission

This is an IMPLEMENTATION task, not only a review. Inspect the actual current HEAD, reproduce the important problems, modify the source/config/test/launcher files, run validation, then perform a second independent audit.

Target: Kaggle T4x2, two NVIDIA T4 GPUs, 16 GB VRAM per GPU, about 29 GB host RAM, 4 CPU cores, 12-hour notebook session and 20 GB auto-saved /kaggle/working.

Core requirement: the project must genuinely train on BOTH GPUs through NCCL DDP. Two T4 GPUs are NOT one shared 32 GB VRAM device. Each rank owns a full model replica and must stay inside its own 16 GB budget.

Do not silently fall back to CPU or one GPU. Do not make speculative CUDA rewrites. Every high-risk change needs a regression/parity test or a measured benchmark.

## Existing audit material

Read these first, but treat them as evidence rather than proof:

- DEEP_AUDIT_REPORT_2026-09-26.md
- DEEP_T4_AUDIT_REPAIR_HANDOFF.md
- draft PR #1: audit-repair-deep-2026-09

The repository already contains substantial hardening around CUDA workspaces, MoE routing, NCCL host-buffer staging, checkpoint versioning, loss scaling, data validation, and signal handling. Preserve correct fixes.

## P0 — Fix the checkpoint RAM failure

training/trainer.cpp currently derives ckpt_ram_budget from physical RAM using approximately 35 percent, capped at 32 GB, with an 8 GB floor.

save_async currently computes approximately snapshot->bytes() * 2 and rejects the checkpoint if that exceeds the budget. The reason is that async/backpressure may temporarily keep two live copies.

The project's own audit records a 480M snapshot around 5.76 GB, roughly 12 bytes per parameter. A 1B snapshot is therefore much larger. On Kaggle's roughly 29 GB RAM, the two-copy heuristic can reject a valid 1B checkpoint or make progress impossible.

Do not simply increase the RAM limit.

Instead audit the complete lifetime of:
- model weights
- optimizer state
- gradients
- snapshot cache
- active writer snapshot
- queued snapshots
- temporary serialization buffers
- .tmp/.bak recovery files
- duplicated shared_ptr references

Then redesign checkpoint backpressure so a valid largest supported T4 recipe can actually make progress without host OOM.

Keep atomic publish and crash recovery. Do not allow a background writer failure to remain hidden for thousands of GPU steps.

Add a regression test where snapshot size is larger than the old two-copy heuristic but still fits the intended safe host-memory budget. Prove that the writer progresses rather than failing deterministically.

Also inspect whether gradients are serialized unnecessarily. If gradients are not needed for exact resume, remove them from the checkpoint format only with versioning and load/save tests. Do not break optimizer/scheduler/resume semantics.

## P0 — Fix SFT architecture parity

configs/sft_en_4xt4.yaml is not a complete architecture match for configs/en_pro.yaml.

Important model-function fields that must be checked include:
- use_qk_norm
- z_loss_scale
- rope_scale
- rope_yarn_mscale
- rope_yarn_low
- rope_yarn_high
- sliding_window
- rope_type

en_pro enables QK-Norm and non-zero z-loss. A pretrained checkpoint followed by SFT must not silently construct a different model function.

Create a canonical architecture identity and enforce it in both checkpoint metadata and config validation where appropriate.

Add a config-parity regression test comparing all model-function fields. Do not compare training-only fields such as learning rate, batch size or optimizer.

An architecture mismatch must fail before long GPU training.

## P0/P1 — Make 2xT4 DDP first-class

kaggle/train_4xt4.sh is currently a 4-GPU launcher. It checks for at least four GPUs and defaults WORLD_SIZE to four. This prevents a normal Kaggle T4x2 run.

Implement a production-safe kaggle/train_2xt4.sh, or refactor into a generic safe DDP launcher plus compatibility wrappers.

The 2-GPU launcher must:
- verify nvidia-smi exists;
- verify at least two GPUs are visible;
- default WORLD_SIZE=2;
- spawn exactly two ranks;
- set RANK and LOCAL_RANK correctly;
- preserve all visible GPUs when the engine maps LOCAL_RANK to device;
- verify the binary was compiled with NCCL;
- remove stale rendezvous state;
- use safe single-node NCCL settings;
- trap SIGINT/SIGTERM;
- kill sibling ranks if one rank fails;
- propagate non-zero status;
- run strict config/data/tokenizer/memory preflight before long training;
- support pilot-only and preflight-only modes;
- keep a safety margin below the 12-hour Kaggle session limit.

Do not merely change the number 4 to 2. Test the complete two-rank lifecycle.

## P1 — NCCL cannot be optional when DDP is requested

CMake currently treats NCCL as optional and can produce a single-GPU build when NCCL is absent.

Keep optional NCCL for portable single-GPU builds, but make an explicit DDP request fail during preflight if the binary lacks NCCL.

kaggle/setup.sh should verify CUDA and NCCL capability before the 2-GPU training path is allowed to start.

Never spend a long build/training session and discover at runtime that DDP was not compiled.

## P1 — Measure real per-GPU VRAM

configs/t4_1b.yaml claims about 14.7 GB per T4 for the 1B recipe with fp32 masters, Lion, batch 1 and sequence 512. That is close to the 16 GB hardware limit.

Do not trust only the arithmetic estimate.

Instrument a real CUDA pilot and record:
- free VRAM before model creation;
- model allocation;
- optimizer allocation;
- forward peak;
- backward peak;
- DDP all-reduce peak;
- optimizer peak;
- checkpoint-capture peak;
- evaluation peak.

Reject unsafe recipes before a long run. Do not silently change batch or sequence length.

Keep the same per-GPU B/T memory envelope when moving from one rank to two ranks. Use gradient accumulation deliberately to control the global token/update size.

## P1 — Global batch and DDP math

Audit the exact contract:
global tokens per optimizer update = batch_size × seq_len × grad_accum × world_size.

Verify gradient all-reduce, token-count all-reduce, clipping, normalization, optimizer stepping and scheduler stepping.

Do NOT blindly multiply learning rate because there are two GPUs.

Define and test whether the 2-GPU recipe preserves:
- per-GPU memory;
- global tokens/update;
- optimizer-step schedule;
- total tokens consumed.

Add a deterministic tiny-model test comparing world_size=1 and world_size=2 normalization.

## P1 — World-size resume

Current trainer code rejects a checkpoint saved with a different DDP world size. Preserve this safety rule unless a fully tested migration mechanism is implemented.

A checkpoint created with WORLD_SIZE=4 should not silently resume with WORLD_SIZE=2 if rank-specific RNG/data streams would change.

For production, it is acceptable to require the same WORLD_SIZE and fail loudly otherwise.

## P1 — Saved-output quota

kaggle/train_1b.sh uses approximately --output-budget-mb 19456. The old 4xT4 launcher does not wire this gate consistently.

The 2xT4 production path must project the Kaggle saved-output footprint before training.

Count:
- repository/build output;
- training shards;
- checkpoints;
- GGUF export;
- temporary files;
- backups;
- persisted copies.

Do not confuse free filesystem space with Kaggle auto-saved output quota.

## P1 — SFT data mixture

Current sft_en_4xt4.yaml uses english_chat 0.65 and english_instruction 0.35, while the flagship SFT recipe also has an english_behavior component.

The user wants natural assistant/dialogue behavior, not only instruction answering.

Inspect the actual shard domains. If english_behavior is present and intended, include it explicitly in the 2xT4 SFT recipe. If it is absent, fail preflight rather than silently pretending it exists.

Preflight must print declared weight, shard count and effective weight for every mixture domain.

## P1 — Data pipeline audit

Audit shard discovery, train/validation separation, masks, SFT loss masks, sequence packing, segment masks, tokenizer vocabulary, corrupted shard handling, file descriptors, prefetch, rank sharding and duplicate/skipped samples.

Existing data hardening should be preserved.

Wrong tokenizer vocabulary, missing declared shard domains or malformed shards must fail before a long GPU run.

## Performance audit

Search globally for hot-path:
- cudaMalloc/cudaFree;
- cudaDeviceSynchronize;
- unnecessary stream synchronization;
- D2H/H2D copies;
- CPU roundtrips;
- per-step allocations;
- file open/close;
- vector growth;
- excessive logging;
- CPU mutex contention.

Existing persistent CUDA workspaces should be verified rather than rewritten.

Known MoE performance leads:

1. cuda/moe.cu may copy tiny expert offsets/counts from device to host before variable-size expert GEMMs. Do not simply delete the synchronization. Measure it and investigate grouped GEMM/cuBLASLt/device-driven alternatives.

2. Per-expert gate/up/down GEMMs may cause many small launches. Measure launch count and cost. Do not replace it with giant padded buffers without proving the memory/performance tradeoff.

Only merge a performance optimization after CPU/CUDA parity and real T4 benchmark evidence.

## CUDA/model correctness

Audit attention, RoPE, QK-Norm, GQA, sliding-window masks, backward recomputation, MoE routing, top-k, jitter, aux loss, expert dimensions, zero-token experts and partial blocks.

Test edge cases:
- T=1;
- odd sequence lengths;
- T=32;
- T=512;
- T=1024;
- K=1;
- K=2;
- zero-token experts;
- highly imbalanced routing;
- frozen parameters;
- zero gradients.

Do not rewrite attention or MoE kernels speculatively.

## Loss scaling / optimizer

T4 production path uses fp32 master parameters with FP16 GEMMs and dynamic loss scaling.

Preserve the measured stable loss-scale initialization unless new tests justify a change.

Add a controlled overflow test proving:
- update is skipped;
- optimizer counter behavior is correct;
- scheduler behavior is correct;
- loss scale decreases;
- training recovers.

Audit AdamW, Lion and Muon state/save/resume semantics. Do not destabilize the shipped Lion path merely to experiment with Muon.

## Tests required after repair

At minimum add/strengthen:

- pretrain/SFT architecture parity test;
- tokenizer/model vocab test;
- large-checkpoint/backpressure test;
- checkpoint interrupted-save test;
- exact resume test;
- world-size mismatch test;
- 2-rank DDP smoke;
- best/last checkpoint collective test;
- one-rank failure test;
- stale NCCL rendezvous test;
- CUDA attention/MoE parity;
- loss-scaling overflow test;
- stable VRAM soak;
- stable host-RSS soak.

The repository already has kaggle/verify_fixes.sh with a 2-rank DDP smoke. Reuse and strengthen it rather than duplicating fragile logic.

## Required validation ladder

1. Release build with -Werror.
2. All CPU tests.
3. CUDA parity tests.
4. Tiny deterministic training.
5. Checkpoint save/load/resume equivalence.
6. Real 2-rank DDP smoke with timeout.
7. Real 2xT4 pilot.
8. Longer 2xT4 soak with telemetry.
9. Second independent source audit after modifications.

Use CUDA_LAUNCH_BLOCKING=1 and compute-sanitizer only for diagnostic tests where appropriate; do not leave expensive debug settings enabled for production.

## Production 2xT4 launcher contract

Create a clear entry point such as kaggle/train_2xt4.sh.

Required flow:

PRECHECK -> NCCL/CUDA CHECK -> CONFIG PARITY -> DATA CHECK -> TOKENIZER CHECK -> VRAM PLAN -> OUTPUT QUOTA -> 2-RANK SMOKE/PILOT -> TRAIN -> CHECKPOINT -> OPTIONAL EXPORT -> FINAL SMOKE.

The launcher should calculate real global throughput from the two ranks, not report one-rank throughput as if it were global.

Keep a safety margin from the Kaggle 12-hour limit. Never promise that arbitrary pretraining will finish in one session; use the measured pilot to determine how many steps fit.

## Important memory rule

Do not change the model architecture just because two GPUs are available.

Each T4 must carry a complete DDP replica. There is no automatic 32 GB shared VRAM.

Keep per-GPU batch/sequence at the validated recipe. If global batch is adjusted through grad_accum, document the exact optimization contract and test it.

## Second audit

After implementation, search again for:
- silent CPU fallback;
- WORLD_SIZE/RANK/LOCAL_RANK mistakes;
- NCCL collective order mismatches;
- unchecked CUDA allocations;
- device/host pointer confusion;
- cudaMemcpy direction mistakes;
- unbounded global/static buffers;
- checkpoint duplicate references;
- .tmp/.bak accumulation;
- output-budget bypasses;
- config keys parsed but ignored;
- hard-coded 4-GPU assumptions;
- hard-coded 1-GPU assumptions;
- rank-specific code that does not execute symmetrically.

## Final report required from Gemini

Return exact modified files and for every fix provide:
- root cause;
- code change;
- why it is safe;
- test proving it;
- benchmark evidence if performance-related;
- measured GPU0 peak VRAM;
- measured GPU1 peak VRAM;
- measured host RAM peak;
- checkpoint transient memory;
- steady-state tokens/sec;
- synchronization/data-wait overhead;
- remaining risks.

Do not claim fixed/verified if the code was not actually executed.

## Success condition

The project is ready only when:

1. both T4 GPUs are genuinely used;
2. each GPU stays inside its own 16 GB budget;
3. DDP collectives are symmetric and do not hang;
4. checkpointing no longer deterministically fails because of the old snapshot×2 RAM heuristic;
5. SFT cannot silently use a different model architecture;
6. NCCL absence is detected before long training;
7. declared data domains cannot silently disappear;
8. host/device memory stays bounded;
9. the pilot measures real 2-GPU throughput;
10. the second audit finds no new critical regression.

Final engineering process:
INSPECT -> REPRODUCE -> FIX -> TEST -> BENCHMARK -> RE-AUDIT -> REPORT

Do the implementation work, not only the report.