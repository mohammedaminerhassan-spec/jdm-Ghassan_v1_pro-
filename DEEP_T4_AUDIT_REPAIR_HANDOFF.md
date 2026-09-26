# Ghassan v1 Pro — Deep T4/Kaggle Audit, Repair & Training-Reliability Handoff

## Mission

This is a DEEP ENGINEERING REPAIR TASK, not a documentation review.

Audit the actual current repository HEAD end-to-end and make the project materially safer, faster, and more reliable for long training runs of Ghassan v1 Pro on a single Kaggle NVIDIA T4-class GPU.

Target properties:
- strong and numerically correct;
- high-throughput on T4;
- memory-efficient;
- stable for long runs;
- resistant to CUDA/CPU crashes, hangs, deadlocks, OOMs, ENOSPC, host OOM, and silent data corruption;
- no accidental CPU fallback when CUDA training was explicitly requested;
- no unbounded memory growth;
- no unnecessary training-time stalls;
- deterministic where the project contract requires it;
- maintainable without fragile over-optimization.

The code may remain feature-rich and large, but runtime behavior must be lean: minimize allocations, synchronizations, copies, duplicate work, and unnecessary CPU orchestration.

## Non-negotiable rules

1. Inspect the ACTUAL CURRENT HEAD first. Do not rely on an old report or stale branch.
2. Audit the whole repository, not only files mentioned below.
3. Do not stop after finding the first bug.
4. Fix issues directly in source/config/test files. Do not merely describe them.
5. Preserve correct existing behavior unless there is evidence it is wrong or harmful.
6. Do not make speculative CUDA rewrites just because they look faster on paper.
7. Every high-risk change needs an appropriate test, parity check, or benchmark.
8. Never trade numerical correctness for a small benchmark gain.
9. Never hide an error by silently falling back to CPU when CUDA was requested.
10. Avoid permanent caches/global allocations unless lifetime, maximum size, reset behavior, and memory budget are explicit.
11. Any hot-path synchronization must have a reason and, where practical, a measurement.
12. Run a second independent audit pass after repairs.
13. Report exact modified files, exact fixes, tests, benchmarks, and remaining risks.
14. Never claim "fixed" or "verified" for code that was not actually exercised.

---

# 1. Verify the existing hardening first

The repository already contains substantial prior fixes. Verify these in the current HEAD instead of blindly rewriting them:

- extended attention API carries sliding-window configuration through forward/backward;
- MoE routing keeps grouped slot data on GPU rather than the old full N*K host slot roundtrip;
- router forward uses cooperative warp-per-token work;
- router backward avoids the old atomics-heavy path;
- MoE auxiliary-loss accumulation uses a persistent device-side accumulator;
- persistent CUDA workspaces reduce allocation churn;
- FP16 fused-QKV cache memory is included in accounting;
- activation memory estimation includes transient attention T^2 storage;
- model loading validates parameter count, names, shapes, and truncation;
- config parsing rejects important ambiguities and invalid optimizer/scheduler values;
- checkpoint/versioning and resume paths contain substantial validation;
- recent Muon changes attempt to make matrix-gradient scaling explicit.

Verify the implementation and tests against current HEAD before deciding what still needs repair.

---

# 2. P0 — Crash, corruption, and undefined-behavior sweep

Search the entire codebase for every path that can cause:

- invalid device pointer;
- out-of-bounds indexing;
- use-after-free;
- double free;
- stale tensor alias;
- wrong device;
- dtype mismatch;
- stride or leading-dimension mismatch;
- integer overflow in element/byte calculations;
- signed/unsigned wraparound;
- empty tensor misuse;
- divide by zero;
- invalid sqrt/log/exp input;
- NaN/Inf propagation;
- uninitialized memory;
- stale CUDA error reported far from the real fault;
- host dereference of device memory;
- device use of freed host memory;
- filesystem race;
- corrupt checkpoint/shard;
- collective mismatch;
- deadlock;
- infinite retry loop;
- failure hidden by fallback logic.

Pay special attention to code that works on CPU but can fail on CUDA.

For every CUDA kernel family verify:
- grid/block assumptions;
- blockDim limits;
- shared-memory size;
- all index arithmetic;
- pointer lifetimes;
- supported N/B/T shapes;
- tiny and partial-block cases;
- maximum supported dimensions;
- launch error checking at useful boundaries;
- stream/lifetime ordering.

Use CUDA_LAUNCH_BLOCKING=1 or equivalent diagnostic execution when reproducing CUDA faults.

---

# 3. P0 — Memory leak and memory-growth audit

Audit all long-lived allocations:

- global/static CUDA buffers;
- workspace pools;
- Tensor caches;
- FP16 weight caches;
- activation scratch;
- optimizer scratch;
- tokenizer caches;
- DataLoader staging;
- shard windows;
- thread-local file handles;
- asynchronous checkpoint snapshots;
- temporary files.

For every allocation answer:
1. Who owns it?
2. Who releases it?
3. Can capacity grow forever?
4. Can a short test leave a huge retained peak?
5. Can a new B/T/device silently create another copy?
6. Is the lifetime intentional?
7. Is the retained maximum included in the T4 budget?

Distinguish intentional bounded caching from real leaks.

Add diagnostics where useful:
- CUDA allocated/reserved bytes;
- workspace capacities;
- optimizer state bytes;
- activation bytes;
- host staging bytes;
- checkpoint snapshot bytes;
- resident host RAM.

Acceptance criterion:
- no unbounded host or device memory growth during a long synthetic training loop with stable shapes;
- changing supported B/T must not leave unlimited stale buffers behind.

---

# 4. P0 — Kaggle host RAM and filesystem survival

Audit separately:
- free RAM;
- process RSS;
- checkpoint snapshot RAM;
- free disk;
- saved-output quota;
- temporary files.

The program must fail EARLY with a useful breakdown instead of dying hours later.

Check checkpoint writes:
- simultaneous snapshots;
- temporary files;
- rename/replace;
- interrupted write;
- disk-full handling;
- failed-save cleanup;
- resume after interrupted save.

Prevent accidental duplication of giant files.

A safe early failure is preferable to a corrupt checkpoint.

---

# 5. P0 — Full training-loop correctness

Audit:
data -> CPU staging -> H2D -> forward -> backward -> gradient accumulation -> reduction -> optimizer -> scheduler -> eval -> checkpoint -> resume.

Verify:
- microstep vs optimizer-step semantics;
- token-weighted loss normalization;
- gradient normalization;
- global clipping;
- loss scaling;
- overflow behavior;
- optimizer counters;
- scheduler counters;
- warmup;
- decay;
- resume identity;
- step limits;
- checkpoint cadence;
- signals;
- time budget;
- final partial accumulation;
- train/validation data separation.

Dangerous silent behavior is forbidden:
- skipped update without a clear reason;
- optimizer step counter advanced when update is skipped;
- scheduler advanced incorrectly;
- schedule changed after resume;
- accidental double weight decay;
- different normalization for last batches;
- accidental data iterator reset that changes intended training distribution.

Add regression tests for fragile contracts.

---

# 6. P0 — CPU/GPU parity and numerical safety

Run CPU and CUDA checks where available for:
- RMSNorm;
- RoPE, including both layouts;
- QK-Norm;
- attention forward/backward;
- dense FFN;
- MoE routing forward/backward;
- top-k selection and weighting;
- MoE aux loss;
- gradient accumulation;
- optimizer steps;
- checkpoint save/load;
- inference logits.

Do not demand bit-identical results where floating-point reordering makes that unreasonable. Define explicit tolerances.

Include edge cases:
- N=1;
- T=1;
- odd lengths;
- partial final blocks;
- non-multiple-of-32 sizes;
- largest supported T;
- K=1 and K=2;
- zero-token experts;
- highly imbalanced expert routing;
- empty/zero gradients;
- frozen parameters.

---

# 7. P0 — T4 VRAM budget audit

Treat GPU memory as a hard contract.

Instrument a real CUDA run and record:
- initial free VRAM;
- model weights;
- gradients;
- optimizer state;
- FP16/GEMM caches;
- activations;
- attention T^2 scratch;
- MoE routing buffers;
- expert GEMM workspaces;
- CUDA context overhead where observable.

Audit the exact flagship recipe in the current config.

The code should detect unsafe combinations before starting a long run:
- model size;
- batch size;
- sequence length;
- optimizer;
- precision;
- activation checkpointing;
- MoE settings.

Do not silently change batch, sequence length, or optimizer to "make it fit". Any automatic reduction must be explicit, logged, deterministic, and documented.

---

# 8. P0 — T4 throughput audit

Benchmark the REAL end-to-end training path.

Measure:
- tokens/sec;
- optimizer steps/sec;
- forward time;
- backward time;
- optimizer time;
- DataLoader wait;
- H2D time;
- synchronization time;
- checkpoint time;
- evaluation time.

Warm up before measuring.

Identify the dominant bottleneck:
- attention;
- MoE router;
- MoE expert GEMMs;
- optimizer;
- data loading;
- CPU preprocessing;
- H2D;
- checkpointing;
- host synchronization.

Do not optimize a subsystem before measuring its fraction of total step time.

---

# 9. H1/H2 — MoE dispatch and GEMM launch efficiency

## H1 — expert-count host synchronization

Current MoE grouping may copy a small expert-offset/count array from device to host before launching variable-size expert GEMMs.

Do NOT simply delete the synchronization.

Investigate:
- cuBLASLt grouped execution;
- grouped GEMM APIs in the installed CUDA/cuBLAS version;
- CUTLASS grouped execution only if integration cost and compatibility are justified;
- other device-driven dispatch designs.

Requirements:
- preserve routing;
- preserve exact expert counts;
- no padding every expert to max count;
- bounded workspace;
- no large CPU roundtrip;
- good T4 tensor-core utilization.

Adopt only if real T4 measurements improve without unacceptable numerical or memory regressions.

## H2 — per-expert GEMM launch overhead

Measure expert GEMM launch count and cost for:
- balanced routing;
- imbalanced routing;
- zero-token experts;
- small N;
- production N.

If grouped execution helps, reduce launch count without creating a giant padded representation.

---

# 10. Attention audit

Audit:
- causal mask;
- sliding-window mask;
- RoPE;
- GQA K/V mapping;
- backward recomputation;
- saved activation lifetime;
- transient T^2 memory;
- stream ordering;
- max-sequence guards.

Forward and backward masks must be mathematically compatible.

Changing sliding_window must not trigger accidental giant allocations.

Benchmark the actual shipped sequence length on T4 before replacing kernels.

---

# 11. MoE routing audit

Audit:
- softmax numerical stability;
- jitter;
- top-k tie breaking;
- top-k probability normalization;
- aux loss;
- aux-free mode;
- router gradient;
- load balance;
- expert starvation;
- shared expert;
- zero-token experts;
- hard-coded local array sizes.

Any configurable expert count/top-k must be validated against kernel assumptions.

A syntactically valid configuration must never be able to trigger a kernel with unsupported K/ne.

Add config validation when needed.

During a pilot, collect expert-load statistics.

---

# 12. Muon audit — do not trust the recent report blindly

The recent Muon change modified the matrix path so effective_scale is applied before momentum accumulation and the final matrix update uses lr after Newton-Schulz.

Validate this mathematically.

Audit:
- raw accumulated gradient;
- token normalization;
- microbatch normalization;
- gradient accumulation;
- dynamic loss scaling;
- global clipping;
- momentum;
- Newton-Schulz normalization;
- decoupled weight decay.

Add explicit tests for:
- equivalent data with grad_accum=1 vs 2/4;
- clipped vs unclipped gradients;
- different loss scales with identical unscaled gradients;
- controlled first-step matrix updates;
- zero-gradient behavior.

Do not claim strict batch-size invariance unless tests establish the exact invariant.

Muon is experimental relative to the shipped Pro recipe. It must not destabilize the main Lion path.

---

# 13. Lion and AdamW audit

Verify:
- configured vs effective LR;
- scheduler integration;
- bias correction where applicable;
- weight decay;
- frozen parameters;
- gradient scale;
- overflow handling;
- state checkpointing.

If an optimizer intentionally modifies configured LR, log BOTH:
- configured LR;
- effective LR.

Resume must preserve optimizer semantics.

---

# 14. Dynamic loss-scaling audit

Check interaction between:
- fp32/bf16/fp16 mode;
- GEMM FP16 mode;
- loss scaling;
- gradient storage dtype;
- overflow detection;
- skipped updates.

Do not apply loss scaling where it has no intended effect.

Create a controlled overflow test:
1. force an overflow;
2. verify one safe skip;
3. verify scale decreases;
4. verify training resumes;
5. verify optimizer and LR counters remain correct.

---

# 15. DataLoader and dataset I/O audit

Training throughput can collapse because of CPU I/O.

Audit:
- shard discovery;
- open/close frequency;
- seek pattern;
- token conversion;
- mask construction;
- per-window allocations;
- per-step allocations;
- filesystem cache behavior;
- thread contention;
- H2D staging.

The inner data path should reuse buffers.

Measure GPU idle time waiting for input.

Malformed shards must fail loudly before a long run.

Verify train/validation separation and mixture weights.

---

# 16. Checkpoint/resume equivalence

A checkpoint should preserve enough state to continue the intended training trajectory.

Audit:
- model;
- optimizer;
- scheduler;
- loss scaler;
- data position if supported;
- RNG state where needed;
- model architecture identity;
- tokenizer identity;
- recipe/config identity.

Run:
1. uninterrupted K-step run;
2. save after M;
3. reload;
4. continue to K;
5. compare loss, parameters, optimizer state, scheduler state within declared tolerances.

Run CPU and CUDA where practical.

---

# 17. Signal and long-run termination

Audit SIGINT/SIGTERM handling.

The process should:
- stop at a safe step boundary;
- not start another huge update after stop request;
- save a final checkpoint when safe and space allows;
- avoid deadlock;
- treat a second signal as emergency termination.

Wall-clock budget and checkpoint cadence must interact predictably.

---

# 18. Build-system audit

Audit:
- Release flags;
- warning-as-error;
- CUDA architecture;
- accidental native CPU flags reaching nvcc;
- fast-math flags affecting NaN/Inf detection;
- separable compilation;
- linker behavior;
- OpenMP;
- NCCL optional path;
- Arrow/Parquet optional path.

T4 builds should avoid unnecessary fatbins and target the real architecture used by the run.

CPU build remains portable.

If CUDA was requested but unavailable, the training entry point must make that failure/fallback explicit. Never silently advertise GPU training while running CPU.

---

# 19. Config/API contract audit

For every shipped YAML key:
- verify it is parsed;
- verify type;
- verify range;
- verify it actually changes the intended code path.

Search for:
- dead keys;
- duplicated keys;
- stale aliases;
- silently ignored keys;
- hard-coded replacements;
- comments disagreeing with implementation.

Prefer fail-fast for dangerous configuration typos.

---

# 20. Global hot-path anti-pattern sweep

Search globally for:
- cudaMalloc/cudaFree inside training loops;
- unnecessary H2D/D2H;
- implicit device synchronization;
- cudaDeviceSynchronize;
- excessive CUDA error checks that synchronize;
- file open/close in inner loops;
- vector growth;
- string formatting in hot loops;
- dynamic allocations;
- repeated tokenizer work;
- repeated shape calculations;
- repeated position generation;
- duplicated kernel setup.

Do not remove useful diagnostics blindly. Separate cold-path diagnostics from hot-path synchronization.

---

# 21. CPU orchestration audit

The GPU can be fast while the CPU starves it.

Check:
- DataLoader worker count;
- OpenMP behavior;
- thread oversubscription;
- mutex contention;
- false sharing;
- host memcpy staging;
- logging frequency;
- checkpoint serialization.

The CPU should not spend most of each step waiting on a tiny synchronous CUDA operation.

---

# 22. Long-run soak test

Create/run a synthetic long-loop test with stable B/T and real optimizer behavior.

Track:
- step;
- tokens/sec;
- loss;
- grad norm;
- loss scale;
- GPU memory;
- host RSS;
- DataLoader wait;
- optimizer time.

Compare early vs late behavior. The requirement is stable operation over hundreds/thousands of steps, not only step 1.

---

# 23. Validation ladder

Run in this order:

1. Build with zero warnings.
2. Run all CPU tests.
3. Run all available CUDA tests.
4. Run parity/gradient tests.
5. Run sanitizers and memory diagnostics where supported.
6. Run tiny deterministic end-to-end training.
7. Run small CUDA MoE training.
8. Run checkpoint/resume equivalence.
9. Run T4 performance pilot.
10. Run longer T4 soak with telemetry.
11. Re-audit modified code for regressions.

Useful tools when available:
- AddressSanitizer;
- UndefinedBehaviorSanitizer;
- ThreadSanitizer for compatible CPU-only paths;
- compute-sanitizer;
- Nsight Systems / Nsight Compute;
- CUDA memory telemetry;
- /usr/bin/time -v;
- RSS monitoring.

Do not invent results for unavailable tools.

---

# 24. Acceptance gates

## Correctness
- no known reproducible crash;
- no known invalid memory access;
- CPU/CUDA parity within declared tolerance;
- optimizer semantics tested;
- checkpoint/resume tested.

## Memory
- no unbounded host/device memory growth;
- peak VRAM measured;
- peak RAM measured;
- checkpoint transient storage budgeted;
- filesystem exhaustion handled.

## Performance
- no unnecessary hot-path allocation;
- no accidental CPU fallback;
- no unexplained synchronization bottleneck;
- real T4 throughput measured before/after major performance work.

## Training reliability
- skipped updates explicit and counted;
- overflow behavior controlled;
- signals safe;
- resume preserves schedule/optimizer identity;
- input pipeline does not starve GPU.

## Maintainability
- no unjustified duplicate implementations;
- no hidden global state without lifetime contract;
- no giant speculative rewrite;
- comments match current code.

---

# 25. Final report requirements

After repairs provide:

1. Exact modified file list.
2. Bug list fixed, with root cause.
3. Performance changes and evidence.
4. Memory changes and evidence.
5. Crash-risk changes.
6. Exact tests and commands.
7. Benchmark numbers before/after where available.
8. Remaining risks.
9. T4 readiness assessment based on measurements.

Do not say "all good" if a high-priority issue was not actually validated.

---

# 26. Special project goal

The goal is NOT to make the code small at any cost.

The target is:

LARGE CAPABILITY + LEAN RUNTIME + PREDICTABLE MEMORY + HIGH GPU UTILIZATION + NUMERICAL CORRECTNESS + NO SURPRISE CRASHES.

Prefer:
- reusable bounded buffers;
- fewer allocations;
- fewer copies;
- fewer synchronizations;
- measured fusion;
- explicit contracts;
- fail-fast safety checks.

Avoid:
- speculative kernel rewrites;
- hidden CPU fallbacks;
- unbounded caches;
- giant padded MoE buffers;
- CPU roundtrips in hot loops;
- logging that synchronizes CUDA;
- optimizations that destroy debuggability.

Final process:

INSPECT -> REPRODUCE -> FIX -> TEST -> BENCHMARK -> RE-AUDIT -> REPORT

Do not finish at the report.
