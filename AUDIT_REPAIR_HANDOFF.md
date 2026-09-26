# Ghassan v1 Pro — Deep Audit & Repair Handoff
Generated from a static audit of the connected repository.

## Important rule
Do NOT blindly rewrite mathematically sensitive CUDA/model code. The current revision already contains many prior hardening fixes. Validate every change with the existing CPU/CUDA parity and training tests.

## Confirmed current status

### Correct / already repaired
1. Sliding-window attention forward/backward now use the extended attention API with `sliding_window`; the old forward/backward mismatch is no longer present in the current `model.cpp`.
2. MoE routing no longer performs the old full N*K host slot roundtrip. Grouped slot data stays on GPU.
3. MoE aux-loss hot path no longer copies full per-expert statistics every layer; it uses a device accumulator and reads the scalar once per microbatch.
4. Router forward uses one warp per token rather than one thread per token.
5. Router backward uses cooperative work and avoids the old atomics-heavy router-gradient path.
6. Persistent CUDA workspaces are used to avoid repeated cudaMalloc/cudaFree in the hot path.
7. FP16 weight-cache memory accounting includes the fused QKV cache.
8. Activation memory estimation uses saturating arithmetic and includes the transient attention T^2 buffer.
9. Raw model loading checks exact parameter count, names, shapes, and truncated reads.
10. Config parsing has fail-fast checks for ambiguous schedules and invalid optimizer/scheduler names.
11. Checkpoint format/versioning, optimizer snapshots, scheduler snapshots and Pro routing/RoPE fields are substantially hardened.

## High-priority performance findings

### H1 — MoE expert-count host synchronization
File: `cuda/moe.cu`

`moe_build_groups()` runs GPU histogram/offset kernels and then:
```cpp
cudaMemcpy(h_offsets, g_grp_cur, sizeof(int) * (ne + 1),
           cudaMemcpyDeviceToHost);
```

This is a synchronous D2H transfer. The payload is tiny, but the synchronization can stall the CPU/GPU pipeline. The counts are required because the current implementation launches variable-M cuBLAS GEMMs separately for each expert.

Impact:
- Particularly relevant for 36-layer / grad_accum=128 training.
- Do NOT remove the synchronization without replacing the variable-size expert GEMM strategy.

Preferred future optimization:
- grouped GEMM / cuBLASLt grouped execution / CUTLASS grouped GEMM;
- or another device-driven dispatch design that does not require CPU-visible expert counts.

Acceptance test:
- same outputs/gradients within existing tolerance;
- no increase in peak VRAM;
- benchmark tokens/sec before/after on T4.

### H2 — Per-expert GEMM launch overhead
File: `cuda/moe.cu`

Routed gate/up/down projections remain separate GEMMs for each expert. The elementwise pack/save/scatter work is already fused, but the variable-M expert GEMMs are still launched independently.

Impact:
- launch overhead and poor efficiency when expert token counts are small/imbalanced;
- especially relevant on T4.

Preferred future optimization:
- grouped GEMM / cuBLASLt grouped matmul;
- preserve exact expert routing and avoid padding every expert to max token count.

Acceptance test:
- compare router load distribution;
- compare FLOPs actually executed;
- compare tok/s and GPU utilization with Nsight or CUDA event telemetry.

## Medium-priority mathematical/design finding

### M1 — Muon batch/gradient-scale semantics need an explicit contract
File: `training/optimizer.cpp`

Muon matrix updates intentionally omit `grad_scale` before Newton-Schulz:
```cpp
ops::copy(..., p->g.f32(), ...);
...
orthogonalize(...);
const float lr_eff = lr * clip_scale;
```

Newton-Schulz removes overall magnitude, but the momentum state is updated before orthogonalization. Therefore the claim that the matrix path is fully batch-size invariant is stronger than what the code mathematically guarantees.

This is NOT a blocker for the shipped `pro_v1.yaml`, which uses Lion.

If Muon is promoted to a primary recipe:
- define whether gradients are token-mean, microbatch-mean, or raw accumulated sums;
- test equivalence across different grad_accum/token counts;
- decide whether grad_scale must enter the momentum update;
- document the intended invariant.

## Compatibility / documentation issue repaired in this audit

File: `model/model.h`

The comments previously described sliding-window attention as having a TODO kernel and RoPE NeoX as unsupported by current kernels. Current code already routes through the extended attention/RoPE APIs and supports the relevant paths. The comments were corrected on branch `audit-repair-deep-2026-09`.

## Important non-bug observations

### Attention memory
Training still uses a transient `[B,H,T,T]` probability buffer for backward recomputation. This is mathematically intentional but remains quadratic in T. On a 16GB T4, long-context training is therefore constrained even though inference attention is tiled.

### Q4 compatibility
The GGUF exporter currently maps requested `q4_k_m/q4_k_s/q4_k` profiles to the implemented Q4_0 writer rather than emitting true K-quant blocks. This is a compatibility/quality limitation, not a training correctness bug. Do not label the exported file as true Q4_K unless a real K-quant writer is implemented.

### Dataset quality
The C++ cleaning/dedup pipeline is infrastructure, not a guarantee that a dataset is semantically high quality. Training quality still depends heavily on the actual source corpus, mixture ratios, dedup quality, language balance, and SFT examples.

## Recommended validation order

1. Build with warnings-as-errors.
2. Run all CPU/unit tests.
3. Run CUDA parity/gradient tests.
4. Run a tiny deterministic MoE training step and compare CPU vs CUDA.
5. Run a short T4 benchmark with telemetry.
6. Only then consider H1/H2 grouped-GEMM optimization.
7. Run checkpoint/resume equivalence after any training-loop change.
8. Run a short real-data pilot before a 9-hour Kaggle run.

## Do not change without tests

- attention backward
- RoPE layout
- QK-Norm backward
- router top-k weighting
- router jitter derivative
- MoE aux-loss normalization
- gradient accumulation/token normalization
- checkpoint binary layout
- tied embedding/lm-head gradient accumulation

## Branch state

Branch used for the safe documentation correction:
`audit-repair-deep-2026-09`

The branch is intentionally separate from main so sensitive performance changes are not merged without validation.
