#pragma once

#include "core/tensor.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// Device-dispatching operator layer. Every op has a CPU reference implementation
// (core/ops_cpu.cpp) and, when built with CUDA, a GPU implementation (cuda/*.cu).
// The model / trainer code is written once against this interface.

namespace gai {
namespace ops {

// ---------------------------------------------------------------- GEMM precision
// CUDA only (CPU always computes in fp32). When enabled, LARGE GEMMs run in
// FP16 on tensor cores (fp32 accumulation, fp32 in/out); small GEMMs stay fp32
// to avoid conversion overhead. Master weights, activations and gradients all
// stay fp32 — this is compute-only mixed precision (safe with loss scaling).
void set_gemm_fp16(bool on);
bool gemm_fp16_enabled();

// BF16 tensor-core GEMM (Ampere+, sm_80 and newer; T4/sm_75 has no BF16 cores)
void set_gemm_bf16(bool on);
bool gemm_bf16_enabled();

// PERF (audit #6/decode): the fp16 fast path engages for GEMMs with
// M*N*K >= threshold (default 1M: conversion overhead eats the win below).
// Decode is M==1, so small projections decode in fp32 by design — the generic
// threshold is NOT a decode bug (converting ~MBs to save ~kMACs loses).
// Exposed for Kaggle/Nsight autotuning (e.g. lower it when a persistent fp16
// weight cache removes the per-call conversion cost).
void set_gemm_fp16_mnk_threshold(i64 mnk);
i64  gemm_fp16_mnk_threshold();

// PERF telemetry (audit P2-2, extended for the repair-prompt acceptance
// criterion 9): lightweight global counters for launch/memory analysis.
// Counting is unconditional and cheap (relaxed atomics); report via
// perf_report() on the main rank at log cadence. All counters are lifetime
// totals — call perf_reset() to delimit a window (e.g. per eval).
//
// What is (and is not) measured: these count host-side dispatches and payload
// bytes, not GPU-side cycles. `moe_*_us` is host dispatch time around the
// (possibly asynchronous) launch, which still bounds framework overhead and
// launch storms; `sync_calls` counts the host-visible synchronizations that
// stall the training thread.
struct PerfCounters {
    u64 gemm_calls      = 0;   // ops::gemm dispatches (both backends)
    u64 gemm_fp16_calls = 0;   // GEMMs routed to the fp16 tensor-core path
    u64 h2d_bytes       = 0;   // host->device payload bytes (device_copy)
    u64 d2h_bytes       = 0;   // device->host payload bytes
    u64 moe_fwd_calls   = 0;   // MoE forward dispatches (one per layer/micro)
    u64 moe_bwd_calls   = 0;   // MoE backward dispatches
    u64 moe_fwd_us      = 0;   // host dispatch microseconds (fwd)
    u64 moe_bwd_us      = 0;   // host dispatch microseconds (bwd)
    u64 sce_calls       = 0;   // softmax_cross_entropy (loss) evaluations
    u64 sce_us          = 0;   // host time inside the loss op
    u64 sync_calls      = 0;   // host-visible device synchronizations
    u64 opt_steps       = 0;   // optimizer update calls
    u64 opt_step_us     = 0;   // host microseconds inside the optimizer
    u64 muon_ns_calls   = 0;   // Newton-Schulz orthogonalize() invocations
    u64 muon_ns_iters   = 0;   // NS iterations executed (calls x ns_steps)
    u64 muon_ns_us      = 0;   // host microseconds inside orthogonalize()
};
PerfCounters perf_counters();
void         perf_reset();
std::string  perf_report();
// Internal notes called by the backends (CUDA fp16 path, H2D/D2H copies).
// Public only so both backends can reach them; prefer perf_counters().
void perf_note_fp16_gemm();
void perf_note_h2d(size_t nbytes);
void perf_note_d2h(size_t nbytes);
void perf_note_sync();                          // one host-visible sync happened
void perf_note_moe_fwd(u64 us);                 // timed MoE forward dispatch
void perf_note_moe_bwd(u64 us);                 // timed MoE backward dispatch
void perf_note_sce(u64 us);                     // timed loss evaluation
void perf_note_opt_step(u64 us);                // timed optimizer update
void perf_note_muon_ns(int iters, u64 us);      // one orthogonalize() call

// DeepSeek-V2 router jitter (train-only noise on router logits for expert
// exploration). 0 = deterministic. Set once per run from ModelConfig.
void set_moe_jitter(float j);
float moe_jitter();
// P2-02 jitter entropy: deterministic (step, rank, sample) context folded
// into the hash so the same position gets different noise each step/rank
// (reproducible from base seed). Set per step by the Trainer.
void set_moe_jitter_seed(u64 seed);
u64 moe_jitter_seed();

// ---------------------------------------------------------------- GEMM
// Row-major C[M,N] = alpha * op(A) * op(B) + beta * C
//   A is [M,K] (or [K,M] when trans_a), B is [K,N] (or [N,K] when trans_b).
void gemm(Device dev,
          bool trans_a, bool trans_b,
          int M, int N, int K,
          float alpha,
          const float* A, int lda,
          const float* B, int ldb,
          float beta,
          float* C, int ldc);

// Convenience: y[M,N] = x[M,K] * W[N,K]^T   (weights stored as [out, in], no bias)
void linear_forward(Device dev, const float* x, const float* w, float* y, int M, int K, int N);
// Split fused [N, qd + 2*kvd] rows into contiguous [N, qd], [N, kvd], [N, kvd] views.
void split_qkv(Device dev, const float* qkv, float* q, float* k, float* v,
               i64 n, int qd, int kvd);
void linear_forward_fp16(Device dev, const float* x, const u16* w, float* y, int M, int K, int N);
void convert_f32_to_f16(Device dev, const float* src, u16* dst, i64 n);
void register_fp16_weight(const float* master, const u16* half, i64 n);
void unregister_fp16_weight(const float* master);
// dx[M,K] += dy[M,N] * W[N,K] ; dw[N,K] += dy[M,N]^T * x[M,K]
void linear_backward(Device dev, const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N);

// ---------------------------------------------------------------- elementwise
void add(Device dev, const float* a, const float* b, float* out, i64 n);
void add_inplace(Device dev, float* a, const float* b, i64 n);
void scale_inplace(Device dev, float* a, float s, i64 n);
void zero(Device dev, float* a, i64 n);
void copy(Device dev, float* dst, const float* src, i64 n);

// ---------------------------------------------------------------- embedding
// ids[B*T] -> out[B*T, D]
void embedding_forward(Device dev, const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab);
// accumulate gradient into dtable
void embedding_backward(Device dev, const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab);

// ---------------------------------------------------------------- rmsnorm
// out[n,d] = x/rms(x) * weight ; rms cached in rrms[n] for backward
void rmsnorm_forward(Device dev, const float* x, const float* weight, float* out,
                     float* rrms, i64 rows, int dim, float eps);
void rmsnorm_backward(Device dev, const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight,
                      i64 rows, int dim);

// ---------------------------------------------------------------- rope
// q[ntok, n_heads, head_dim] and k[ntok, n_kv, head_dim], positions[ntok]
void rope_forward(Device dev, float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta);
void rope_backward(Device dev, float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta);
// Pro: rope_type 0=interleaved, 1=neox; yarn_* full YaRN ramp (1.0/32.0/1.0 = legacy).
void rope_forward_ex(Device dev, float* q, float* k, const i32* pos,
                     i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                     int rope_type, float yarn_low, float yarn_high, float yarn_scale);
void rope_backward_ex(Device dev, float* dq, float* dk, const i32* pos,
                       i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                       int rope_type, float yarn_low, float yarn_high, float yarn_scale);
void rope_forward_cached(Device dev, float* q, float* k, const i32* pos,
                         const float* inv_freq, i64 ntok, int n_heads, int n_kv,
                         int head_dim, int rope_type);
void rope_backward_cached(Device dev, float* dq, float* dk, const i32* pos,
                          const float* inv_freq, i64 ntok, int n_heads, int n_kv,
                          int head_dim, int rope_type);

// ---------------------------------------------------------------- swiglu
// out = silu(g) * u   (g,u,out all [n, d])
void swiglu_forward(Device dev, const float* g, const float* u, float* out, i64 n);
void swiglu_backward(Device dev, const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n);

// ---------------------------------------------------------------- MoE
// Top-k routed SwiGLU Mixture-of-Experts + shared expert (DeepSeek-style).
//   x[N,d]; router_w[ne,d]; gates/ups[ne*E,d]; downs[ne*d,E] ([ne,d,E]);
//   sh_g/sh_u[E,d]; sh_d[d,E].
//   out[N,d] is OVERWRITTEN (shared expert + weighted routed experts).
//   probs_cache[N,ne] (f32), idx_cache[N,K] (i32), w_cache[N,K] (f32) are
//   optional training caches (pass null for inference).
//   s_gate/s_up/s_act[N,K,E] is caller-owned scratch, reused across layers.
// Backward ACCUMULATES into dx and all d* (grad-accumulation safe; the caller
// zeroes dx once per micro-batch). aux_frac[ne] (nullable) carries the current
// token fraction per expert; aux_scale==0 disables the load-balance term.
void moe_forward(Device dev,
                 const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K);
// Aux-loss-free: router_bias[ne] (device, nullable) steers top-k selection only.
void moe_forward_bias(Device dev,
                      const float* x, const float* router_w, const float* router_bias,
                      const float* gates, const float* ups, const float* downs,
                      const float* sh_g, const float* sh_u, const float* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K);
void moe_backward(Device dev,
                  const float* x, const float* router_w,
                  const float* gates, const float* ups, const float* downs,
                  const float* sh_g, const float* sh_u, const float* sh_d,
                  const float* probs, const i32* idx, const float* tw,
                  const float* s_gate, const float* s_up, const float* s_act,
                  const float* aux_frac, float aux_scale,
                  const float* dout,
                  float* dx,
                  float* drouter_w, float* dgates, float* dups, float* ddowns,
                  float* dsh_g, float* dsh_u, float* dsh_d,
                  float* s_dact,
                  i64 N, int d, int E, int ne, int K);

// GPU fast path for the MoE load-balance fractions (CUDA only; CPU returns false).
// Fills frac_dev[ne] on DEVICE without any N*ne / N*K host copy; optionally
// fills h_frac[ne] / h_psum[ne] on HOST with tiny copies for the loss scalar
// (pass nullptrs on the training hot path: zero syncs per layer).
// d_raw_accum (or null): device scalar folded with this layer's raw term;
// reset per microbatch with moe_aux_reset() and read once with moe_aux_read()
// (ONE sync per microbatch instead of 2x ne-float copies per layer).
// Returns true when the GPU path was taken.
bool moe_aux_gpu(Device dev,
                 const float* probs, const i32* idx, float* frac_dev,
                 float* h_frac, float* h_psum,
                 i64 N, int K, int ne, double* d_raw_accum = nullptr);
double* moe_aux_begin(Device dev);  // zero + return device raw accum (CUDA; CPU null)
double moe_aux_end(Device dev);     // single-sync read of the accumulator (CUDA; CPU 0.0)

// ---------------------------------------------------------------- attention
// Batched causal grouped-query attention.
//   q  [B, T, H,  hd]
//   k  [B, T, KV, hd]
//   v  [B, T, KV, hd]
//   out[B, T, H,  hd]
//   probs (optional, size B*H*T*T) cached for backward; if null the fwd is memory-lean.
void attention_forward(Device dev,
                       const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale);
void attention_backward(Device dev,
                        const float* q, const float* k, const float* v,
                        const float* probs, const float* dout,
                        float* dq, float* dk, float* dv,
                        int B, int T, int H, int KV, int hd, float scale);
// SWA: window 0 = full causal, >0 = sliding window (Mistral-style).
void attention_forward_ex(Device dev,
                          const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids = nullptr);
void attention_backward_ex(Device dev,
                           const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window);

// Single-token decode against a KV cache.
//   q      [H, hd]
//   kcache [max_len, KV, hd] (only first cur_len valid)
void attention_decode(Device dev,
                      const float* q, const float* kcache, const float* vcache,
                      float* out, int H, int KV, int hd, int cur_len, int max_len,
                      float scale, float* scratch);
void attention_decode_ex(Device dev,
                         const float* q, const float* kcache, const float* vcache,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window);
void attention_decode_ring(Device dev,
                           const float* q, const float* kcache, const float* vcache,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window);

// ---------------------------------------------------------------- loss
// logits[n, V], targets[n] (-100 = ignore). Returns sum of losses and count.
// If dlogits != null it is filled with dL/dlogits in SUM form (caller scales
// by its loss scale only; no divide-by-n_valid roundtrip, audit P1).
// z_scale adds the z-loss stabilizer: loss += z_scale * mean(logZ^2),
// grad += 2*z_scale*logZ*p (SUM form; prevents logit explosion at 1B+).
void softmax_cross_entropy(Device dev,
                           const float* logits, const i32* targets,
                           float* dlogits, i64 n, int V,
                           double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);
// F-10: device-side loss/count accumulation for the chunked training loop.
// softmax_cross_entropy() above performs one host sync per call; with
// grad_accum=128 that is 128+ blocking reductions per optimizer step. The
// accumulate API folds loss+count into a persistent device-side accumulator
// across all chunks/micros with ZERO host traffic, and sce_acc_end() performs
// the single synchronized reduction per optimizer step. dlogits are still
// written per chunk (the backward needs them immediately); only the scalar
// loss/count stay on device. The old single-shot op is unchanged and stays
// for eval/inference ( Generator::score_tokens, Trainer::evaluate ).
void sce_acc_begin(Device dev);      // zero the persistent accumulators
void sce_accumulate(Device dev,
                    const float* logits, const i32* targets, float* dlogits,
                    i64 n, int V, float z_scale = 0.0f);
void sce_acc_end(Device dev, double* out_loss_sum, i64* out_count);
// F-02: acc[e] += 1 per routed slot (atomic on CUDA). The aux-free bias path
// counts on-device into the model's persistent [L*ne] buffer; the host loop
// in cpu::moe_count_slots is the reference.
void moe_count_slots(Device dev, const i32* idx, float* acc, i64 NK, int ne);

// ---------------------------------------------------------------- optimizer
void adamw_step(Device dev, float* w, const float* g, float* m, float* v,
                i64 n, float lr, float beta1, float beta2, float eps,
                float weight_decay, float bias_correction1, float bias_correction2,
                float grad_scale);

void lion_step(Device dev, float* w, const float* g, float* m,
               i64 n, float lr, float beta1, float beta2, float weight_decay,
               float grad_scale);

double global_sq_norm(Device dev, const float* g, i64 n);
// Fused: single-sync norm over many grad tensors (see ops_cpu.h).
double global_sq_norm_multi(Device dev,
                            const std::vector<std::pair<const float*, i64>>& parts);

// ---- inference fast sampling (audit P1: no full-vocab D2H per token)
// top-K (descending, ties by lowest id) of logits[V] into out_vals/ids[K].
// K<=0 or K>V fails fast. CUDA keeps everything on device; the caller copies
// only K pairs to the host.
void topk_select(Device dev, const float* logits, int V, int K,
                 float* out_vals, i32* out_ids);
// Full-vocab argmax with penalties pre-applied by the caller. Exact greedy.
i32 argmax_token(Device dev, const float* logits, int V);
// In-place CTRL-style penalties over the full row (exact Sampler mirror).
// hist[0..hist_n) is the penalty window slice (caller-capped, <= 2048).
void apply_rep_penalties(Device dev, float* logits, int V,
                         const i32* hist, int hist_n,
                         float rep, float freq, float pres);

} // namespace ops
} // namespace gai
