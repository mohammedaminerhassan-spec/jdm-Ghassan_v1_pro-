#pragma once

#include "core/tensor.h"
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

// BF16 tensor-core GEMM (Ampere/T4+)
void set_gemm_bf16(bool on);
bool gemm_bf16_enabled();

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
// fills h_frac[ne] / h_psum[ne] on HOST with tiny copies for the loss scalar.
// Returns true when the GPU path was taken.
bool moe_aux_gpu(Device dev,
                 const float* probs, const i32* idx, float* frac_dev,
                 float* h_frac, float* h_psum,
                 i64 N, int K, int ne);

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

// Single-token decode against a KV cache.
//   q      [H, hd]
//   kcache [max_len, KV, hd] (only first cur_len valid)
void attention_decode(Device dev,
                      const float* q, const float* kcache, const float* vcache,
                      float* out, int H, int KV, int hd, int cur_len, int max_len,
                      float scale, float* scratch);

// ---------------------------------------------------------------- loss
// logits[n, V], targets[n] (-100 = ignore). Returns sum of losses and count.
// If dlogits != null it is filled with dL/dlogits (already divided by n_valid).
// z_scale adds the z-loss stabilizer: loss += z_scale * mean(logZ^2),
// grad += 2*z_scale*logZ*p/n_valid (prevents logit explosion at 1B+).
void softmax_cross_entropy(Device dev,
                           const float* logits, const i32* targets,
                           float* dlogits, i64 n, int V,
                           double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);

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

} // namespace ops
} // namespace gai
