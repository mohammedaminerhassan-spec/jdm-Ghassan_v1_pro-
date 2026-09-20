#pragma once

#include "core/common.h"
#include <vector>
#include <utility>

// CPU reference / production kernels. These are also the correctness reference
// used by tests/test_cuda_parity.
namespace gai {
namespace cpu {

void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc);

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N);
void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N);

void add(const float* a, const float* b, float* out, i64 n);
void add_inplace(float* a, const float* b, i64 n);
void scale_inplace(float* a, float s, i64 n);

void embedding_forward(const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab);
void embedding_backward(const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab);

void rmsnorm_forward(const float* x, const float* weight, float* out, float* rrms,
                     i64 rows, int dim, float eps);
void rmsnorm_backward(const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight, i64 rows, int dim);

void rope_forward(float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta);
void rope_backward(float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta);
// Pro extensions (DeepSeek-V3 / LLaMA-3 class): rope_type 0=interleaved pairs
// (2i,2i+1), 1=neox half-rotate (first half x second half). yarn_low/high/scale
// implement full YaRN ramp: dims with wavelength < low stay linear, > high get
// full NTK scaling, middle interpolated. window 0=full causal, >0 sliding window.
void rope_forward_ex(float* q, float* k, const i32* pos,
                     i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                     int rope_type, float yarn_low, float yarn_high, float yarn_scale);
void rope_backward_ex(float* dq, float* dk, const i32* pos,
                      i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                      int rope_type, float yarn_low, float yarn_high, float yarn_scale);

void swiglu_forward(const float* g, const float* u, float* out, i64 n);
void swiglu_backward(const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n);

void moe_forward(const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K);
// Aux-loss-free (DeepSeek-V3 §3.2): bias added ONLY for top-k selection,
// weights stay softmax(router) without bias (no aux grad, EMA bias update).
void moe_forward_bias(const float* x, const float* router_w, const float* router_bias,
                      const float* gates, const float* ups, const float* downs,
                      const float* sh_g, const float* sh_u, const float* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K);
void moe_backward(const float* x, const float* router_w,
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

void attention_forward(const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale);
void attention_backward(const float* q, const float* k, const float* v,
                        const float* probs, const float* dout,
                        float* dq, float* dk, float* dv,
                        int B, int T, int H, int KV, int hd, float scale);
void attention_decode(const float* q, const float* kcache, const float* vcache,
                      float* out, int H, int KV, int hd, int cur_len, int max_len,
                      float scale, float* scratch);
// SWA variants: window 0 = full causal (legacy), >0 = attend only to last `window` keys.
void attention_forward_ex(const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window);
void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window);
void attention_decode_ex(const float* q, const float* kcache, const float* vcache,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window);

void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);

void adamw_step(float* w, const float* g, float* m, float* v, i64 n,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                float bc1, float bc2, float grad_scale);

// Lion (EvoLved Sign Momentum): single momentum state, ~50% optimizer memory
// vs AdamW. Update: c = b1*m + (1-b1)*g; w -= lr*(sign(c) + wd*w); m = b2*m + (1-b2)*g.
void lion_step(float* w, const float* g, float* m, i64 n,
               float lr, float beta1, float beta2, float weight_decay,
               float grad_scale);

double global_sq_norm(const float* g, i64 n);
// FIX (10/10 DeepSeek-style): fused multi-tensor norm — 1 sync instead of
// ~200 (was 200 cudaMemcpy D2H per optimizer step, the main GPU-starvation
// bottleneck on T4). CPU reference loops; CUDA launches all tiles async then
// a single D2H.
double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts);

// DeepSeek-V2 router jitter (CPU mirror; set via ops::set_moe_jitter).
void set_moe_jitter_cpu(float j);
void set_moe_jitter_seed_cpu(u64 seed);

// in-place softmax over a row (fp32 accumulators, max-subtracted)
void softmax_row(float* x, int n);

// ---- inference fast-sampling helpers (audit P1; CPU reference mirrors CUDA)
// top-K selection, descending by value (ties: lowest id first, deterministic).
// K is clamped to [1, V] by the caller contract (GAI_CHECK inside).
void topk_select(const float* logits, int V, int K, float* out_vals, i32* out_ids);
// full-vocab argmax, first-max wins on ties (matches Sampler::sample greedy).
i32 argmax_token(const float* logits, int V);
// CTRL-style repetition/frequency/presence penalties, EXACT mirror of
// Sampler::apply_penalties (divide positive / multiply negative by rep,
// subtract freq*n and pres). Order: rep, freq, pres — keep in sync.
void apply_rep_penalties(float* logits, int V, const i32* hist, int hist_n,
                         float rep, float freq, float pres);

} // namespace cpu
} // namespace gai
