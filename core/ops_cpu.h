#pragma once

#include "core/common.h"
#include "core/dtype.h"
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
void linear_forward_fp16(const float* x, const u16* w, float* y, int M, int K, int N);
void split_qkv(const float* qkv, float* q, float* k, float* v, i64 n, int qd, int kvd);
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
void rope_forward_cached(float* q, float* k, const i32* pos, const float* inv_freq,
                         i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type);
void rope_backward_cached(float* dq, float* dk, const i32* pos, const float* inv_freq,
                          i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type);

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

// ---- F-03 fused grouped helpers -------------------------------------------
// These mirror the CUDA fused kernels in cuda/moe.cu one-to-one (same
// indexing, same grouped-slot layout) so tests/test_moe_fused.cpp can prove
// the fusion logic bit-exact against moe_forward() above. They are the CPU
// reference for the fusion, not the production CPU path (which stays with the
// token-major reference above).
//
// Grouped-slot layout (identical to moe_build_groups in cuda/moe.cu):
//   grouped[s] for s in [0,NK) holds the token-space slot t*K+k, packed so
//   that expert e owns grouped[offsets[e] .. offsets[e+1]).
//   counts[e] = offsets[e+1]-offsets[e]; NK = N*K.
//
// slot: token-space position t*K+k (which token/k). grouped-position s: rank
// in the grouped order. All three helpers iterate grouped positions.
void moe_group_slots(const i32* idx, i64 N, int K, int ne,
                     i32* grouped, int* counts, int* offsets);
// out[s] = x[grouped[s]/K]  (pack every expert's input rows in one pass)
void moe_pack_all(const float* x, const i32* grouped, float* out,
                  i64 NK, int d, int K);
// G/U/A[s] = s_*[grouped[s]]  (gather every expert's saved activations once;
// the inverse of moe_save3_all, for the backward pass)
void moe_gather3_all(const float* s_gate, const float* s_up, const float* s_act,
                     const i32* grouped, float* G, float* U, float* A,
                     i64 NK, int E);
// S[s] = dout[t] * w[t,k]  (pack + scale the upstream grads in one pass)
void moe_scale_all(const float* dout, const float* w, const i32* grouped,
                   float* S, i64 NK, int d, int K);
// s_*[slot] = block[s]  (save every expert's G/U/A in one pass)
void moe_save3_all(const float* G, const float* U, const float* A,
                   const i32* grouped, float* s_gate, float* s_up, float* s_act,
                   i64 NK, int E);
// out[t] += w[t,k] * Y[s]  (weighted scatter-add over every slot at once)
void moe_scatter_add_all(float* out, const float* Y, const i32* grouped,
                         const float* w, i64 NK, int d, int K);
// F-02: acc[e] += 1 for every slot assigned to expert e. The CUDA backend runs
// this as one atomic kernel into a persistent [L*ne] device buffer (zero D2H
// per layer/micro); the loop below is the CPU reference for the same math.
void moe_count_slots(const i32* idx, float* acc, i64 NK, int ne);
// Full fused forward (same contract as moe_forward): pack once, per-expert
// GEMMs on packed blocks, one swiglu, one save, one scatter-add.
void moe_forward_fused(const float* x, const float* router_w,
                       const float* gates, const float* ups, const float* downs,
                       const float* sh_g, const float* sh_u, const float* sh_d,
                       float* out,
                       float* probs_cache, i32* idx_cache, float* w_cache,
                       float* s_gate, float* s_up, float* s_act,
                       float* Xpack, float* Gpack, float* Upack, float* Apack,
                       float* Ypack, i32* grouped, int* counts, int* offsets,
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
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids = nullptr);
void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window);
void attention_decode_ex(const float* q, const float* kcache, const float* vcache,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window);
void attention_decode_ring(const float* q, const float* kcache, const float* vcache,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window);

void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);
// F-10: host-side accumulate API (trivial: file-static doubles). The CUDA
// backend keeps the accumulators on device; see cuda/kernels.cu.
void sce_acc_begin();
void sce_accumulate(const float* logits, const i32* targets, float* dlogits,
                    i64 n, int V, float z_scale = 0.0f);
void sce_acc_end(double* out_loss_sum, i64* out_count);

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
