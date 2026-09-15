#pragma once

#include "core/common.h"
#include <utility>
#include <vector>

// GPU implementations, mirroring core/ops_cpu.h exactly so core/ops.cpp can
// dispatch with a single macro. All pointers are device pointers.
namespace gai {
namespace cuda_ops {

void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc);

// FP16 tensor-core GEMMs (large matrices only; see kernels.cu heuristic).
// Master weights stay fp32 — conversion happens on the fly per call.
void set_fp16_gemm(bool on);
bool fp16_gemm_enabled();

// BF16 tensor-core GEMMs (T4/Ampere+)
void set_bf16_gemm(bool on);
bool bf16_gemm_enabled();

// DeepSeek router jitter (host-side mirror; kernels take it as a param).
void set_moe_jitter(float j);
void set_moe_jitter_seed(u64 seed);

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N);
void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N);

void add(const float* a, const float* b, float* out, i64 n);
void add_inplace(float* a, const float* b, i64 n);
void scale_inplace(float* a, float s, i64 n);
void zero(float* a, i64 n);
void copy(float* dst, const float* src, i64 n);

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

// GPU load-balance fractions (no N*ne / N*K host roundtrip).
// frac[ne] stays on device; h_frac/h_psum are optional tiny host copies.
void moe_aux_frac_gpu(const float* probs, const i32* idx, float* frac,
                      float* h_frac, float* h_psum,
                      i64 N, int K, int ne);

void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);

void adamw_step(float* w, const float* g, float* m, float* v, i64 n,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                float bc1, float bc2, float grad_scale);

void lion_step(float* w, const float* g, float* m, i64 n,
               float lr, float beta1, float beta2, float weight_decay,
               float grad_scale);

// Scale gradients in-place by a constant factor (for distributed training)
void scale_grad(float* g, i64 n, float scale);

double global_sq_norm(const float* g, i64 n);
// Fused multi-tensor norm: N kernels async + ONE D2H (was N syncs).
double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts);

// Scratch pools used by reduction/MoE kernels. Pools grow monotonically
// during the run (by design, not a leak) and are released here / at shutdown.
void  free_workspace();
void  moe_free_workspace();

} // namespace cuda_ops
} // namespace gai
