#include "core/ops.h"
#include "core/ops_cpu.h"

#ifdef GAI_CUDA
#include "cuda/cuda_ops.h"
#endif

#include <atomic>
#include <cstring>

// Dispatch (T4-ONLY): Device::CUDA goes to cuda_ops::*, everything else to cpu::*.
// No TPU/Metal/Vulkan branches exist by design — zero dead code paths.

#ifdef GAI_CUDA
#define GAI_DISPATCH(dev, call) \
    do { if ((dev) == Device::CUDA) { cuda_ops::call; return; } cpu::call; } while (0)
#define GAI_DISPATCH_RET(dev, call) \
    do { if ((dev) == Device::CUDA) { return cuda_ops::call; } return cpu::call; } while (0)
#else
#define GAI_DISPATCH(dev, call) \
    do { (void)(dev); cpu::call; } while (0)
#define GAI_DISPATCH_RET(dev, call) \
    do { (void)(dev); return cpu::call; } while (0)
#endif

namespace gai {
namespace ops {

static std::atomic<bool> g_gemm_fp16{true};
static std::atomic<bool> g_gemm_bf16{true};
static std::atomic<float> g_moe_jitter{0.0f};
static std::atomic<u64> g_jitter_seed{0};

void set_moe_jitter(float j) {
    if (j < 0.0f) j = 0.0f;
    if (j > 0.5f) j = 0.5f;
    g_moe_jitter.store(j, std::memory_order_relaxed);
    cpu::set_moe_jitter_cpu(j);
#ifdef GAI_CUDA
    cuda_ops::set_moe_jitter(j);
#endif
}
float moe_jitter() { return g_moe_jitter.load(std::memory_order_relaxed); }

void set_moe_jitter_seed(u64 s) {
    g_jitter_seed.store(s, std::memory_order_relaxed);
    cpu::set_moe_jitter_seed_cpu(s);
#ifdef GAI_CUDA
    cuda_ops::set_moe_jitter_seed(s);
#endif
}
u64 moe_jitter_seed() { return g_jitter_seed.load(std::memory_order_relaxed); }

void set_gemm_fp16(bool on) {
    g_gemm_fp16.store(on, std::memory_order_relaxed);
#ifdef GAI_CUDA
    cuda_ops::set_fp16_gemm(on);
#endif
}

bool gemm_fp16_enabled() { return g_gemm_fp16.load(std::memory_order_relaxed); }

void set_gemm_bf16(bool on) {
    g_gemm_bf16.store(on, std::memory_order_relaxed);
#ifdef GAI_CUDA
    cuda_ops::set_bf16_gemm(on);
#endif
}

bool gemm_bf16_enabled() { return g_gemm_bf16.load(std::memory_order_relaxed); }

void gemm(Device dev, bool ta, bool tb, int M, int N, int K, float alpha,
          const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc) {
    GAI_DISPATCH(dev, gemm(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}

void linear_forward(Device dev, const float* x, const float* w, float* y, int M, int K, int N) {
    GAI_DISPATCH(dev, linear_forward(x, w, y, M, K, N));
}

void linear_backward(Device dev, const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N) {
    GAI_DISPATCH(dev, linear_backward(x, w, dy, dx, dw, M, K, N));
}

void add(Device dev, const float* a, const float* b, float* out, i64 n) {
    GAI_DISPATCH(dev, add(a, b, out, n));
}

void add_inplace(Device dev, float* a, const float* b, i64 n) {
    GAI_DISPATCH(dev, add_inplace(a, b, n));
}

void scale_inplace(Device dev, float* a, float s, i64 n) {
    GAI_DISPATCH(dev, scale_inplace(a, s, n));
}

void zero(Device dev, float* a, i64 n) {
    if (n <= 0 || a == nullptr) return;
#ifdef GAI_CUDA
    if (dev == Device::CUDA) { cuda_ops::zero(a, n); return; }
#endif
    (void)dev;
    std::memset(a, 0, sizeof(float) * static_cast<size_t>(n));
}

void copy(Device dev, float* dst, const float* src, i64 n) {
    if (n <= 0 || dst == nullptr || src == nullptr) return;
#ifdef GAI_CUDA
    if (dev == Device::CUDA) { cuda_ops::copy(dst, src, n); return; }
#endif
    (void)dev;
    std::memcpy(dst, src, sizeof(float) * static_cast<size_t>(n));
}

void embedding_forward(Device dev, const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab) {
    GAI_DISPATCH(dev, embedding_forward(ids, table, out, ntok, dim, vocab));
}

void embedding_backward(Device dev, const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab) {
    GAI_DISPATCH(dev, embedding_backward(ids, dout, dtable, ntok, dim, vocab));
}

void rmsnorm_forward(Device dev, const float* x, const float* weight, float* out,
                     float* rrms, i64 rows, int dim, float eps) {
    GAI_DISPATCH(dev, rmsnorm_forward(x, weight, out, rrms, rows, dim, eps));
}

void rmsnorm_backward(Device dev, const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight,
                      i64 rows, int dim) {
    GAI_DISPATCH(dev, rmsnorm_backward(x, weight, dout, rrms, dx, dweight, rows, dim));
}

void rope_forward(Device dev, float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {
    GAI_DISPATCH(dev, rope_forward(q, k, pos, ntok, n_heads, n_kv, head_dim, theta));
}

void rope_backward(Device dev, float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {
    GAI_DISPATCH(dev, rope_backward(dq, dk, pos, ntok, n_heads, n_kv, head_dim, theta));
}

void swiglu_forward(Device dev, const float* g, const float* u, float* out, i64 n) {
    GAI_DISPATCH(dev, swiglu_forward(g, u, out, n));
}

void swiglu_backward(Device dev, const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n) {
    GAI_DISPATCH(dev, swiglu_backward(g, u, dout, dg, du, n));
}

void attention_forward(Device dev, const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale) {
    GAI_DISPATCH(dev, attention_forward(q, k, v, out, probs, B, T, H, KV, hd, scale));
}

void attention_backward(Device dev, const float* q, const float* k, const float* v,
                        const float* probs, const float* dout,
                        float* dq, float* dk, float* dv,
                        int B, int T, int H, int KV, int hd, float scale) {
    GAI_DISPATCH(dev, attention_backward(q, k, v, probs, dout, dq, dk, dv, B, T, H, KV, hd, scale));
}

void attention_decode(Device dev, const float* q, const float* kc, const float* vc,
                       float* out, int H, int KV, int hd, int cur_len, int max_len,
                       float scale, float* scratch) {
    GAI_DISPATCH(dev, attention_decode(q, kc, vc, out, H, KV, hd, cur_len, max_len, scale, scratch));
}

// MoE has CPU + CUDA backends only (T4-only build).
void moe_forward(Device dev,
                 const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::moe_forward(x, router_w, gates, ups, downs, sh_g, sh_u, sh_d,
                              out, probs_cache, idx_cache, w_cache,
                              s_gate, s_up, s_act, N, d, E, ne, K);
        return;
    }
#endif
    (void)dev;
    cpu::moe_forward(x, router_w, gates, ups, downs, sh_g, sh_u, sh_d,
                     out, probs_cache, idx_cache, w_cache,
                     s_gate, s_up, s_act, N, d, E, ne, K);
}

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
                  i64 N, int d, int E, int ne, int K) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::moe_backward(x, router_w, gates, ups, downs, sh_g, sh_u, sh_d,
                               probs, idx, tw, s_gate, s_up, s_act,
                               aux_frac, aux_scale, dout, dx,
                               drouter_w, dgates, dups, ddowns,
                               dsh_g, dsh_u, dsh_d, s_dact, N, d, E, ne, K);
        return;
    }
#endif
    (void)dev;
    cpu::moe_backward(x, router_w, gates, ups, downs, sh_g, sh_u, sh_d,
                      probs, idx, tw, s_gate, s_up, s_act,
                      aux_frac, aux_scale, dout, dx,
                      drouter_w, dgates, dups, ddowns,
                      dsh_g, dsh_u, dsh_d, s_dact, N, d, E, ne, K);
}

bool moe_aux_gpu(Device dev,
                 const float* probs, const i32* idx, float* frac_dev,
                 float* h_frac, float* h_psum,
                 i64 N, int K, int ne) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::moe_aux_frac_gpu(probs, idx, frac_dev, h_frac, h_psum, N, K, ne);
        return true;
    }
#endif
    (void)dev; (void)probs; (void)idx; (void)frac_dev;
    (void)h_frac; (void)h_psum; (void)N; (void)K; (void)ne;
    return false;
}

void softmax_cross_entropy(Device dev, const float* logits, const i32* targets,
                           float* dlogits, i64 n, int V,
                           double* out_loss_sum, i64* out_count,
                           float z_scale) {
    GAI_DISPATCH(dev, softmax_cross_entropy(logits, targets, dlogits, n, V, out_loss_sum, out_count, z_scale));
}

void adamw_step(Device dev, float* w, const float* g, float* m, float* v, i64 n,
                float lr, float b1, float b2, float eps, float wd,
                float bc1, float bc2, float grad_scale) {
    GAI_DISPATCH(dev, adamw_step(w, g, m, v, n, lr, b1, b2, eps, wd, bc1, bc2, grad_scale));
}

void lion_step(Device dev, float* w, const float* g, float* m, i64 n,
               float lr, float b1, float b2, float wd, float grad_scale) {
    GAI_DISPATCH(dev, lion_step(w, g, m, n, lr, b1, b2, wd, grad_scale));
}

double global_sq_norm(Device dev, const float* g, i64 n) {
    GAI_DISPATCH_RET(dev, global_sq_norm(g, n));
}

double global_sq_norm_multi(Device dev,
                            const std::vector<std::pair<const float*, i64>>& parts) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) return cuda_ops::global_sq_norm_multi(parts);
#endif
    (void)dev;
    return cpu::global_sq_norm_multi(parts);
}

} // namespace ops
} // namespace gai
