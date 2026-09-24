#include "core/ops.h"
#include "core/ops_cpu.h"
#include "core/common.h"

#ifdef GAI_CUDA
#include "cuda/cuda_ops.h"
#endif

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

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
// P3-5: bf16 path defaults OFF (was true). The CUDA kernel checks fp16 first,
// so this changes nothing while fp16 is on; but with fp16 off the old default
// silently routed large GEMMs into cublasGemmEx BF16 — fatal on T4/sm_75
// (no BF16 cores). BF16 is now strictly opt-in (the Trainer enables it only
// after probing sm_80+). Mirrors cuda/kernels.cu g_bf16_gemm below.
static std::atomic<bool> g_gemm_bf16{false};
static std::atomic<float> g_moe_jitter{0.0f};
static std::atomic<u64> g_jitter_seed{0};
static std::mutex g_fp16_weights_mutex;
static std::unordered_map<const float*, std::pair<const u16*, i64>> g_fp16_weights;

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

// PERF: fp16 fast-path engagement threshold on M*N*K (default 1M). Below it
// the f32->f16 conversion traffic costs more than the tensor-core win —
// this is why M==1 decode projections decode in fp32 by design, not by bug.
static std::atomic<i64> g_mnk_threshold{1024LL * 1024LL};
void set_gemm_fp16_mnk_threshold(i64 mnk) {
    if (mnk < 0) mnk = 0;
    g_mnk_threshold.store(mnk, std::memory_order_relaxed);
}
i64 gemm_fp16_mnk_threshold() { return g_mnk_threshold.load(std::memory_order_relaxed); }

// PERF telemetry (audit P2-2): relaxed atomics, negligible overhead (~ns).
static std::atomic<u64> g_perf_gemm{0};
static std::atomic<u64> g_perf_fp16{0};
static std::atomic<u64> g_perf_h2d{0};
static std::atomic<u64> g_perf_d2h{0};
void perf_note_fp16_gemm() { g_perf_fp16.fetch_add(1, std::memory_order_relaxed); }
void perf_note_h2d(size_t n) { g_perf_h2d.fetch_add(static_cast<u64>(n), std::memory_order_relaxed); }
void perf_note_d2h(size_t n) { g_perf_d2h.fetch_add(static_cast<u64>(n), std::memory_order_relaxed); }
PerfCounters perf_counters() {
    PerfCounters c;
    c.gemm_calls      = g_perf_gemm.load(std::memory_order_relaxed);
    c.gemm_fp16_calls = g_perf_fp16.load(std::memory_order_relaxed);
    c.h2d_bytes       = g_perf_h2d.load(std::memory_order_relaxed);
    c.d2h_bytes       = g_perf_d2h.load(std::memory_order_relaxed);
    return c;
}
void perf_reset() {
    g_perf_gemm.store(0, std::memory_order_relaxed);
    g_perf_fp16.store(0, std::memory_order_relaxed);
    g_perf_h2d.store(0, std::memory_order_relaxed);
    g_perf_d2h.store(0, std::memory_order_relaxed);
}
std::string perf_report() {
    PerfCounters c = perf_counters();
    return strfmt("gemm=%s (fp16 %s) h2d=%s d2h=%s",
                  human_count(c.gemm_calls).c_str(),
                  human_count(c.gemm_fp16_calls).c_str(),
                  human_bytes(c.h2d_bytes).c_str(),
                  human_bytes(c.d2h_bytes).c_str());
}

void gemm(Device dev, bool ta, bool tb, int M, int N, int K, float alpha,
          const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc) {
    g_perf_gemm.fetch_add(1, std::memory_order_relaxed);
    GAI_DISPATCH(dev, gemm(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}

void linear_forward(Device dev, const float* x, const float* w, float* y, int M, int K, int N) {
    g_perf_gemm.fetch_add(1, std::memory_order_relaxed);
    const u16* half = nullptr;
    i64 count = 0;
    {
        std::lock_guard<std::mutex> lock(g_fp16_weights_mutex);
        auto it = g_fp16_weights.find(w);
        if (it != g_fp16_weights.end()) {
            half = it->second.first;
            count = it->second.second;
        }
    }
    if (half && count == static_cast<i64>(K) * N) {
        g_perf_fp16.fetch_add(1, std::memory_order_relaxed);
        linear_forward_fp16(dev, x, half, y, M, K, N);
        return;
    }
    GAI_DISPATCH(dev, linear_forward(x, w, y, M, K, N));
}

void linear_forward_fp16(Device dev, const float* x, const u16* w, float* y, int M, int K, int N) {
    GAI_DISPATCH(dev, linear_forward_fp16(x, w, y, M, K, N));
}

void convert_f32_to_f16(Device dev, const float* src, u16* dst, i64 n) {
    if (dev == Device::CPU) {
        for (i64 i = 0; i < n; ++i) dst[i] = fp32_to_fp16(src[i]);
        return;
    }
#ifdef GAI_CUDA
    cuda_ops::convert_f32_to_f16(src, dst, n);
#endif
}

void register_fp16_weight(const float* master, const u16* half, i64 n) {
    GAI_CHECK(master != nullptr && half != nullptr && n > 0, "invalid fp16 weight registration");
    std::lock_guard<std::mutex> lock(g_fp16_weights_mutex);
    g_fp16_weights[master] = {half, n};
}

void unregister_fp16_weight(const float* master) {
    std::lock_guard<std::mutex> lock(g_fp16_weights_mutex);
    g_fp16_weights.erase(master);
}

void split_qkv(Device dev, const float* qkv, float* q, float* k, float* v,
               i64 n, int qd, int kvd) {
    GAI_DISPATCH(dev, split_qkv(qkv, q, k, v, n, qd, kvd));
}

void linear_backward(Device dev, const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N) {
    // Two GEMMs inside (dx and dw) — count both so the counter reflects
    // launched work, matching what Nsight would show.
    g_perf_gemm.fetch_add(2, std::memory_order_relaxed);
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
    // DeepSeek fail-fast: n<=0 is a valid no-op, but null with n>0 hides a
    // missing grad scratch (stale Activations) -> silent wrong training.
    if (n <= 0) return;
    GAI_CHECK(a != nullptr, "ops::zero on null pointer (missing scratch?)");
#ifdef GAI_CUDA
    if (dev == Device::CUDA) { cuda_ops::zero(a, n); return; }
#endif
    (void)dev;
    std::memset(a, 0, sizeof(float) * static_cast<size_t>(n));
}

void copy(Device dev, float* dst, const float* src, i64 n) {
    if (n <= 0) return;
    GAI_CHECK(dst != nullptr && src != nullptr, "ops::copy on null pointer");
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

void rope_forward_ex(Device dev, float* q, float* k, const i32* pos,
                     i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                     int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::rope_forward_ex(q, k, pos, ntok, n_heads, n_kv, head_dim, theta,
                                  rope_type, yarn_low, yarn_high, yarn_scale);
        return;
    }
#endif
    cpu::rope_forward_ex(q, k, pos, ntok, n_heads, n_kv, head_dim, theta,
                         rope_type, yarn_low, yarn_high, yarn_scale);
}

void rope_backward_ex(Device dev, float* dq, float* dk, const i32* pos,
                      i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                      int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::rope_backward_ex(dq, dk, pos, ntok, n_heads, n_kv, head_dim, theta,
                                   rope_type, yarn_low, yarn_high, yarn_scale);
        return;
    }
#endif
    cpu::rope_backward_ex(dq, dk, pos, ntok, n_heads, n_kv, head_dim, theta,
                          rope_type, yarn_low, yarn_high, yarn_scale);
}

void rope_forward_cached(Device dev, float* q, float* k, const i32* pos,
                         const float* inv_freq, i64 ntok, int n_heads, int n_kv,
                         int head_dim, int rope_type) {
    GAI_DISPATCH(dev, rope_forward_cached(q, k, pos, inv_freq, ntok, n_heads, n_kv,
                                          head_dim, rope_type));
}

void rope_backward_cached(Device dev, float* dq, float* dk, const i32* pos,
                          const float* inv_freq, i64 ntok, int n_heads, int n_kv,
                          int head_dim, int rope_type) {
    GAI_DISPATCH(dev, rope_backward_cached(dq, dk, pos, inv_freq, ntok, n_heads, n_kv,
                                           head_dim, rope_type));
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

void attention_decode_ring(Device dev,
                           const float* q, const float* kcache, const float* vcache,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::attention_decode_ring(q, kcache, vcache, out, H, KV, hd, ring_start,
                                        pinned_prefix, cur_len, cache_max, scale,
                                        scratch, window);
        return;
    }
#endif
    cpu::attention_decode_ring(q, kcache, vcache, out, H, KV, hd, ring_start,
                               pinned_prefix, cur_len, cache_max, scale,
                               scratch, window);
}

void attention_forward_ex(Device dev,
                          const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::attention_forward_ex(q, k, v, out, probs, B, T, H, KV, hd, scale, window,
                                       segment_ids);
        return;
    }
#endif
    cpu::attention_forward_ex(q, k, v, out, probs, B, T, H, KV, hd, scale, window,
                              segment_ids);
}

void attention_backward_ex(Device dev,
                           const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::attention_backward_ex(q, k, v, probs, dout, dq, dk, dv, B, T, H, KV, hd, scale, window);
        return;
    }
#endif
    cpu::attention_backward_ex(q, k, v, probs, dout, dq, dk, dv, B, T, H, KV, hd, scale, window);
}

void attention_decode_ex(Device dev,
                         const float* q, const float* kc, const float* vc,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::attention_decode_ex(q, kc, vc, out, H, KV, hd, cur_len, max_len, scale, scratch, window);
        return;
    }
#endif
    cpu::attention_decode_ex(q, kc, vc, out, H, KV, hd, cur_len, max_len, scale, scratch, window);
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

void moe_forward_bias(Device dev,
                      const float* x, const float* router_w, const float* router_bias,
                      const float* gates, const float* ups, const float* downs,
                      const float* sh_g, const float* sh_u, const float* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K) {
    if (!router_bias) {
        moe_forward(dev, x, router_w, gates, ups, downs, sh_g, sh_u, sh_d, out,
                    probs_cache, idx_cache, w_cache, s_gate, s_up, s_act, N, d, E, ne, K);
        return;
    }
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::moe_forward_bias(x, router_w, router_bias, gates, ups, downs, sh_g, sh_u, sh_d,
                                   out, probs_cache, idx_cache, w_cache,
                                   s_gate, s_up, s_act, N, d, E, ne, K);
        return;
    }
#endif
    (void)dev;
    cpu::moe_forward_bias(x, router_w, router_bias, gates, ups, downs, sh_g, sh_u, sh_d,
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
                 i64 N, int K, int ne, double* d_raw_accum) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda_ops::moe_aux_frac_gpu(probs, idx, frac_dev, h_frac, h_psum, N, K, ne,
                                   d_raw_accum);
        return true;
    }
#endif
    (void)dev; (void)probs; (void)idx; (void)frac_dev;
    (void)h_frac; (void)h_psum; (void)N; (void)K; (void)ne; (void)d_raw_accum;
    return false;
}

double* moe_aux_begin(Device dev) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) return cuda_ops::moe_aux_begin();
#endif
    (void)dev;
    return nullptr;
}

double moe_aux_end(Device dev) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) return cuda_ops::moe_aux_end();
#endif
    (void)dev;
    return 0.0;
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

void topk_select(Device dev, const float* logits, int V, int K,
                 float* out_vals, i32* out_ids) {
    GAI_DISPATCH(dev, topk_select(logits, V, K, out_vals, out_ids));
}

i32 argmax_token(Device dev, const float* logits, int V) {
    GAI_DISPATCH_RET(dev, argmax_token(logits, V));
}

void apply_rep_penalties(Device dev, float* logits, int V,
                         const i32* hist, int hist_n,
                         float rep, float freq, float pres) {
    GAI_DISPATCH(dev, apply_rep_penalties(logits, V, hist, hist_n, rep, freq, pres));
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
