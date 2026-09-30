#pragma once

#include "core/tensor.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace gai {
namespace ops {

void set_gemm_fp16(bool on);
bool gemm_fp16_enabled();

void set_gemm_bf16(bool on);
bool gemm_bf16_enabled();

void set_gemm_fp16_mnk_threshold(i64 mnk);
i64  gemm_fp16_mnk_threshold();

struct PerfCounters {
    u64 gemm_calls      = 0;
    u64 gemm_fp16_calls = 0;
    u64 h2d_bytes       = 0;
    u64 d2h_bytes       = 0;
    u64 moe_fwd_calls   = 0;
    u64 moe_bwd_calls   = 0;
    u64 moe_fwd_us      = 0;
    u64 moe_bwd_us      = 0;
    u64 sce_calls       = 0;
    u64 sce_us          = 0;
    u64 sync_calls      = 0;
    u64 opt_steps       = 0;
    u64 opt_step_us     = 0;
    u64 muon_ns_calls   = 0;
    u64 muon_ns_iters   = 0;
    u64 muon_ns_us      = 0;
};
PerfCounters perf_counters();
void         perf_reset();
std::string  perf_report();

void perf_note_fp16_gemm();
void perf_note_h2d(size_t nbytes);
void perf_note_d2h(size_t nbytes);
void perf_note_sync();
void perf_note_moe_fwd(u64 us);
void perf_note_moe_bwd(u64 us);
void perf_note_sce(u64 us);
void perf_note_opt_step(u64 us);
void perf_note_muon_ns(int iters, u64 us);

// Async phase timers for step-time attribution (which part of a training step
// consumes wall time: attention vs MoE vs loader). CUDA path records stream-0
// events (zero host blocking); CPU path is a no-op. Single training thread
// per process is assumed (same contract as the rest of the engine).
enum class TrainPhase : int { AttnFwd = 0, MoeFwd = 1, AttnBwd = 2, MoeBwd = 3, Load = 4 };
void phase_start(Device dev, TrainPhase ph);
void phase_stop(Device dev, TrainPhase ph);
void phase_reset(Device dev);
// Per-step totals since the last call (also resets). Empty on CPU.
std::string phase_report(Device dev);

void set_moe_jitter(float j);
float moe_jitter();

void set_moe_jitter_seed(u64 seed);
u64 moe_jitter_seed();

void gemm(Device dev,
          bool trans_a, bool trans_b,
          int M, int N, int K,
          float alpha,
          const float* A, int lda,
          const float* B, int ldb,
          float beta,
          float* C, int ldc);

void linear_forward(Device dev, const float* x, const float* w, float* y, int M, int K, int N);

void split_qkv(Device dev, const float* qkv, float* q, float* k, float* v,
               i64 n, int qd, int kvd);
void linear_forward_fp16(Device dev, const float* x, const u16* w, float* y, int M, int K, int N);
void convert_f32_to_f16(Device dev, const float* src, u16* dst, i64 n);
void register_fp16_weight(const float* master, const u16* half, i64 n);
void unregister_fp16_weight(const float* master);

void linear_backward(Device dev, const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N);

void add(Device dev, const float* a, const float* b, float* out, i64 n);
void add_inplace(Device dev, float* a, const float* b, i64 n);
void scale_inplace(Device dev, float* a, float s, i64 n);
void zero(Device dev, float* a, i64 n);
void copy(Device dev, float* dst, const float* src, i64 n);

void embedding_forward(Device dev, const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab);

void embedding_backward(Device dev, const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab);

void rmsnorm_forward(Device dev, const float* x, const float* weight, float* out,
                     float* rrms, i64 rows, int dim, float eps);
void rmsnorm_backward(Device dev, const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight,
                      i64 rows, int dim);

void rope_forward(Device dev, float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta);
void rope_backward(Device dev, float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta);

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

void swiglu_forward(Device dev, const float* g, const float* u, float* out, i64 n);
void swiglu_backward(Device dev, const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n);

void moe_forward(Device dev,
                 const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K);

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

bool moe_aux_gpu(Device dev,
                 const float* probs, const i32* idx, float* frac_dev,
                 float* h_frac, float* h_psum,
                 i64 N, int K, int ne, double* d_raw_accum = nullptr);
double* moe_aux_begin(Device dev);
double moe_aux_end(Device dev);

void attention_forward(Device dev,
                       const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale);
void attention_backward(Device dev,
                        const float* q, const float* k, const float* v,
                        const float* probs, const float* dout,
                        float* dq, float* dk, float* dv,
                        int B, int T, int H, int KV, int hd, float scale);

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

void softmax_cross_entropy(Device dev,
                           const float* logits, const i32* targets,
                           float* dlogits, i64 n, int V,
                           double* out_loss_sum, i64* out_count,
                           float z_scale = 0.0f);

void sce_acc_begin(Device dev);
void sce_accumulate(Device dev,
                    const float* logits, const i32* targets, float* dlogits,
                    i64 n, int V, float z_scale = 0.0f);
void sce_acc_end(Device dev, double* out_loss_sum, i64* out_count);

void moe_count_slots(Device dev, const i32* idx, float* acc, i64 NK, int ne);

void adamw_step(Device dev, float* w, const float* g, float* m, float* v,
                i64 n, float lr, float beta1, float beta2, float eps,
                float weight_decay, float bias_correction1, float bias_correction2,
                float grad_scale);

void lion_step(Device dev, float* w, const float* g, float* m,
               i64 n, float lr, float beta1, float beta2, float weight_decay,
               float grad_scale);

double global_sq_norm(Device dev, const float* g, i64 n);

double global_sq_norm_multi(Device dev,
                            const std::vector<std::pair<const float*, i64>>& parts);

void topk_select(Device dev, const float* logits, int V, int K,
                 float* out_vals, i32* out_ids);

i32 argmax_token(Device dev, const float* logits, int V);

void apply_rep_penalties(Device dev, float* logits, int V,
                         const i32* hist, int hist_n,
                         float rep, float freq, float pres);

}
}
