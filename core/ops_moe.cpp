// core/ops_moe.cpp — CPU reference implementation of top-k routed SwiGLU MoE.
// DeepSeek-style: softmax router over all experts, top-k selected per token
// (raw softmax weights, no renormalisation), plus one always-on shared expert.
// Forward is parallel over tokens; backward runs serially over tokens because
// concurrent tokens may accumulate into the same expert rows.

#include "core/ops_cpu.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <atomic>

// DeepSeek-V2 router jitter (train-only, hash-based, no RNG state).
static std::atomic<float> g_jitter_cpu{0.0f};
static std::atomic<uint64_t> g_jitter_seed_cpu{0};
namespace gai {
namespace cpu {
void set_moe_jitter_cpu(float j) {
    if (j < 0.0f) j = 0.0f;
    if (j > 0.5f) j = 0.5f;
    g_jitter_cpu.store(j, std::memory_order_relaxed);
}
void set_moe_jitter_seed_cpu(u64 seed) {
    g_jitter_seed_cpu.store(static_cast<uint64_t>(seed), std::memory_order_relaxed);
}
}
}
static inline float jitter_u_cpu(gai::i64 tt, int ee) {
    uint64_t seed = g_jitter_seed_cpu.load(std::memory_order_relaxed);
    uint32_t h = static_cast<uint32_t>(static_cast<uint64_t>(tt) * 2654435761ULL) ^
                 static_cast<uint32_t>(ee * 40503u + 1u) ^
                 static_cast<uint32_t>(seed & 0xFFFFFFFFu) ^
                 static_cast<uint32_t>((seed >> 32) * 2246822519ULL);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return (static_cast<float>(h % 1000000u) / 1000000.0f) - 0.5f;
}

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace cpu {

namespace {

// thread-local scratch so the parallel forward never allocates
struct FwdScratch {
    std::vector<float> logits;   // [ne]
    std::vector<float> g, u, a;  // [E]
    std::vector<float> tmp;      // [d]
    void ensure(int ne, int E, int d) {
        if ((int)logits.size() < ne) logits.resize(ne);
        if ((int)g.size() < E) { g.resize(E); u.resize(E); a.resize(E); }
        if ((int)tmp.size() < d) tmp.resize(d);
    }
};
struct BwdScratch {
    std::vector<float> dlogit;   // [ne]
    std::vector<float> dp;       // [ne]
    std::vector<float> g, u, a;  // [E] (shared-expert recompute)
    std::vector<float> dg, du, da; // [E]
    std::vector<float> dx;       // [d] per-token accumulator
    std::vector<float> tmp;      // [d]
    std::vector<float> oute;     // [d] expert output (router grad)
    void ensure(int ne, int E, int d) {
        if ((int)dlogit.size() < ne) { dlogit.resize(ne); dp.resize(ne); }
        if ((int)g.size() < E) { g.resize(E); u.resize(E); a.resize(E);
                                 dg.resize(E); du.resize(E); da.resize(E); }
        if ((int)dx.size() < d) { dx.resize(d); tmp.resize(d); oute.resize(d); }
    }
};

inline float dot_row(const float* a, const float* b, int n) {
    float s = 0.0f;
    int i = 0;
    for (; i + 4 <= n; i += 4)
        s += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3];
    for (; i < n; ++i) s += a[i]*b[i];
    return s;
}

// K passes of argmax, excluding already-picked experts. K is tiny (<=8).
inline void topk_pick(const float* p, int ne, int K, i32* idx, float* w) {
    for (int k = 0; k < K; ++k) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < ne; ++e) {
            bool taken = false;
            for (int j = 0; j < k; ++j) if (idx[j] == e) { taken = true; break; }
            if (!taken && p[e] > bv) { bv = p[e]; best = e; }
        }
        if (best < 0) best = 0;
        idx[k] = best;
        w[k] = p[best];
    }
}

} // namespace

void moe_forward(const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K) {
    if (N <= 0) return;
    // FIX: stack buffers t_idx[8]/t_w[8] below overflow when K>8 (direct
    // cpu::moe_* call bypassing ModelConfig::validate) -> stack smash,
    // silent corruption on training. Fail fast (Kaggle/T4 + low-PC safety).
    GAI_CHECK(K >= 1 && K <= 8, "moe_forward: K must be in [1,8]");
    GAI_CHECK(ne > 0 && ne <= 64, "moe_forward: ne out of range");
    GAI_CHECK(d > 0 && E > 0, "moe_forward: bad dims");
#ifdef GAI_OPENMP
    #pragma omp parallel if(N > 4)
#endif
    {
        FwdScratch sc;
        sc.ensure(ne, E, d);
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
        for (i64 t = 0; t < N; ++t) {
            const float* xt = x + t * d;
            float* outt = out + t * d;

            for (int e = 0; e < ne; ++e)
                sc.logits[e] = dot_row(xt, router_w + (size_t)e * d, d);
            // DeepSeek-V2 jitter (train-only: probs_cache!=null => training).
            {
                float jj = g_jitter_cpu.load(std::memory_order_relaxed);
                if (jj > 0.0f && probs_cache) {
                    for (int e = 0; e < ne; ++e)
                        sc.logits[e] *= (1.0f + jj * 2.0f * jitter_u_cpu(t, e));
                }
            }
            softmax_row(sc.logits.data(), ne);

            i32 t_idx[8];
            float t_w[8];
            topk_pick(sc.logits.data(), ne, K, t_idx, t_w);
            float sum_w = 0.0f;
            for (int k = 0; k < K; ++k) sum_w += t_w[k];
            float inv_w = sum_w > 1e-8f ? (1.0f / sum_w) : 0.0f;
            for (int k = 0; k < K; ++k) t_w[k] *= inv_w;

            if (probs_cache) std::memcpy(probs_cache + t * ne, sc.logits.data(),
                                         sizeof(float) * (size_t)ne);
            if (idx_cache) std::memcpy(idx_cache + t * K, t_idx, sizeof(i32) * (size_t)K);
            if (w_cache) std::memcpy(w_cache + t * K, t_w, sizeof(float) * (size_t)K);

            // shared expert (always on unless the model was built with
            // moe_shared=false, in which case the output starts at zero)
            if (sh_g && sh_u && sh_d) {
                linear_forward(xt, sh_g, sc.g.data(), 1, d, E);
                linear_forward(xt, sh_u, sc.u.data(), 1, d, E);
                swiglu_forward(sc.g.data(), sc.u.data(), sc.a.data(), E);
                linear_forward(sc.a.data(), sh_d, outt, 1, E, d);
            } else {
                std::fill(outt, outt + d, 0.0f);
            }

            // routed experts
            for (int k = 0; k < K; ++k) {
                int e = t_idx[k];
                float w = t_w[k];
                size_t slot = ((size_t)t * K + k) * (size_t)E;
                const float* ge = gates + (size_t)e * E * d;
                const float* ue = ups   + (size_t)e * E * d;
                const float* de = downs + (size_t)e * d * E;
                float* sg = s_gate + slot;
                float* su = s_up + slot;
                float* sa = s_act + slot;
                linear_forward(xt, ge, sg, 1, d, E);
                linear_forward(xt, ue, su, 1, d, E);
                swiglu_forward(sg, su, sa, E);
                linear_forward(sa, de, sc.tmp.data(), 1, E, d);
                for (int j = 0; j < d; ++j) outt[j] += w * sc.tmp[j];
            }
        }
    }
}

void moe_forward_bias(const float* x, const float* router_w, const float* router_bias,
                      const float* gates, const float* ups, const float* downs,
                      const float* sh_g, const float* sh_u, const float* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K) {
    if (N <= 0) return;
    GAI_CHECK(K >= 1 && K <= 8, "moe_forward_bias: K must be in [1,8]");
    GAI_CHECK(ne > 0 && ne <= 64, "moe_forward_bias: ne out of range");
    GAI_CHECK(d > 0 && E > 0, "moe_forward_bias: bad dims");
#ifdef GAI_OPENMP
    #pragma omp parallel if(N > 4)
#endif
    {
        FwdScratch sc;
        sc.ensure(ne, E, d);
        std::vector<float> sel;
        sel.resize(static_cast<size_t>(ne));
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
        for (i64 t = 0; t < N; ++t) {
            const float* xt = x + t * d;
            float* outt = out + t * d;

            for (int e = 0; e < ne; ++e)
                sc.logits[e] = dot_row(xt, router_w + (size_t)e * d, d);
            {
                float jj = g_jitter_cpu.load(std::memory_order_relaxed);
                if (jj > 0.0f && probs_cache) {
                    for (int e = 0; e < ne; ++e)
                        sc.logits[e] *= (1.0f + jj * 2.0f * jitter_u_cpu(t, e));
                }
            }
            softmax_row(sc.logits.data(), ne);

            // Selection uses logits+bias (bias steers load, carries no grad).
            for (int e = 0; e < ne; ++e)
                sel[static_cast<size_t>(e)] = sc.logits[static_cast<size_t>(e)] +
                    (router_bias ? router_bias[e] : 0.0f);
            i32 t_idx[8];
            float t_sel[8];
            topk_pick(sel.data(), ne, K, t_idx, t_sel);
            // Weights stay softmax(router) at selected experts, renormalized.
            float t_w[8];
            float sum_w = 0.0f;
            for (int k = 0; k < K; ++k) {
                t_w[k] = sc.logits[static_cast<size_t>(t_idx[k])];
                sum_w += t_w[k];
            }
            float inv_w = sum_w > 1e-8f ? (1.0f / sum_w) : 0.0f;
            for (int k = 0; k < K; ++k) t_w[k] *= inv_w;

            if (probs_cache) std::memcpy(probs_cache + t * ne, sc.logits.data(),
                                         sizeof(float) * (size_t)ne);
            if (idx_cache) std::memcpy(idx_cache + t * K, t_idx, sizeof(i32) * (size_t)K);
            if (w_cache) std::memcpy(w_cache + t * K, t_w, sizeof(float) * (size_t)K);

            if (sh_g && sh_u && sh_d) {
                linear_forward(xt, sh_g, sc.g.data(), 1, d, E);
                linear_forward(xt, sh_u, sc.u.data(), 1, d, E);
                swiglu_forward(sc.g.data(), sc.u.data(), sc.a.data(), E);
                linear_forward(sc.a.data(), sh_d, outt, 1, E, d);
            } else {
                std::fill(outt, outt + d, 0.0f);
            }

            for (int k = 0; k < K; ++k) {
                int e = t_idx[k];
                float w = t_w[k];
                size_t slot = ((size_t)t * K + k) * (size_t)E;
                const float* ge = gates + (size_t)e * E * d;
                const float* ue = ups   + (size_t)e * E * d;
                const float* de = downs + (size_t)e * d * E;
                float* sg = s_gate + slot;
                float* su = s_up + slot;
                float* sa = s_act + slot;
                linear_forward(xt, ge, sg, 1, d, E);
                linear_forward(xt, ue, su, 1, d, E);
                swiglu_forward(sg, su, sa, E);
                linear_forward(sa, de, sc.tmp.data(), 1, E, d);
                for (int j = 0; j < d; ++j) outt[j] += w * sc.tmp[j];
            }
        }
    }
}

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
                  i64 N, int d, int E, int ne, int K) {
    if (N <= 0) return;
    // Same stack-safety contract as forward (see above).
    GAI_CHECK(K >= 1 && K <= 8, "moe_backward: K must be in [1,8]");
    GAI_CHECK(ne > 0 && ne <= 64, "moe_backward: ne out of range");
    // NOTE: serial over tokens BY DESIGN on CPU. Concurrent tokens accumulate
    // into the same expert rows (dsh_*/dgate*/dup*/ddown*), so a naive
    // `#pragma omp parallel for` would race. The T4/CUDA path (cuda/moe.cu)
    // is the parallel implementation used in training: grouped dispatch +
    // batched cuBLAS GEMMs, one thread-block per expert group. Keep this CPU
    // reference serial for gradient-parity with CUDA (tests/test_cuda.cpp).
    BwdScratch sc;
    sc.ensure(ne, E, d);

    const float aux_coef = (aux_frac && aux_scale != 0.0f)
        ? aux_scale * (float)ne / (float)N : 0.0f;

    // NOTE: serial over tokens (shared expert rows would race otherwise).
    for (i64 t = 0; t < N; ++t) {
        const float* xt   = x + t * d;
        const float* dt   = dout + t * d;
        float* dxt        = dx + t * d;
        const float* pt   = probs + t * ne;
        const i32* idxt   = idx + t * K;
        const float* twt  = tw + t * K;

        std::fill(sc.dx.begin(), sc.dx.begin() + d, 0.0f);
        std::fill(sc.dp.begin(), sc.dp.begin() + ne, 0.0f);

        // ---- shared expert: recompute forward, then backward
        // NOTE: swiglu_backward ACCUMULATES (+=, like linear_backward), so its
        // outputs must be zeroed first (the dense path does the same via ops::zero).
        if (sh_g && sh_u && sh_d && dsh_g && dsh_u && dsh_d) {
            linear_forward(xt, sh_g, sc.g.data(), 1, d, E);
            linear_forward(xt, sh_u, sc.u.data(), 1, d, E);
            swiglu_forward(sc.g.data(), sc.u.data(), sc.a.data(), E);
            std::fill(sc.da.begin(), sc.da.begin() + E, 0.0f);
            linear_backward(sc.a.data(), sh_d, dt, sc.da.data(), dsh_d, 1, E, d);
            std::fill(sc.dg.begin(), sc.dg.begin() + E, 0.0f);
            std::fill(sc.du.begin(), sc.du.begin() + E, 0.0f);
            swiglu_backward(sc.g.data(), sc.u.data(), sc.da.data(),
                            sc.dg.data(), sc.du.data(), E);
            linear_backward(xt, sh_g, sc.dg.data(), sc.dx.data(), dsh_g, 1, d, E);
            linear_backward(xt, sh_u, sc.du.data(), sc.dx.data(), dsh_u, 1, d, E);
        }

        // ---- routed experts
        float g_expert[8] = {0.0f};
        float sum_p = 0.0f;
        for (int k = 0; k < K; ++k) {
            int e = idxt[k];
            float w = twt[k];
            if (e < 0 || e >= ne) continue;
            sum_p += pt[e];
            size_t slot = ((size_t)t * K + k) * (size_t)E;
            const float* sg = s_gate + slot;
            const float* su = s_up + slot;
            const float* ge = gates + (size_t)e * E * d;
            const float* ue = ups   + (size_t)e * E * d;
            const float* de = downs + (size_t)e * d * E;
            float* dge = dgates + (size_t)e * E * d;
            float* due = dups   + (size_t)e * E * d;
            float* dde = ddowns + (size_t)e * d * E;
            float* dact = s_dact + slot;

            // dact = (dt * w) @ de^T ; dde += (dt*w)^T @ act_saved
            const float* sa = s_act + slot;
            for (int j = 0; j < d; ++j) sc.tmp[j] = dt[j] * w;
            std::fill(dact, dact + E, 0.0f);
            linear_backward(sa, de, sc.tmp.data(), dact, dde, 1, E, d);
            std::fill(sc.dg.begin(), sc.dg.begin() + E, 0.0f);
            std::fill(sc.du.begin(), sc.du.begin() + E, 0.0f);
            swiglu_backward(sg, su, dact, sc.dg.data(), sc.du.data(), E);
            linear_backward(xt, ge, sc.dg.data(), sc.dx.data(), dge, 1, d, E);
            linear_backward(xt, ue, sc.du.data(), sc.dx.data(), due, 1, d, E);

            // router gradient needs the expert output: out_e = act @ de^T
            linear_forward(sa, de, sc.oute.data(), 1, E, d);
            g_expert[k] = dot_row(dt, sc.oute.data(), d);
        }

        // Backprop through top-k normalization: w_k = p_{e_k} / S, S = sum(p_{e_j})
        // dL/dp_{e_k} = (g_k - bar_g) / S, where bar_g = sum(g_j * w_j)
        float inv_S = sum_p > 1e-8f ? (1.0f / sum_p) : 0.0f;
        float bar_g = 0.0f;
        for (int k = 0; k < K; ++k) bar_g += g_expert[k] * twt[k];
        for (int k = 0; k < K; ++k) {
            int e = idxt[k];
            if (e >= 0 && e < ne) {
                sc.dp[e] += (g_expert[k] - bar_g) * inv_S;
            }
        }

        // aux load-balance term. NOTE: aux_scale here is pre-scaled by the
        // caller (Model::forward_backward) to aux_orig*eff_scale*N so both
        // main (sum*dscale) and aux live in the same scaled+sum space (P0-02).
        // Hence aux_coef = aux_scale*ne/N = aux_orig*ne*eff_scale (sum, scaled).
        if (aux_coef != 0.0f)
            for (int e = 0; e < ne; ++e) sc.dp[e] += aux_coef * aux_frac[e];

        // softmax backward: dlogit = p * (dp - dot(p, dp))
        // P0-06 FIX: forward is z'=z*(1+j*2*u) (jitter), so chain rule is
        // dL/dz = (1+j*2*u)*dL/dz'. Old code omitted the factor (invisible at
        // j=0, wrong router grad at j>0). Same hash as forward => exact match.
        float pdot = 0.0f;
        for (int e = 0; e < ne; ++e) pdot += pt[e] * sc.dp[e];
        {
            float jj = g_jitter_cpu.load(std::memory_order_relaxed);
            for (int e = 0; e < ne; ++e) {
                float dl = pt[e] * (sc.dp[e] - pdot);
                if (jj > 0.0f) dl *= (1.0f + jj * 2.0f * jitter_u_cpu(t, e));
                sc.dlogit[e] = dl;
            float* dr = drouter_w + (size_t)e * d;
            for (int j = 0; j < d; ++j) dr[j] += dl * xt[j];
            const float* rr = router_w + (size_t)e * d;
            for (int j = 0; j < d; ++j) sc.dx[j] += dl * rr[j];
        }
        }

        for (int j = 0; j < d; ++j) dxt[j] += sc.dx[j];
    }
}

// ---------------------------------------------------------------- F-03 fused grouped helpers
// CPU reference for the fused CUDA kernels in cuda/moe.cu (k_pack_all,
// k_save3_all, k_scatter_add_all). Same grouped-slot layout, same indexing,
// verified bit-exact against moe_forward() by tests/test_moe_fused.cpp.

void moe_group_slots(const i32* idx, i64 N, int K, int ne,
                     i32* grouped, int* counts, int* offsets) {
    // Mirrors k_group_hist + k_group_offsets + k_group_fill exactly:
    // histogram over experts, exclusive prefix sum, stable fill.
    for (int e = 0; e < ne; ++e) counts[e] = 0;
    const i64 NK = N * K;
    for (i64 s = 0; s < NK; ++s) {
        const int e = idx[s];
        if (e >= 0 && e < ne) counts[e]++;
    }
    offsets[0] = 0;
    for (int e = 0; e < ne; ++e) offsets[e + 1] = offsets[e] + counts[e];
    std::vector<int> cursor(static_cast<size_t>(ne));
    for (int e = 0; e < ne; ++e) cursor[static_cast<size_t>(e)] = offsets[e];
    for (i64 t = 0; t < N; ++t) {
        for (int k = 0; k < K; ++k) {
            const int e = idx[static_cast<size_t>(t) * K + k];
            if (e < 0 || e >= ne) continue;
            grouped[cursor[static_cast<size_t>(e)]++] = static_cast<i32>(t * K + k);
        }
    }
}

void moe_pack_all(const float* x, const i32* grouped, float* out,
                  i64 NK, int d, int K) {
    // out[s] = x[grouped[s]/K]. One pass over every grouped slot.
    for (i64 s = 0; s < NK; ++s) {
        const i64 t = static_cast<i64>(grouped[s]) / K;
        std::memcpy(out + s * d, x + t * d, sizeof(float) * static_cast<size_t>(d));
    }
}

void moe_gather3_all(const float* s_gate, const float* s_up, const float* s_act,
                     const i32* grouped, float* G, float* U, float* A,
                     i64 NK, int E) {
    // G/U/A[s] = s_*[grouped[s]]. Inverse of moe_save3_all.
    for (i64 s = 0; s < NK; ++s) {
        const i64 slot = grouped[s];
        std::memcpy(G + s * E, s_gate + slot * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(U + s * E, s_up + slot * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(A + s * E, s_act + slot * E, sizeof(float) * static_cast<size_t>(E));
    }
}

void moe_scale_all(const float* dout, const float* w, const i32* grouped,
                   float* S, i64 NK, int d, int K) {
    // S[s] = dout[t] * w[t,k]. Fused pack+scale of the upstream grads.
    for (i64 s = 0; s < NK; ++s) {
        const i64 slot = grouped[s];
        const i64 t = slot / K;
        const i64 k = slot % K;
        const float wv = w ? w[t * K + k] : 1.0f;
        const float* r = dout + t * d;
        float* o = S + s * d;
        for (int j = 0; j < d; ++j) o[j] = r[j] * wv;
    }
}

void moe_save3_all(const float* G, const float* U, const float* A,
                   const i32* grouped, float* s_gate, float* s_up, float* s_act,
                   i64 NK, int E) {
    // s_*[grouped[s]] = block[s]. Same destination order as the per-expert
    // k_scatter_copy sequence (slot-major), just emitted in one pass.
    for (i64 s = 0; s < NK; ++s) {
        const i64 slot = grouped[s];
        std::memcpy(s_gate + slot * E, G + s * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(s_up + slot * E, U + s * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(s_act + slot * E, A + s * E, sizeof(float) * static_cast<size_t>(E));
    }
}

void moe_scatter_add_all(float* out, const float* Y, const i32* grouped,
                         const float* w, i64 NK, int d, int K) {
    // out[t] += w[t,k] * Y[s], single pass. K=2 slots of one token share the
    // destination row, so (like the CUDA kernel) this accumulates; the caller
    // must have zeroed `out` for the routed contribution first.
    for (i64 s = 0; s < NK; ++s) {
        const i64 slot = grouped[s];
        const i64 t = slot / K;
        const i64 k = slot % K;
        const float wv = w ? w[t * K + k] : 1.0f;
        float* o = out + t * d;
        const float* r = Y + s * d;
        for (int j = 0; j < d; ++j) o[j] += wv * r[j];
    }
}

void moe_count_slots(const i32* idx, float* acc, i64 NK, int ne) {
    // F-02 CPU reference: identical math to k_count_slots in cuda/moe.cu.
    for (i64 s = 0; s < NK; ++s) {
        const int e = idx[s];
        if (e >= 0 && e < ne) acc[e] += 1.0f;
    }
}

void moe_forward_fused(const float* x, const float* router_w,
                       const float* gates, const float* ups, const float* downs,
                       const float* sh_g, const float* sh_u, const float* sh_d,
                       float* out,
                       float* probs_cache, i32* idx_cache, float* w_cache,
                       float* s_gate, float* s_up, float* s_act,
                       float* Xpack, float* Gpack, float* Upack, float* Apack,
                       float* Ypack, i32* grouped, int* counts, int* offsets,
                       i64 N, int d, int E, int ne, int K) {
    if (N <= 0) return;
    GAI_CHECK(K >= 1 && K <= 8, "moe_forward_fused: K must be in [1,8]");
    GAI_CHECK(ne > 0 && ne <= 64, "moe_forward_fused: ne out of range");
    const i64 NK = N * K;

    // 1. route exactly like moe_forward (same router math, same caches).
    // NOTE: this duplicates the routing loop rather than calling moe_forward
    // because the fused path needs idx/w in hand before packing; the math is
    // copied verbatim so a divergence is a compile-visible diff, not drift.
    {
        FwdScratch sc;
        sc.ensure(ne, E, d);
        for (i64 t = 0; t < N; ++t) {
            const float* xt = x + t * d;
            for (int e = 0; e < ne; ++e)
                sc.logits[e] = dot_row(xt, router_w + (size_t)e * d, d);
            {
                float jj = g_jitter_cpu.load(std::memory_order_relaxed);
                if (jj > 0.0f && probs_cache) {
                    for (int e = 0; e < ne; ++e)
                        sc.logits[e] *= (1.0f + jj * 2.0f * jitter_u_cpu(t, e));
                }
            }
            softmax_row(sc.logits.data(), ne);
            i32 t_idx[8];
            float t_w[8];
            topk_pick(sc.logits.data(), ne, K, t_idx, t_w);
            float sum_w = 0.0f;
            for (int k = 0; k < K; ++k) sum_w += t_w[k];
            float inv_w = sum_w > 1e-8f ? (1.0f / sum_w) : 0.0f;
            for (int k = 0; k < K; ++k) t_w[k] *= inv_w;
            if (probs_cache) std::memcpy(probs_cache + t * ne, sc.logits.data(),
                                         sizeof(float) * (size_t)ne);
            if (idx_cache) std::memcpy(idx_cache + t * K, t_idx, sizeof(i32) * (size_t)K);
            if (w_cache) std::memcpy(w_cache + t * K, t_w, sizeof(float) * (size_t)K);
        }
    }

    const i32* idx_use = idx_cache ? idx_cache : nullptr;
    const float* w_use = w_cache ? w_cache : nullptr;
    // The fused path needs materialized routing; without caches there is no
    // grouped layout to pack from.
    GAI_CHECK(idx_use != nullptr && w_use != nullptr,
              "moe_forward_fused requires idx/w caches (training/inference always pass them)");

    // 2. shared expert first (writes the base `out`, like the reference).
    if (sh_g && sh_u && sh_d) {
        std::vector<float> g(static_cast<size_t>(N) * E), u(static_cast<size_t>(N) * E),
            a(static_cast<size_t>(N) * E);
        for (i64 t = 0; t < N; ++t) {
            linear_forward(x + t * d, sh_g, g.data() + t * E, 1, d, E);
            linear_forward(x + t * d, sh_u, u.data() + t * E, 1, d, E);
        }
        swiglu_forward(g.data(), u.data(), a.data(), N * E);
        for (i64 t = 0; t < N; ++t)
            linear_forward(a.data() + t * E, sh_d, out + t * d, 1, E, d);
    } else {
        for (i64 t = 0; t < N; ++t)
            std::fill(out + t * d, out + t * d + d, 0.0f);
    }

    // 3. group on the (already routed) indices.
    moe_group_slots(idx_use, N, K, ne, grouped, counts, offsets);

    // 4. pack every expert's input rows in one pass.
    moe_pack_all(x, grouped, Xpack, NK, d, K);

    // 5. per-expert gate/up GEMMs on packed blocks (same math as reference,
    //    only the input layout changed from token-major to grouped).
    for (int e = 0; e < ne; ++e) {
        const int ns = counts[e];
        if (ns == 0) continue;
        const float* ge = gates + (size_t)e * E * d;
        const float* ue = ups   + (size_t)e * E * d;
        float* Gp = Gpack + (size_t)offsets[e] * E;
        float* Up = Upack + (size_t)offsets[e] * E;
        const float* Xp = Xpack + (size_t)offsets[e] * d;
        for (int s = 0; s < ns; ++s) {
            linear_forward(Xp + (size_t)s * d, ge, Gp + (size_t)s * E, 1, d, E);
            linear_forward(Xp + (size_t)s * d, ue, Up + (size_t)s * E, 1, d, E);
        }
    }

    // 6. one swiglu over the whole packed block.
    swiglu_forward(Gpack, Upack, Apack, NK * E);

    // 7. one save of every expert's G/U/A.
    moe_save3_all(Gpack, Upack, Apack, grouped, s_gate, s_up, s_act, NK, E);

    // 8. per-expert down GEMMs into the packed output block.
    for (int e = 0; e < ne; ++e) {
        const int ns = counts[e];
        if (ns == 0) continue;
        const float* de = downs + (size_t)e * d * E;
        const float* Ap = Apack + (size_t)offsets[e] * E;
        float* Yp = Ypack + (size_t)offsets[e] * d;
        for (int s = 0; s < ns; ++s)
            linear_forward(Ap + (size_t)s * E, de, Yp + (size_t)s * d, 1, E, d);
    }

    // 9. one weighted scatter-add over every slot.
    moe_scatter_add_all(out, Ypack, grouped, w_use, NK, d, K);
}

} // namespace cpu
} // namespace gai
