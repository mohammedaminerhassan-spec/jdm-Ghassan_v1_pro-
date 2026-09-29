#include "core/ops_cpu.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <atomic>

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
    uint64_t h = static_cast<uint64_t>(tt) * 0x9E3779B97F4A7C15ULL;
    h ^= static_cast<uint64_t>(ee) * 0xC2B2AE3D27D4EB4FULL;
    h ^= (seed + 0x9E3779B97F4A7C15ULL);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (static_cast<float>((h >> 40) & 0xFFFFFF) / static_cast<float>(1 << 24)) - 0.5f;
}

namespace gai {
namespace cpu {

float jitter_u_for_test(i64 tt, int ee) { return jitter_u_cpu(tt, ee); }
}
}

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace cpu {

namespace {

struct FwdScratch {
    std::vector<float> logits;
    std::vector<float> g, u, a;
    std::vector<float> tmp;
    void ensure(int ne, int E, int d) {
        if ((int)logits.size() < ne) logits.resize(ne);
        if ((int)g.size() < E) { g.resize(E); u.resize(E); a.resize(E); }
        if ((int)tmp.size() < d) tmp.resize(d);
    }
};
struct BwdScratch {
    std::vector<float> dlogit;
    std::vector<float> dp;
    std::vector<float> g, u, a;
    std::vector<float> dg, du, da;
    std::vector<float> dx;
    std::vector<float> tmp;
    std::vector<float> oute;
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

}

void moe_forward(const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K) {
    if (N <= 0) return;

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

            for (int e = 0; e < ne; ++e)
                sel[static_cast<size_t>(e)] = sc.logits[static_cast<size_t>(e)] +
                    (router_bias ? router_bias[e] : 0.0f);
            i32 t_idx[8];
            float t_sel[8];
            topk_pick(sel.data(), ne, K, t_idx, t_sel);

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

    GAI_CHECK(K >= 1 && K <= 8, "moe_backward: K must be in [1,8]");
    GAI_CHECK(ne > 0 && ne <= 64, "moe_backward: ne out of range");

    BwdScratch sc;
    sc.ensure(ne, E, d);

    const float aux_coef = (aux_frac && aux_scale != 0.0f)
        ? aux_scale * (float)ne / (float)N : 0.0f;

    for (i64 t = 0; t < N; ++t) {
        const float* xt   = x + t * d;
        const float* dt   = dout + t * d;
        float* dxt        = dx + t * d;
        const float* pt   = probs + t * ne;
        const i32* idxt   = idx + t * K;
        const float* twt  = tw + t * K;

        std::fill(sc.dx.begin(), sc.dx.begin() + d, 0.0f);
        std::fill(sc.dp.begin(), sc.dp.begin() + ne, 0.0f);

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

            const float* sa = s_act + slot;
            for (int j = 0; j < d; ++j) sc.tmp[j] = dt[j] * w;
            std::fill(dact, dact + E, 0.0f);
            linear_backward(sa, de, sc.tmp.data(), dact, dde, 1, E, d);
            std::fill(sc.dg.begin(), sc.dg.begin() + E, 0.0f);
            std::fill(sc.du.begin(), sc.du.begin() + E, 0.0f);
            swiglu_backward(sg, su, dact, sc.dg.data(), sc.du.data(), E);
            linear_backward(xt, ge, sc.dg.data(), sc.dx.data(), dge, 1, d, E);
            linear_backward(xt, ue, sc.du.data(), sc.dx.data(), due, 1, d, E);

            linear_forward(sa, de, sc.oute.data(), 1, E, d);
            g_expert[k] = dot_row(dt, sc.oute.data(), d);
        }

        float inv_S = sum_p > 1e-8f ? (1.0f / sum_p) : 0.0f;
        float bar_g = 0.0f;
        for (int k = 0; k < K; ++k) bar_g += g_expert[k] * twt[k];
        for (int k = 0; k < K; ++k) {
            int e = idxt[k];
            if (e >= 0 && e < ne) {
                sc.dp[e] += (g_expert[k] - bar_g) * inv_S;
            }
        }

        if (aux_coef != 0.0f)
            for (int e = 0; e < ne; ++e) sc.dp[e] += aux_coef * aux_frac[e];

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

void moe_group_slots(const i32* idx, i64 N, int K, int ne,
                     i32* grouped, int* counts, int* offsets) {

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

    for (i64 s = 0; s < NK; ++s) {
        const i64 t = static_cast<i64>(grouped[s]) / K;
        std::memcpy(out + s * d, x + t * d, sizeof(float) * static_cast<size_t>(d));
    }
}

void moe_gather3_all(const float* s_gate, const float* s_up, const float* s_act,
                     const i32* grouped, float* G, float* U, float* A,
                     i64 NK, int E) {

    for (i64 s = 0; s < NK; ++s) {
        const i64 slot = grouped[s];
        std::memcpy(G + s * E, s_gate + slot * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(U + s * E, s_up + slot * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(A + s * E, s_act + slot * E, sizeof(float) * static_cast<size_t>(E));
    }
}

void moe_scale_all(const float* dout, const float* w, const i32* grouped,
                   float* S, i64 NK, int d, int K) {

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

    for (i64 s = 0; s < NK; ++s) {
        const i64 slot = grouped[s];
        std::memcpy(s_gate + slot * E, G + s * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(s_up + slot * E, U + s * E, sizeof(float) * static_cast<size_t>(E));
        std::memcpy(s_act + slot * E, A + s * E, sizeof(float) * static_cast<size_t>(E));
    }
}

void moe_scatter_add_all(float* out, const float* Y, const i32* grouped,
                         const float* w, i64 NK, int d, int K) {

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

    GAI_CHECK(idx_use != nullptr && w_use != nullptr,
              "moe_forward_fused requires idx/w caches (training/inference always pass them)");

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

    moe_group_slots(idx_use, N, K, ne, grouped, counts, offsets);

    moe_pack_all(x, grouped, Xpack, NK, d, K);

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

    swiglu_forward(Gpack, Upack, Apack, NK * E);

    moe_save3_all(Gpack, Upack, Apack, grouped, s_gate, s_up, s_act, NK, E);

    for (int e = 0; e < ne; ++e) {
        const int ns = counts[e];
        if (ns == 0) continue;
        const float* de = downs + (size_t)e * d * E;
        const float* Ap = Apack + (size_t)offsets[e] * E;
        float* Yp = Ypack + (size_t)offsets[e] * d;
        for (int s = 0; s < ns; ++s)
            linear_forward(Ap + (size_t)s * E, de, Yp + (size_t)s * d, 1, E, d);
    }

    moe_scatter_add_all(out, Ypack, grouped, w_use, NK, d, K);
}

}
}
