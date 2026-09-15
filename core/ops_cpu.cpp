#include "core/ops_cpu.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {
namespace cpu {

// ================================================================ GEMM
// Row-major. Four transpose cases with loop orders chosen for cache locality.
// The dominant case in this model is NT (C = A * B^T) because weights are
// stored [out_features, in_features].

static inline float dot_f32(const float* a, const float* b, int n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = s0 + s1 + s2 + s3;
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline void axpy_f32(float* y, const float* x, float a, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        y[i]     += a * x[i];
        y[i + 1] += a * x[i + 1];
        y[i + 2] += a * x[i + 2];
        y[i + 3] += a * x[i + 3];
    }
    for (; i < n; ++i) y[i] += a * x[i];
}

void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0) return;

    // scale / clear C
    if (beta == 0.0f) {
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 8) schedule(static)
#endif
        for (int m = 0; m < M; ++m) std::memset(C + static_cast<size_t>(m) * ldc, 0, sizeof(float) * static_cast<size_t>(N));
    } else if (beta != 1.0f) {
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 8) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            float* c = C + static_cast<size_t>(m) * ldc;
            for (int n = 0; n < N; ++n) c[n] *= beta;
        }
    }
    if (K <= 0 || alpha == 0.0f) return;

    if (!trans_a && trans_b) {
        // C[m,n] += alpha * dot(A[m,:], B[n,:])
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M * static_cast<long long>(N) > 4096) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            const float* a = A + static_cast<size_t>(m) * lda;
            float*       c = C + static_cast<size_t>(m) * ldc;
            for (int n = 0; n < N; ++n) {
                c[n] += alpha * dot_f32(a, B + static_cast<size_t>(n) * ldb, K);
            }
        }
    } else if (!trans_a && !trans_b) {
        // C[m,:] += alpha * sum_k A[m,k] * B[k,:]
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 2) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            const float* a = A + static_cast<size_t>(m) * lda;
            float*       c = C + static_cast<size_t>(m) * ldc;
            for (int k = 0; k < K; ++k) {
                float av = a[k];
                if (av != 0.0f) axpy_f32(c, B + static_cast<size_t>(k) * ldb, alpha * av, N);
            }
        }
    } else if (trans_a && !trans_b) {
        // A is [K,M]: C[m,:] += alpha * sum_k A[k,m] * B[k,:]
        // parallelise over M with a strided read of A (correct, no races)
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 2) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            float* c = C + static_cast<size_t>(m) * ldc;
            for (int k = 0; k < K; ++k) {
                float av = A[static_cast<size_t>(k) * lda + m];
                if (av != 0.0f) axpy_f32(c, B + static_cast<size_t>(k) * ldb, alpha * av, N);
            }
        }
    } else {
        // A [K,M], B [N,K]
#ifdef GAI_OPENMP
        #pragma omp parallel for if(M > 2) schedule(static)
#endif
        for (int m = 0; m < M; ++m) {
            float* c = C + static_cast<size_t>(m) * ldc;
            for (int n = 0; n < N; ++n) {
                float s = 0.f;
                const float* b = B + static_cast<size_t>(n) * ldb;
                for (int k = 0; k < K; ++k) s += A[static_cast<size_t>(k) * lda + m] * b[k];
                c[n] += alpha * s;
            }
        }
    }
}

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N) {
    gemm(false, true, M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
}

void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N) {
    if (dx) gemm(false, false, M, K, N, 1.0f, dy, N, w, K, 1.0f, dx, K);   // dx += dy * W
    if (dw) gemm(true,  false, N, K, M, 1.0f, dy, N, x, K, 1.0f, dw, K);   // dw += dy^T * x
}

// ================================================================ elementwise
void add(const float* a, const float* b, float* out, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void add_inplace(float* a, const float* b, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) a[i] += b[i];
}

void scale_inplace(float* a, float s, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) a[i] *= s;
}

// ================================================================ embedding
void embedding_forward(const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab) {
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {
        i32 id = ids[t];
        float* o = out + t * dim;
        if (id < 0 || id >= vocab) { std::memset(o, 0, sizeof(float) * static_cast<size_t>(dim)); continue; }
        std::memcpy(o, table + static_cast<size_t>(id) * dim, sizeof(float) * static_cast<size_t>(dim));
    }
}

void embedding_backward(const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab) {
    // Serial accumulation over tokens avoids atomics; dim-loop is vectorised.
    for (i64 t = 0; t < ntok; ++t) {
        i32 id = ids[t];
        if (id < 0 || id >= vocab) continue;
        float* d = dtable + static_cast<size_t>(id) * dim;
        const float* g = dout + t * dim;
        for (int i = 0; i < dim; ++i) d[i] += g[i];
    }
}

// ================================================================ rmsnorm
void rmsnorm_forward(const float* x, const float* weight, float* out, float* rrms,
                     i64 rows, int dim, float eps) {
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 r = 0; r < rows; ++r) {
        const float* xr = x + r * dim;
        float ss = 0.f;
        for (int i = 0; i < dim; ++i) ss += xr[i] * xr[i];
        float inv = 1.0f / std::sqrt(ss / static_cast<float>(dim) + eps);
        if (rrms) rrms[r] = inv;
        float* o = out + r * dim;
        for (int i = 0; i < dim; ++i) o[i] = xr[i] * inv * weight[i];
    }
}

void rmsnorm_backward(const float* x, const float* weight, const float* dout,
                      const float* rrms, float* dx, float* dweight, i64 rows, int dim) {
    // dweight needs cross-row accumulation -> per-thread buffers.
    int nthreads = 1;
#ifdef GAI_OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<float> dw_buf(static_cast<size_t>(nthreads) * static_cast<size_t>(dim), 0.0f);

#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 r = 0; r < rows; ++r) {
        int tid = 0;
#ifdef GAI_OPENMP
        tid = omp_get_thread_num();
#endif
        const float* xr = x + r * dim;
        const float* gr = dout + r * dim;
        float inv = rrms[r];
        float* dwl = dw_buf.data() + static_cast<size_t>(tid) * static_cast<size_t>(dim);

        // dot = sum_i g_i * w_i * x_i
        float dot = 0.f;
        for (int i = 0; i < dim; ++i) dot += gr[i] * weight[i] * xr[i];
        float coef = inv * inv * inv / static_cast<float>(dim) * dot;

        float* dxr = dx + r * dim;
        for (int i = 0; i < dim; ++i) {
            dwl[i] += gr[i] * xr[i] * inv;
            dxr[i] += gr[i] * weight[i] * inv - xr[i] * coef;
        }
    }

    if (dweight) {
        for (int t = 0; t < nthreads; ++t) {
            const float* src = dw_buf.data() + static_cast<size_t>(t) * static_cast<size_t>(dim);
            for (int i = 0; i < dim; ++i) dweight[i] += src[i];
        }
    }
}

// ================================================================ rope
// Interleaved-pair convention: dims (2i, 2i+1) rotate together.
static inline void rope_pair(float& a, float& b, float c, float s) {
    float na = a * c - b * s;
    float nb = a * s + b * c;
    a = na;
    b = nb;
}

void rope_forward(float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {
    if (!q && !k) return;
    int half = head_dim / 2;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {
        float p = static_cast<float>(pos[t]);
        for (int i = 0; i < half; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
            float ang  = p * freq;
            float c = std::cos(ang), s = std::sin(ang);
            if (q) {
                for (int h = 0; h < n_heads; ++h) {
                    float* qh = q + (t * n_heads + h) * head_dim;
                    rope_pair(qh[2 * i], qh[2 * i + 1], c, s);
                }
            }
            if (k) {
                for (int h = 0; h < n_kv; ++h) {
                    float* kh = k + (t * n_kv + h) * head_dim;
                    rope_pair(kh[2 * i], kh[2 * i + 1], c, s);
                }
            }
        }
    }
}

void rope_backward(float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {
    // rotation is orthogonal -> backward is rotation by -angle
    // Null-safe: dq/dk may be null when unrotating Q-only or K-only staging
    // buffers (QK-Norm exact backward). Zero-head loops never dereference.
    if (!dq && !dk) return;
    int half = head_dim / 2;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {
        float p = static_cast<float>(pos[t]);
        for (int i = 0; i < half; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
            float ang  = p * freq;
            float c = std::cos(ang), s = -std::sin(ang);
            if (dq) {
                for (int h = 0; h < n_heads; ++h) {
                    float* qh = dq + (t * n_heads + h) * head_dim;
                    rope_pair(qh[2 * i], qh[2 * i + 1], c, s);
                }
            }
            if (dk) {
                for (int h = 0; h < n_kv; ++h) {
                    float* kh = dk + (t * n_kv + h) * head_dim;
                    rope_pair(kh[2 * i], kh[2 * i + 1], c, s);
                }
            }
        }
    }
}

// ================================================================ swiglu
static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void swiglu_forward(const float* g, const float* u, float* out, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) out[i] = silu(g[i]) * u[i];
}

void swiglu_backward(const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) {
        float gv = g[i];
        float sg = 1.0f / (1.0f + std::exp(-gv));
        float sl = gv * sg;
        float dsilu = sg * (1.0f + gv * (1.0f - sg));
        dg[i] += dout[i] * u[i] * dsilu;
        du[i] += dout[i] * sl;
    }
}

// ================================================================ softmax
void softmax_row(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

// ================================================================ attention
// q [B,T,H,hd], k/v [B,T,KV,hd], out [B,T,H,hd]
// probs (optional) [B,H,T,T] lower-triangular (upper part left as 0).
// FIX (10/10): old code did `std::vector<float> s(T)` per (b,h) per layer
// (624 allocs/step at L=26) — allocator churn that looked like a CPU leak.
// Reuse a thread-local scratch that only grows (same monotonic-pool idea as
// CUDA workspaces). Identical math, zero per-row mallocs.
void attention_forward(const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale) {
    const int group = H / KV;
    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

#ifdef GAI_OPENMP
    #pragma omp parallel
#endif
    {
        thread_local std::vector<float> tls;
        if (tls.size() < static_cast<size_t>(T)) tls.resize(static_cast<size_t>(T));
        float* s = tls.data();
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
    for (int bh = 0; bh < B * H; ++bh) {
        {
            int b = bh / H;
            int h = bh % H;
            int kvh = h / group;
            for (int t = 0; t < T; ++t) {
                const float* qh = q + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                int len = t + 1;
                float mx = -3.4e38f;
                for (int j = 0; j < len; ++j) {
                    const float* kh = k + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                    float d = dot_f32(qh, kh, hd) * scale;
                    s[static_cast<size_t>(j)] = d;
                    mx = std::max(mx, d);
                }
                float sum = 0.f;
                for (int j = 0; j < len; ++j) { s[static_cast<size_t>(j)] = std::exp(s[static_cast<size_t>(j)] - mx); sum += s[static_cast<size_t>(j)]; }
                float inv = 1.0f / sum;
                float* o = out + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
                for (int j = 0; j < len; ++j) {
                    float p = s[static_cast<size_t>(j)] * inv;
                    s[static_cast<size_t>(j)] = p;
                    const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                    axpy_f32(o, vh, p, hd);
                }
                if (probs) {
                    float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    std::memcpy(pr, s, sizeof(float) * static_cast<size_t>(len));
                    if (len < T) std::memset(pr + len, 0, sizeof(float) * static_cast<size_t>(T - len));
                }
            }
        }
    }
    }
}

void attention_backward(const float* q, const float* k, const float* v,
                        const float* probs, const float* dout,
                        float* dq, float* dk, float* dv,
                        int B, int T, int H, int KV, int hd, float scale) {
    const int group = H / KV;
    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

    // Parallelise over (b, kv-head) so that dk/dv writes never race:
    // all query heads in a group map to the same kv head, handled by one thread.
    // Flattened loop (no collapse clause: MSVC warns C4849 on it).
    // FIX (10/10): same thread-local reuse as forward (was per-(b,kv) alloc).
#ifdef GAI_OPENMP
    #pragma omp parallel
#endif
    {
        thread_local std::vector<float> tld;
        if (tld.size() < static_cast<size_t>(T)) tld.resize(static_cast<size_t>(T));
#ifdef GAI_OPENMP
        #pragma omp for schedule(static)
#endif
    for (int bkv = 0; bkv < B * KV; ++bkv) {
        {
            int b = bkv / KV;
            int kvh = bkv % KV;
            float* dsv = tld.data();
            for (int hg = 0; hg < group; ++hg) {
                int h = kvh * group + hg;
                for (int t = 0; t < T; ++t) {
                    int len = t + 1;
                    const float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    const float* go = dout + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    const float* qh = q + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    float* dqh = dq + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;

                    // dP_j = dot(go, v_j) ; also accumulate dv_j += p_j * go
                    float dot_pg = 0.f;
                    for (int j = 0; j < len; ++j) {
                        const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        float dpj = dot_f32(go, vh, hd);
                        dsv[static_cast<size_t>(j)] = dpj;
                        dot_pg += pr[j] * dpj;
                        float* dvh = dv + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        axpy_f32(dvh, go, pr[j], hd);
                    }
                    // softmax jacobian: dS_j = p_j * (dP_j - sum_l p_l dP_l)
                    for (int j = 0; j < len; ++j) {
                        float ds = pr[j] * (dsv[static_cast<size_t>(j)] - dot_pg) * scale;
                        if (ds == 0.0f) continue;
                        const float* kh = k + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        float* dkh = dk + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        axpy_f32(dqh, kh, ds, hd);
                        axpy_f32(dkh, qh, ds, hd);
                    }
                }
            }
        }
    }
    }
}

void attention_decode(const float* q, const float* kcache, const float* vcache,
                      float* out, int H, int KV, int hd, int cur_len, int max_len,
                      float scale, float* scratch) {
    const int group = H / KV;
    (void)max_len;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; ++h) {
        int kvh = h / group;
        float* s = scratch + static_cast<size_t>(h) * static_cast<size_t>(cur_len);
        const float* qh = q + static_cast<size_t>(h) * hd;
        float mx = -3.4e38f;
        for (int j = 0; j < cur_len; ++j) {
            const float* kh = kcache + (static_cast<size_t>(j) * KV + kvh) * hd;
            float d = dot_f32(qh, kh, hd) * scale;
            s[j] = d;
            mx = std::max(mx, d);
        }
        float sum = 0.f;
        for (int j = 0; j < cur_len; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float inv = 1.0f / sum;
        float* o = out + static_cast<size_t>(h) * hd;
        std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
        for (int j = 0; j < cur_len; ++j) {
            const float* vh = vcache + (static_cast<size_t>(j) * KV + kvh) * hd;
            axpy_f32(o, vh, s[j] * inv, hd);
        }
    }
}

// ================================================================ loss
void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale) {
    double total = 0.0;
    i64 count = 0;

    // count valid first so dlogits can be scaled in the same pass
    for (i64 i = 0; i < n; ++i) if (targets[i] >= 0 && targets[i] < V) ++count;
    float invc = count > 0 ? 1.0f / static_cast<float>(count) : 0.0f;

#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:total)
#endif
    for (i64 i = 0; i < n; ++i) {
        const float* row = logits + i * V;
        i32 tgt = targets[i];
        float* d = dlogits ? dlogits + i * V : nullptr;

        if (tgt < 0 || tgt >= V) {
            if (d) std::memset(d, 0, sizeof(float) * static_cast<size_t>(V));
            continue;
        }
        float mx = row[0];
        for (int j = 1; j < V; ++j) mx = std::max(mx, row[j]);
        double sum = 0.0;
        for (int j = 0; j < V; ++j) sum += std::exp(static_cast<double>(row[j] - mx));
        double logZ = std::log(sum) + static_cast<double>(mx);
        total += logZ - static_cast<double>(row[tgt]);
        if (z_scale != 0.0f) total += (double)z_scale * logZ * logZ;

        if (d) {
            float invsum = static_cast<float>(1.0 / sum);
            float zcorr = (z_scale != 0.0f) ? (2.0f * z_scale * (float)logZ * invc) : 0.0f;
            for (int j = 0; j < V; ++j) {
                float p = std::exp(row[j] - mx) * invsum;
                d[j] = p * (invc + zcorr);
            }
            d[tgt] -= invc;
        }
    }
    if (out_loss_sum) *out_loss_sum = total;
    if (out_count)    *out_count = count;
}

// ================================================================ optimizer
void adamw_step(float* w, const float* g, float* m, float* v, i64 n,
                float lr, float beta1, float beta2, float eps, float weight_decay,
                float bc1, float bc2, float grad_scale) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) {
        float gi = g[i] * grad_scale;
        float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float mh = mi / bc1;
        float vh = vi / bc2;
        float upd = mh / (std::sqrt(vh) + eps);
        if (weight_decay != 0.0f) upd += weight_decay * w[i];
        w[i] -= lr * upd;
    }
}

void lion_step(float* w, const float* g, float* m, i64 n,
               float lr, float beta1, float beta2, float weight_decay,
               float grad_scale) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(n > 8192) schedule(static)
#endif
    for (i64 i = 0; i < n; ++i) {
        float gi = g[i] * grad_scale;
        float mi = m[i];
        float c = beta1 * mi + (1.0f - beta1) * gi;
        float upd = (c > 0.0f) ? 1.0f : ((c < 0.0f) ? -1.0f : 0.0f);
        if (weight_decay != 0.0f) upd += weight_decay * w[i];
        w[i] -= lr * upd;
        m[i] = beta2 * mi + (1.0f - beta2) * gi;
    }
}

double global_sq_norm(const float* g, i64 n) {
    double s = 0.0;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:s)
#endif
    for (i64 i = 0; i < n; ++i) s += static_cast<double>(g[i]) * static_cast<double>(g[i]);
    return s;
}

double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts) {
    double s = 0.0;
    for (auto& pr : parts) s += global_sq_norm(pr.first, pr.second);
    return s;
}

} // namespace cpu
} // namespace gai
