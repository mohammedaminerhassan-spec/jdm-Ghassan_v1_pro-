#include "core/ops_cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>
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

// P1-8: explicit AVX2+FMA kernels with RUNTIME dispatch (llama.cpp-style).
// The default build stays baseline x86-64 (SSE2) so binaries run on old PCs;
// on AVX2+FMA machines the hot dot/axpy loops below switch to intrinsics.
// Scalar path is untouched (used on ARM/MSVC/old x86 and via GAI_NO_SIMD=1).
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && !defined(_MSC_VER)
#define GAI_HAVE_AVX2_DISPATCH 1
#include <immintrin.h>
namespace simd {
// Cached once per process (GAI_NO_SIMD=1 forces scalar, e.g. for A/B tests).
inline bool supported() {
    static const bool v = [] {
        const char* e = std::getenv("GAI_NO_SIMD");
        if (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y')) return false;
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return v;
}
__attribute__((target("avx2,fma"))) inline float dot_avx2(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    int i = 0;
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    const __m128 lo = _mm256_castps256_ps128(s);
    const __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 h = _mm_add_ps(lo, hi);
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    float out = _mm_cvtss_f32(h);
    for (; i < n; ++i) out += a[i] * b[i];
    return out;
}
__attribute__((target("avx2,fma"))) inline void axpy_avx2(float* y, const float* x, float a, int n) {
    const __m256 va = _mm256_set1_ps(a);
    int i = 0;
    const int n32 = n & ~31;
    for (; i < n32; i += 32) {
        __m256 y0 = _mm256_loadu_ps(y + i);
        __m256 y1 = _mm256_loadu_ps(y + i + 8);
        __m256 y2 = _mm256_loadu_ps(y + i + 16);
        __m256 y3 = _mm256_loadu_ps(y + i + 24);
        y0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i),      y0);
        y1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8),  y1);
        y2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 16), y2);
        y3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 24), y3);
        _mm256_storeu_ps(y + i,      y0);
        _mm256_storeu_ps(y + i + 8,  y1);
        _mm256_storeu_ps(y + i + 16, y2);
        _mm256_storeu_ps(y + i + 24, y3);
    }
    for (; i < n; ++i) y[i] += a * x[i];
}
} // namespace simd
#else
#define GAI_HAVE_AVX2_DISPATCH 0
#endif

static inline float dot_f32(const float* a, const float* b, int n) {
#if GAI_HAVE_AVX2_DISPATCH
    if (n >= 32 && simd::supported()) return simd::dot_avx2(a, b, n);
#endif
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
#if GAI_HAVE_AVX2_DISPATCH
    if (n >= 32 && simd::supported()) {
        simd::axpy_avx2(y, x, a, n);
        return;
    }
#endif
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

    // ---------------------------------------------------------------- GEMV
    // FIX (P1-1): every GEMM below parallelises over M. At decode time M==1
    // (one token), so the omp-for ran over a single iteration and ONE thread
    // did 100% of the work -> single-threaded token generation on every core
    // count. Split the M==1 NT case (the shape every linear_forward uses)
    // over N instead. omp_in_parallel() guard: moe_forward already runs
    // inside a parallel region during training, and nested regions would add
    // overhead for no gain. N>2048 threshold is MEASURED (laptop: serial+SIMD
    // wins at N=768 (0.159 vs 0.181ms) and ties at N=2048; GEMV decode is
    // bandwidth-bound so threads add little on small rows). The V-head
    // (N=32000, ~half of all decode weight traffic) always stays parallel,
    // which is where engaging all cores actually matters.
    if (M == 1 && !trans_a && trans_b) {
        const float* a = A;
        float*       c = C;
#ifdef GAI_OPENMP
        const bool nested = omp_in_parallel();
        #pragma omp parallel for schedule(static) if(!nested && N > 2048)
#endif
        for (int n = 0; n < N; ++n)
            c[n] += alpha * dot_f32(a, B + static_cast<size_t>(n) * ldb, K);
        return;
    }

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

void linear_forward_fp16(const float* x, const u16* w, float* y, int M, int K, int N) {
#ifdef GAI_OPENMP
    #pragma omp parallel for if(M * N > 256) schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            const u16* row = w + static_cast<size_t>(n) * K;
            const float* xrow = x + static_cast<size_t>(m) * K;
            for (int k = 0; k < K; ++k) sum += xrow[k] * fp16_to_fp32(row[k]);
            y[static_cast<size_t>(m) * N + n] = sum;
        }
    }
}

void split_qkv(const float* qkv, float* q, float* k, float* v, i64 n, int qd, int kvd) {
    if (n <= 0) return;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < n; ++t) {
        const float* row = qkv + t * (qd + 2 * kvd);
        std::memcpy(q + t * qd, row, sizeof(float) * static_cast<size_t>(qd));
        std::memcpy(k + t * kvd, row + qd, sizeof(float) * static_cast<size_t>(kvd));
        std::memcpy(v + t * kvd, row + qd + kvd, sizeof(float) * static_cast<size_t>(kvd));
    }
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
    // P1-7: the old code was serial over tokens (one thread for millions of
    // adds at V=32000). Tokens sharing an id race on the same row, so a naive
    // parallel-for is wrong. Instead: group positions by id (sort), then run
    // groups in parallel — groups touch DISJOINT rows (race-free), and each
    // row still accumulates positions in ascending order, i.e. BIT-IDENTICAL
    // to the serial loop below (kept for small batches, zero overhead).
    if (ntok <= 0) return;
    if (ntok < 256) {
        for (i64 t = 0; t < ntok; ++t) {
            i32 id = ids[t];
            if (id < 0 || id >= vocab) continue;
            float* d = dtable + static_cast<size_t>(id) * dim;
            const float* g = dout + t * dim;
            for (int i = 0; i < dim; ++i) d[i] += g[i];
        }
        return;
    }
    // NOTE: these MUST be frame-local, not thread_local: the parallel loop
    // below runs on worker threads, and a thread_local here would resolve to
    // each worker's own EMPTY copy (out-of-bounds reads). Workers share the
    // calling thread's vectors read-only, which is safe. Two small mallocs
    // per call (~40KB) are negligible next to the megabytes accumulated.
    std::vector<std::pair<i32, i64>> order;
    std::vector<size_t> bounds;
    order.reserve(static_cast<size_t>(ntok));
    for (i64 t = 0; t < ntok; ++t) {
        const i32 id = ids[t];
        if (id < 0 || id >= vocab) continue;
        order.emplace_back(id, t);
    }
    if (order.empty()) return;
    // FIX P2 (bit-identical claim): std::sort is unstable so positions within
    // the same id reorder nondeterministically and intra-row add order differs
    // vs the serial path (~ulp drift). stable_sort keeps (id,t) order exact.
    std::stable_sort(order.begin(), order.end(),
              [](const std::pair<i32, i64>& a, const std::pair<i32, i64>& b) {
                  return a.first < b.first;
              });
    bounds.clear();
    bounds.reserve(order.size() + 1);
    bounds.push_back(0);
    for (size_t i = 1; i < order.size(); ++i) {
        if (order[i].first != order[i - 1].first) bounds.push_back(i);
    }
    bounds.push_back(order.size());
    const size_t ngroups = bounds.size() - 1;
    // PRO-HARDEN (MSVC C3016): متغير حلقة OpenMP يجب أن يكون signed على MSVC
    // (size_t unsigned يفشل البناء على Windows بينما GCC يقبله). long long
    // signed يعمل على كل المنصات الثلاث.
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) if(ngroups > 4)
#endif
    for (long long g = 0; g < static_cast<long long>(ngroups); ++g) {
        const size_t gs = static_cast<size_t>(g);
        const i32 id = order[bounds[gs]].first;
        float* d = dtable + static_cast<size_t>(id) * dim;
        for (size_t k = bounds[gs]; k < bounds[gs + 1]; ++k) {
            const float* gg = dout + order[k].second * dim;
            for (int i = 0; i < dim; ++i) d[i] += gg[i];
        }
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
        // DeepSeek numerical rule: variance accumulate in f64. In f32,
        // exploded activations overflow to inf -> inv=0 -> silent zero-out.
        double ss = 0.0;
        for (int i = 0; i < dim; ++i) ss += static_cast<double>(xr[i]) * xr[i];
        double inv_d = 1.0 / std::sqrt(ss / static_cast<double>(dim) + static_cast<double>(eps));
        // Clamp non-finite (eps<=0 or inf input can still produce inf/NaN).
        float inv = std::isfinite(inv_d) ? static_cast<float>(inv_d) : 0.0f;
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

// FIX (P1-2): `freq` depends only on `i`, never on the token, but the old
// code called std::pow() inside the token loop -> ntok*half pow() calls per
// layer (~2.5M transcendentals per forward at T=1024, hd=64, L=26). Hoist the
// frequency table out of the loop: `half` pow() calls per call instead.
// PRO-HARDEN: cache خيطي keyed بـ(head_dim,theta) يلغي ~6656 malloc/step
// (L26*accum128) التي كانت تهش heap الـCPU وتبطئ T4 host-side.
static const float* rope_freqs(int half, int head_dim, float theta,
                               std::vector<float>& buf) {
    thread_local int cached_half = -1;
    thread_local int cached_hd = -1;
    thread_local float cached_theta = 0.0f;
    thread_local std::vector<float> cached;
    if (cached_half == half && cached_hd == head_dim && cached_theta == theta &&
        !cached.empty()) {
        return cached.data();
    }
    buf.resize(static_cast<size_t>(half));
    for (int i = 0; i < half; ++i)
        buf[static_cast<size_t>(i)] =
            1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
    cached = buf;
    cached_half = half;
    cached_hd = head_dim;
    cached_theta = theta;
    return buf.data();
}

// Full YaRN ramp per-dim frequency (YaRN paper §3): wavelength = 2π/freq.
// < low -> linear (no scaling), > high -> full 1/scale, else interpolated.
static inline float yarn_freq_single(int i, int head_dim, float theta,
                                     float yarn_scale, float yarn_low, float yarn_high) {
    float base = 1.0f / std::pow(theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
    if (yarn_scale <= 1.0f) return base;
    const float wl = 2.0f * 3.14159265358979f / base;
    if (wl < yarn_low) return base;
    if (wl > yarn_high) return base / yarn_scale;
    float span = yarn_high - yarn_low;
    float t = (span > 0.0f) ? (wl - yarn_low) / span : 1.0f;
    return base * (1.0f - t + t / yarn_scale);
}

static void rope_apply(float* q, float* k, const i32* pos,
                       i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                       int rope_type, float yarn_low, float yarn_high, float yarn_scale,
                       float sign) {
    if (!q && !k) return;
    int half = head_dim / 2;
    thread_local std::vector<float> fbuf;
    fbuf.resize(static_cast<size_t>(half));
    if (yarn_scale > 1.0f) {
        for (int i = 0; i < half; ++i)
            fbuf[static_cast<size_t>(i)] = yarn_freq_single(i, head_dim, theta, yarn_scale, yarn_low, yarn_high);
    } else {
        std::vector<float> tmp;
        const float* f = rope_freqs(half, head_dim, theta, tmp);
        for (int i = 0; i < half; ++i) fbuf[static_cast<size_t>(i)] = f[i];
    }
    const float* freqs = fbuf.data();
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {
        float p = static_cast<float>(pos[t]);
        for (int i = 0; i < half; ++i) {
            float ang  = p * freqs[i];
            float c = std::cos(ang), s = std::sin(ang) * sign;
            if (rope_type == 1) {
                // NeoX half-rotate: pair (i, i+half).
                if (q) {
                    for (int h = 0; h < n_heads; ++h) {
                        float* qh = q + (t * n_heads + h) * head_dim;
                        rope_pair(qh[i], qh[i + half], c, s);
                    }
                }
                if (k) {
                    for (int h = 0; h < n_kv; ++h) {
                        float* kh = k + (t * n_kv + h) * head_dim;
                        rope_pair(kh[i], kh[i + half], c, s);
                    }
                }
            } else {
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
}

void rope_forward(float* q, float* k, const i32* pos,
                  i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {
    rope_apply(q, k, pos, ntok, n_heads, n_kv, head_dim, theta, 0, 1.0f, 32.0f, 1.0f, 1.0f);
}

void rope_backward(float* dq, float* dk, const i32* pos,
                   i64 ntok, int n_heads, int n_kv, int head_dim, float theta) {
    // rotation is orthogonal -> backward is rotation by -angle
    rope_apply(dq, dk, pos, ntok, n_heads, n_kv, head_dim, theta, 0, 1.0f, 32.0f, 1.0f, -1.0f);
}

void rope_forward_ex(float* q, float* k, const i32* pos,
                     i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                     int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
    rope_apply(q, k, pos, ntok, n_heads, n_kv, head_dim, theta,
               rope_type, yarn_low, yarn_high, yarn_scale, 1.0f);
}

void rope_backward_ex(float* dq, float* dk, const i32* pos,
                      i64 ntok, int n_heads, int n_kv, int head_dim, float theta,
                      int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
    rope_apply(dq, dk, pos, ntok, n_heads, n_kv, head_dim, theta,
               rope_type, yarn_low, yarn_high, yarn_scale, -1.0f);
}

static void rope_apply_cached(float* q, float* k, const i32* pos, const float* inv_freq,
                              i64 ntok, int n_heads, int n_kv, int head_dim,
                              int rope_type, float sign) {
    const int half = head_dim / 2;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (i64 t = 0; t < ntok; ++t) {
        for (int h = 0; h < n_heads; ++h) {
            float* row = q + (t * n_heads + h) * head_dim;
            for (int i = 0; i < half; ++i) {
                float c = std::cos(static_cast<float>(pos[t]) * inv_freq[i]);
                float s = std::sin(static_cast<float>(pos[t]) * inv_freq[i]) * sign;
                if (rope_type == 1) {
                    rope_pair(row[i], row[i + half], c, s);
                } else {
                    rope_pair(row[2 * i], row[2 * i + 1], c, s);
                }
            }
        }
        for (int h = 0; h < n_kv; ++h) {
            float* row = k + (t * n_kv + h) * head_dim;
            for (int i = 0; i < half; ++i) {
                float c = std::cos(static_cast<float>(pos[t]) * inv_freq[i]);
                float s = std::sin(static_cast<float>(pos[t]) * inv_freq[i]) * sign;
                if (rope_type == 1) {
                    rope_pair(row[i], row[i + half], c, s);
                } else {
                    rope_pair(row[2 * i], row[2 * i + 1], c, s);
                }
            }
        }
    }
}

void rope_forward_cached(float* q, float* k, const i32* pos, const float* inv_freq,
                         i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type) {
    rope_apply_cached(q, k, pos, inv_freq, ntok, n_heads, n_kv, head_dim, rope_type, 1.0f);
}

void rope_backward_cached(float* dq, float* dk, const i32* pos, const float* inv_freq,
                          i64 ntok, int n_heads, int n_kv, int head_dim, int rope_type) {
    rope_apply_cached(dq, dk, pos, inv_freq, ntok, n_heads, n_kv, head_dim, rope_type, -1.0f);
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
    // FIX: guard n<=0 (MoE ne==0 / vocab==0 misconfig) — old code read x[0]
    // unconditionally -> OOB read, NaN logits, training crash.
    if (n <= 0 || !x) return;
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
    attention_decode_ex(q, kcache, vcache, out, H, KV, hd, cur_len, max_len, scale, scratch, 0);
}

void attention_forward_ex(const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids) {
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
                int j0 = (window > 0) ? (t - window + 1 > 0 ? t - window + 1 : 0) : 0;
                int len = t + 1 - j0;
                const i32 seg_t = segment_ids ? segment_ids[static_cast<size_t>(b) * T + t] : 0;
                float mx = -3.4e38f;
                for (int j = j0; j <= t; ++j) {
                    i32 seg_j = segment_ids ? segment_ids[static_cast<size_t>(b) * T + j] : 0;
                    if (segment_ids && seg_t >= 0 && seg_j != seg_t) {
                        s[static_cast<size_t>(j - j0)] = -3.4e38f;
                        continue;
                    }
                    const float* kh = k + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                    float d = dot_f32(qh, kh, hd) * scale;
                    s[static_cast<size_t>(j - j0)] = d;
                    mx = std::max(mx, d);
                }
                float sum = 0.f;
                for (int j = 0; j < len; ++j) {
                    float value = s[static_cast<size_t>(j)];
                    value = value < -3.0e38f ? 0.0f : std::exp(value - mx);
                    s[static_cast<size_t>(j)] = value;
                    sum += s[static_cast<size_t>(j)];
                }
                float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                float* o = out + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
                for (int j = 0; j < len; ++j) {
                    float p = s[static_cast<size_t>(j)] * inv;
                    s[static_cast<size_t>(j)] = p;
                    const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j + j0) * KV + kvh) * hd;
                    axpy_f32(o, vh, p, hd);
                }
                if (probs) {
                    float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    std::memset(pr, 0, sizeof(float) * static_cast<size_t>(T));
                    std::memcpy(pr + j0, s, sizeof(float) * static_cast<size_t>(len));
                }
            }
        }
    }
    }
}

void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window) {
    const int group = H / KV;
    const size_t qs  = static_cast<size_t>(T) * H  * hd;
    const size_t kvs = static_cast<size_t>(T) * KV * hd;

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
                    int j0 = (window > 0) ? (t - window + 1 > 0 ? t - window + 1 : 0) : 0;
                    const float* pr = probs + ((static_cast<size_t>(b) * H + h) * T + t) * T;
                    const float* go = dout + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    const float* qh = q + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;
                    float* dqh = dq + static_cast<size_t>(b) * qs + (static_cast<size_t>(t) * H + h) * hd;

                    float dot_pg = 0.f;
                    for (int j = j0; j <= t; ++j) {
                        const float* vh = v + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        float dpj = dot_f32(go, vh, hd);
                        dsv[static_cast<size_t>(j - j0)] = dpj;
                        dot_pg += pr[j] * dpj;
                        float* dvh = dv + static_cast<size_t>(b) * kvs + (static_cast<size_t>(j) * KV + kvh) * hd;
                        axpy_f32(dvh, go, pr[j], hd);
                    }
                    for (int j = j0; j <= t; ++j) {
                        float ds = pr[j] * (dsv[static_cast<size_t>(j - j0)] - dot_pg) * scale;
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

void attention_decode_ex(const float* q, const float* kcache, const float* vcache,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window) {
    const int group = H / KV;
    (void)max_len;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; ++h) {
        int kvh = h / group;
        float* s = scratch + static_cast<size_t>(h) * static_cast<size_t>(cur_len);
        const float* qh = q + static_cast<size_t>(h) * hd;
        int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
        float mx = -3.4e38f;
        for (int j = j0; j < cur_len; ++j) {
            const float* kh = kcache + (static_cast<size_t>(j) * KV + kvh) * hd;
            float d = dot_f32(qh, kh, hd) * scale;
            s[j] = d;
            mx = std::max(mx, d);
        }
        float sum = 0.f;
        for (int j = j0; j < cur_len; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float inv = 1.0f / sum;
        float* o = out + static_cast<size_t>(h) * hd;
        std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
        for (int j = j0; j < cur_len; ++j) {
            const float* vh = vcache + (static_cast<size_t>(j) * KV + kvh) * hd;
            axpy_f32(o, vh, s[j] * inv, hd);
        }
    }
}

static inline size_t decode_ring_slot(int j, int pinned_prefix, int ring_start, int ring_capacity) {
    if (j < pinned_prefix) return static_cast<size_t>(j);
    return static_cast<size_t>(pinned_prefix) +
           static_cast<size_t>((ring_start + (j - pinned_prefix)) % ring_capacity);
}

void attention_decode_ring(const float* q, const float* kcache, const float* vcache,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window) {
    GAI_CHECK(H > 0 && KV > 0 && hd > 0, "attention_decode_ring: empty shape");
    GAI_CHECK(H % KV == 0, "attention_decode_ring: H must be a multiple of KV");
    GAI_CHECK(cur_len >= 0 && cache_max >= 0 && cur_len <= cache_max,
              "attention_decode_ring: length exceeds cache capacity");
    GAI_CHECK(pinned_prefix >= 0 && pinned_prefix <= cur_len,
              "attention_decode_ring: pinned prefix out of range");
    const int ring_capacity = cache_max - pinned_prefix;
    if (cur_len > pinned_prefix) {
        GAI_CHECK(ring_capacity > 0, "attention_decode_ring: ring has no rolling slots");
        GAI_CHECK(ring_start >= 0 && ring_start < ring_capacity,
                  "attention_decode_ring: ring start out of range");
    }
    const int group = H / KV;
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; ++h) {
        int kvh = h / group;
        float* s = scratch + static_cast<size_t>(h) * static_cast<size_t>(cur_len);
        const float* qh = q + static_cast<size_t>(h) * hd;
        int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
        float mx = -3.4e38f;
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            const float* kh = kcache + (slot * static_cast<size_t>(KV) + kvh) * hd;
            float d = dot_f32(qh, kh, hd) * scale;
            s[j] = d;
            mx = std::max(mx, d);
        }
        float sum = 0.f;
        for (int j = j0; j < cur_len; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float inv = 1.0f / sum;
        float* o = out + static_cast<size_t>(h) * hd;
        std::memset(o, 0, sizeof(float) * static_cast<size_t>(hd));
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            const float* vh = vcache + (slot * static_cast<size_t>(KV) + kvh) * hd;
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

    // AUDIT P1 (contract change, mirrors CUDA): dlogits are SUM grads
    // (caller scales by its loss scale only). Count is still reported.
    for (i64 i = 0; i < n; ++i) if (targets[i] >= 0 && targets[i] < V) ++count;

    // P3-3: OpenMP reduction order is unspecified, so the loss scalar is not
    // bit-reproducible run to run. Default keeps the parallel sum (fast);
    // GAI_DETERMINISTIC=1 forces the serial order for debugging.
    [[maybe_unused]] const bool deterministic = [] {
        const char* e = std::getenv("GAI_DETERMINISTIC");
        return e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
    }();
#ifdef GAI_OPENMP
    #pragma omp parallel for schedule(static) reduction(+:total) if(!deterministic)
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
            float scale = 1.0f + ((z_scale != 0.0f) ? (2.0f * z_scale * (float)logZ) : 0.0f);
            for (int j = 0; j < V; ++j) {
                float p = std::exp(row[j] - mx) * invsum;
                d[j] = p * scale;
            }
            d[tgt] -= 1.0f;
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

// ---------------------------------------------------------------- fast sampling
void topk_select(const float* logits, int V, int K, float* out_vals, i32* out_ids) {
    GAI_CHECK(V > 0 && K >= 1 && K <= V, "topk_select: K out of range");
    std::vector<int> idx(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) idx[static_cast<size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) {
                          if (logits[a] != logits[b]) return logits[a] > logits[b];
                          return a < b;
                      });
    for (int k = 0; k < K; ++k) {
        out_vals[k] = logits[idx[static_cast<size_t>(k)]];
        out_ids[k] = static_cast<i32>(idx[static_cast<size_t>(k)]);
    }
}

i32 argmax_token(const float* logits, int V) {
    GAI_CHECK(V > 0, "argmax_token: empty vocabulary");
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < V; ++i)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return static_cast<i32>(best);
}

void apply_rep_penalties(float* logits, int V, const i32* hist, int hist_n,
                         float rep, float freq, float pres) {
    if (hist_n <= 0 || V <= 0) return;
    const bool use_rep = rep != 1.0f;
    const bool use_freq = freq != 0.0f;
    const bool use_pres = pres != 0.0f;
    if (!use_rep && !use_freq && !use_pres) return;
    thread_local std::vector<int> counts;
    if (counts.size() < static_cast<size_t>(V)) counts.resize(static_cast<size_t>(V));
    std::fill(counts.begin(), counts.end(), 0);
    for (int i = 0; i < hist_n; ++i) {
        const i32 tok = hist[i];
        if (tok >= 0 && tok < V) ++counts[static_cast<size_t>(tok)];
    }
    for (int tok = 0; tok < V; ++tok) {
        const int n = counts[static_cast<size_t>(tok)];
        if (n == 0) continue;
        float& l = logits[tok];
        if (use_rep) l = (l > 0.0f) ? l / rep : l * rep;
        if (use_freq) l -= freq * static_cast<float>(n);
        if (use_pres) l -= pres;
    }
}

} // namespace cpu
} // namespace gai
