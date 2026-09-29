#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <vector>

namespace gai {
namespace cuda_ops {

#define CU_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

static inline int grid_for(i64 n, int block) {
    i64 g = (n + block - 1) / block;
    if (g < 1) g = 1;
    if (g > 65535 * 64) g = 65535 * 64;
    return static_cast<int>(g);
}

static constexpr size_t kWsAlign = 16;
static inline uint8_t* carve16(uint8_t*& ws, size_t bytes) {
    uint8_t* p = ws;
    ws += (bytes + kWsAlign - 1) / kWsAlign * kWsAlign;
    return p;
}

static inline size_t ws_pad_for(int blocks) {
    return static_cast<size_t>(blocks) * kWsAlign;
}

__device__ __forceinline__ bool f4_ok(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & (uintptr_t)(kWsAlign - 1)) == 0;
}

static void*  g_moe_ws = nullptr;
static size_t g_moe_ws_bytes = 0;

static size_t moe_round(size_t bytes) {
    const size_t chunk = (64ull << 20);
    return ((bytes + chunk - 1) / chunk) * chunk;
}

static void* moe_workspace(size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes <= g_moe_ws_bytes) return g_moe_ws;
    size_t want = moe_round(bytes);
    if (want <= g_moe_ws_bytes) return g_moe_ws;
    if (g_moe_ws) CU_CHECK(cudaFree(g_moe_ws));
    gai::cuda::check_free_vram(want, "MoE workspace");
    CU_CHECK(cudaMalloc(&g_moe_ws, want));
    g_moe_ws_bytes = want;
    return g_moe_ws;
}

void moe_reserve_workspace(size_t bytes) {
    if (bytes == 0) return;
    moe_workspace(bytes);
}

static i32*   g_grp_grouped = nullptr;
static size_t g_grp_grouped_cap = 0;
static int*   g_grp_cnt = nullptr;
static int*   g_grp_cur = nullptr;
static int*   g_grp_off = nullptr;

static int    g_grp_ne_cap = 0;

static float g_moe_jitter = 0.0f;
static unsigned long long g_moe_jitter_seed = 0;
void set_moe_jitter(float j) { g_moe_jitter = (j<0?0:(j>0.5f?0.5f:j)); }
void set_moe_jitter_seed(u64 s) { g_moe_jitter_seed = s; }

static void grp_ensure(size_t nk, int ne) {
    if (nk > g_grp_grouped_cap) {

        size_t want = nk + nk / 8 + 1024;
        if (g_grp_grouped) CU_CHECK(cudaFree(g_grp_grouped));
        gai::cuda::check_free_vram(sizeof(i32) * want, "MoE grouped slots");
        CU_CHECK(cudaMalloc(&g_grp_grouped, sizeof(i32) * (want > 0 ? want : 1)));
        g_grp_grouped_cap = want;
    }
    if (ne > g_grp_ne_cap) {
        if (g_grp_cnt) CU_CHECK(cudaFree(g_grp_cnt));
        if (g_grp_cur) CU_CHECK(cudaFree(g_grp_cur));
        if (g_grp_off) CU_CHECK(cudaFree(g_grp_off));
        gai::cuda::check_free_vram(sizeof(int) * (size_t)(ne + 1), "MoE group counters");
        CU_CHECK(cudaMalloc(&g_grp_cnt, sizeof(int) * (size_t)ne));
        CU_CHECK(cudaMalloc(&g_grp_cur, sizeof(int) * (size_t)(ne + 1)));
        CU_CHECK(cudaMalloc(&g_grp_off, sizeof(int) * (size_t)(ne + 1)));
        g_grp_ne_cap = ne;
    }
}

size_t moe_pool_bytes() { return g_moe_ws_bytes; }

__device__ __forceinline__ float jitter_u(i64 t, int e, unsigned long long seed) {

    uint64_t h = (uint64_t)t * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)e * 0xC2B2AE3D27D4EB4FULL;
    h ^= (seed + 0x9E3779B97F4A7C15ULL);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;

    return ((h >> 40) & 0xFFFFFF) / (float)(1 << 24) - 0.5f;
}

__global__ void k_route(const float* logits, float* probs, i32* idx, float* w,
                        i64 N, int ne, int K, float jitter, unsigned long long seed) {
    const i64 t = (i64)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int W = (int)blockDim.x;
    if (t >= N) return;

    const float* lg = logits + t * ne;
    const bool train = (probs != nullptr);
    __shared__ float sred[256];

    float lmx = -FLT_MAX;
    for (int e = tid; e < ne; e += W) {
        float v = lg[e];
        if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
        if (v > lmx) lmx = v;
    }
    sred[tid] = lmx;
    __syncthreads();
    float mx = sred[0];
    for (int i = 1; i < W; ++i) mx = fmaxf(mx, sred[i]);

    float lsum = 0.0f;
    for (int e = tid; e < ne; e += W) {
        float v = lg[e];
        if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
        lsum += expf(v - mx);
    }
    sred[tid] = lsum;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < W; ++i) sum += sred[i];
    const float inv = 1.0f / sum;

    if (probs) {
        float* pr = probs + t * ne;
        for (int e = tid; e < ne; e += W) {
            float v = lg[e];
            if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
            pr[e] = expf(v - mx) * inv;
        }
    }
    if (tid == 0) {
        i32 picked[8];
        float picked_w[8];
        float sum_w = 0.0f;
        for (int k = 0; k < K; ++k) {
            int best = -1;
            float bv = -FLT_MAX;
            for (int e = 0; e < ne; ++e) {
                bool taken = false;
                for (int j = 0; j < k; ++j) if (picked[j] == e) { taken = true; break; }
                if (taken) continue;
                float v = lg[e];
                if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
                float p = expf(v - mx) * inv;
                if (p > bv) { bv = p; best = e; }
            }
            if (best < 0) best = 0;
            picked[k] = best;
            if (idx) idx[t * K + k] = best;
            float vb = lg[best];
            if (train && jitter > 0.0f) vb *= (1.0f + jitter * 2.0f * jitter_u(t, best, seed));
            float pw = expf(vb - mx) * inv;
            picked_w[k] = pw;
            sum_w += pw;
        }
        if (w) {
            float inv_w = sum_w > 1e-8f ? (1.0f / sum_w) : 0.0f;
            for (int k = 0; k < K; ++k) {
                w[t * K + k] = picked_w[k] * inv_w;
            }
        }
    }
}

__global__ void k_route_bias(const float* logits, const float* bias,
                             float* probs, i32* idx, float* w,
                             i64 N, int ne, int K, float jitter, unsigned long long seed) {
    const i64 t = (i64)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int W = (int)blockDim.x;
    if (t >= N) return;
    const float* lg = logits + t * ne;
    const bool train = (probs != nullptr);
    __shared__ float sred[256];
    float lmx = -FLT_MAX;
    for (int e = tid; e < ne; e += W) {
        float v = lg[e];
        if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
        if (v > lmx) lmx = v;
    }
    sred[tid] = lmx;
    __syncthreads();
    float mx = sred[0];
    for (int i = 1; i < W; ++i) mx = fmaxf(mx, sred[i]);
    float lsum = 0.0f;
    for (int e = tid; e < ne; e += W) {
        float v = lg[e];
        if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
        lsum += expf(v - mx);
    }
    sred[tid] = lsum;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < W; ++i) sum += sred[i];
    const float inv = 1.0f / sum;
    if (probs) {
        float* pr = probs + t * ne;
        for (int e = tid; e < ne; e += W) {
            float v = lg[e];
            if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
            pr[e] = expf(v - mx) * inv;
        }
    }
    if (tid == 0) {
        i32 picked[8];
        float picked_w[8];
        float sum_w = 0.0f;
        for (int k = 0; k < K; ++k) {
            int best = -1;
            float bv = -FLT_MAX;
            for (int e = 0; e < ne; ++e) {
                bool taken = false;
                for (int j = 0; j < k; ++j) if (picked[j] == e) { taken = true; break; }
                if (taken) continue;
                float v = lg[e];
                if (train && jitter > 0.0f) v *= (1.0f + jitter * 2.0f * jitter_u(t, e, seed));
                float p = expf(v - mx) * inv + (bias ? bias[e] : 0.0f);
                if (p > bv) { bv = p; best = e; }
            }
            if (best < 0) best = 0;
            picked[k] = best;
            if (idx) idx[t * K + k] = best;
            float vb = lg[best];
            if (train && jitter > 0.0f) vb *= (1.0f + jitter * 2.0f * jitter_u(t, best, seed));
            float pw = expf(vb - mx) * inv;
            picked_w[k] = pw;
            sum_w += pw;
        }
        if (w) {
            float inv_w = sum_w > 1e-8f ? (1.0f / sum_w) : 0.0f;
            for (int k = 0; k < K; ++k) w[t * K + k] = picked_w[k] * inv_w;
        }
    }
}

__global__ void k_gather_tok(const float* src, const i32* slots, float* dst,
                             i64 nslots, int rowlen, int K) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < nslots; s += stride) {
        i32 t = slots[s] / K;
        const float* r = src + (i64)t * rowlen;
        float* o = dst + s * rowlen;
        int j = 0;
        if ((rowlen & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            float4* o4 = reinterpret_cast<float4*>(o);
            for (; j < rowlen / 4; ++j) o4[j] = r4[j];
        } else {
            for (; j < rowlen; ++j) o[j] = r[j];
        }
    }
}

__global__ void k_gather(const float* src, const i32* slots, float* dst,
                         i64 nslots, int rowlen) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < nslots; s += stride) {
        const float* r = src + (i64)slots[s] * rowlen;
        float* o = dst + s * rowlen;
        if ((rowlen & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            float4* o4 = reinterpret_cast<float4*>(o);
            for (int j = 0; j < rowlen / 4; ++j) o4[j] = r4[j];
        } else {
            for (int j = 0; j < rowlen; ++j) o[j] = r[j];
        }
    }
}

__global__ void k_scatter_copy(const float* src, const i32* slots, float* dst,
                               i64 nslots, int rowlen) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < nslots; s += stride) {
        const float* r = src + s * rowlen;
        float* o = dst + (i64)slots[s] * rowlen;
        if ((rowlen & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            float4* o4 = reinterpret_cast<float4*>(o);
            for (int j = 0; j < rowlen / 4; ++j) o4[j] = r4[j];
        } else {
            for (int j = 0; j < rowlen; ++j) o[j] = r[j];
        }
    }
}

__global__ void k_scatter_add(float* dst, const float* src, const i32* slots,
                              const float* w, i64 nslots, int d, int K) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < nslots; s += stride) {
        i32 slot = slots[s];
        i32 t = slot / K;
        i32 k = slot % K;
        float wv = w ? w[(i64)t * K + k] : 1.0f;
        float* o = dst + (i64)t * d;
        const float* r = src + s * d;
        if ((d & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            for (int j = 0; j < d / 4; ++j) {
                float4 v = r4[j];
                atomicAdd(&o[j * 4 + 0], wv * v.x);
                atomicAdd(&o[j * 4 + 1], wv * v.y);
                atomicAdd(&o[j * 4 + 2], wv * v.z);
                atomicAdd(&o[j * 4 + 3], wv * v.w);
            }
        } else {
            for (int j = 0; j < d; ++j) atomicAdd(&o[j], wv * r[j]);
        }
    }
}

__global__ void k_pack_all(const float* x, const i32* grouped, float* out,
                           i64 NK, int d, int K) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < NK; s += stride) {
        i32 t = grouped[s] / K;
        const float* r = x + (i64)t * d;
        float* o = out + s * d;
        int j = 0;
        if ((d & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            float4* o4 = reinterpret_cast<float4*>(o);
            for (; j < d / 4; ++j) o4[j] = r4[j];
        } else {
            for (; j < d; ++j) o[j] = r[j];
        }
    }
}

__global__ void k_save3_all(const float* G, const float* U, const float* A,
                            const i32* grouped,
                            float* s_gate, float* s_up, float* s_act,
                            i64 NK, int E) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < NK; s += stride) {
        i64 slot = grouped[s];
        const float* g = G + s * E;
        const float* u = U + s * E;
        const float* a = A + s * E;
        float* og = s_gate + slot * E;
        float* ou = s_up + slot * E;
        float* oa = s_act + slot * E;
        int j = 0;
        if ((E & 3) == 0 && f4_ok(g) && f4_ok(u) && f4_ok(a) && f4_ok(og) && f4_ok(ou) && f4_ok(oa)) {
            const float4* g4 = reinterpret_cast<const float4*>(g);
            const float4* u4 = reinterpret_cast<const float4*>(u);
            const float4* a4 = reinterpret_cast<const float4*>(a);
            float4* og4 = reinterpret_cast<float4*>(og);
            float4* ou4 = reinterpret_cast<float4*>(ou);
            float4* oa4 = reinterpret_cast<float4*>(oa);
            for (; j < E / 4; ++j) { og4[j] = g4[j]; ou4[j] = u4[j]; oa4[j] = a4[j]; }
        } else {
            for (; j < E; ++j) { og[j] = g[j]; ou[j] = u[j]; oa[j] = a[j]; }
        }
    }
}

__global__ void k_scatter_add_all(float* out, const float* Y, const i32* grouped,
                                  const float* w, i64 NK, int d, int K) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < NK; s += stride) {
        i32 slot = grouped[s];
        i32 t = slot / K;
        i32 k = slot % K;
        float wv = w ? w[(i64)t * K + k] : 1.0f;
        float* o = out + (i64)t * d;
        const float* r = Y + s * d;
        if ((d & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            for (int j = 0; j < d / 4; ++j) {
                float4 v = r4[j];
                atomicAdd(&o[j * 4 + 0], wv * v.x);
                atomicAdd(&o[j * 4 + 1], wv * v.y);
                atomicAdd(&o[j * 4 + 2], wv * v.z);
                atomicAdd(&o[j * 4 + 3], wv * v.w);
            }
        } else {
            for (int j = 0; j < d; ++j) atomicAdd(&o[j], wv * r[j]);
        }
    }
}

__global__ void k_gather3_all(const float* s_gate, const float* s_up, const float* s_act,
                              const i32* grouped, float* G, float* U, float* A,
                              i64 NK, int E) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < NK; s += stride) {
        i64 slot = grouped[s];
        const float* g = s_gate + slot * E;
        const float* u = s_up + slot * E;
        const float* a = s_act + slot * E;
        float* og = G + s * E;
        float* ou = U + s * E;
        float* oa = A + s * E;
        int j = 0;
        if ((E & 3) == 0 && f4_ok(g) && f4_ok(u) && f4_ok(a) && f4_ok(og) && f4_ok(ou) && f4_ok(oa)) {
            const float4* g4 = reinterpret_cast<const float4*>(g);
            const float4* u4 = reinterpret_cast<const float4*>(u);
            const float4* a4 = reinterpret_cast<const float4*>(a);
            float4* og4 = reinterpret_cast<float4*>(og);
            float4* ou4 = reinterpret_cast<float4*>(ou);
            float4* oa4 = reinterpret_cast<float4*>(oa);
            for (; j < E / 4; ++j) { og4[j] = g4[j]; ou4[j] = u4[j]; oa4[j] = a4[j]; }
        } else {
            for (; j < E; ++j) { og[j] = g[j]; ou[j] = u[j]; oa[j] = a[j]; }
        }
    }
}

__global__ void k_scale_all(const float* dout, const float* w, const i32* grouped,
                            float* S, i64 NK, int d, int K) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < NK; s += stride) {
        i64 slot = grouped[s];
        i64 t = slot / K;
        i64 k = slot % K;
        float wv = w ? w[t * K + k] : 1.0f;
        const float* r = dout + t * d;
        float* o = S + s * d;
        int j = 0;
        if ((d & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            float4* o4 = reinterpret_cast<float4*>(o);
            for (; j < d / 4; ++j) {
                float4 v = r4[j];
                v.x *= wv; v.y *= wv; v.z *= wv; v.w *= wv;
                o4[j] = v;
            }
        } else {
            for (; j < d; ++j) o[j] = r[j] * wv;
        }
    }
}

__global__ void k_count_slots(const i32* idx, float* acc, i64 NK, int ne) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (; s < NK; s += stride) {
        int e = idx[s];
        if (e >= 0 && e < ne) atomicAdd(&acc[e], 1.0f);
    }
}

void moe_count_slots(const i32* idx, float* acc, i64 NK, int ne) {
    if (NK <= 0 || ne <= 0) return;
    GAI_CHECK(ne >= 1 && ne <= 64, "moe_count_slots: ne out of kernel range [1,64]");
    k_count_slots<<<grid_for(NK, 256), 256>>>(idx, acc, NK, ne);
    CU_CHECK(cudaGetLastError());
}

__global__ void k_scale_rows(const float* src, const float* w, const i32* slots,
                             float* dst, i64 nslots, int d, int K) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < nslots; s += stride) {
        i32 slot = slots[s];
        i32 t = slot / K;
        i32 k = slot % K;
        float wv = w[(i64)t * K + k];
        const float* r = src + (i64)t * d;
        float* o = dst + s * d;
        if ((d & 3) == 0 && f4_ok(r) && f4_ok(o)) {
            const float4* r4 = reinterpret_cast<const float4*>(r);
            float4* o4 = reinterpret_cast<float4*>(o);
            for (int j = 0; j < d / 4; ++j) {
                float4 v = r4[j];
                v.x *= wv; v.y *= wv; v.z *= wv; v.w *= wv;
                o4[j] = v;
            }
        } else {
            for (int j = 0; j < d; ++j) o[j] = r[j] * wv;
        }
    }
}

__device__ __forceinline__ float silu_f(float v) {
    return v / (1.0f + expf(-v));
}

__global__ void k_swiglu(const float* g, const float* u, float* o, i64 n) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (; i < n; i += stride) o[i] = silu_f(g[i]) * u[i];
}

__global__ void k_swiglu_bwd_assign(const float* g, const float* u, const float* dout,
                             float* dg, float* du, i64 n) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        float gv = g[i];
        float s = 1.0f / (1.0f + expf(-gv));
        dg[i] = dout[i] * (s * (1.0f + gv * (1.0f - s))) * u[i];
        du[i] = dout[i] * s * gv;
    }
}

__global__ void k_dp_dot(const float* dout, const float* eout, const i32* slots,
                         float* dp, i64 nslots, int d, int K) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < nslots; s += stride) {
        i32 t = slots[s] / K;
        const float* a = dout + (i64)t * d;
        const float* b = eout + s * d;
        float acc = 0.0f;
        for (int j = 0; j < d; ++j) acc += a[j] * b[j];
        dp[s] = acc;
    }
}

__global__ void k_router_dl(const float* x, const float* router_w,
                            const float* probs, const float* dp_slots,
                            const i32* idx, const float* aux_frac,
                            float aux_coef, float jitter, unsigned long long seed,
                            float* DL, float* dx,
                            i64 N, int d, int ne, int K) {
    const i64 t = (i64)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int W = (int)blockDim.x;
    if (t >= N) return;
    (void)x;
    const float* pt = probs + t * ne;
    __shared__ float sdp[64];
    __shared__ float sdl[64];

    if (tid == 0) {
        for (int e = 0; e < ne; ++e)
            sdp[e] = aux_frac ? aux_coef * aux_frac[e] : 0.0f;

        float sum_p = 0.0f;
        for (int k = 0; k < K; ++k) {
            int e = idx[t * K + k];
            if (e >= 0 && e < ne) sum_p += pt[e];
        }
        float inv_S = sum_p > 1e-8f ? (1.0f / sum_p) : 0.0f;
        float bar_g = 0.0f;
        for (int k = 0; k < K; ++k) {
            int e = idx[t * K + k];
            float wk = (e >= 0 && e < ne) ? pt[e] * inv_S : 0.0f;
            bar_g += dp_slots[t * K + k] * wk;
        }
        for (int k = 0; k < K; ++k) {
            int e = idx[t * K + k];
            if (e >= 0 && e < ne) {
                sdp[e] += (dp_slots[t * K + k] - bar_g) * inv_S;
            }
        }
        float pdot = 0.0f;
        for (int e = 0; e < ne; ++e) pdot += pt[e] * sdp[e];
        for (int e = 0; e < ne; ++e) {
            float dl = pt[e] * (sdp[e] - pdot);

            if (jitter > 0.0f) {
                uint64_t hh = (uint64_t)t * 0x9E3779B97F4A7C15ULL;
                hh ^= (uint64_t)e * 0xC2B2AE3D27D4EB4FULL;
                hh ^= (seed + 0x9E3779B97F4A7C15ULL);
                hh ^= hh >> 33; hh *= 0xff51afd7ed558ccdULL;
                hh ^= hh >> 33; hh *= 0xc4ceb9fe1a85ec53ULL;
                hh ^= hh >> 33;
                float u = ((hh >> 40) & 0xFFFFFF) / (float)(1 << 24) - 0.5f;
                dl *= (1.0f + jitter * 2.0f * u);
            }
            sdl[e] = dl;
        }
    }
    __syncthreads();

    float* DLt = DL + t * ne;
    for (int e = tid; e < ne; e += W) DLt[e] = sdl[e];

    float* dxt = dx + t * d;
    for (int j = tid; j < d; j += W) {
        float acc = 0.0f;
        for (int e = 0; e < ne; ++e) acc += sdl[e] * router_w[(i64)e * d + j];
        dxt[j] += acc;
    }
}

__global__ void k_group_hist(const i32* idx, int* cnt, i64 NK, int ne) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < NK; s += stride) {
        int e = idx[s];
        if (e >= 0 && e < ne) atomicAdd(&cnt[e], 1);
    }
}

__global__ void k_group_offsets(const int* cnt, int* offsets, int* cursors, int ne) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int running = 0;
    for (int e = 0; e < ne; ++e) {
        offsets[e] = running;
        cursors[e] = running;
        running += cnt[e] < 0 ? 0 : cnt[e];
    }
    offsets[ne] = running;
}

__global__ void k_group_fill(const i32* idx, i32* grouped, int* cursors,
                             i64 NK, int ne) {
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x; s < NK; s += stride) {
        int e = idx[s];
        if (e < 0 || e >= ne) continue;
        int pos = atomicAdd(&cursors[e], 1);
        grouped[pos] = (i32)s;
    }
}

static int*  g_h_offsets_pinned = nullptr;
static int   g_h_offsets_cap = 0;
static cudaEvent_t g_offsets_event = nullptr;

static void ensure_offsets_pinned(int ne) {
    if (ne <= g_h_offsets_cap) return;
    if (g_h_offsets_pinned) CU_CHECK(cudaFreeHost(g_h_offsets_pinned));
    if (!g_offsets_event) CU_CHECK(cudaEventCreate(&g_offsets_event));
    CU_CHECK(cudaMallocHost(&g_h_offsets_pinned, sizeof(int) * (size_t)(ne + 1)));
    g_h_offsets_cap = ne;
}

static void moe_build_groups_launch(const i32* d_idx, i64 N, int K, int ne,
                                    i32** grouped_out) {
    const i64 NK = N * K;
    GAI_CHECK(ne > 0 && ne <= 64, "moe grouping: ne out of range");
    grp_ensure((size_t)(NK > 0 ? NK : 1), ne);
    ensure_offsets_pinned(ne);
    *grouped_out = g_grp_grouped;
    if (NK <= 0) return;
    CU_CHECK(cudaMemset(g_grp_cnt, 0, sizeof(int) * (size_t)ne));
    k_group_hist<<<grid_for(NK, 256), 256>>>(d_idx, g_grp_cnt, NK, ne);

    k_group_offsets<<<1, 1>>>(g_grp_cnt, g_grp_off, g_grp_cur, ne);
    CU_CHECK(cudaGetLastError());
    k_group_fill<<<grid_for(NK, 256), 256>>>(d_idx, g_grp_grouped, g_grp_cur, NK, ne);
    CU_CHECK(cudaGetLastError());

    CU_CHECK(cudaMemcpyAsync(g_h_offsets_pinned, g_grp_off,
                             sizeof(int) * (size_t)(ne + 1),
                             cudaMemcpyDeviceToHost, 0));
    CU_CHECK(cudaEventRecord(g_offsets_event, 0));
}

static void moe_build_groups_sync(i64 NK, int ne, int* h_counts, int* h_offsets) {
    for (int e = 0; e < ne; ++e) h_counts[e] = 0;
    for (int e = 0; e <= ne; ++e) h_offsets[e] = 0;
    if (NK <= 0) return;

    CU_CHECK(cudaEventSynchronize(g_offsets_event));
    for (int e = 0; e <= ne; ++e) h_offsets[e] = g_h_offsets_pinned[e];
    for (int e = 0; e < ne; ++e) {
        int c = h_offsets[e + 1] - h_offsets[e];
        h_counts[e] = c < 0 ? 0 : c;
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

    GAI_CHECK(N <= 2147483647LL, "moe forward: N exceeds INT_MAX (B*T too large)");
    GAI_CHECK(K >= 1 && K <= 8, "moe forward: top_k out of kernel range [1,8]");
    GAI_CHECK(ne >= 1 && ne <= 64, "moe forward: num_experts out of kernel range [1,64]");
    const i64 NK = N * K;

    uint8_t* ws = (uint8_t*)moe_workspace(
        sizeof(float) * (size_t)N * ne +
        sizeof(i32) * (size_t)NK +
        sizeof(float) * (size_t)NK +
        sizeof(float) * (size_t)NK * E * 3 +
        sizeof(float) * (size_t)NK * d +
        ws_pad_for(7));
    float* rlogits = (float*)carve16(ws, sizeof(float) * (size_t)N * ne);
    i32*   tmp_idx = (i32*)carve16(ws, sizeof(i32) * (size_t)NK);
    float* tmp_w   = (float*)carve16(ws, sizeof(float) * (size_t)NK);
    float* Gblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Ublk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Ablk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Xblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * d);

    linear_forward(x, router_w, rlogits, (int)N, d, ne);

    i32* idx_use = idx_cache ? idx_cache : tmp_idx;
    float* w_use = w_cache ? w_cache : tmp_w;

    k_route<<<(unsigned)N, 32>>>(rlogits, probs_cache, idx_use, w_use, N, ne, K, g_moe_jitter, g_moe_jitter_seed);
    CU_CHECK(cudaGetLastError());

    i32* grouped = nullptr;
    int h_cnt[64], h_off[65];
    moe_build_groups_launch(idx_use, N, K, ne, &grouped);

    if (sh_g && sh_u && sh_d) {
        linear_forward(x, sh_g, Gblk, (int)N, d, E);
        linear_forward(x, sh_u, Ublk, (int)N, d, E);
        k_swiglu<<<grid_for(N * E, 256), 256>>>(Gblk, Ublk, Ablk, N * E);
        CU_CHECK(cudaGetLastError());
        linear_forward(Ablk, sh_d, out, (int)N, E, d);
    } else {
        CU_CHECK(cudaMemset(out, 0, sizeof(float) * (size_t)N * d));
    }

    moe_build_groups_sync(NK, ne, h_cnt, h_off);
    if (h_off[ne] <= 0) return;

    k_pack_all<<<grid_for(NK, 256), 256>>>(x, grouped, Xblk, NK, d, K);
    CU_CHECK(cudaGetLastError());
    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        const float* ge = gates + (size_t)e * E * d;
        const float* ue = ups   + (size_t)e * E * d;
        linear_forward(Xblk + (size_t)h_off[e] * d, ge, Gblk + (size_t)h_off[e] * E, (int)ns, d, E);
        linear_forward(Xblk + (size_t)h_off[e] * d, ue, Ublk + (size_t)h_off[e] * E, (int)ns, d, E);
    }
    k_swiglu<<<grid_for(NK * E, 256), 256>>>(Gblk, Ublk, Ablk, NK * E);
    CU_CHECK(cudaGetLastError());
    k_save3_all<<<grid_for(NK, 256), 256>>>(Gblk, Ublk, Ablk, grouped, s_gate, s_up, s_act, NK, E);
    CU_CHECK(cudaGetLastError());
    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        const float* de = downs + (size_t)e * d * E;
        linear_forward(Ablk + (size_t)h_off[e] * E, de, Xblk + (size_t)h_off[e] * d, (int)ns, E, d);
    }
    k_scatter_add_all<<<grid_for(NK, 256), 256>>>(out, Xblk, grouped, w_use, NK, d, K);
    CU_CHECK(cudaGetLastError());
}

void moe_forward_bias(const float* x, const float* router_w, const float* router_bias,
                      const float* gates, const float* ups, const float* downs,
                      const float* sh_g, const float* sh_u, const float* sh_d,
                      float* out,
                      float* probs_cache, i32* idx_cache, float* w_cache,
                      float* s_gate, float* s_up, float* s_act,
                      i64 N, int d, int E, int ne, int K) {
    if (N <= 0) return;
    if (!router_bias) {
        moe_forward(x, router_w, gates, ups, downs, sh_g, sh_u, sh_d, out,
                    probs_cache, idx_cache, w_cache, s_gate, s_up, s_act, N, d, E, ne, K);
        return;
    }

    GAI_CHECK(N <= 2147483647LL, "moe forward bias: N exceeds INT_MAX (B*T too large)");
    GAI_CHECK(K >= 1 && K <= 8, "moe routing bias: top_k out of kernel range [1,8]");
    GAI_CHECK(ne >= 1 && ne <= 64, "moe routing bias: ne out of kernel range [1,64]");
    const i64 NK = N * K;

    uint8_t* ws = (uint8_t*)moe_workspace(
        sizeof(float) * (size_t)N * ne +
        sizeof(i32) * (size_t)NK +
        sizeof(float) * (size_t)NK +
        sizeof(float) * (size_t)NK * E * 3 +
        sizeof(float) * (size_t)NK * d +
        ws_pad_for(7));
    float* rlogits = (float*)carve16(ws, sizeof(float) * (size_t)N * ne);
    i32*   tmp_idx = (i32*)carve16(ws, sizeof(i32) * (size_t)NK);
    float* tmp_w   = (float*)carve16(ws, sizeof(float) * (size_t)NK);
    float* Gblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Ublk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Ablk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Xblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * d);

    linear_forward(x, router_w, rlogits, (int)N, d, ne);

    i32* idx_use = idx_cache ? idx_cache : tmp_idx;
    float* w_use = w_cache ? w_cache : tmp_w;

    k_route_bias<<<(unsigned)N, 32>>>(rlogits, router_bias, probs_cache, idx_use, w_use,
                                     N, ne, K, g_moe_jitter, g_moe_jitter_seed);
    CU_CHECK(cudaGetLastError());

    i32* grouped = nullptr;
    int h_cnt[64], h_off[65];
    moe_build_groups_launch(idx_use, N, K, ne, &grouped);

    if (sh_g && sh_u && sh_d) {
        linear_forward(x, sh_g, Gblk, (int)N, d, E);
        linear_forward(x, sh_u, Ublk, (int)N, d, E);
        k_swiglu<<<grid_for(N * E, 256), 256>>>(Gblk, Ublk, Ablk, N * E);
        CU_CHECK(cudaGetLastError());
        linear_forward(Ablk, sh_d, out, (int)N, E, d);
    } else {
        CU_CHECK(cudaMemset(out, 0, sizeof(float) * (size_t)N * d));
    }

    moe_build_groups_sync(NK, ne, h_cnt, h_off);
    if (h_off[ne] <= 0) return;

    k_pack_all<<<grid_for(NK, 256), 256>>>(x, grouped, Xblk, NK, d, K);
    CU_CHECK(cudaGetLastError());
    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        const float* ge = gates + (size_t)e * E * d;
        const float* ue = ups   + (size_t)e * E * d;
        linear_forward(Xblk + (size_t)h_off[e] * d, ge, Gblk + (size_t)h_off[e] * E, (int)ns, d, E);
        linear_forward(Xblk + (size_t)h_off[e] * d, ue, Ublk + (size_t)h_off[e] * E, (int)ns, d, E);
    }
    k_swiglu<<<grid_for(NK * E, 256), 256>>>(Gblk, Ublk, Ablk, NK * E);
    CU_CHECK(cudaGetLastError());
    k_save3_all<<<grid_for(NK, 256), 256>>>(Gblk, Ublk, Ablk, grouped, s_gate, s_up, s_act, NK, E);
    CU_CHECK(cudaGetLastError());
    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        const float* de = downs + (size_t)e * d * E;
        linear_forward(Ablk + (size_t)h_off[e] * E, de, Xblk + (size_t)h_off[e] * d, (int)ns, E, d);
    }
    k_scatter_add_all<<<grid_for(NK, 256), 256>>>(out, Xblk, grouped, w_use, NK, d, K);
    CU_CHECK(cudaGetLastError());
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
    GAI_CHECK(N <= 2147483647LL, "moe backward: N exceeds INT_MAX (B*T too large)");
    GAI_CHECK(K >= 1 && K <= 8, "moe backward: top_k out of kernel range [1,8]");
    GAI_CHECK(ne >= 1 && ne <= 64, "moe backward: num_experts out of kernel range [1,64]");
    const i64 NK = N * K;
    const float aux_coef = (aux_frac && aux_scale != 0.0f)
        ? aux_scale * (float)ne / (float)N : 0.0f;
    (void)s_dact;

    uint8_t* ws = (uint8_t*)moe_workspace(
        sizeof(float) * (size_t)N * E * 3 +
        sizeof(float) * (size_t)N * E * 3 +
        sizeof(float) * (size_t)NK +
        sizeof(float) * (size_t)NK * E * 4 +
        sizeof(float) * (size_t)NK * d * 2 +
        sizeof(float) * (size_t)NK +
        sizeof(float) * (size_t)N * ne +
        ws_pad_for(15));

    float* shg = (float*)carve16(ws, sizeof(float) * (size_t)N * E);
    float* shu = (float*)carve16(ws, sizeof(float) * (size_t)N * E);
    float* sha = (float*)carve16(ws, sizeof(float) * (size_t)N * E);
    float* shdg = (float*)carve16(ws, sizeof(float) * (size_t)N * E);
    float* shdu = (float*)carve16(ws, sizeof(float) * (size_t)N * E);
    float* shda = (float*)carve16(ws, sizeof(float) * (size_t)N * E);
    float* dpfull = (float*)carve16(ws, sizeof(float) * (size_t)NK);
    float* Gblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Ublk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Ablk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Dblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * E);
    float* Xblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * d);
    float* Sblk = (float*)carve16(ws, sizeof(float) * (size_t)NK * d);
    float* Pblk = (float*)carve16(ws, sizeof(float) * (size_t)NK);
    float* DLblk = (float*)carve16(ws, sizeof(float) * (size_t)N * ne);

    if (sh_g && sh_u && sh_d && dsh_g && dsh_u && dsh_d) {
        linear_forward(x, sh_g, shg, (int)N, d, E);
        linear_forward(x, sh_u, shu, (int)N, d, E);
        k_swiglu<<<grid_for(N * E, 256), 256>>>(shg, shu, sha, N * E);
        CU_CHECK(cudaGetLastError());

    gemm(false, false, (int)N, E, d, 1.0f, dout, d, sh_d, E, 0.0f, shda, E);
    k_swiglu_bwd_assign<<<grid_for(N * E, 256), 256>>>(shg, shu, shda, shdg, shdu, N * E);
        CU_CHECK(cudaGetLastError());
        linear_backward(sha, sh_d, dout, nullptr, dsh_d, (int)N, E, d);
        linear_backward(x, sh_g, shdg, dx, dsh_g, (int)N, d, E);
        linear_backward(x, sh_u, shdu, dx, dsh_u, (int)N, d, E);
    }

    CU_CHECK(cudaMemset(dpfull, 0, sizeof(float) * (size_t)NK));

    i32* grouped = nullptr;
    int h_cnt[64], h_off[65];
    moe_build_groups_launch(idx, N, K, ne, &grouped);

    k_gather3_all<<<grid_for(NK, 256), 256>>>(s_gate, s_up, s_act, grouped, Gblk, Ublk, Ablk, NK, E);
    CU_CHECK(cudaGetLastError());
    k_pack_all<<<grid_for(NK, 256), 256>>>(x, grouped, Xblk, NK, d, K);
    CU_CHECK(cudaGetLastError());
    k_scale_all<<<grid_for(NK, 256), 256>>>(dout, tw, grouped, Sblk, NK, d, K);
    CU_CHECK(cudaGetLastError());

    moe_build_groups_sync(NK, ne, h_cnt, h_off);
    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        const float* de = downs + (size_t)e * d * E;

        gemm(false, false, (int)ns, E, d, 1.0f,
             Sblk + (size_t)h_off[e] * d, d, de, E, 0.0f,
             Dblk + (size_t)h_off[e] * E, E);
    }
    k_swiglu_bwd_assign<<<grid_for(NK * E, 256), 256>>>(Gblk, Ublk, Dblk,
                                                        Gblk, Ublk, NK * E);
    CU_CHECK(cudaGetLastError());

    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        i32* d_slots = grouped + h_off[e];
        const float* ge = gates + (size_t)e * E * d;
        const float* ue = ups   + (size_t)e * E * d;
        const float* de = downs + (size_t)e * d * E;
        float* dge = dgates + (size_t)e * E * d;
        float* due = dups   + (size_t)e * E * d;
        float* dde = ddowns + (size_t)e * d * E;
        const float* Gp = Gblk + (size_t)h_off[e] * E;
        const float* Up = Ublk + (size_t)h_off[e] * E;
        const float* Ap = Ablk + (size_t)h_off[e] * E;
        const float* Xp = Xblk + (size_t)h_off[e] * d;
        const float* Sp = Sblk + (size_t)h_off[e] * d;

        linear_backward(Ap, de, Sp, nullptr, dde, (int)ns, E, d);

        zero(Sblk + (size_t)h_off[e] * d, (i64)ns * d);
        linear_backward(Xp, ge, Gp, Sblk + (size_t)h_off[e] * d, dge, (int)ns, d, E);
        k_scatter_add<<<grid_for(ns, 256), 256>>>(dx, Sblk + (size_t)h_off[e] * d, d_slots,
                                                  nullptr, ns, d, K);
        CU_CHECK(cudaGetLastError());

        zero(Sblk + (size_t)h_off[e] * d, (i64)ns * d);
        linear_backward(Xp, ue, Up, Sblk + (size_t)h_off[e] * d, due, (int)ns, d, E);
        k_scatter_add<<<grid_for(ns, 256), 256>>>(dx, Sblk + (size_t)h_off[e] * d, d_slots,
                                                  nullptr, ns, d, K);
        CU_CHECK(cudaGetLastError());

        linear_forward(Ap, de, Sblk + (size_t)h_off[e] * d, (int)ns, E, d);
        k_dp_dot<<<grid_for(ns, 256), 256>>>(dout, Sblk + (size_t)h_off[e] * d, d_slots, Pblk, ns, d, K);
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Pblk, d_slots, dpfull, ns, 1);
        CU_CHECK(cudaGetLastError());
    }

    GAI_CHECK(K >= 1 && K <= 8, "moe routing backward: top_k out of kernel range [1,8]");
    GAI_CHECK(ne >= 1 && ne <= 64, "moe routing backward: num_experts out of kernel range [1,64]");
    k_router_dl<<<(unsigned)N, 32>>>(x, router_w, probs, dpfull, idx, aux_frac,
                                    aux_coef, g_moe_jitter, g_moe_jitter_seed, DLblk, dx, N, d, ne, K);
    CU_CHECK(cudaGetLastError());
    gemm(true, false, ne, d, (int)N, 1.0f, DLblk, ne, x, d, 1.0f, drouter_w, d);
}

__global__ void k_aux_accum(const float* probs, const i32* idx,
                            int* cnt, float* psum,
                            i64 N, int K, int ne) {
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    for (int k = 0; k < K; ++k) {
        int e = idx[t * K + k];
        if (e >= 0 && e < ne) atomicAdd(&cnt[e], 1);
    }
    const float* pt = probs + t * ne;
    for (int e = 0; e < ne; ++e) atomicAdd(&psum[e], pt[e]);
}

__global__ void k_aux_finalize(const int* cnt, float* frac,
                               i64 N, int K, int ne) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    float denom = (float)((double)N * (double)K);
    frac[e] = denom > 0.0f ? (float)cnt[e] / denom : 0.0f;
}

static int*   g_aux_cnt = nullptr;
static float* g_aux_psum = nullptr;
static int    g_aux_cap = 0;

static void aux_ensure(int ne) {
    if (ne <= g_aux_cap) return;
    if (g_aux_cnt) { cudaFree(g_aux_cnt); g_aux_cnt = nullptr; }
    if (g_aux_psum) { cudaFree(g_aux_psum); g_aux_psum = nullptr; }
    CU_CHECK(cudaMalloc(&g_aux_cnt, sizeof(int) * (size_t)ne));
    CU_CHECK(cudaMalloc(&g_aux_psum, sizeof(float) * (size_t)ne));
    g_aux_cap = ne;
}

static double* g_aux_raw = nullptr;

double* moe_aux_begin() {
    if (!g_aux_raw) CU_CHECK(cudaMalloc(&g_aux_raw, sizeof(double)));
    CU_CHECK(cudaMemset(g_aux_raw, 0, sizeof(double)));
    return g_aux_raw;
}

double moe_aux_end() {
    if (!g_aux_raw) return 0.0;
    double h = 0.0;
    CU_CHECK(cudaMemcpy(&h, g_aux_raw, sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

__global__ void k_aux_raw_add(const float* psum, const float* frac,
                              double* accum, i64 N, int ne) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double raw = 0.0;
    for (int e = 0; e < ne; ++e) {
        double m = N > 0 ? (double)psum[e] / (double)N : 0.0;
        raw += m * (double)frac[e];
    }
    raw *= (double)ne;
    *accum += raw;
}

void moe_aux_frac_gpu(const float* probs, const i32* idx, float* frac,
                      float* h_frac, float* h_psum,
                      i64 N, int K, int ne, double* d_raw_accum) {
    if (N <= 0 || ne <= 0) {
        if (frac && ne > 0) CU_CHECK(cudaMemset(frac, 0, sizeof(float) * (size_t)ne));
        if (h_frac) for (int e = 0; e < ne; ++e) h_frac[e] = 0.0f;
        if (h_psum) for (int e = 0; e < ne; ++e) h_psum[e] = 0.0f;
        return;
    }
    aux_ensure(ne);
    CU_CHECK(cudaMemset(g_aux_cnt, 0, sizeof(int) * (size_t)ne));
    CU_CHECK(cudaMemset(g_aux_psum, 0, sizeof(float) * (size_t)ne));
    k_aux_accum<<<grid_for(N, 256), 256>>>(probs, idx, g_aux_cnt, g_aux_psum, N, K, ne);
    CU_CHECK(cudaGetLastError());
    k_aux_finalize<<<(ne + 255) / 256, 256>>>(g_aux_cnt, frac, N, K, ne);
    CU_CHECK(cudaGetLastError());

    if (d_raw_accum && N > 0 && ne > 0)
        k_aux_raw_add<<<1, 1>>>(g_aux_psum, frac, d_raw_accum, N, ne);

    if (h_frac) CU_CHECK(cudaMemcpy(h_frac, frac, sizeof(float) * (size_t)ne,
                                    cudaMemcpyDeviceToHost));
    if (h_psum) CU_CHECK(cudaMemcpy(h_psum, g_aux_psum, sizeof(float) * (size_t)ne,
                                    cudaMemcpyDeviceToHost));
}

void moe_free_workspace() {
    if (g_moe_ws) { cudaFree(g_moe_ws); g_moe_ws = nullptr; g_moe_ws_bytes = 0; }
    if (g_grp_grouped) { cudaFree(g_grp_grouped); g_grp_grouped = nullptr; g_grp_grouped_cap = 0; }
    if (g_grp_cnt) { cudaFree(g_grp_cnt); g_grp_cnt = nullptr; }
    if (g_grp_cur) { cudaFree(g_grp_cur); g_grp_cur = nullptr; }
    if (g_grp_off) { cudaFree(g_grp_off); g_grp_off = nullptr; }
    g_grp_ne_cap = 0;
    if (g_aux_cnt) { cudaFree(g_aux_cnt); g_aux_cnt = nullptr; }
    if (g_aux_psum) { cudaFree(g_aux_psum); g_aux_psum = nullptr; }
    g_aux_cap = 0;
    if (g_aux_raw) { cudaFree(g_aux_raw); g_aux_raw = nullptr; }
    if (g_offsets_event) { cudaEventDestroy(g_offsets_event); g_offsets_event = nullptr; }
    if (g_h_offsets_pinned) { cudaFreeHost(g_h_offsets_pinned); g_h_offsets_pinned = nullptr; }
    g_h_offsets_cap = 0;
}

}
}
