// cuda/moe.cu — GPU implementation of top-k routed SwiGLU MoE.
// Grouped dispatch: routing is one kernel (one block per token); every expert
// GEMM runs through cuBLAS on gathered row blocks; scatter/gather and the
// small elementwise steps are fused kernels. The math mirrors core/ops_moe.cpp
// exactly (fp32 summation order may differ slightly, like all our kernels).
//
// Indexing convention: a "slot" s = t*K+k lives in [N*K] space. Tensors in
// slot space ([N,K,E] caches) are indexed by slot directly; token-space
// tensors (x, dout, dx) are indexed by t = slot/K via k_gather_tok.

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
    return static_cast<int>(g);
}

// NOTE: GEMMs go through cuda_ops::linear_forward / linear_backward / gemm
// (kernels.cu), so the MoE automatically uses the FP16 tensor-core fast path
// for large matrices. No local BLAS wrappers here (keeps one code path).
// ---------------------------------------------------------------- workspace
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
    CU_CHECK(cudaMalloc(&g_moe_ws, want));
    g_moe_ws_bytes = want;
    return g_moe_ws;
}

// Persistent grouped-slot buffers: avoids cudaMalloc/cudaFree on every
// layer in every micro-batch (was 1600+ allocs per optimizer step) and
// avoids the old read_slots() host roundtrip of N*K ints per layer.
// Only ne counters cross the host (tiny sync); slot data stays on GPU.
// All buffers grow monotonically and live for the process lifetime.
static i32*   g_grp_grouped = nullptr;
static size_t g_grp_grouped_cap = 0;
static int*   g_grp_cnt = nullptr;
static int*   g_grp_cur = nullptr;
static int    g_grp_ne_cap = 0;

static float g_moe_jitter = 0.0f;
static unsigned long long g_moe_jitter_seed = 0;
void set_moe_jitter(float j) { g_moe_jitter = (j<0?0:(j>0.5f?0.5f:j)); }
void set_moe_jitter_seed(u64 s) { g_moe_jitter_seed = s; }

static void grp_ensure(size_t nk, int ne) {
    if (nk > g_grp_grouped_cap) {
        // Over-allocate 12% headroom so B/T jitter doesn't realloc every step.
        size_t want = nk + nk / 8 + 1024;
        if (g_grp_grouped) CU_CHECK(cudaFree(g_grp_grouped));
        CU_CHECK(cudaMalloc(&g_grp_grouped, sizeof(i32) * (want > 0 ? want : 1)));
        g_grp_grouped_cap = want;
    }
    if (ne > g_grp_ne_cap) {
        if (g_grp_cnt) CU_CHECK(cudaFree(g_grp_cnt));
        if (g_grp_cur) CU_CHECK(cudaFree(g_grp_cur));
        CU_CHECK(cudaMalloc(&g_grp_cnt, sizeof(int) * (size_t)ne));
        CU_CHECK(cudaMalloc(&g_grp_cur, sizeof(int) * (size_t)ne));
        g_grp_ne_cap = ne;
    }
}

// ================================================================ kernels
// One block per token: stable softmax over ne logits + K argmax passes.
// Caches are optional (null = inference; the local picked list is used).
// DeepSeek-V2 jitter: train-only multiplicative noise on router logits.
// probs!=nullptr means training (caches requested); inference (null) stays
// deterministic. Noise is hash-based (no RNG state, bit-reproducible per
// token) and folds the per-step seed (step/rank/base) for temporal diversity.
__device__ __forceinline__ float jitter_u(i64 t, int e, unsigned long long seed) {
    uint32_t h = (uint32_t)(t * 2654435761ULL) ^ (uint32_t)(e * 40503u + 1u)
               ^ (uint32_t)(seed & 0xFFFFFFFFu) ^ (uint32_t)((seed >> 32) * 2246822519ULL);
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return ((h % 1000000u) / 1000000.0f) - 0.5f; // [-0.5,0.5)
}
// Warp-per-token router (audit P0 #4): one TOKEN per block, blockDim.x
// threads cooperating (launch with 32 = one warp). The old launch used
// (N blocks x 1 thread): 31/32 warp slots idle per token, so the
// expf-heavy softmax could not hide latency across 36 layers x grad_accum
// micros. Math matches the old kernel: parallel max is exact (order-free);
// the softmax denominator reduction may reorder fp adds (~1ulp vs serial);
// top-k stays serial on lane 0 with the same strict-greater tie-break, so
// picks/weights agree up to that 1ulp (parity tolerance is 5%).
__global__ void k_route(const float* logits, float* probs, i32* idx, float* w,
                        i64 N, int ne, int K, float jitter, unsigned long long seed) {
    const i64 t = (i64)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int W = (int)blockDim.x;
    if (t >= N) return;
    const float* lg = logits + t * ne;
    const bool train = (probs != nullptr);
    __shared__ float sred[256];  // partial max/sum (launch W <= 256)
    // -- max (order-independent: exact)
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
    // -- denominator (parallel partials; ~1ulp vs old serial order)
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
    // -- probs row (cooperative) + top-k (lane 0, same tie-break as before)
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

// Aux-loss-free (DeepSeek-V3 §3.2): selection on probs+bias, weights stay
// softmax(logits). Bias steers load with no grad (EMA update on host).
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

// dst[s] = src[t] for slot s = t*K+k (token-space gather)
__global__ void k_gather_tok(const float* src, const i32* slots, float* dst,
                             i64 nslots, int rowlen, int K) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    i32 t = slots[s] / K;
    const float* r = src + (i64)t * rowlen;
    float* o = dst + s * rowlen;
    for (int j = 0; j < rowlen; ++j) o[j] = r[j];
}

// dst[s] = src[slot]  (slot-space gather)
__global__ void k_gather(const float* src, const i32* slots, float* dst,
                         i64 nslots, int rowlen) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    const float* r = src + (i64)slots[s] * rowlen;
    float* o = dst + s * rowlen;
    for (int j = 0; j < rowlen; ++j) o[j] = r[j];
}

// dst[slot] = src[s]  (slot-space scatter)
__global__ void k_scatter_copy(const float* src, const i32* slots, float* dst,
                               i64 nslots, int rowlen) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    const float* r = src + s * rowlen;
    float* o = dst + (i64)slots[s] * rowlen;
    for (int j = 0; j < rowlen; ++j) o[j] = r[j];
}

// dst[t] += w[t,k] * src[s]  (atomic: slots of one token share dst rows)
__global__ void k_scatter_add(float* dst, const float* src, const i32* slots,
                              const float* w, i64 nslots, int d, int K) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    i32 slot = slots[s];
    i32 t = slot / K;
    i32 k = slot % K;
    float wv = w ? w[(i64)t * K + k] : 1.0f;
    float* o = dst + (i64)t * d;
    const float* r = src + s * d;
    for (int j = 0; j < d; ++j) atomicAdd(&o[j], wv * r[j]);
}

// dst[s] = src[t] * w[t,k]
__global__ void k_scale_rows(const float* src, const float* w, const i32* slots,
                             float* dst, i64 nslots, int d, int K) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    i32 slot = slots[s];
    i32 t = slot / K;
    i32 k = slot % K;
    float wv = w[(i64)t * K + k];
    const float* r = src + (i64)t * d;
    float* o = dst + s * d;
    for (int j = 0; j < d; ++j) o[j] = r[j] * wv;
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
        du[i] = dout[i] * s * gv;   // du = dout * silu(g), silu(g) = g*sigmoid(g)
    }
}

// dp[s] = dot(dout[t], expert_out[s])
__global__ void k_dp_dot(const float* dout, const float* eout, const i32* slots,
                         float* dp, i64 nslots, int d, int K) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nslots) return;
    i32 t = slots[s] / K;
    const float* a = dout + (i64)t * d;
    const float* b = eout + s * d;
    float acc = 0.0f;
    for (int j = 0; j < d; ++j) acc += a[j] * b[j];
    dp[s] = acc;
}

// One block per token: softmax backward over the router distribution plus the
// aux load-balance term. Writes per-token DL[t,ne] rows (dL/dlogits) and
// accumulates the dx part directly (dx[t] is exclusive to this token).
// drouter is computed AFTERWARDS as DL^T @ x via cuBLAS (no atomics):
// the old version did ne*d atomicAdds per token (12.5M/layer at N=2048),
// all colliding on ne*d rows -- the hottest serialization point on T4.
// Warp-per-token router backward (same audit P0 #4 fix as k_route).
// Old kernel ran the whole token serially in one thread — including the
// ne*d router-gradient accumulation (49k MACs at ne=64,d=768) in a single
// thread. New layout: lane 0 runs the cheap scalar prologue (identical code
// and summation order, so DL is bit-exact vs the old kernel), then all
// threads cooperate: DL row writes are disjoint per expert, and dx rows are
// j-partitioned (each thread owns strided j's, e ascending — the same final
// summation order per element as before, so dx is bit-exact too, accumulated
// into the pre-zeroed dx row with += as the caller requires).
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
    (void)x;  // x enters only through the later DL^T @ x GEMM, not here
    const float* pt = probs + t * ne;
    __shared__ float sdp[64];  // dp_full workspace (ne <= 64, checked on host)
    __shared__ float sdl[64];  // final DL row workspace

    if (tid == 0) {
        for (int e = 0; e < ne; ++e)
            sdp[e] = aux_frac ? aux_coef * aux_frac[e] : 0.0f;

        // Backprop through top-k normalization: w_k = p_{e_k} / S, S = sum(p_{e_j})
        // dL/dp_{e_k} = (g_k - bar_g) / S, where bar_g = sum(g_j * w_j)
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
            // P0-06 FIX: forward jitter z'=z*(1+j*2*u) needs chain-rule factor.
            // Same hash as k_route forward (deterministic, no state).
            if (jitter > 0.0f) {
                uint32_t h = (uint32_t)(t * 2654435761ULL) ^ (uint32_t)(e * 40503u + 1u)
                           ^ (uint32_t)(seed & 0xFFFFFFFFu) ^ (uint32_t)((seed >> 32) * 2246822519ULL);
                h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
                float u = ((h % 1000000u) / 1000000.0f) - 0.5f;
                dl *= (1.0f + jitter * 2.0f * u);
            }
            sdl[e] = dl;
        }
    }
    __syncthreads();
    // DL row: one writer per expert (disjoint, race-free).
    float* DLt = DL + t * ne;
    for (int e = tid; e < ne; e += W) DLt[e] = sdl[e];
    // dx row: j-partitioned accumulation into the pre-zeroed row (disjoint,
    // race-free, no atomics — dx[t] is exclusive to this token's block).
    float* dxt = dx + t * d;
    for (int j = tid; j < d; j += W) {
        float acc = 0.0f;
        for (int e = 0; e < ne; ++e) acc += sdl[e] * router_w[(i64)e * d + j];
        dxt[j] += acc;
    }
}

// GPU grouping: slots stay on device, only ne counters cross the host.
// grouped[s] layout after build: expert 0 slots, then expert 1, ...
// h_counts[e] = slots for expert e, h_offsets[e] = start index in grouped.
__global__ void k_group_hist(const i32* idx, int* cnt, i64 NK, int ne) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= NK) return;
    int e = idx[s];
    if (e >= 0 && e < ne) atomicAdd(&cnt[e], 1);
}

__global__ void k_group_fill(const i32* idx, i32* grouped, int* cursors,
                             i64 NK, int ne) {
    i64 s = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= NK) return;
    int e = idx[s];
    if (e < 0 || e >= ne) return;
    int pos = atomicAdd(&cursors[e], 1);
    grouped[pos] = (i32)s;
}

// Fills h_counts[ne], h_offsets[ne+1] on host; grouped slots on device.
// h_counts/h_offsets must have room for ne / ne+1 ints. grouped_out receives
// the device pointer (valid until the next call with larger NK).
static void moe_build_groups(const i32* d_idx, i64 N, int K, int ne,
                             i32** grouped_out, int* h_counts, int* h_offsets) {
    const i64 NK = N * K;
    GAI_CHECK(ne > 0 && ne <= 64, "moe grouping: ne out of range");
    grp_ensure((size_t)(NK > 0 ? NK : 1), ne);
    *grouped_out = g_grp_grouped;
    for (int e = 0; e < ne; ++e) h_counts[e] = 0;
    for (int e = 0; e <= ne; ++e) h_offsets[e] = 0;
    if (NK <= 0) return;
    CU_CHECK(cudaMemset(g_grp_cnt, 0, sizeof(int) * (size_t)ne));
    k_group_hist<<<grid_for(NK, 256), 256>>>(d_idx, g_grp_cnt, NK, ne);
    CU_CHECK(cudaGetLastError());
    // Tiny sync: ne ints instead of NK ints (8 vs 4096 at N=2048,K=2).
    // PERF NOTE (unmeasured here — needs T4 + Nsight): the payload is tiny so
    // this is latency, not bandwidth. At grad_accum=128 x 36 MoE layers the
    // repeated D2H stalls can still show up as a CPU/GPU gap. Profile step
    // time with Nsight Systems on the real GPU before optimizing further
    // (candidate: fully device-side grouping when it proves hot).
    CU_CHECK(cudaMemcpy(h_counts, g_grp_cnt, sizeof(int) * (size_t)ne,
                        cudaMemcpyDeviceToHost));
    h_offsets[0] = 0;
    for (int e = 0; e < ne; ++e) {
        int c = h_counts[e] < 0 ? 0 : h_counts[e];
        h_counts[e] = c;
        h_offsets[e + 1] = h_offsets[e] + c;
    }
    if (h_offsets[ne] <= 0) return;
    // cursors start at per-expert offsets; atomicAdd walks them forward.
    CU_CHECK(cudaMemcpy(g_grp_cur, h_offsets, sizeof(int) * (size_t)ne,
                        cudaMemcpyHostToDevice));
    k_group_fill<<<grid_for(NK, 256), 256>>>(d_idx, g_grp_grouped, g_grp_cur, NK, ne);
    CU_CHECK(cudaGetLastError());
}

// ================================================================ forward
void moe_forward(const float* x, const float* router_w,
                 const float* gates, const float* ups, const float* downs,
                 const float* sh_g, const float* sh_u, const float* sh_d,
                 float* out,
                 float* probs_cache, i32* idx_cache, float* w_cache,
                 float* s_gate, float* s_up, float* s_act,
                 i64 N, int d, int E, int ne, int K) {
    if (N <= 0) return;
    const i64 NK = N * K;

    uint8_t* ws = (uint8_t*)moe_workspace(
        sizeof(float) * (size_t)N * ne +      // rlogits
        sizeof(i32) * (size_t)NK +            // tmp idx (inference only)
        sizeof(float) * (size_t)NK +          // tmp w   (inference only)
        sizeof(float) * (size_t)NK * E * 3 +  // expert block G/U/A
        sizeof(float) * (size_t)NK * d);      // expert block X/Eg
    float* rlogits = (float*)ws; ws += sizeof(float) * (size_t)N * ne;
    i32*   tmp_idx = (i32*)ws;   ws += sizeof(i32) * (size_t)NK;
    float* tmp_w   = (float*)ws; ws += sizeof(float) * (size_t)NK;
    float* Gblk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Ublk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Ablk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Xblk = (float*)ws; ws += sizeof(float) * (size_t)NK * d;

    // 1. router logits
    linear_forward(x, router_w, rlogits, (int)N, d, ne);

    // 2. route (one warp per token: 32 threads cooperate on each token's
    // ne-expert row instead of 1 thread doing it serially — audit P0 #4)
    i32* idx_use = idx_cache ? idx_cache : tmp_idx;
    float* w_use = w_cache ? w_cache : tmp_w;
    GAI_CHECK(K >= 1 && K <= 8, "moe routing: top_k out of kernel range [1,8]");
    GAI_CHECK(ne >= 1 && ne <= 64, "moe routing: num_experts out of kernel range [1,64]");
    k_route<<<(unsigned)N, 32>>>(rlogits, probs_cache, idx_use, w_use, N, ne, K, g_moe_jitter, g_moe_jitter_seed);
    CU_CHECK(cudaGetLastError());

    // 3. shared expert: out = down(swiglu(g(x), u(x)))
    // Runs on N tokens (not NK slots): only first N*E rows valid.
    if (sh_g && sh_u && sh_d) {
        linear_forward(x, sh_g, Gblk, (int)N, d, E);
        linear_forward(x, sh_u, Ublk, (int)N, d, E);
        k_swiglu<<<grid_for(N * E, 256), 256>>>(Gblk, Ublk, Ablk, N * E);
        CU_CHECK(cudaGetLastError());
        linear_forward(Ablk, sh_d, out, (int)N, E, d);
    } else {
        CU_CHECK(cudaMemset(out, 0, sizeof(float) * (size_t)N * d));
    }

    // 4. routed experts, grouped on the GPU (tiny ne-only host sync).
    i32* grouped = nullptr;
    int h_cnt[64], h_off[65];
    moe_build_groups(idx_use, N, K, ne, &grouped, h_cnt, h_off);
    if (h_off[ne] <= 0) return;

    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        i32* d_slots = grouped + h_off[e];
        const float* ge = gates + (size_t)e * E * d;
        const float* ue = ups   + (size_t)e * E * d;
        const float* de = downs + (size_t)e * d * E;
        k_gather_tok<<<grid_for(ns, 256), 256>>>(x, d_slots, Xblk, ns, d, K);
        CU_CHECK(cudaGetLastError());
        linear_forward(Xblk, ge, Gblk, (int)ns, d, E);
        linear_forward(Xblk, ue, Ublk, (int)ns, d, E);
        k_swiglu<<<grid_for(ns * E, 256), 256>>>(Gblk, Ublk, Ablk, ns * E);
        CU_CHECK(cudaGetLastError());
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Gblk, d_slots, s_gate, ns, E);
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Ublk, d_slots, s_up, ns, E);
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Ablk, d_slots, s_act, ns, E);
        CU_CHECK(cudaGetLastError());
        linear_forward(Ablk, de, Xblk, (int)ns, E, d);
        k_scatter_add<<<grid_for(ns, 256), 256>>>(out, Xblk, d_slots, w_use, ns, d, K);
        CU_CHECK(cudaGetLastError());
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
    if (!router_bias) {
        moe_forward(x, router_w, gates, ups, downs, sh_g, sh_u, sh_d, out,
                    probs_cache, idx_cache, w_cache, s_gate, s_up, s_act, N, d, E, ne, K);
        return;
    }
    const i64 NK = N * K;

    uint8_t* ws = (uint8_t*)moe_workspace(
        sizeof(float) * (size_t)N * ne +
        sizeof(i32) * (size_t)NK +
        sizeof(float) * (size_t)NK +
        sizeof(float) * (size_t)NK * E * 3 +
        sizeof(float) * (size_t)NK * d);
    float* rlogits = (float*)ws; ws += sizeof(float) * (size_t)N * ne;
    i32*   tmp_idx = (i32*)ws;   ws += sizeof(i32) * (size_t)NK;
    float* tmp_w   = (float*)ws; ws += sizeof(float) * (size_t)NK;
    float* Gblk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Ublk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Ablk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Xblk = (float*)ws; ws += sizeof(float) * (size_t)NK * d;

    linear_forward(x, router_w, rlogits, (int)N, d, ne);

    i32* idx_use = idx_cache ? idx_cache : tmp_idx;
    float* w_use = w_cache ? w_cache : tmp_w;
    GAI_CHECK(K >= 1 && K <= 8, "moe routing bias: top_k out of range [1,8]");
    GAI_CHECK(ne >= 1 && ne <= 64, "moe routing bias: ne out of range [1,64]");
    k_route_bias<<<(unsigned)N, 32>>>(rlogits, router_bias, probs_cache, idx_use, w_use,
                                     N, ne, K, g_moe_jitter, g_moe_jitter_seed);
    CU_CHECK(cudaGetLastError());

    if (sh_g && sh_u && sh_d) {
        linear_forward(x, sh_g, Gblk, (int)N, d, E);
        linear_forward(x, sh_u, Ublk, (int)N, d, E);
        k_swiglu<<<grid_for(N * E, 256), 256>>>(Gblk, Ublk, Ablk, N * E);
        CU_CHECK(cudaGetLastError());
        linear_forward(Ablk, sh_d, out, (int)N, E, d);
    } else {
        CU_CHECK(cudaMemset(out, 0, sizeof(float) * (size_t)N * d));
    }

    i32* grouped = nullptr;
    int h_cnt[64], h_off[65];
    moe_build_groups(idx_use, N, K, ne, &grouped, h_cnt, h_off);
    if (h_off[ne] <= 0) return;

    for (int e = 0; e < ne; ++e) {
        i64 ns = (i64)h_cnt[e];
        if (ns == 0) continue;
        i32* d_slots = grouped + h_off[e];
        const float* ge = gates + (size_t)e * E * d;
        const float* ue = ups   + (size_t)e * E * d;
        const float* de = downs + (size_t)e * d * E;
        k_gather_tok<<<grid_for(ns, 256), 256>>>(x, d_slots, Xblk, ns, d, K);
        CU_CHECK(cudaGetLastError());
        linear_forward(Xblk, ge, Gblk, (int)ns, d, E);
        linear_forward(Xblk, ue, Ublk, (int)ns, d, E);
        k_swiglu<<<grid_for(ns * E, 256), 256>>>(Gblk, Ublk, Ablk, ns * E);
        CU_CHECK(cudaGetLastError());
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Gblk, d_slots, s_gate, ns, E);
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Ublk, d_slots, s_up, ns, E);
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Ablk, d_slots, s_act, ns, E);
        CU_CHECK(cudaGetLastError());
        linear_forward(Ablk, de, Xblk, (int)ns, E, d);
        k_scatter_add<<<grid_for(ns, 256), 256>>>(out, Xblk, d_slots, w_use, ns, d, K);
        CU_CHECK(cudaGetLastError());
    }
}

// ================================================================ backward
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
    const i64 NK = N * K;
    const float aux_coef = (aux_frac && aux_scale != 0.0f)
        ? aux_scale * (float)ne / (float)N : 0.0f;
    (void)s_dact;  // CUDA path uses workspace blocks below instead

    uint8_t* ws = (uint8_t*)moe_workspace(
        sizeof(float) * (size_t)N * E * 3 +   // shared g/u/a
        sizeof(float) * (size_t)N * E * 3 +   // shared dg/du/dact
        sizeof(float) * (size_t)NK +          // dp_slots_full
        sizeof(float) * (size_t)NK * E * 4 +  // expert G/U/A/Dact
        sizeof(float) * (size_t)NK * d * 2 +  // expert Xgather/Sscaled
        sizeof(float) * (size_t)NK +          // expert dp piece
        sizeof(float) * (size_t)N * ne);      // router DL [N,ne]
    float* shg = (float*)ws; ws += sizeof(float) * (size_t)N * E;
    float* shu = (float*)ws; ws += sizeof(float) * (size_t)N * E;
    float* sha = (float*)ws; ws += sizeof(float) * (size_t)N * E;
    float* shdg = (float*)ws; ws += sizeof(float) * (size_t)N * E;
    float* shdu = (float*)ws; ws += sizeof(float) * (size_t)N * E;
    float* shda = (float*)ws; ws += sizeof(float) * (size_t)N * E;
    float* dpfull = (float*)ws; ws += sizeof(float) * (size_t)NK;
    float* Gblk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Ublk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Ablk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Dblk = (float*)ws; ws += sizeof(float) * (size_t)NK * E;
    float* Xblk = (float*)ws; ws += sizeof(float) * (size_t)NK * d;
    float* Sblk = (float*)ws; ws += sizeof(float) * (size_t)NK * d;
    float* Pblk = (float*)ws; ws += sizeof(float) * (size_t)NK;
    float* DLblk = (float*)ws; ws += sizeof(float) * (size_t)N * ne;

    // ---- shared expert: recompute, then fully batched backward
    if (sh_g && sh_u && sh_d && dsh_g && dsh_u && dsh_d) {
        linear_forward(x, sh_g, shg, (int)N, d, E);
        linear_forward(x, sh_u, shu, (int)N, d, E);
        k_swiglu<<<grid_for(N * E, 256), 256>>>(shg, shu, sha, N * E);
        CU_CHECK(cudaGetLastError());
    // dact[N,E] = dout[N,d] @ sh_d[d,E] (NO transpose): linear_forward()
    // always applies W^T, so issue the GEMM directly here.
    gemm(false, false, (int)N, E, d, 1.0f, dout, d, sh_d, E, 0.0f, shda, E);
    k_swiglu_bwd_assign<<<grid_for(N * E, 256), 256>>>(shg, shu, shda, shdg, shdu, N * E);
        CU_CHECK(cudaGetLastError());
        linear_backward(sha, sh_d, dout, nullptr, dsh_d, (int)N, E, d);
        linear_backward(x, sh_g, shdg, dx, dsh_g, (int)N, d, E);
        linear_backward(x, sh_u, shdu, dx, dsh_u, (int)N, d, E);
    }

    // ---- routed experts, grouped on the GPU (tiny ne-only host sync).
    CU_CHECK(cudaMemset(dpfull, 0, sizeof(float) * (size_t)NK));
    i32* grouped = nullptr;
    int h_cnt[64], h_off[65];
    moe_build_groups(idx, N, K, ne, &grouped, h_cnt, h_off);

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

        k_gather<<<grid_for(ns, 256), 256>>>(s_act, d_slots, Ablk, ns, E);
        k_gather<<<grid_for(ns, 256), 256>>>(s_gate, d_slots, Gblk, ns, E);
        k_gather<<<grid_for(ns, 256), 256>>>(s_up, d_slots, Ublk, ns, E);
        k_gather_tok<<<grid_for(ns, 256), 256>>>(x, d_slots, Xblk, ns, d, K);
        k_scale_rows<<<grid_for(ns, 256), 256>>>(dout, tw, d_slots, Sblk, ns, d, K);
        CU_CHECK(cudaGetLastError());

        // dact[ns,E] = Sblk[ns,d] @ de[d,E] (NO transpose): direct GEMM.
        gemm(false, false, (int)ns, E, d, 1.0f, Sblk, d, de, E, 0.0f, Dblk, E);
        k_swiglu_bwd_assign<<<grid_for(ns * E, 256), 256>>>(Gblk, Ublk, Dblk,
                                                     Gblk, Ublk, ns * E);
        CU_CHECK(cudaGetLastError());
        // Gblk/Ublk now hold dg/du. Down weight grad only (dx=NULL):
        // dde[d,E] += Sblk[ns,d]^T @ Ablk[ns,E].
        linear_backward(Ablk, de, Sblk, nullptr, dde, (int)ns, E, d);
        // Gate path: dx piece [ns,d] into Sblk. Sblk still holds scaled-dout
        // and linear_backward accumulates (beta=1), so it must be zeroed first.
        // dge[E,d] += Gblk[ns,E]^T @ Xblk[ns,d].
        zero(Sblk, (i64)ns * d);
        linear_backward(Xblk, ge, Gblk, Sblk, dge, (int)ns, d, E);
        k_scatter_add<<<grid_for(ns, 256), 256>>>(dx, Sblk, d_slots,
                                                  nullptr, ns, d, K);
        CU_CHECK(cudaGetLastError());
        // Up path: same Sblk reuse (re-zero: it holds the gate dx-piece now).
        zero(Sblk, (i64)ns * d);
        linear_backward(Xblk, ue, Ublk, Sblk, due, (int)ns, d, E);
        k_scatter_add<<<grid_for(ns, 256), 256>>>(dx, Sblk, d_slots,
                                                  nullptr, ns, d, K);
        CU_CHECK(cudaGetLastError());

        // expert outputs for the router gradient: out_e = A @ de^T
        linear_forward(Ablk, de, Sblk, (int)ns, E, d);
        k_dp_dot<<<grid_for(ns, 256), 256>>>(dout, Sblk, d_slots, Pblk, ns, d, K);
        k_scatter_copy<<<grid_for(ns, 256), 256>>>(Pblk, d_slots, dpfull, ns, 1);
        CU_CHECK(cudaGetLastError());
    }

    // ---- router backward: per-token DL rows (no atomics), then
    // drouter[ne,d] += DL[N,ne]^T @ x[N,d] via cuBLAS (beta=1 accumulates
    // across micro-batches, same as the old atomic version).
    // NOTE: aux_coef here already includes the P0-02 pre-scale
    // (aux_orig*eff_scale*N) from Model::forward_backward, so both paths
    // share one scaled+sum space. P0-06 jitter factor inside the kernel.
    GAI_CHECK(K >= 1 && K <= 8, "moe routing backward: top_k out of kernel range [1,8]");
    GAI_CHECK(ne >= 1 && ne <= 64, "moe routing backward: num_experts out of kernel range [1,64]");
    k_router_dl<<<(unsigned)N, 32>>>(x, router_w, probs, dpfull, idx, aux_frac,
                                    aux_coef, g_moe_jitter, g_moe_jitter_seed, DLblk, dx, N, d, ne, K);
    CU_CHECK(cudaGetLastError());
    gemm(true, false, ne, d, (int)N, 1.0f, DLblk, ne, x, d, 1.0f, drouter_w, d);
}

// ================================================================ aux fractions (GPU)
// Computes DeepSeek load-balance fractions WITHOUT any N*ne / N*K host
// roundtrip. Old path copied probs[N,ne]+idx[N,K] to CPU per layer
// (832 large syncs per step). New path keeps everything on GPU; the
// caller copies back only frac[ne] (8 floats) for the loss scalar + stats.
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

// Device-side aux-raw accumulator: per-layer raw scalars (needed for the
// returned training loss) add up ON DEVICE across the layer loop, so the
// host reads ONE float per microbatch instead of 2x ne-float D2H per layer
// (72 syncs/microbatch at 36 layers). moe_aux_begin() at microbatch start,
// moe_aux_end() once at the end.
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

// Single thread: ne <= 64. frac[e] already holds the fraction cnt/(N*K)
// (see k_aux_finalize), so raw = ne*sum_e((psum[e]/N) * frac[e]) — the exact
// host formula in Model::moe_layer_aux, accumulated in double precision.
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

// frac[ne] is left ON DEVICE for moe_backward (no host roundtrip).
// If h_frac / h_psum are non-null, they are filled on the HOST with tiny
// D2H copies (ne floats each) so the caller can finish the double-precision
// raw scalar + routing stats without ever moving N*ne / N*K buffers.
// Pass d_raw_accum (from moe_aux_begin()) to ALSO fold this layer's raw
// scalar into the device accumulator; the normal path passes the persistent
// buffer and reads it once per microbatch (moe_aux_end), while log steps
// additionally pass h buffers for the balance stats.
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
    // Device raw fold (no sync): the training hot path always passes the
    // persistent accumulator and reads it once per microbatch.
    if (d_raw_accum && N > 0 && ne > 0)
        k_aux_raw_add<<<1, 1>>>(g_aux_psum, frac, d_raw_accum, N, ne);
    // Tiny copies only (ne <= 64), and ONLY for log-cadence balance stats.
    // The training hot path passes nullptrs here: zero syncs per layer.
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
    g_grp_ne_cap = 0;
    if (g_aux_cnt) { cudaFree(g_aux_cnt); g_aux_cnt = nullptr; }
    if (g_aux_psum) { cudaFree(g_aux_psum); g_aux_psum = nullptr; }
    g_aux_cap = 0;
    if (g_aux_raw) { cudaFree(g_aux_raw); g_aux_raw = nullptr; }
}

} // namespace cuda_ops
} // namespace gai
