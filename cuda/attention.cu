// Causal grouped-query attention on the GPU.
//
//  * prefill  : flash-style tiled kernel with online softmax. K/V tiles live in
//               shared memory and the [T,T] score matrix is never materialised,
//               so memory is O(T) per query row instead of O(T^2).
//               When `probs` is requested (training) the row is written out, which
//               is the only case that costs O(T^2) memory.
//  * backward : one block per (batch, kv-head, query-tile); dk/dv accumulate via
//               atomics restricted to the block's own kv head.
//  * decode   : split over heads, one block per head, warp-parallel over KV length.

#include "cuda/cuda_ops.h"

#include <cuda_runtime.h>
#include <cfloat>

namespace gai {
namespace cuda_ops {

#define CU_CHECK2(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

static constexpr int WARP_A = 32;
static constexpr int KV_TILE = 64;      // KV positions processed per shared-memory tile
static constexpr int MAX_HD  = 128;     // head_dim upper bound (ours is 64)

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- forward
// grid  = (T, H, B) ; block = 32 threads (one warp per query position).
// Shared layout: sQ[hd] | sK[KV_TILE*hd] | sV[KV_TILE*hd] | sS[KV_TILE] | sAcc[hd]
// Everything that must be shared across lanes lives in shared memory, so register
// pressure stays flat regardless of head_dim.
__global__ void k_attn_fwd(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, float* __restrict__ out,
                           float* __restrict__ probs,
                           int T, int H, int KV, int hd, float scale) {
    extern __shared__ float sh[];
    float* sQ   = sh;
    float* sK   = sQ + hd;
    float* sV   = sK + KV_TILE * hd;
    float* sS   = sV + KV_TILE * hd;
    float* sAcc = sS + KV_TILE;

    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int group = H / KV;
    const int kvh = h / group;
    const int lane = threadIdx.x;

    const size_t qs  = size_t(T) * H * hd;
    const size_t kvs = size_t(T) * KV * hd;
    const float* qh = q + size_t(b) * qs + (size_t(t) * H + h) * hd;

    for (int i = lane; i < hd; i += WARP_A) { sQ[i] = qh[i]; sAcc[i] = 0.0f; }
    __syncwarp();

    float run_max = -FLT_MAX;
    float run_sum = 0.0f;
    const int len = t + 1;

    for (int base = 0; base < len; base += KV_TILE) {
        const int tile = min(KV_TILE, len - base);

        for (int idx = lane; idx < tile * hd; idx += WARP_A) {
            int j = idx / hd;
            int c = idx - j * hd;
            size_t off = size_t(b) * kvs + (size_t(base + j) * KV + kvh) * hd + c;
            sK[idx] = k[off];
            sV[idx] = v[off];
        }
        __syncwarp();

        // scores for this tile
        float tmax = -FLT_MAX;
        for (int j = lane; j < tile; j += WARP_A) {
            const float* kj = sK + j * hd;
            float s = 0.0f;
            #pragma unroll 4
            for (int c = 0; c < hd; ++c) s += sQ[c] * kj[c];
            s *= scale;
            sS[j] = s;
            tmax = fmaxf(tmax, s);
            // Raw scores are cached; the row is exponentiated and normalised once at
            // the end, because the running max is only final after the last tile.
            if (probs) probs[((size_t(b) * H + h) * T + t) * T + base + j] = s;
        }
        #pragma unroll
        for (int o = WARP_A / 2; o > 0; o >>= 1)
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
        __syncwarp();

        const float new_max = fmaxf(run_max, tmax);
        const float rescale = (run_max == -FLT_MAX) ? 0.0f : __expf(run_max - new_max);

        float tsum = 0.0f;
        for (int j = lane; j < tile; j += WARP_A) {
            float p = __expf(sS[j] - new_max);
            sS[j] = p;
            tsum += p;
        }
        #pragma unroll
        for (int o = WARP_A / 2; o > 0; o >>= 1) tsum += __shfl_xor_sync(0xffffffffu, tsum, o);
        __syncwarp();

        run_sum = run_sum * rescale + tsum;
        run_max = new_max;

        // rescale the running accumulator and add this tile's contribution
        for (int c = lane; c < hd; c += WARP_A) {
            float a = sAcc[c] * rescale;
            for (int j = 0; j < tile; ++j) a += sS[j] * sV[j * hd + c];
            sAcc[c] = a;
        }

        __syncwarp();
    }

    const float inv = 1.0f / run_sum;
    float* o = out + size_t(b) * qs + (size_t(t) * H + h) * hd;
    for (int c = lane; c < hd; c += WARP_A) o[c] = sAcc[c] * inv;

    if (probs) {
        float* pr = probs + ((size_t(b) * H + h) * T + t) * T;
        for (int j = lane; j < len; j += WARP_A) pr[j] = __expf(pr[j] - run_max) * inv;
        for (int j = len + lane; j < T; j += WARP_A) pr[j] = 0.0f;
    }
}

void attention_forward(const float* q, const float* k, const float* v,
                       float* out, float* probs,
                       int B, int T, int H, int KV, int hd, float scale) {
    if (B <= 0 || T <= 0) return;
    GAI_CHECK(H > 0 && H <= 65535 && B <= 65535,
              "cuda attention_forward: grid dimensions out of range");
    GAI_CHECK(hd <= MAX_HD, "cuda attention: head_dim too large");
    GAI_CHECK(H % KV == 0, "cuda attention_forward: H must be a multiple of KV");
    dim3 grid(T, H, B);
    // Shared layout: sQ[hd] | sK[KV_TILE*hd] | sV[KV_TILE*hd] | sS[KV_TILE] | sAcc[hd]
    // T4 has 48KB shared/block: hd=64 -> ~33KB ok, hd=128 -> ~66KB OOM.
    // Fail fast with a clear message instead of illegal launch.
    size_t sh = sizeof(float) * (size_t(hd) + size_t(KV_TILE) * hd * 2 + KV_TILE + size_t(hd));
    GAI_CHECK(sh <= 48 * 1024, "cuda attention_forward: shared memory over T4 limit (use smaller head_dim)");
    k_attn_fwd<<<grid, WARP_A, sh>>>(q, k, v, out, probs, T, H, KV, hd, scale);
    CU_CHECK2(cudaGetLastError());
}

// SWA (Mistral-style sliding window): simple correct kernel for window>0.
// window==0 delegates to the flash tiled path above (zero regression risk).
__global__ void k_attn_fwd_swa(const float* __restrict__ q, const float* __restrict__ k,
                               const float* __restrict__ v, float* __restrict__ out,
                               float* __restrict__ probs,
                               int T, int H, int KV, int hd, float scale, int window) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int group = H / KV;
    const int kvh = h / group;
    const int lane = threadIdx.x;
    const size_t qs  = size_t(T) * H * hd;
    const size_t kvs = size_t(T) * KV * hd;
    const float* qh = q + size_t(b) * qs + (size_t(t) * H + h) * hd;
    float* o = out + size_t(b) * qs + (size_t(t) * H + h) * hd;
    int j0 = (window > 0 && t + 1 > window) ? t + 1 - window : 0;
    // scores in global probs row scratch (probs required for SWA training path;
    // inference prefill passes probs=null and uses registers only via swa path below).
    // To keep this kernel simple it recomputes per lane with warp reductions.
    __shared__ float sAcc[128];
    for (int c = lane; c < hd; c += WARP_A) sAcc[c] = 0.0f;
    __syncwarp();
    float mx = -FLT_MAX;
    for (int j = j0 + lane; j <= t; j += WARP_A) {
        const float* kh = k + size_t(b) * kvs + (size_t(j) * KV + kvh) * hd;
        float d = 0.0f;
        for (int c = 0; c < hd; ++c) d += qh[c] * kh[c];
        d *= scale;
        mx = fmaxf(mx, d);
    }
#pragma unroll
    for (int o2 = WARP_A / 2; o2 > 0; o2 >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o2));
    // FIX P0-2 (SWA 32x overcount): every lane computes the SAME full sum
    // (j loop is not strided, sAcc partition is over c not j), so a warp
    // reduction would multiply sum by 32 and produce 1/32 outputs.
    // All lanes already hold identical sum -> use directly, no reduction.
    float sum = 0.0f;
    for (int j = j0; j <= t; ++j) {
        const float* kh = k + size_t(b) * kvs + (size_t(j) * KV + kvh) * hd;
        float d = 0.0f;
        for (int c = 0; c < hd; ++c) d += qh[c] * kh[c];
        d *= scale;
        float p = __expf(d - mx);
        sum += p;
        const float* vh = v + size_t(b) * kvs + (size_t(j) * KV + kvh) * hd;
        for (int c = lane; c < hd; c += WARP_A) sAcc[c] += p * vh[c];
        if (probs) probs[((size_t(b) * H + h) * T + t) * T + j] = p;
    }
    __syncwarp();  // ensure all sAcc partitions visible before normalize
    float inv = 1.0f / sum;
    for (int c = lane; c < hd; c += WARP_A) o[c] = sAcc[c] * inv;
    if (probs) {
        float* pr = probs + ((size_t(b) * H + h) * T + t) * T;
        for (int j = j0 + lane; j <= t; j += WARP_A) pr[j] *= inv;
        for (int j = lane; j < j0; j += WARP_A) pr[j] = 0.0f;
        for (int j = t + 1 + lane; j < T; j += WARP_A) pr[j] = 0.0f;
    }
}

void attention_forward_ex(const float* q, const float* k, const float* v,
                          float* out, float* probs,
                          int B, int T, int H, int KV, int hd, float scale, int window) {
    if (window <= 0) { attention_forward(q, k, v, out, probs, B, T, H, KV, hd, scale); return; }
    if (B <= 0 || T <= 0) return;
    GAI_CHECK(H > 0 && H <= 65535 && B <= 65535,
              "cuda attention_ex: grid dimensions out of range");
    GAI_CHECK(hd <= MAX_HD, "cuda attention_ex: head_dim too large");
    GAI_CHECK(H % KV == 0, "cuda attention_ex: H must be a multiple of KV");
    dim3 grid(T, H, B);
    k_attn_fwd_swa<<<grid, WARP_A>>>(q, k, v, out, probs, T, H, KV, hd, scale, window);
    CU_CHECK2(cudaGetLastError());
}

// ---------------------------------------------------------------- backward
// Requires the cached probs (training path).
// Grid = (T, KV, B), block = 32 (one warp per query position).
// Each block owns ALL query heads of one KV group, so same-KV writes from
// different query heads never race across blocks. Inside the block the KV
// row range is processed in KV_TILE-sized tiles that always fit in shared
// memory (no overflow for any T), and dk/dv hit global memory with exactly
// one atomicAdd per element per block. dq rows are exclusive per (t, head),
// written with plain adds. Global head indices (gh) are used everywhere.
__global__ void k_attn_bwd(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, const float* __restrict__ probs,
                           const float* __restrict__ dout,
                           float* __restrict__ dq, float* __restrict__ dk,
                           float* __restrict__ dv,
                           int T, int H, int KV, int hd, float scale) {
    const int t = blockIdx.x;
    const int kvh = blockIdx.y;
    const int b = blockIdx.z;
    const int group = H / KV;
    const int lane = threadIdx.x;

    const size_t qs  = size_t(T) * H * hd;
    const size_t kvs = size_t(T) * KV * hd;
    const int len = t + 1;

    extern __shared__ float shmem[];
    float* s_dk  = shmem;                        // [KV_TILE * hd] tile scratch
    float* s_dv  = shmem + KV_TILE * hd;         // [KV_TILE * hd] tile scratch
    float* s_dot = shmem + 2 * KV_TILE * hd;     // [group] row sums, lane 0 writes

    // Phase 1: dot_pg[h] = sum_j p_j * dot(go, v_j) over the FULL row.
    for (int h = 0; h < group; ++h) {
        const int gh = kvh * group + h;
        const float* pr = probs + ((size_t(b) * H + gh) * T + t) * T;
        const float* go = dout + size_t(b) * qs + (size_t(t) * H + gh) * hd;
        float dot_pg = 0.0f;
        for (int j = lane; j < len; j += WARP_A) {
            const float* vh = v + size_t(b) * kvs + (size_t(j) * KV + kvh) * hd;
            float dpj = 0.0f;
            #pragma unroll 4
            for (int c = 0; c < hd; ++c) dpj += go[c] * vh[c];
            dot_pg += pr[j] * dpj;
        }
        #pragma unroll
        for (int o = WARP_A / 2; o > 0; o >>= 1)
            dot_pg += __shfl_xor_sync(0xffffffffu, dot_pg, o);
        if (lane == 0) s_dot[h] = dot_pg;
    }
    __syncthreads();

    // Phase 2: tiled accumulation. Within a tile, (j, c) pairs are owned by
    // exactly one thread (j strided by lane, c looped fully), so plain +=
    // into shared memory is race-free. dq uses per-thread registers.
    for (int base = 0; base < len; base += KV_TILE) {
        const int tile = min(KV_TILE, len - base);
        for (int idx = lane; idx < tile * hd; idx += WARP_A) {
            s_dk[idx] = 0.0f;
            s_dv[idx] = 0.0f;
        }
        __syncthreads();

        for (int h = 0; h < group; ++h) {
            const int gh = kvh * group + h;
            const float* pr = probs + ((size_t(b) * H + gh) * T + t) * T;
            const float* go = dout + size_t(b) * qs + (size_t(t) * H + gh) * hd;
            const float* qh = q + size_t(b) * qs + (size_t(t) * H + gh) * hd;
            const float dot_pg = s_dot[h];

            float dqacc[MAX_HD];
            for (int c = 0; c < hd; ++c) dqacc[c] = 0.0f;

            for (int j = lane; j < tile; j += WARP_A) {
                const int jj = base + j;
                const float* vh = v + size_t(b) * kvs + (size_t(jj) * KV + kvh) * hd;
                float dpj = 0.0f;
                #pragma unroll 4
                for (int c = 0; c < hd; ++c) dpj += go[c] * vh[c];
                const float p = pr[jj];
                for (int c = 0; c < hd; ++c) s_dv[j * hd + c] += p * go[c];
                const float ds = p * (dpj - dot_pg) * scale;
                if (ds != 0.0f) {
                    const float* kh = k + size_t(b) * kvs + (size_t(jj) * KV + kvh) * hd;
                    for (int c = 0; c < hd; ++c) {
                        dqacc[c] += ds * kh[c];
                        s_dk[j * hd + c] += ds * qh[c];
                    }
                }
            }
            // dq row (b,t,gh) is exclusive to this block: plain add, lane 0.
            float* dqh = dq + size_t(b) * qs + (size_t(t) * H + gh) * hd;
            for (int c = 0; c < hd; ++c) {
                const float s = warp_reduce_sum(dqacc[c]);
                if (lane == 0) dqh[c] += s;
            }
        }
        __syncthreads();

        // Flush this tile to global dk/dv (one atomic per element per block).
        for (int j = lane; j < tile; j += WARP_A) {
            const int jj = base + j;
            float* dvh = dv + size_t(b) * kvs + (size_t(jj) * KV + kvh) * hd;
            float* dkh = dk + size_t(b) * kvs + (size_t(jj) * KV + kvh) * hd;
            for (int c = 0; c < hd; ++c) {
                atomicAdd(dvh + c, s_dv[j * hd + c]);
                atomicAdd(dkh + c, s_dk[j * hd + c]);
            }
        }
        __syncthreads();
    }
}

void attention_backward(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale) {
    if (B <= 0 || T <= 0) return;
    GAI_CHECK(KV > 0 && KV <= 65535 && B <= 65535,
              "cuda attention_backward: grid dimensions out of range");
    GAI_CHECK(probs != nullptr, "cuda attention_backward requires cached probs");
    GAI_CHECK(hd <= MAX_HD, "cuda attention: head_dim too large");
    GAI_CHECK(H % KV == 0, "cuda attention_backward: H must be a multiple of KV");
    dim3 grid(T, KV, B);
    // Shared: s_dk[KV_TILE*hd] + s_dv[KV_TILE*hd] + s_dot[group].
    // hd=64, group=3 -> ~32KB, fits the 48KB T4 limit for any T.
    const int group = H / KV;
    const size_t shmem = sizeof(float) * (size_t(2) * KV_TILE * hd + group);
    GAI_CHECK(shmem <= 48 * 1024, "cuda attention_backward: shared memory over T4 limit");
    k_attn_bwd<<<grid, WARP_A, shmem>>>(q, k, v, probs, dout, dq, dk, dv, T, H, KV, hd, scale);
    CU_CHECK2(cudaGetLastError());
}

// ---------------------------------------------------------------- decode
// One block per query head, 128 threads striding over the KV length.
__global__ void k_attn_decode(const float* __restrict__ q, const float* __restrict__ kc,
                              const float* __restrict__ vc, float* __restrict__ out,
                              int H, int KV, int hd, int cur_len, float scale,
                              float* __restrict__ scratch) {
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int group = H / KV;
    const int kvh = h / group;
    const float* qh = q + size_t(h) * hd;
    float* s = scratch + size_t(h) * cur_len;

    // scores
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cur_len; j += blockDim.x) {
        const float* kh = kc + (size_t(j) * KV + kvh) * hd;
        float d = 0.0f;
        #pragma unroll 4
        for (int c = 0; c < hd; ++c) d += qh[c] * kh[c];
        d *= scale;
        s[j] = d;
        local_max = fmaxf(local_max, d);
    }
    // block max
    int lane = threadIdx.x % WARP_A, wid = threadIdx.x / WARP_A;
    #pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
    if (lane == 0) sm[wid] = local_max;
    __syncthreads();
    int nw = (blockDim.x + WARP_A - 1) / WARP_A;
    if (threadIdx.x == 0) {
        float m = sm[0];
        for (int i = 1; i < nw; ++i) m = fmaxf(m, sm[i]);
        sm[32] = m;
    }
    __syncthreads();
    float mx = sm[32];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cur_len; j += blockDim.x) {
        float p = __expf(s[j] - mx);
        s[j] = p;
        local_sum += p;
    }
    #pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1) local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
    if (lane == 0) sm[wid] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += sm[i];
        sm[33] = t;
    }
    __syncthreads();
    float inv = 1.0f / sm[33];

    float* o = out + size_t(h) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j < cur_len; ++j) a += s[j] * vc[(size_t(j) * KV + kvh) * hd + c];
        o[c] = a * inv;
    }
}

void attention_decode(const float* q, const float* kc, const float* vc,
                      float* out, int H, int KV, int hd, int cur_len, int max_len,
                      float scale, float* scratch) {
    // FIX P2 (OOB read): cur_len was never checked against the KV allocation
    // (max_len). An undersized cache silently read past K/V. Fail fast.
    GAI_CHECK(cur_len <= max_len,
              "cuda attention_decode: cur_len exceeds KV cache max_len (increase max_context)");
    GAI_CHECK(cur_len >= 0 && max_len >= 0, "cuda attention_decode: negative length");
    if (H <= 0 || cur_len <= 0) return;
    int block = 128;
    size_t sh = sizeof(float) * 40;
    k_attn_decode<<<H, block, sh>>>(q, kc, vc, out, H, KV, hd, cur_len, scale, scratch);
    CU_CHECK2(cudaGetLastError());
}

void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window) {
    // SWA forward_ex already zeroed probs outside the window, so the standard
    // tiled backward (which multiplies by pr[j]) naturally respects the mask.
    // Delegate directly — no separate kernel, zero regression risk.
    (void)window;
    attention_backward(q, k, v, probs, dout, dq, dk, dv, B, T, H, KV, hd, scale);
}

__global__ void k_attn_decode_ex(const float* __restrict__ q, const float* __restrict__ kc,
                                 const float* __restrict__ vc, float* __restrict__ out,
                                 int H, int KV, int hd, int cur_len, float scale,
                                 float* __restrict__ scratch, int j0) {
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int group = H / KV;
    const int kvh = h / group;
    const float* qh = q + size_t(h) * hd;
    float* s = scratch + size_t(h) * cur_len;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x + j0; j < cur_len; j += blockDim.x) {
        const float* kh = kc + (size_t(j) * KV + kvh) * hd;
        float d = 0.0f;
#pragma unroll 4
        for (int c = 0; c < hd; ++c) d += qh[c] * kh[c];
        d *= scale;
        s[j] = d;
        local_max = fmaxf(local_max, d);
    }
    int lane = threadIdx.x % WARP_A, wid = threadIdx.x / WARP_A;
#pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, o));
    if (lane == 0) sm[wid] = local_max;
    __syncthreads();
    int nw = (blockDim.x + WARP_A - 1) / WARP_A;
    if (threadIdx.x == 0) {
        float m = sm[0];
        for (int i = 1; i < nw; ++i) m = fmaxf(m, sm[i]);
        sm[32] = m;
    }
    __syncthreads();
    float mx = sm[32];

    float local_sum = 0.0f;
    for (int j = threadIdx.x + j0; j < cur_len; j += blockDim.x) {
        float p = __expf(s[j] - mx);
        s[j] = p;
        local_sum += p;
    }
#pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1) local_sum += __shfl_xor_sync(0xffffffffu, local_sum, o);
    if (lane == 0) sm[wid] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < nw; ++i) t += sm[i];
        sm[33] = t;
    }
    __syncthreads();
    float inv = 1.0f / sm[33];

    float* o = out + size_t(h) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float a = 0.0f;
        for (int j = j0; j < cur_len; ++j) a += s[j] * vc[(size_t(j) * KV + kvh) * hd + c];
        o[c] = a * inv;
    }
}

void attention_decode_ex(const float* q, const float* kc, const float* vc,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window) {
    GAI_CHECK(cur_len <= max_len,
              "cuda attention_decode_ex: cur_len exceeds KV cache max_len (increase max_context)");
    GAI_CHECK(cur_len >= 0 && max_len >= 0, "cuda attention_decode_ex: negative length");
    if (H <= 0 || cur_len <= 0) return;
    if (window <= 0) { attention_decode(q, kc, vc, out, H, KV, hd, cur_len, max_len, scale, scratch); return; }
    int j0 = (cur_len > window) ? cur_len - window : 0;
    int block = 128;
    size_t sh = sizeof(float) * 40;
    k_attn_decode_ex<<<H, block, sh>>>(q, kc, vc, out, H, KV, hd, cur_len, scale, scratch, j0);
    CU_CHECK2(cudaGetLastError());
}

} // namespace cuda_ops
} // namespace gai
