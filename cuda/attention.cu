#include "cuda/cuda_ops.h"

#include <cuda_runtime.h>
#include <cfloat>

namespace gai {
namespace cuda_ops {

#define CU_CHECK2(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

static constexpr int WARP_A = 32;
static constexpr int KV_TILE = 64;
static constexpr int MAX_HD  = 128;

static constexpr int DECODE_BLOCK = 128;
static constexpr size_t DECODE_SHMEM_FLOATS = 40;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_A / 2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__global__ void k_attn_fwd(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, float* __restrict__ out,
                           float* __restrict__ probs, const i32* __restrict__ segment_ids,
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
    const i32 seg_t = segment_ids ? segment_ids[size_t(b) * T + t] : 0;

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

        float tmax = -FLT_MAX;
        for (int j = lane; j < tile; j += WARP_A) {
            int pos = base + j;
            i32 seg_j = segment_ids ? segment_ids[size_t(b) * T + pos] : 0;
            float s = 0.0f;
            if (!segment_ids || seg_t < 0 || seg_j == seg_t) {
                const float* kj = sK + j * hd;
                #pragma unroll 4
                for (int c = 0; c < hd; ++c) s += sQ[c] * kj[c];
                s *= scale;
            } else {
                s = -FLT_MAX;
            }
            sS[j] = s;
            tmax = fmaxf(tmax, s);

            if (probs) probs[((size_t(b) * H + h) * T + t) * T + pos] = s;
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

    size_t sh = sizeof(float) * (size_t(hd) + size_t(KV_TILE) * hd * 2 + KV_TILE + size_t(hd));
    GAI_CHECK(sh <= 48 * 1024, "cuda attention_forward: shared memory over T4 limit (use smaller head_dim)");
    k_attn_fwd<<<grid, WARP_A, sh>>>(q, k, v, out, probs, nullptr, T, H, KV, hd, scale);
    CU_CHECK2(cudaGetLastError());
}

__global__ void k_attn_fwd_swa(const float* __restrict__ q, const float* __restrict__ k,
                               const float* __restrict__ v, float* __restrict__ out,
                               float* __restrict__ probs, const i32* __restrict__ segment_ids,
                               int T, int H, int KV, int hd, float scale, int window) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int group = H / KV;
    const int kvh = h / group;
    const int lane = threadIdx.x;
    const i32 seg_t = segment_ids ? segment_ids[size_t(b) * T + t] : 0;
    const size_t qs  = size_t(T) * H * hd;
    const size_t kvs = size_t(T) * KV * hd;
    const float* qh = q + size_t(b) * qs + (size_t(t) * H + h) * hd;
    float* o = out + size_t(b) * qs + (size_t(t) * H + h) * hd;
    int j0 = (window > 0 && t + 1 > window) ? t + 1 - window : 0;

    __shared__ float sAcc[128];
    for (int c = lane; c < hd; c += WARP_A) sAcc[c] = 0.0f;
    __syncwarp();
    float mx = -FLT_MAX;
    for (int j = j0 + lane; j <= t; j += WARP_A) {
        i32 seg_j = segment_ids ? segment_ids[size_t(b) * T + j] : 0;
        if (segment_ids && seg_t >= 0 && seg_j != seg_t) continue;
        const float* kh = k + size_t(b) * kvs + (size_t(j) * KV + kvh) * hd;
        float d = 0.0f;
        for (int c = 0; c < hd; ++c) d += qh[c] * kh[c];
        d *= scale;
        mx = fmaxf(mx, d);
    }
#pragma unroll
    for (int o2 = WARP_A / 2; o2 > 0; o2 >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o2));

    float sum = 0.0f;
    for (int j = j0; j <= t; ++j) {
        i32 seg_j = segment_ids ? segment_ids[size_t(b) * T + j] : 0;
        if (segment_ids && seg_t >= 0 && seg_j != seg_t) {
            if (probs) probs[((size_t(b) * H + h) * T + t) * T + j] = 0.0f;
            continue;
        }
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
    __syncwarp();
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
                          int B, int T, int H, int KV, int hd, float scale, int window,
                          const i32* segment_ids) {
    if (B <= 0 || T <= 0) return;
    GAI_CHECK(H > 0 && H <= 65535 && B <= 65535,
              "cuda attention_ex: grid dimensions out of range");
    GAI_CHECK(hd <= MAX_HD, "cuda attention_ex: head_dim too large");
    GAI_CHECK(H % KV == 0, "cuda attention_ex: H must be a multiple of KV");
    dim3 grid(T, H, B);
    if (window <= 0) {
        size_t sh = sizeof(float) * (size_t(hd) + size_t(KV_TILE) * hd * 2 + KV_TILE + size_t(hd));
        GAI_CHECK(sh <= 48 * 1024, "cuda attention_forward_ex: shared memory over T4 limit");
        k_attn_fwd<<<grid, WARP_A, sh>>>(q, k, v, out, probs, segment_ids,
                                         T, H, KV, hd, scale);
    } else {
        k_attn_fwd_swa<<<grid, WARP_A>>>(q, k, v, out, probs, segment_ids,
                                         T, H, KV, hd, scale, window);
    }
    CU_CHECK2(cudaGetLastError());
}

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
    float* s_dk  = shmem;
    float* s_dv  = shmem + KV_TILE * hd;
    float* s_dot = shmem + 2 * KV_TILE * hd;

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

            float* dqh = dq + size_t(b) * qs + (size_t(t) * H + gh) * hd;
            for (int c = 0; c < hd; ++c) {
                const float s = warp_reduce_sum(dqacc[c]);
                if (lane == 0) dqh[c] += s;
            }
        }
        __syncthreads();

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

    const int group = H / KV;
    const size_t shmem = sizeof(float) * (size_t(2) * KV_TILE * hd + group);
    GAI_CHECK(shmem <= 48 * 1024, "cuda attention_backward: shared memory over T4 limit");
    k_attn_bwd<<<grid, WARP_A, shmem>>>(q, k, v, probs, dout, dq, dk, dv, T, H, KV, hd, scale);
    CU_CHECK2(cudaGetLastError());
}

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

    GAI_CHECK(cur_len <= max_len,
              "cuda attention_decode: cur_len exceeds KV cache max_len (increase max_context)");
    GAI_CHECK(cur_len >= 0 && max_len >= 0, "cuda attention_decode: negative length");
    GAI_CHECK(H % KV == 0, "cuda attention_decode: H must be a multiple of KV");
    GAI_CHECK(scratch != nullptr, "cuda attention_decode: null scratch (need H*cur_len floats)");
    if (H <= 0 || cur_len <= 0) return;
    int block = DECODE_BLOCK;
    GAI_CHECK((block + WARP_A - 1) / WARP_A + 2 <= (int)DECODE_SHMEM_FLOATS,
              "cuda attention_decode: block exceeds shared-memory budget");
    size_t sh = sizeof(float) * DECODE_SHMEM_FLOATS;
    k_attn_decode<<<H, block, sh>>>(q, kc, vc, out, H, KV, hd, cur_len, scale, scratch);
    CU_CHECK2(cudaGetLastError());
}

void attention_backward_ex(const float* q, const float* k, const float* v,
                           const float* probs, const float* dout,
                           float* dq, float* dk, float* dv,
                           int B, int T, int H, int KV, int hd, float scale, int window) {

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
    float inv = sm[33] > 0.0f ? 1.0f / sm[33] : 0.0f;

    float* o = out + size_t(h) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float a = 0.0f;
        for (int j = j0; j < cur_len; ++j) a += s[j] * vc[(size_t(j) * KV + kvh) * hd + c];
        o[c] = a * inv;
    }
}

__device__ __forceinline__ size_t decode_ring_slot(int j, int pinned_prefix,
                                                     int ring_start, int ring_capacity) {
    if (j < pinned_prefix) return (size_t)j;
    int cap = ring_capacity > 0 ? ring_capacity : 1;
    return (size_t)pinned_prefix + (size_t)((ring_start + (j - pinned_prefix)) % cap);
}

__global__ void k_attn_decode_ring(const float* __restrict__ q, const float* __restrict__ kc,
                                   const float* __restrict__ vc, float* __restrict__ out,
                                   int H, int KV, int hd, int cur_len, int ring_start,
                                   int pinned_prefix, int ring_capacity, float scale,
                                   float* __restrict__ scratch, int j0) {
    extern __shared__ float sm[];
    const int h = blockIdx.x;
    const int group = H / KV;
    const int kvh = h / group;
    const float* qh = q + size_t(h) * hd;
    float* s = scratch + size_t(h) * cur_len;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x + j0; j < cur_len; j += blockDim.x) {
        size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
        const float* kh = kc + (slot * size_t(KV) + kvh) * hd;
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
    float inv = sm[33] > 0.0f ? 1.0f / sm[33] : 0.0f;

    float* o = out + size_t(h) * hd;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        float a = 0.0f;
        for (int j = j0; j < cur_len; ++j) {
            size_t slot = decode_ring_slot(j, pinned_prefix, ring_start, ring_capacity);
            a += s[j] * vc[(slot * size_t(KV) + kvh) * hd + c];
        }
        o[c] = a * inv;
    }
}

void attention_decode_ex(const float* q, const float* kc, const float* vc,
                         float* out, int H, int KV, int hd, int cur_len, int max_len,
                         float scale, float* scratch, int window) {
    GAI_CHECK(cur_len <= max_len,
              "cuda attention_decode_ex: cur_len exceeds KV cache max_len (increase max_context)");
    GAI_CHECK(cur_len >= 0 && max_len >= 0, "cuda attention_decode_ex: negative length");
    GAI_CHECK(H % KV == 0, "cuda attention_decode_ex: H must be a multiple of KV");
    GAI_CHECK(scratch != nullptr, "cuda attention_decode_ex: null scratch (need H*cur_len floats)");
    if (H <= 0 || cur_len <= 0) return;
    if (window <= 0) { attention_decode(q, kc, vc, out, H, KV, hd, cur_len, max_len, scale, scratch); return; }
    int j0 = (cur_len > window) ? cur_len - window : 0;
    int block = DECODE_BLOCK;
    GAI_CHECK((block + WARP_A - 1) / WARP_A + 2 <= (int)DECODE_SHMEM_FLOATS,
              "cuda attention_decode_ex: block exceeds shared-memory budget");
    size_t sh = sizeof(float) * DECODE_SHMEM_FLOATS;
    k_attn_decode_ex<<<H, block, sh>>>(q, kc, vc, out, H, KV, hd, cur_len, scale, scratch, j0);
    CU_CHECK2(cudaGetLastError());
}

void attention_decode_ring(const float* q, const float* kc, const float* vc,
                           float* out, int H, int KV, int hd, int ring_start,
                           int pinned_prefix, int cur_len, int cache_max,
                           float scale, float* scratch, int window) {
    GAI_CHECK(H > 0 && KV > 0 && hd > 0, "cuda attention_decode_ring: empty shape");
    GAI_CHECK(H % KV == 0, "cuda attention_decode_ring: H must be a multiple of KV");
    GAI_CHECK(cur_len >= 0 && cache_max >= 0 && cur_len <= cache_max,
              "cuda attention_decode_ring: length exceeds cache capacity");
    GAI_CHECK(pinned_prefix >= 0 && pinned_prefix <= cur_len,
              "cuda attention_decode_ring: pinned prefix out of range");
    const int ring_capacity = cache_max - pinned_prefix;
    if (cur_len > pinned_prefix) {
        GAI_CHECK(ring_capacity > 0, "cuda attention_decode_ring: ring has no rolling slots");
        GAI_CHECK(ring_start >= 0 && ring_start < ring_capacity,
                  "cuda attention_decode_ring: ring start out of range");
    }
    if (H <= 0 || cur_len <= 0) return;
    int j0 = (window > 0 && cur_len > window) ? cur_len - window : 0;
    int block = DECODE_BLOCK;
    GAI_CHECK((block + WARP_A - 1) / WARP_A + 2 <= (int)DECODE_SHMEM_FLOATS,
              "cuda attention_decode_ring: block exceeds shared-memory budget");
    size_t sh = sizeof(float) * DECODE_SHMEM_FLOATS;
    k_attn_decode_ring<<<H, block, sh>>>(q, kc, vc, out, H, KV, hd, cur_len, ring_start,
                                         pinned_prefix, ring_capacity, scale, scratch, j0);
    CU_CHECK2(cudaGetLastError());
}

}
}
