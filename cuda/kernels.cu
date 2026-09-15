// Elementwise, normalization, embedding, activation, loss and optimizer kernels.
// Attention lives in cuda/attention.cu.

#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#if __CUDACC_VER_MAJOR__ >= 11
#include <cuda_bf16.h>
#endif
#include <cfloat>
#include <cmath>
#include <vector>

namespace gai {
namespace cuda_ops {

#define CU_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)

static constexpr int WARP = 32;

static inline int grid_for(i64 n, int block) {
    i64 g = (n + block - 1) / block;
    if (g < 1) g = 1;
    if (g > 65535 * 64) g = 65535 * 64;
    return static_cast<int>(g);
}

// ---------------------------------------------------------------- workspace
// FIX: round up to 64MB chunks so warmup size variations don't cause
// repeated cudaFree/cudaMalloc (fragmentation + sync stalls that looked
// like a GPU leak on nvidia-smi). Monotonic by design, freed at shutdown.
static void*  g_ws = nullptr;
static size_t g_ws_bytes = 0;

static size_t round_ws(size_t bytes) {
    const size_t chunk = (64ull << 20);
    return ((bytes + chunk - 1) / chunk) * chunk;
}

static void* workspace(size_t bytes) {
    if (bytes <= g_ws_bytes) return g_ws;
    size_t want = round_ws(bytes);
    if (want <= g_ws_bytes) return g_ws;
    if (g_ws) CU_CHECK(cudaFree(g_ws));
    CU_CHECK(cudaMalloc(&g_ws, want));
    g_ws_bytes = want;
    return g_ws;
}

void free_workspace() {
    if (g_ws) cudaFree(g_ws);
    g_ws = nullptr;
    g_ws_bytes = 0;
    moe_free_workspace();
}

static bool g_fp16_gemm = true;
void set_fp16_gemm(bool on) { g_fp16_gemm = on; }
bool fp16_gemm_enabled() { return g_fp16_gemm; }

static bool g_bf16_gemm = true;
void set_bf16_gemm(bool on) { g_bf16_gemm = on; }
bool bf16_gemm_enabled() { return g_bf16_gemm; }

// f32 row-major [rows,cols] (stride ld_src) -> contiguous fp16 block
__global__ void k_f32_to_f16_strided(const float* src, __half* dst,
                                     int rows, int cols, int ld_src) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 n = (i64)rows * cols;
    if (i >= n) return;
    int r = (int)(i / cols), c = (int)(i % cols);
    dst[i] = __float2half(src[(i64)r * ld_src + c]);
}

// f32 row-major [rows,cols] (stride ld_src) -> contiguous bf16 block
#if __CUDACC_VER_MAJOR__ >= 11
__global__ void k_f32_to_bf16_strided(const float* src, __nv_bfloat16* dst,
                                      int rows, int cols, int ld_src) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 n = (i64)rows * cols;
    if (i >= n) return;
    int r = (int)(i / cols), c = (int)(i % cols);
    dst[i] = __float2bfloat16_rn(src[(i64)r * ld_src + c]);
}
#endif

// Large-GEMM fast path: fp16 tensor cores with fp32 accumulation, fp32 I/O.
// Master weights never leave fp32 — conversion is per-call scratch.
static void gemm_fp16(bool trans_a, bool trans_b, int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb, float beta, float* C, int ldc) {
    const int rA = trans_a ? K : M, cA = trans_a ? M : K;
    const int rB = trans_b ? N : K, cB = trans_b ? K : N;
    const size_t nA = (size_t)rA * cA, nB = (size_t)rB * cB;
    __half* Ah = (__half*)workspace(sizeof(__half) * (nA + nB));
    __half* Bh = Ah + nA;
    k_f32_to_f16_strided<<<grid_for((i64)nA, 256), 256>>>(A, Ah, rA, cA, lda);
    k_f32_to_f16_strided<<<grid_for((i64)nB, 256), 256>>>(B, Bh, rB, cB, ldb);
    CU_CHECK(cudaGetLastError());
    // Same operand/flag mapping as gemm(): contiguous strides are the widths.
    cublasHandle_t h = reinterpret_cast<cublasHandle_t>(cuda::cublas_handle());
    // NOTE: must be cublasGemmEx (19 args). The legacy cublasSgemmEx takes
    // only 16 args and was removed in recent CUDA toolkits.
    cublasStatus_t s = cublasGemmEx(h,
                                     trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                     trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha,
                                     Bh, CUDA_R_16F, cB,
                                     Ah, CUDA_R_16F, cA,
                                     &beta,
                                     C, CUDA_R_32F, ldc,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) GAI_FAIL("cublasGemmEx failed");
}

// BF16 tensor-core GEMM (T4/Ampere+, CUDA 11+ only — our only GPU target).
#if __CUDACC_VER_MAJOR__ >= 11
static void gemm_bf16(bool trans_a, bool trans_b, int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb, float beta, float* C, int ldc) {
    const int rA = trans_a ? K : M, cA = trans_a ? M : K;
    const int rB = trans_b ? N : K, cB = trans_b ? K : N;
    const size_t nA = (size_t)rA * cA, nB = (size_t)rB * cB;
    __nv_bfloat16* Ah = (__nv_bfloat16*)workspace(sizeof(__nv_bfloat16) * (nA + nB));
    __nv_bfloat16* Bh = Ah + nA;
    k_f32_to_bf16_strided<<<grid_for((i64)nA, 256), 256>>>(A, Ah, rA, cA, lda);
    k_f32_to_bf16_strided<<<grid_for((i64)nB, 256), 256>>>(B, Bh, rB, cB, ldb);
    CU_CHECK(cudaGetLastError());
    cublasHandle_t h = reinterpret_cast<cublasHandle_t>(cuda::cublas_handle());
    cublasStatus_t s = cublasGemmEx(h,
                                     trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                     trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha,
                                     Bh, CUDA_R_16BF, cB,
                                     Ah, CUDA_R_16BF, cA,
                                     &beta,
                                     C, CUDA_R_32F, ldc,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) GAI_FAIL("cublasGemmEx BF16 failed");
}
#endif // __CUDACC_VER_MAJOR__ >= 11

// ---------------------------------------------------------------- reductions
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

// block-wide sum, 1024 threads max (32 warps)
__device__ __forceinline__ float block_sum(float v, float* smem) {
    int lane = threadIdx.x % WARP;
    int wid  = threadIdx.x / WARP;
    v = warp_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_max(float v, float* smem) {
    int lane = threadIdx.x % WARP;
    int wid  = threadIdx.x / WARP;
    v = warp_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : -FLT_MAX;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    return smem[0];
}

// ================================================================ GEMM (cuBLAS)
// We store everything row-major; cuBLAS is column-major. A row-major
// C[MxN] = A[MxK] * B[KxN] equals, in column-major terms,
// C'[NxM] = B'[NxK] * A'[KxM], which is what we issue below.
void gemm(bool trans_a, bool trans_b, int M, int N, int K,
          float alpha, const float* A, int lda, const float* B, int ldb,
          float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0 || alpha == 0.0f) {
        if (beta == 0.0f) {
            CU_CHECK(cudaMemset2D(C, sizeof(float) * ldc, 0, sizeof(float) * N, M));
        } else if (beta != 1.0f) {
            scale_inplace(C, beta, static_cast<i64>(M) * N);
        }
        return;
    }
    // FP16 tensor-core fast path for large GEMMs (compute-only mixed precision).
    // Small GEMMs stay fp32: conversion overhead would eat the win.
    const i64 mnk = (i64)M * N * K;
    if (g_fp16_gemm && mnk >= (1024LL * 1024LL)) {
        gemm_fp16(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    // BF16 tensor-core fast path (T4/Ampere+, CUDA 11+)
#if __CUDACC_VER_MAJOR__ >= 11
    if (g_bf16_gemm && mnk >= (1024LL * 1024LL)) {
        gemm_bf16(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#else
    (void)g_bf16_gemm;
#endif
    cublasHandle_t h = reinterpret_cast<cublasHandle_t>(cuda::cublas_handle());
    cublasOperation_t opA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;   // note the swap
    cublasOperation_t opB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasStatus_t s = cublasSgemm(h, opA, opB, N, M, K,
                                   &alpha, B, ldb, A, lda, &beta, C, ldc);
    if (s != CUBLAS_STATUS_SUCCESS) GAI_FAIL("cublasSgemm failed");
}

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N) {
    gemm(false, true, M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
}

void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N) {
    if (dx) gemm(false, false, M, K, N, 1.0f, dy, N, w, K, 1.0f, dx, K);
    if (dw) gemm(true,  false, N, K, M, 1.0f, dy, N, x, K, 1.0f, dw, K);
}

// ================================================================ elementwise
__global__ void k_add(const float* a, const float* b, float* o, i64 n) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) o[i] = a[i] + b[i];
}
__global__ void k_add_inplace(float* a, const float* b, i64 n) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) a[i] += b[i];
}
__global__ void k_scale(float* a, float s, i64 n) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) a[i] *= s;
}

void add(const float* a, const float* b, float* o, i64 n) {
    if (n <= 0) return;
    k_add<<<grid_for(n, 256), 256>>>(a, b, o, n);
    CU_CHECK(cudaGetLastError());
}
void add_inplace(float* a, const float* b, i64 n) {
    if (n <= 0) return;
    k_add_inplace<<<grid_for(n, 256), 256>>>(a, b, n);
    CU_CHECK(cudaGetLastError());
}
void scale_inplace(float* a, float s, i64 n) {
    if (n <= 0) return;
    k_scale<<<grid_for(n, 256), 256>>>(a, s, n);
    CU_CHECK(cudaGetLastError());
}
void zero(float* a, i64 n) {
    if (n <= 0) return;
    CU_CHECK(cudaMemset(a, 0, sizeof(float) * size_t(n)));
}
void copy(float* dst, const float* src, i64 n) {
    if (n <= 0) return;
    CU_CHECK(cudaMemcpy(dst, src, sizeof(float) * size_t(n), cudaMemcpyDeviceToDevice));
}

// ================================================================ embedding
__global__ void k_embed_fwd(const i32* ids, const float* table, float* out,
                            i64 ntok, int dim, int vocab) {
    i64 t = blockIdx.x;
    if (t >= ntok) return;
    i32 id = ids[t];
    float* o = out + t * dim;
    if (id < 0 || id >= vocab) {
        for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = 0.0f;
        return;
    }
    const float* src = table + i64(id) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = src[i];
}

__global__ void k_embed_bwd(const i32* ids, const float* dout, float* dtable,
                            i64 ntok, int dim, int vocab) {
    i64 t = blockIdx.x;
    if (t >= ntok) return;
    i32 id = ids[t];
    if (id < 0 || id >= vocab) return;
    float* d = dtable + i64(id) * dim;
    const float* g = dout + t * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) atomicAdd(d + i, g[i]);
}

void embedding_forward(const i32* ids, const float* table, float* out,
                       i64 ntok, int dim, int vocab) {
    if (ntok <= 0) return;
    int block = dim >= 256 ? 256 : ((dim + 31) / 32) * 32;
    if (block < 32) block = 32;
    k_embed_fwd<<<static_cast<int>(ntok), block>>>(ids, table, out, ntok, dim, vocab);
    CU_CHECK(cudaGetLastError());
}

void embedding_backward(const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab) {
    if (ntok <= 0) return;
    int block = dim >= 256 ? 256 : ((dim + 31) / 32) * 32;
    if (block < 32) block = 32;
    k_embed_bwd<<<static_cast<int>(ntok), block>>>(ids, dout, dtable, ntok, dim, vocab);
    CU_CHECK(cudaGetLastError());
}

// ================================================================ rmsnorm
__global__ void k_rmsnorm_fwd(const float* x, const float* w, float* out, float* rrms,
                              int dim, float eps) {
    extern __shared__ float smem[];
    i64 r = blockIdx.x;
    const float* xr = x + r * dim;
    float* o = out + r * dim;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i];
        ss += v * v;
    }
    ss = block_sum(ss, smem);
    float inv = rsqrtf(ss / float(dim) + eps);
    if (threadIdx.x == 0 && rrms) rrms[r] = inv;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = xr[i] * inv * w[i];
}

__global__ void k_rmsnorm_bwd_dx(const float* x, const float* w, const float* dout,
                                 const float* rrms, float* dx, int dim) {
    extern __shared__ float smem[];
    i64 r = blockIdx.x;
    const float* xr = x + r * dim;
    const float* gr = dout + r * dim;
    float* dxr = dx + r * dim;
    float inv = rrms[r];

    float dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dot += gr[i] * w[i] * xr[i];
    dot = block_sum(dot, smem);
    float coef = inv * inv * inv / float(dim) * dot;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dxr[i] += gr[i] * w[i] * inv - xr[i] * coef;
    }
}

// dweight[i] = sum_r dout[r,i] * x[r,i] * rrms[r]  -> one block per column tile
__global__ void k_rmsnorm_bwd_dw(const float* x, const float* dout, const float* rrms,
                                 float* dweight, i64 rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float acc = 0.0f;
    for (i64 r = 0; r < rows; ++r) acc += dout[r * dim + i] * x[r * dim + i] * rrms[r];
    dweight[i] += acc;
}

static int norm_block(int dim) {
    int b = 256;
    if (dim >= 1024) b = 512;
    if (dim < 128) b = 64;
    return b;
}

void rmsnorm_forward(const float* x, const float* w, float* out, float* rrms,
                     i64 rows, int dim, float eps) {
    if (rows <= 0) return;
    int block = norm_block(dim);
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    k_rmsnorm_fwd<<<static_cast<int>(rows), block, sh>>>(x, w, out, rrms, dim, eps);
    CU_CHECK(cudaGetLastError());
}

void rmsnorm_backward(const float* x, const float* w, const float* dout,
                      const float* rrms, float* dx, float* dweight, i64 rows, int dim) {
    if (rows <= 0) return;
    int block = norm_block(dim);
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    k_rmsnorm_bwd_dx<<<static_cast<int>(rows), block, sh>>>(x, w, dout, rrms, dx, dim);
    CU_CHECK(cudaGetLastError());
    if (dweight) {
        int b = 128;
        k_rmsnorm_bwd_dw<<<(dim + b - 1) / b, b>>>(x, dout, rrms, dweight, rows, dim);
        CU_CHECK(cudaGetLastError());
    }
}

// ================================================================ rope
__global__ void k_rope(float* q, float* k, const i32* pos, i64 ntok,
                       int n_heads, int n_kv, int hd, float theta, float sign) {
    int half = hd / 2;
    i64 t = blockIdx.x;
    if (t >= ntok) return;
    float p = float(pos[t]);

    int total = (n_heads + n_kv) * half;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int i = idx % half;
        int h = idx / half;
        float freq = __powf(theta, -(2.0f * float(i)) / float(hd));
        float ang  = p * freq;
        float c = __cosf(ang);
        float s = __sinf(ang) * sign;

        float* base;
        if (h < n_heads) {
            if (!q) continue;
            base = q + (t * n_heads + h) * hd;
        } else {
            if (!k) continue;
            base = k + (t * n_kv + (h - n_heads)) * hd;
        }
        float a = base[2 * i], b = base[2 * i + 1];
        base[2 * i]     = a * c - b * s;
        base[2 * i + 1] = a * s + b * c;
    }
}

void rope_forward(float* q, float* k, const i32* pos, i64 ntok,
                  int n_heads, int n_kv, int hd, float theta) {
    if (ntok <= 0) return;
    k_rope<<<static_cast<int>(ntok), 256>>>(q, k, pos, ntok, n_heads, n_kv, hd, theta, 1.0f);
    CU_CHECK(cudaGetLastError());
}

void rope_backward(float* dq, float* dk, const i32* pos, i64 ntok,
                   int n_heads, int n_kv, int hd, float theta) {
    if (ntok <= 0) return;
    k_rope<<<static_cast<int>(ntok), 256>>>(dq, dk, pos, ntok, n_heads, n_kv, hd, theta, -1.0f);
    CU_CHECK(cudaGetLastError());
}

// ================================================================ swiglu
__global__ void k_swiglu_fwd(const float* g, const float* u, float* o, i64 n) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        float gv = g[i];
        o[i] = (gv / (1.0f + __expf(-gv))) * u[i];
    }
}

__global__ void k_swiglu_bwd(const float* g, const float* u, const float* dout,
                             float* dg, float* du, i64 n) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        float gv = g[i];
        float sg = 1.0f / (1.0f + __expf(-gv));
        float sl = gv * sg;
        float dsilu = sg * (1.0f + gv * (1.0f - sg));
        float d = dout[i];
        dg[i] += d * u[i] * dsilu;
        du[i] += d * sl;
    }
}

void swiglu_forward(const float* g, const float* u, float* o, i64 n) {
    if (n <= 0) return;
    k_swiglu_fwd<<<grid_for(n, 256), 256>>>(g, u, o, n);
    CU_CHECK(cudaGetLastError());
}

void swiglu_backward(const float* g, const float* u, const float* dout,
                     float* dg, float* du, i64 n) {
    if (n <= 0) return;
    k_swiglu_bwd<<<grid_for(n, 256), 256>>>(g, u, dout, dg, du, n);
    CU_CHECK(cudaGetLastError());
}

// ================================================================ cross entropy
// One block per row. Fused: online max, sum, loss and dlogits in a single pass over V,
// which avoids materialising a second [N, V] probability buffer.
// z_scale adds z-loss: loss += z*logZ^2, grad += 2*z*logZ*p/n_valid.
__global__ void k_ce(const float* logits, const i32* targets, float* dlogits,
                     int V, float* loss_out, float invc, float z_scale) {
    extern __shared__ float smem[];
    i64 r = blockIdx.x;
    const float* row = logits + r * V;
    i32 tgt = targets[r];
    float* d = dlogits ? dlogits + r * V : nullptr;

    if (tgt < 0 || tgt >= V) {
        if (d) for (int j = threadIdx.x; j < V; j += blockDim.x) d[j] = 0.0f;
        if (threadIdx.x == 0) loss_out[r] = 0.0f;
        return;
    }

    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, row[j]);
    mx = block_max(mx, smem);

    float sum = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) sum += __expf(row[j] - mx);
    sum = block_sum(sum, smem);

    float logZ = __logf(sum) + mx;
    if (threadIdx.x == 0) loss_out[r] = (logZ - row[tgt]) + z_scale * logZ * logZ;

    if (d) {
        float invsum = 1.0f / sum;
        float zcorr = (z_scale != 0.0f) ? (2.0f * z_scale * logZ * invc) : 0.0f;
        float scale = invc + zcorr;
        for (int j = threadIdx.x; j < V; j += blockDim.x) {
            d[j] = __expf(row[j] - mx) * invsum * scale;
        }
        __syncthreads();
        if (threadIdx.x == 0) d[tgt] -= invc;
    }
}

__global__ void k_count_valid(const i32* targets, i64 n, int V, int* out) {
    extern __shared__ float smem[];
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    float c = 0.0f;
    for (; i < n; i += stride) if (targets[i] >= 0 && targets[i] < V) c += 1.0f;
    c = block_sum(c, smem);
    if (threadIdx.x == 0) atomicAdd(out, int(c));
}

__global__ void k_sum_partial(const float* x, i64 n, float* out) {
    extern __shared__ float smem[];
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    float s = 0.0f;
    for (; i < n; i += stride) s += x[i];
    s = block_sum(s, smem);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

void softmax_cross_entropy(const float* logits, const i32* targets, float* dlogits,
                           i64 n, int V, double* out_loss_sum, i64* out_count,
                           float z_scale) {
    if (n <= 0) {
        if (out_loss_sum) *out_loss_sum = 0.0;
        if (out_count) *out_count = 0;
        return;
    }
    // buffers: [n floats losses] [1 int count] [grid floats partial]
    const int block = 256;
    const int red_grid = 64;
    size_t need = sizeof(float) * size_t(n) + sizeof(int) * 4 + sizeof(float) * red_grid;
    char* ws = static_cast<char*>(workspace(need));
    float* losses = reinterpret_cast<float*>(ws);
    int*   dcount = reinterpret_cast<int*>(ws + sizeof(float) * size_t(n));
    float* partial= reinterpret_cast<float*>(ws + sizeof(float) * size_t(n) + sizeof(int) * 4);

    CU_CHECK(cudaMemset(dcount, 0, sizeof(int)));
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    k_count_valid<<<red_grid, block, sh>>>(targets, n, V, dcount);
    CU_CHECK(cudaGetLastError());

    int hcount = 0;
    CU_CHECK(cudaMemcpy(&hcount, dcount, sizeof(int), cudaMemcpyDeviceToHost));
    if (out_count) *out_count = hcount;
    float invc = hcount > 0 ? 1.0f / float(hcount) : 0.0f;

    int ce_block = V >= 2048 ? 512 : 256;
    size_t ce_sh = sizeof(float) * ((ce_block + WARP - 1) / WARP + 1);
    k_ce<<<static_cast<int>(n), ce_block, ce_sh>>>(logits, targets, dlogits, V, losses, invc, z_scale);
    CU_CHECK(cudaGetLastError());

    k_sum_partial<<<red_grid, block, sh>>>(losses, n, partial);
    CU_CHECK(cudaGetLastError());

    float hp[red_grid];
    CU_CHECK(cudaMemcpy(hp, partial, sizeof(float) * red_grid, cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < red_grid; ++i) total += hp[i];
    if (out_loss_sum) *out_loss_sum = total;
}

// ================================================================ adamw
__global__ void k_adamw(float* w, const float* g, float* m, float* v, i64 n,
                        float lr, float b1, float b2, float eps, float wd,
                        float bc1, float bc2, float gscale) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        float gi = g[i] * gscale;
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float upd = (mi / bc1) / (sqrtf(vi / bc2) + eps);
        if (wd != 0.0f) upd += wd * w[i];
        w[i] -= lr * upd;
    }
}

void adamw_step(float* w, const float* g, float* m, float* v, i64 n,
                float lr, float b1, float b2, float eps, float wd,
                float bc1, float bc2, float gscale) {
    if (n <= 0) return;
    k_adamw<<<grid_for(n, 256), 256>>>(w, g, m, v, n, lr, b1, b2, eps, wd, bc1, bc2, gscale);
    CU_CHECK(cudaGetLastError());
}

__global__ void k_lion(float* w, const float* g, float* m, i64 n,
                       float lr, float b1, float b2, float wd, float gscale) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        float gi = g[i] * gscale;
        float mi = m[i];
        float c = b1 * mi + (1.0f - b1) * gi;
        float upd = (c > 0.0f) ? 1.0f : ((c < 0.0f) ? -1.0f : 0.0f);
        if (wd != 0.0f) upd += wd * w[i];
        w[i] -= lr * upd;
        m[i] = b2 * mi + (1.0f - b2) * gi;
    }
}

void lion_step(float* w, const float* g, float* m, i64 n,
               float lr, float b1, float b2, float wd, float gscale) {
    if (n <= 0) return;
    k_lion<<<grid_for(n, 256), 256>>>(w, g, m, n, lr, b1, b2, wd, gscale);
    CU_CHECK(cudaGetLastError());
}

__global__ void k_sqnorm(const float* g, i64 n, float* out) {
    extern __shared__ float smem[];
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    float s = 0.0f;
    for (; i < n; i += stride) { float v = g[i]; s += v * v; }
    s = block_sum(s, smem);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// Scale gradients in-place by a constant factor (for distributed training)
__global__ void k_scale_grad(float* g, i64 n, float scale) {
    i64 i = blockIdx.x * i64(blockDim.x) + threadIdx.x;
    i64 stride = i64(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) g[i] *= scale;
}

void scale_grad(float* g, i64 n, float scale) {
    if (n <= 0) return;
    k_scale_grad<<<grid_for(n, 256), 256>>>(g, n, scale);
    CU_CHECK(cudaGetLastError());
}

double global_sq_norm(const float* g, i64 n) {
    if (n <= 0) return 0.0;
    const int block = 256;
    const int grid = 64;
    float* partial = static_cast<float*>(workspace(sizeof(float) * grid));
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    k_sqnorm<<<grid, block, sh>>>(g, n, partial);
    CU_CHECK(cudaGetLastError());
    float hp[grid];
    CU_CHECK(cudaMemcpy(hp, partial, sizeof(float) * grid, cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (int i = 0; i < grid; ++i) s += hp[i];
    return s;
}

// FIX (10/10): fused norm — all per-tensor tiles launch async into slices of
// one monotonic workspace buffer, then a SINGLE D2H + host reduce.
// 200 params: 200 launches + 1 sync instead of 200 syncs (~15ms saved/step
// on T4, the dominant optimizer overhead after MoE grouping).
double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts) {
    const int block = 256;
    const int grid = 16; // smaller per-tensor grid: 16*200=3200 partials max
    size_t nparts = 0;
    for (auto& pr : parts) if (pr.second > 0) ++nparts;
    if (nparts == 0) return 0.0;
    float* partial = static_cast<float*>(workspace(sizeof(float) * grid * nparts));
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    size_t off = 0;
    for (auto& pr : parts) {
        if (pr.second <= 0) continue;
        k_sqnorm<<<grid, block, sh>>>(pr.first, pr.second, partial + off);
        off += grid;
    }
    CU_CHECK(cudaGetLastError());
    std::vector<float> hp(off);
    CU_CHECK(cudaMemcpy(hp.data(), partial, sizeof(float) * off, cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (float v : hp) s += (double)v;
    return s;
}

} // namespace cuda_ops
} // namespace gai
