#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"
#include "core/ops.h"

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

static void*  g_ws = nullptr;
static size_t g_ws_bytes = 0;

struct SceAcc;
static SceAcc* g_sce_acc = nullptr;

static size_t round_ws(size_t bytes) {
    const size_t chunk = (64ull << 20);
    return ((bytes + chunk - 1) / chunk) * chunk;
}

static void* workspace(size_t bytes) {
    if (bytes <= g_ws_bytes) return g_ws;
    size_t want = round_ws(bytes);
    if (want <= g_ws_bytes) return g_ws;
    if (g_ws) CU_CHECK(cudaFree(g_ws));
    gai::cuda::check_free_vram(want, "GEMM workspace");
    CU_CHECK(cudaMalloc(&g_ws, want));
    g_ws_bytes = want;
    return g_ws;
}

void reserve_workspaces(size_t gemm_bytes, size_t moe_bytes) {
    if (gemm_bytes > 0) workspace(gemm_bytes);
    if (moe_bytes > 0) moe_reserve_workspace(moe_bytes);
}

void free_sampling_workspace();
void free_workspace() {
    if (g_ws) cudaFree(g_ws);
    g_ws = nullptr;
    g_ws_bytes = 0;
    if (g_sce_acc) { cudaFree(g_sce_acc); g_sce_acc = nullptr; }
    free_sampling_workspace();
    moe_free_workspace();
}

size_t pool_bytes() { return g_ws_bytes; }

static i32* g_pen_hist = nullptr;
static int* g_pen_freq = nullptr;
static int g_pen_freq_cap = 0;
static i32* g_argmax_id = nullptr;
static float* g_topk_tv = nullptr;
static i32* g_topk_ti = nullptr;
static int g_topk_cap = 0;

void free_sampling_workspace() {
    if (g_pen_hist) { cudaFree(g_pen_hist); g_pen_hist = nullptr; }
    if (g_pen_freq) { cudaFree(g_pen_freq); g_pen_freq = nullptr; }
    g_pen_freq_cap = 0;
    if (g_argmax_id) { cudaFree(g_argmax_id); g_argmax_id = nullptr; }
    if (g_topk_tv) { cudaFree(g_topk_tv); g_topk_tv = nullptr; }
    if (g_topk_ti) { cudaFree(g_topk_ti); g_topk_ti = nullptr; }
    g_topk_cap = 0;
}

static bool g_fp16_gemm = true;
void set_fp16_gemm(bool on) { g_fp16_gemm = on; }
bool fp16_gemm_enabled() { return g_fp16_gemm; }

static bool g_bf16_gemm = false;
void set_bf16_gemm(bool on) { g_bf16_gemm = on; }
bool bf16_gemm_enabled() { return g_bf16_gemm; }

__global__ void k_f32_to_f16_strided(const float* src, __half* dst,
                                     int rows, int cols, int ld_src) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 n = (i64)rows * cols;
    if (i >= n) return;
    int r = (int)(i / cols), c = (int)(i % cols);
    dst[i] = __float2half(src[(i64)r * ld_src + c]);
}

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

    cublasHandle_t h = reinterpret_cast<cublasHandle_t>(cuda::cublas_handle());

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

#if __CUDACC_VER_MAJOR__ >= 11
static void gemm_bf16(bool trans_a, bool trans_b, int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb, float beta, float* C, int ldc) {

    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        if (prop.major < 8)
            GAI_FAIL("gemm_bf16 on sm_75 (T4): no BF16 tensor cores (need Ampere+). "
                     "Use fp32 masters + fp16 GEMMs instead.");
    }
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
#endif

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

static const char* cublas_status_name(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "NOT_SUPPORTED";
        default:                             return "UNKNOWN";
    }
}

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

    const i64 mnk = (i64)M * N * K;
    const i64 mnk_thr = ops::gemm_fp16_mnk_threshold();
    if (g_fp16_gemm && mnk >= mnk_thr) {
        ops::perf_note_fp16_gemm();
        gemm_fp16(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

#if __CUDACC_VER_MAJOR__ >= 11
    if (g_bf16_gemm && mnk >= mnk_thr) {
        ops::perf_note_fp16_gemm();
        gemm_bf16(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#else
    (void)g_bf16_gemm;
#endif
    cublasHandle_t h = reinterpret_cast<cublasHandle_t>(cuda::cublas_handle());
    cublasOperation_t opA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasStatus_t s = cublasSgemm(h, opA, opB, N, M, K,
                                   &alpha, B, ldb, A, lda, &beta, C, ldc);
    if (s != CUBLAS_STATUS_SUCCESS) {

        int cur_dev = -1;
        cudaGetDevice(&cur_dev);
        GAI_FAIL(strfmt("cublasSgemm failed: status=%d (%s) | M=%d N=%d K=%d "
                        "lda=%d ldb=%d ldc=%d | handle-device=%d current-device=%d | "
                        "hint: EXECUTION_FAILED with sane dims points at a wedged GPU "
                        "(nvidia-smi -r or session restart) or a prior async fault; "
                        "NOT_INITIALIZED with mismatched devices points at a device slip; "
                        "INVALID_VALUE points at a host-side shape bug",
                        static_cast<int>(s), cublas_status_name(s),
                        M, N, K, lda, ldb, ldc,
                        cuda::cublas_device(), cur_dev));
    }
}

void linear_forward(const float* x, const float* w, float* y, int M, int K, int N) {
    gemm(false, true, M, N, K, 1.0f, x, K, w, K, 0.0f, y, N);
}

__global__ void k_f32_to_f16_flat(const float* src, __half* dst, i64 n) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = __float2half(src[i]);
}

void convert_f32_to_f16(const float* src, u16* dst, i64 n) {
    if (n <= 0) return;
    k_f32_to_f16_flat<<<grid_for(n, 256), 256>>>(src, reinterpret_cast<__half*>(dst), n);
    CU_CHECK(cudaGetLastError());
}

void linear_forward_fp16(const float* x, const u16* w, float* y, int M, int K, int N) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        CU_CHECK(cudaMemset(y, 0, sizeof(float) * static_cast<size_t>(M) * N));
        return;
    }
    __half* xh = static_cast<__half*>(workspace(sizeof(__half) * static_cast<size_t>(M) * K));
    convert_f32_to_f16(x, reinterpret_cast<u16*>(xh), static_cast<i64>(M) * K);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t h = reinterpret_cast<cublasHandle_t>(cuda::cublas_handle());
    cublasStatus_t status = cublasGemmEx(h,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         N, M, K,
                                         &alpha,
                                         w, CUDA_R_16F, K,
                                         xh, CUDA_R_16F, K,
                                         &beta,
                                         y, CUDA_R_32F, N,
                                         CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) GAI_FAIL("cublasGemmEx fp16 linear failed");
}

__global__ void k_split_qkv(const float* qkv, float* q, float* k, float* v,
                            i64 n, int qd, int kvd) {
    const i64 width = (i64)qd + 2 * kvd;
    i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 stride = (i64)gridDim.x * blockDim.x;
    for (; idx < n * width; idx += stride) {
        i64 t = idx / width;
        int c = (int)(idx % width);
        float value = qkv[idx];
        if (c < qd) {
            q[t * qd + c] = value;
        } else if (c < qd + kvd) {
            k[t * kvd + (c - qd)] = value;
        } else {
            v[t * kvd + (c - qd - kvd)] = value;
        }
    }
}

void split_qkv(const float* qkv, float* q, float* k, float* v, i64 n, int qd, int kvd) {
    if (n <= 0) return;
    k_split_qkv<<<grid_for(n * ((i64)qd + 2 * kvd), 256), 256>>>(qkv, q, k, v, n, qd, kvd);
    CU_CHECK(cudaGetLastError());
}

void linear_backward(const float* x, const float* w, const float* dy,
                     float* dx, float* dw, int M, int K, int N) {
    if (dx) gemm(false, false, M, K, N, 1.0f, dy, N, w, K, 1.0f, dx, K);
    if (dw) gemm(true,  false, N, K, M, 1.0f, dy, N, x, K, 1.0f, dw, K);
}

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

    GAI_CHECK(ntok <= 2147483647LL, "embedding_forward: ntok exceeds INT_MAX");
    int block = dim >= 256 ? 256 : ((dim + 31) / 32) * 32;
    if (block < 32) block = 32;
    k_embed_fwd<<<static_cast<int>(ntok), block>>>(ids, table, out, ntok, dim, vocab);
    CU_CHECK(cudaGetLastError());
}

void embedding_backward(const i32* ids, const float* dout, float* dtable,
                        i64 ntok, int dim, int vocab) {
    if (ntok <= 0) return;
    GAI_CHECK(ntok <= 2147483647LL, "embedding_backward: ntok exceeds INT_MAX");
    int block = dim >= 256 ? 256 : ((dim + 31) / 32) * 32;
    if (block < 32) block = 32;
    k_embed_bwd<<<static_cast<int>(ntok), block>>>(ids, dout, dtable, ntok, dim, vocab);
    CU_CHECK(cudaGetLastError());
}

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

    if (!isfinite(inv)) inv = 0.0f;
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

__global__ void k_rmsnorm_bwd_dw(const float* x, const float* dout, const float* rrms,
                                 float* dweight, i64 rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double acc = 0.0;
    for (i64 r = 0; r < rows; ++r)
        acc += (double)dout[r * dim + i] * (double)x[r * dim + i] * (double)rrms[r];
    dweight[i] += (float)acc;
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
    GAI_CHECK(rows <= 2147483647LL, "rmsnorm_forward: rows exceeds INT_MAX");
    int block = norm_block(dim);
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    k_rmsnorm_fwd<<<static_cast<int>(rows), block, sh>>>(x, w, out, rrms, dim, eps);
    CU_CHECK(cudaGetLastError());
}

void rmsnorm_backward(const float* x, const float* w, const float* dout,
                      const float* rrms, float* dx, float* dweight, i64 rows, int dim) {
    if (rows <= 0) return;
    GAI_CHECK(rows <= 2147483647LL, "rmsnorm_backward: rows exceeds INT_MAX");
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

        double freq_d = pow((double)theta, -(2.0 * (double)i) / (double)hd);
        double ang_d  = (double)p * freq_d;
        float c, s;
        sincosf((float)ang_d, &s, &c);
        s *= sign;

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
    GAI_CHECK(ntok <= 2147483647LL, "rope_forward: ntok exceeds INT_MAX");
    k_rope<<<static_cast<int>(ntok), 256>>>(q, k, pos, ntok, n_heads, n_kv, hd, theta, 1.0f);
    CU_CHECK(cudaGetLastError());
}

void rope_backward(float* dq, float* dk, const i32* pos, i64 ntok,
                   int n_heads, int n_kv, int hd, float theta) {
    if (ntok <= 0) return;
    GAI_CHECK(ntok <= 2147483647LL, "rope_backward: ntok exceeds INT_MAX");
    k_rope<<<static_cast<int>(ntok), 256>>>(dq, dk, pos, ntok, n_heads, n_kv, hd, theta, -1.0f);
    CU_CHECK(cudaGetLastError());
}

__global__ void k_rope_ex(float* q, float* k, const i32* pos, i64 ntok,
                          int n_heads, int n_kv, int hd, float theta, float sign,
                          int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
    int half = hd / 2;
    i64 t = blockIdx.x;
    if (t >= ntok) return;
    float p = float(pos[t]);

    int total = (n_heads + n_kv) * half;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int i = idx % half;
        int h = idx / half;
        double base_d = pow((double)theta, -(2.0 * (double)i) / (double)hd);
        if (yarn_scale > 1.0) {
            double wl = 6.28318530717959 / base_d;
            if (wl > (double)yarn_high) base_d /= (double)yarn_scale;
            else if (wl > (double)yarn_low) {
                double span = (double)yarn_high - (double)yarn_low;
                double tt = span > 0.0 ? (wl - (double)yarn_low) / span : 1.0;
                base_d *= (1.0 - tt + tt / (double)yarn_scale);
            }
        }
        double ang_d  = (double)p * base_d;
        float c, s;
        sincosf((float)ang_d, &s, &c);
        s *= sign;

        float* base;
        if (h < n_heads) {
            if (!q) continue;
            base = q + (t * n_heads + h) * hd;
        } else {
            if (!k) continue;
            base = k + (t * n_kv + (h - n_heads)) * hd;
        }
        if (rope_type == 1) {
            float a = base[i], b = base[i + half];
            base[i]        = a * c - b * s;
            base[i + half] = a * s + b * c;
        } else {
            float a = base[2 * i], b = base[2 * i + 1];
            base[2 * i]     = a * c - b * s;
            base[2 * i + 1] = a * s + b * c;
        }
    }
}

__global__ void k_rope_freq(float* q, float* k, const i32* pos, const float* inv_freq,
                            i64 ntok, int n_heads, int n_kv, int hd, int rope_type,
                            float sign) {
    const int half = hd / 2;
    i64 t = blockIdx.x;
    if (t >= ntok) return;
    const int total = (n_heads + n_kv) * half;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int i = idx % half;
        int h = idx / half;
        float angle = static_cast<float>(pos[t]) * inv_freq[i];
        float sn;
        float cs;
        sincosf(angle, &sn, &cs);
        sn *= sign;
        float* base;
        if (h < n_heads) {
            if (!q) continue;
            base = q + (t * n_heads + h) * hd;
        } else {
            if (!k) continue;
            base = k + (t * n_kv + (h - n_heads)) * hd;
        }
        if (rope_type == 1) {
            float a = base[i];
            float b = base[i + half];
            base[i] = a * cs - b * sn;
            base[i + half] = a * sn + b * cs;
        } else {
            float a = base[2 * i];
            float b = base[2 * i + 1];
            base[2 * i] = a * cs - b * sn;
            base[2 * i + 1] = a * sn + b * cs;
        }
    }
}

void rope_forward_ex(float* q, float* k, const i32* pos, i64 ntok,
                     int n_heads, int n_kv, int hd, float theta,
                     int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
    if (ntok <= 0) return;
    GAI_CHECK(ntok <= 2147483647LL, "rope_forward_ex: ntok exceeds INT_MAX");
    k_rope_ex<<<static_cast<int>(ntok), 256>>>(q, k, pos, ntok, n_heads, n_kv, hd, theta, 1.0f,
                                               rope_type, yarn_low, yarn_high, yarn_scale);
    CU_CHECK(cudaGetLastError());
}

void rope_backward_ex(float* dq, float* dk, const i32* pos, i64 ntok,
                      int n_heads, int n_kv, int hd, float theta,
                      int rope_type, float yarn_low, float yarn_high, float yarn_scale) {
    if (ntok <= 0) return;
    GAI_CHECK(ntok <= 2147483647LL, "rope_backward_ex: ntok exceeds INT_MAX");
    k_rope_ex<<<static_cast<int>(ntok), 256>>>(dq, dk, pos, ntok, n_heads, n_kv, hd, theta, -1.0f,
                                               rope_type, yarn_low, yarn_high, yarn_scale);
    CU_CHECK(cudaGetLastError());
}

void rope_forward_cached(float* q, float* k, const i32* pos, const float* inv_freq,
                         i64 ntok, int n_heads, int n_kv, int hd, int rope_type) {
    if (ntok <= 0) return;
    GAI_CHECK(ntok <= 2147483647LL, "rope_forward_cached: ntok exceeds INT_MAX");
    k_rope_freq<<<static_cast<int>(ntok), 256>>>(q, k, pos, inv_freq, ntok,
                                                 n_heads, n_kv, hd, rope_type, 1.0f);
    CU_CHECK(cudaGetLastError());
}

void rope_backward_cached(float* dq, float* dk, const i32* pos, const float* inv_freq,
                          i64 ntok, int n_heads, int n_kv, int hd, int rope_type) {
    if (ntok <= 0) return;
    GAI_CHECK(ntok <= 2147483647LL, "rope_backward_cached: ntok exceeds INT_MAX");
    k_rope_freq<<<static_cast<int>(ntok), 256>>>(dq, dk, pos, inv_freq, ntok,
                                                 n_heads, n_kv, hd, rope_type, -1.0f);
    CU_CHECK(cudaGetLastError());
}

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

__global__ void k_ce(const float* logits, const i32* targets, float* dlogits,
                     int V, float* loss_out, int* d_count, float z_scale) {
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
    if (threadIdx.x == 0) atomicAdd(d_count, 1);

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
        float scale = 1.0f + ((z_scale != 0.0f) ? (2.0f * z_scale * logZ) : 0.0f);
        for (int j = threadIdx.x; j < V; j += blockDim.x) {
            d[j] = __expf(row[j] - mx) * invsum * scale;
        }
        __syncthreads();
        if (threadIdx.x == 0) d[tgt] -= 1.0f;
    }
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

    GAI_CHECK(n <= 2147483647LL, "cuda sce: rows exceed INT_MAX");

    const int block = 256;
    const int red_grid = 64;
    struct CeReduce { int count; float partial[64]; };
    static_assert(sizeof(CeReduce) == sizeof(int) + 64 * sizeof(float), "CeReduce packing");
    size_t need = sizeof(float) * size_t(n) + sizeof(CeReduce);
    char* ws = static_cast<char*>(workspace(need));
    float* losses = reinterpret_cast<float*>(ws);
    CeReduce* red = reinterpret_cast<CeReduce*>(ws + sizeof(float) * size_t(n));

    CU_CHECK(cudaMemset(red, 0, sizeof(CeReduce)));

    int ce_block = V >= 2048 ? 512 : 256;
    size_t ce_sh = sizeof(float) * ((ce_block + WARP - 1) / WARP + 1);
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    k_ce<<<static_cast<int>(n), ce_block, ce_sh>>>(logits, targets, dlogits, V,
                                                   losses, &red->count, z_scale);
    CU_CHECK(cudaGetLastError());

    k_sum_partial<<<red_grid, block, sh>>>(losses, n, red->partial);
    CU_CHECK(cudaGetLastError());

    CeReduce hred;
    CU_CHECK(cudaMemcpy(&hred, red, sizeof(CeReduce), cudaMemcpyDeviceToHost));
    if (out_count) *out_count = hred.count;
    double total = 0.0;
    for (int i = 0; i < red_grid; ++i) total += hred.partial[i];
    if (out_loss_sum) *out_loss_sum = total;
}

struct SceAcc { double loss; long long count; };

static void sce_acc_ensure() {
    if (!g_sce_acc) CU_CHECK(cudaMalloc(&g_sce_acc, sizeof(SceAcc)));
}
void sce_acc_begin() {
    sce_acc_ensure();
    CU_CHECK(cudaMemset(g_sce_acc, 0, sizeof(SceAcc)));
}

__global__ void k_sce_fold(const float* partial, const int* count,
                           SceAcc* acc) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double s = 0.0;
    for (int i = 0; i < 64; ++i) s += (double)partial[i];
    acc->loss += s;
    acc->count += (long long)*count;
}

void sce_accumulate(const float* logits, const i32* targets, float* dlogits,
                    i64 n, int V, float z_scale) {
    if (n <= 0) return;
    GAI_CHECK(n <= 2147483647LL, "cuda sce_accumulate: rows exceed INT_MAX");
    sce_acc_ensure();

    const int block = 256;
    const int red_grid = 64;
    struct CeReduce { int count; float partial[64]; };
    size_t need = sizeof(float) * size_t(n) + sizeof(CeReduce);
    char* ws = static_cast<char*>(workspace(need));
    float* losses = reinterpret_cast<float*>(ws);
    CeReduce* red = reinterpret_cast<CeReduce*>(ws + sizeof(float) * size_t(n));

    CU_CHECK(cudaMemset(red, 0, sizeof(CeReduce)));

    int ce_block = V >= 2048 ? 512 : 256;
    size_t ce_sh = sizeof(float) * ((ce_block + WARP - 1) / WARP + 1);
    size_t sh = sizeof(float) * ((block + WARP - 1) / WARP + 1);
    k_ce<<<static_cast<int>(n), ce_block, ce_sh>>>(logits, targets, dlogits, V,
                                                   losses, &red->count, z_scale);
    CU_CHECK(cudaGetLastError());

    k_sum_partial<<<red_grid, block, sh>>>(losses, n, red->partial);
    CU_CHECK(cudaGetLastError());

    k_sce_fold<<<1, 1>>>(red->partial, &red->count, g_sce_acc);
    CU_CHECK(cudaGetLastError());
}

void sce_acc_end(double* out_loss_sum, i64* out_count) {
    sce_acc_ensure();

    SceAcc hacc{0.0, 0};
    CU_CHECK(cudaMemcpy(&hacc, g_sce_acc, sizeof(SceAcc), cudaMemcpyDeviceToHost));
    if (out_loss_sum) *out_loss_sum = hacc.loss;
    if (out_count) *out_count = (i64)hacc.count;
}

__global__ void k_rep_hist(const i32* hist, int n, int* freq, int V) {
    int i = (int)((i64)blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int t = hist[i];
    if (t >= 0 && t < V) atomicAdd(&freq[t], 1);
}
__global__ void k_rep_apply(float* logits, int V, const int* freq,
                            float rep, float freq_p, float pres) {
    int i = (int)((i64)blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= V) return;
    int cnt = freq[i];
    if (cnt == 0) return;
    float l = logits[i];
    if (rep != 1.0f) l = (l > 0.0f) ? l / rep : l * rep;
    if (freq_p != 0.0f) l -= freq_p * (float)cnt;
    if (pres != 0.0f) l -= pres;
    logits[i] = l;
}

void apply_rep_penalties(float* logits, int V, const i32* hist, int hist_n,
                         float rep, float freq, float pres) {
    if (V <= 0 || hist_n <= 0) return;
    if (rep == 1.0f && freq == 0.0f && pres == 0.0f) return;
    GAI_CHECK(hist_n <= 2048, "penalties: history window exceeds 2048 (use legacy path)");
    if (!g_pen_hist) CU_CHECK(cudaMalloc(&g_pen_hist, sizeof(i32) * 2048));
    if (g_pen_freq_cap < V) {
        if (g_pen_freq) { CU_CHECK(cudaFree(g_pen_freq)); g_pen_freq = nullptr; }
        CU_CHECK(cudaMalloc(&g_pen_freq, sizeof(int) * (size_t)V));
        g_pen_freq_cap = V;
    }

    CU_CHECK(cudaMemcpy(g_pen_hist, hist, sizeof(i32) * (size_t)hist_n,
                        cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemset(g_pen_freq, 0, sizeof(int) * (size_t)V));
    k_rep_hist<<<grid_for((i64)hist_n, 256), 256>>>(g_pen_hist, hist_n, g_pen_freq, V);
    CU_CHECK(cudaGetLastError());
    k_rep_apply<<<grid_for((i64)V, 256), 256>>>(logits, V, g_pen_freq,
                                                rep, freq, pres);
    CU_CHECK(cudaGetLastError());
}

__global__ void k_argmax(const float* x, int V, i32* out_id) {
    __shared__ float sv[1024];
    __shared__ i32 si[1024];
    int t = (int)threadIdx.x;
    float bv = -FLT_MAX;
    i32 bi = 2147483647;
    for (int i = t; i < V; i += 1024) {
        float v = x[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    sv[t] = bv;
    si[t] = bi;
    __syncthreads();
    for (int s = 512; s > 0; s >>= 1) {
        if (t < s) {
            float a = sv[t], b = sv[t + s];
            i32 ia = si[t], ib = si[t + s];
            if (b > a || (b == a && ib < ia)) { sv[t] = b; si[t] = ib; }
        }
        __syncthreads();
    }
    if (t == 0) *out_id = si[0];
}

i32 argmax_token(const float* logits, int V) {
    GAI_CHECK(V > 0, "argmax_token: empty vocabulary");
    if (!g_argmax_id) CU_CHECK(cudaMalloc(&g_argmax_id, sizeof(i32)));
    k_argmax<<<1, 1024>>>(logits, V, g_argmax_id);
    CU_CHECK(cudaGetLastError());
    i32 h = 0;
    CU_CHECK(cudaMemcpy(&h, g_argmax_id, sizeof(i32), cudaMemcpyDeviceToHost));
    return h;
}

__global__ void k_topk_s1(const float* x, int V, float* tvals, i32* tids, int K) {
    __shared__ float sv[512];
    __shared__ i32 si[512];
    int t = (int)threadIdx.x;
    int b = (int)blockIdx.x;
    int chunk = (V + 63) / 64;
    int lo = b * chunk;
    int hi = lo + chunk < V ? lo + chunk : V;
    float v0 = -FLT_MAX, v1 = -FLT_MAX;
    i32 i0 = 0, i1 = 0;
    for (int i = lo + t; i < hi; i += 256) {
        float v = x[i];
        if (v > v0) { v1 = v0; i1 = i0; v0 = v; i0 = i; }
        else if (v > v1) { v1 = v; i1 = i; }
    }
    sv[2 * t] = v0; si[2 * t] = i0; sv[2 * t + 1] = v1; si[2 * t + 1] = i1;
    __syncthreads();
    if (t == 0) {

        for (int k = 0; k < K; ++k) {
            float bv = -FLT_MAX;
            i32 bi = 0;
            int bj = -1;
            for (int j = 0; j < 512; ++j)
                if (sv[j] > bv) { bv = sv[j]; bi = si[j]; bj = j; }
            tvals[b * K + k] = bv; tids[b * K + k] = bi;
            if (bj >= 0) sv[bj] = -FLT_MAX;
        }
    }
}

__global__ void k_topk_s2(float* tvals, i32* tids, int M,
                          float* ovals, i32* oids, int K) {

    if ((int)threadIdx.x == 0 && (int)blockIdx.x == 0) {
        for (int k = 0; k < K; ++k) {
            float bv = -FLT_MAX;
            i32 bi = 0;
            int bj = -1;
            for (int j = 0; j < M; ++j) {
                float v = tvals[j];
                if (v > bv) { bv = v; bi = tids[j]; bj = j; }
            }
            ovals[k] = bv;
            oids[k] = bi;
            if (bj >= 0) tvals[bj] = -FLT_MAX;
        }
    }
}

void topk_select(const float* logits, int V, int K, float* out_vals, i32* out_ids) {

    GAI_CHECK(K >= 1 && K <= 128, "topk_select: K out of range [1,128]");
    GAI_CHECK(V >= 64 * K, "topk_select: vocab too small for K (block filler would leak)");
    GAI_CHECK(V <= 64 * 512, "topk_select: vocab exceeds tiled kernel coverage");
    const int need = 64 * K;

    if (!g_topk_tv || !g_topk_ti || g_topk_cap < need) {
        if (g_topk_tv) { cudaFree(g_topk_tv); g_topk_tv = nullptr; }
        if (g_topk_ti) { cudaFree(g_topk_ti); g_topk_ti = nullptr; }
        CU_CHECK(cudaMalloc(&g_topk_tv, sizeof(float) * (size_t)(64 * 128)));
        CU_CHECK(cudaMalloc(&g_topk_ti, sizeof(i32) * (size_t)(64 * 128)));
        g_topk_cap = 64 * 128;
    }
    k_topk_s1<<<64, 256>>>(logits, V, g_topk_tv, g_topk_ti, K);
    CU_CHECK(cudaGetLastError());
    k_topk_s2<<<1, 256>>>(g_topk_tv, g_topk_ti, need, out_vals, out_ids, K);
    CU_CHECK(cudaGetLastError());
}

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

    double s = 0.0;
    for (; i < n; i += stride) { double v = (double)g[i]; s += v * v; }

    float sf = (float)s;
    sf = block_sum(sf, smem);
    if (threadIdx.x == 0) out[blockIdx.x] = sf;
}

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

double global_sq_norm_multi(const std::vector<std::pair<const float*, i64>>& parts) {
    const int block = 256;
    const int grid = 16;
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

}
}
