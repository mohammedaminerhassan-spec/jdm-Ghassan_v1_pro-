#include "cuda/cuda_utils.h"
#include "core/device.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace gai {
namespace cuda {

static std::string  g_last_error;
static cublasHandle_t g_cublas = nullptr;
static bool         g_initialized = false;

const char* last_error() { return g_last_error.c_str(); }

void check(cudaError_t e, const char* what, const char* file, int line) {
    if (e == cudaSuccess) return;
    g_last_error = std::string(what) + ": " + cudaGetErrorString(e);
    GAI_FAIL(std::string(file) + ":" + std::to_string(line) + " CUDA " + g_last_error);
}

#define CUDA_CHECK(x) ::gai::cuda::check((x), #x, __FILE__, __LINE__)

static void check_blas(cublasStatus_t s, const char* what) {
    if (s == CUBLAS_STATUS_SUCCESS) return;
    GAI_FAIL(std::string("cuBLAS failure in ") + what + " (status " + std::to_string(int(s)) + ")");
}

void probe(DeviceInfo& info) {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count <= 0) {
        g_last_error = (e == cudaSuccess) ? "no CUDA device" : cudaGetErrorString(e);
        info.cuda_available = false;
        return;
    }

    // DDP: each rank pins its own GPU via LOCAL_RANK (torchrun/Kaggle).
    // Without it, every rank would pile onto the same device.
    int best = 0;
    size_t best_mem = 0;
    if (const char* lr = std::getenv("LOCAL_RANK")) {
        int want = std::atoi(lr);
        if (want >= 0 && want < count) {
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, want) == cudaSuccess) {
                best = want;
                best_mem = p.totalGlobalMem;
            }
        }
    }
    if (best_mem == 0) {
        // pick the device with the most memory
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp p{};
            if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
            if (p.totalGlobalMem > best_mem) { best_mem = p.totalGlobalMem; best = i; }
        }
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, best) != cudaSuccess) {
        info.cuda_available = false;
        return;
    }
    if (cudaSetDevice(best) != cudaSuccess) {
        info.cuda_available = false;
        return;
    }

    size_t freem = 0, totalm = 0;
    cudaMemGetInfo(&freem, &totalm);

    info.cuda_available = true;
    info.device_count   = count;
    info.device_index   = best;
    info.name           = prop.name;
    info.cc_major       = prop.major;
    info.cc_minor       = prop.minor;
    info.total_mem      = totalm;
    info.free_mem       = freem;
    info.supports_fp16  = (prop.major > 5) || (prop.major == 5 && prop.minor >= 3);
    info.supports_bf16  = (prop.major >= 8);          // Ampere and newer
    info.supports_tf32  = (prop.major >= 8);
    g_initialized = true;
}

void* malloc_device(size_t nbytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, nbytes));
    return p;
}

void free_device(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void memset_zero(void* ptr, size_t nbytes) {
    CUDA_CHECK(cudaMemset(ptr, 0, nbytes));
}

void copy(void* dst, bool dst_dev, const void* src, bool src_dev, size_t nbytes) {
    cudaMemcpyKind kind = cudaMemcpyHostToHost;
    if (dst_dev && src_dev)        kind = cudaMemcpyDeviceToDevice;
    else if (dst_dev && !src_dev)  kind = cudaMemcpyHostToDevice;
    else if (!dst_dev && src_dev)  kind = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, kind));
}

void synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void* cublas_handle() {
    if (!g_cublas) {
        check_blas(cublasCreate(&g_cublas), "cublasCreate");
        // TF32 is a large free speedup on Ampere+ and is numerically fine for
        // this model size; explicitly opt in unless GAI_TF32=0 (bit-exact fp32).
        // FIX (10/10): T4 is sm75 (no TF32 cores) — old code set TF32 math
        // unconditionally, which is a no-op on T4 but misleads profiling.
        // Auto-detect: TF32 only on sm>=80, otherwise DEFAULT_MATH.
        const char* tf = std::getenv("GAI_TF32");
        bool want_tf32 = true;
        if (tf && (tf[0] == '0' || tf[0] == 'n' || tf[0] == 'N')) want_tf32 = false;
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            if (prop.major < 8) want_tf32 = false; // Turing/Pascal: no TF32
        }
        cublasSetMathMode(g_cublas, want_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH
                                              : CUBLAS_DEFAULT_MATH);
    }
    return reinterpret_cast<void*>(g_cublas);
}

void shutdown() {
    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = nullptr;
    }
    if (g_initialized) cudaDeviceReset();
    g_initialized = false;
}

} // namespace cuda
} // namespace gai
