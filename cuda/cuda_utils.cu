#include "cuda/cuda_utils.h"
#include "cuda/cuda_ops.h"
#include "core/device.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace gai {
namespace cuda {

static std::string  g_last_error;
static cublasHandle_t g_cublas = nullptr;
static int          g_cublas_dev = -1;
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
    info.supports_bf16  = (prop.major >= 8);
    info.supports_tf32  = (prop.major >= 8);
    g_initialized = true;
}

void* malloc_device(size_t nbytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, nbytes));
    return p;
}

void free_device(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

void memset_zero(void* ptr, size_t nbytes) {
    CUDA_CHECK(cudaMemset(ptr, 0, nbytes));
}

bool is_device_memory(const void* ptr) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, const_cast<void*>(ptr)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeDevice;
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

size_t free_bytes_live() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return free_b;
}

void check_free_vram(size_t need, const char* what) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
    if (need > free_b) {
        GAI_FAIL(strfmt("CUDA OOM: %s needs %s but only %s free of %s total",
                        what ? what : "allocation",
                        human_bytes(need).c_str(),
                        human_bytes(free_b).c_str(),
                        human_bytes(total_b).c_str()));
    }
}

void* cublas_handle() {
    if (!g_cublas) {
        check_blas(cublasCreate(&g_cublas), "cublasCreate");

        if (cudaGetDevice(&g_cublas_dev) != cudaSuccess) g_cublas_dev = -1;

        const char* tf = std::getenv("GAI_TF32");
        bool want_tf32 = true;
        if (tf && (tf[0] == '0' || tf[0] == 'n' || tf[0] == 'N')) want_tf32 = false;
        int dev = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        if (prop.major < 8) want_tf32 = false;
        }
        check_blas(cublasSetMathMode(g_cublas, want_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH
                                                        : CUBLAS_DEFAULT_MATH),
                   "cublasSetMathMode");
    }
    return reinterpret_cast<void*>(g_cublas);
}

int cublas_device() { return g_cublas_dev; }

void shutdown() {
    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = nullptr;
    }
    cuda_ops::free_workspace();
    cuda_ops::free_sampling_workspace();
    cuda_ops::moe_free_workspace();
    cuda_ops::attn_free_dots();
    cuda_ops::phase_shutdown();

    if (g_initialized) {
        cudaDeviceSynchronize();
        const char* r = std::getenv("GAI_CUDA_RESET");
        if (r && (r[0] == '1' || r[0] == 'y' || r[0] == 'Y')) cudaDeviceReset();
    }
    g_initialized = false;
}

}  // namespace cuda

namespace cuda_ops {

// Async phase timers: stream-0 event pairs in a ring. Record calls never
// block the host; only phase_report() synchronizes (once per log line).
// Ring is sized for log_every<=40 at the largest recipes; single training
// thread per process is assumed (same contract as the MoE workspace).
namespace {
constexpr size_t kPhaseRing = 16384;
struct PhaseSlot { cudaEvent_t s = nullptr, e = nullptr; int ph = -1; bool used = false; };
std::vector<PhaseSlot> g_slots;
size_t g_head = 0;
void phase_pool() {
    if (!g_slots.empty()) return;
    g_slots.resize(kPhaseRing);
    for (size_t i = 0; i < kPhaseRing; ++i) {
        if (cudaEventCreate(&g_slots[i].s) != cudaSuccess ||
            cudaEventCreate(&g_slots[i].e) != cudaSuccess)
            GAI_FAIL("phase timer pool: cudaEventCreate failed");
    }
}
}  // namespace

void phase_start(int ph) {
    phase_pool();
    PhaseSlot& sl = g_slots[g_head];
    sl.ph = ph;
    sl.used = true;
    if (cudaEventRecord(sl.s, 0) != cudaSuccess)
        GAI_FAIL("phase_start: cudaEventRecord failed");
}
void phase_stop(int ph) {
    (void)ph;
    phase_pool();
    PhaseSlot& sl = g_slots[g_head];
    if (cudaEventRecord(sl.e, 0) != cudaSuccess)
        GAI_FAIL("phase_stop: cudaEventRecord failed");
    g_head = (g_head + 1) % kPhaseRing;
}
void phase_reset() {
    for (auto& sl : g_slots) { sl.used = false; sl.ph = -1; }
}
void phase_shutdown() {
    for (auto& sl : g_slots) {
        if (sl.s) { cudaEventDestroy(sl.s); sl.s = nullptr; }
        if (sl.e) { cudaEventDestroy(sl.e); sl.e = nullptr; }
        sl.used = false;
    }
    g_slots.clear();
    g_head = 0;
}
std::string phase_report() {
    static const char* names[] = {"attnF", "moeF", "attnB", "moeB", "load"};
    double ms[5] = {};
    for (auto& sl : g_slots) {
        if (!sl.used || sl.ph < 0 || sl.ph > 4) continue;
        float d = 0.0f;
        if (cudaEventElapsedTime(&d, sl.s, sl.e) != cudaSuccess)
            GAI_FAIL("phase_report: cudaEventElapsedTime failed");
        ms[sl.ph] += static_cast<double>(d);
        sl.used = false;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "ph [%s %.1f %s %.1f %s %.1f %s %.1f]",
                  names[0], ms[0], names[1], ms[1], names[2], ms[2], names[3], ms[3]);
    return std::string(buf);
}

}  // namespace cuda_ops
}
