#include "core/device.h"

#include <cstdlib>
#include <cstring>
#include <new>

#ifdef GAI_CUDA
#include "cuda/cuda_utils.h"
#endif

namespace gai {

bool is_gpu(Device d) { return d == Device::CUDA; }

// ---------------------------------------------------------------- info
// T4-ONLY: CPU + CUDA sm_75. No Metal/Vulkan/TPU probing by design.
static DeviceInfo probe_device() {
    DeviceInfo info;
#ifdef GAI_CUDA
    cuda::probe(info);
#endif
    if (!info.cuda_available) {
        info.name = "cpu";
    }
    return info;
}

const DeviceInfo& device_info() {
    static DeviceInfo info = probe_device();
    return info;
}

bool cuda_available() { return device_info().cuda_available; }

Device best_device() {
    // T4-ONLY: CUDA when compiled + available, else CPU. No fallback guessing.
    if (cuda_available()) return Device::CUDA;
    return Device::CPU;
}

void print_device_report() {
    const DeviceInfo& d = device_info();
    log_info("---------------- device (T4-only) ----------------");
    if (d.cuda_available) {
        log_info(strfmt("  CUDA device %d/%d : %s (sm_%d%d)",
                        d.device_index, d.device_count, d.name.c_str(), d.cc_major, d.cc_minor));
        log_info(strfmt("  memory           : %s free / %s total",
                        human_bytes(d.free_mem).c_str(), human_bytes(d.total_mem).c_str()));
        log_info(strfmt("  fp16 %s | bf16 %s | tf32 %s",
                        d.supports_fp16 ? "yes" : "no",
                        d.supports_bf16 ? "yes" : "no",
                        d.supports_tf32 ? "yes" : "no"));
    } else {
        log_info("  GPU              : not available (CPU backend)");
    }
    log_info(strfmt("  cpu threads      : %d", num_threads()));
    log_info("----------------------------------------");
}

// ---------------------------------------------------------------- memory
static void* aligned_alloc_64(size_t n) {
    if (n == 0) return nullptr;
    size_t rounded = (n + 63) & ~static_cast<size_t>(63);
#if defined(_MSC_VER)
    void* p = _aligned_malloc(rounded, 64);
#else
    void* p = nullptr;
    #if defined(_WIN32)
        p = __mingw_aligned_malloc(rounded, 64);
    #else
        if (posix_memalign(&p, 64, rounded) != 0) p = nullptr;
    #endif
#endif
    if (!p) throw std::bad_alloc();
    return p;
}

static void aligned_free_64(void* p) {
    if (!p) return;
#if defined(_MSC_VER)
    _aligned_free(p);
#elif defined(_WIN32)
    __mingw_aligned_free(p);
#else
    free(p);
#endif
}

void* device_alloc(size_t nbytes, Device dev, DType /*dt*/) {
    if (nbytes == 0) return nullptr;
    if (dev == Device::CUDA) {
#ifdef GAI_CUDA
        return cuda::malloc_device(nbytes);
#else
        GAI_FAIL("CUDA allocation requested but the build has no CUDA support (CPU-only build)");
#endif
    }
    return aligned_alloc_64(nbytes);
}

void device_free(void* ptr, Device dev) {
    if (!ptr) return;
    if (dev == Device::CUDA) {
#ifdef GAI_CUDA
        cuda::free_device(ptr);
#endif
        return;
    }
    aligned_free_64(ptr);
}

void device_memset_zero(void* ptr, size_t nbytes, Device dev) {
    if (!ptr || nbytes == 0) return;
    if (dev == Device::CUDA) {
#ifdef GAI_CUDA
        cuda::memset_zero(ptr, nbytes);
#endif
        return;
    }
    std::memset(ptr, 0, nbytes);
}

void device_copy(void* dst, Device dst_dev, const void* src, Device src_dev, size_t nbytes) {
    if (nbytes == 0) return;
    if (dst == nullptr || src == nullptr) GAI_FAIL("device_copy: null pointer");
    if (dst_dev == Device::CPU && src_dev == Device::CPU) {
        std::memcpy(dst, src, nbytes);
        return;
    }
#ifdef GAI_CUDA
    if ((dst_dev == Device::CUDA || dst_dev == Device::CPU) &&
        (src_dev == Device::CUDA || src_dev == Device::CPU)) {
        cuda::copy(dst, dst_dev == Device::CUDA, src, src_dev == Device::CUDA, nbytes);
        return;
    }
#endif
    GAI_FAIL("device copy involving unsupported device combination (T4-only: cpu/cuda)");
}

void device_synchronize(Device dev) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) cuda::synchronize();
#else
    (void)dev;
#endif
}

} // namespace gai
