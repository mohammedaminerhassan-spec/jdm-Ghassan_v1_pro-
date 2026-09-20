#include "core/device.h"
#include "core/ops.h"

#include <cstdlib>
#include <cstring>
#include <new>

#ifdef GAI_CUDA
#include "cuda/cuda_utils.h"
#endif

namespace gai {

bool is_gpu(Device d) { return d == Device::CUDA; }

// ---------------------------------------------------------------- info
// Portable backends: CPU (OpenMP, everywhere: Windows/Linux/macOS) +
// optional CUDA (NVIDIA GPUs; T4 sm_75 is the reference target, any arch
// works when built with matching -DCMAKE_CUDA_ARCHITECTURES).
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
    // CUDA when compiled + available, else portable CPU. The CPU backend is
    // the universal fallback (local PCs, macOS, Linux without NVIDIA).
    if (cuda_available()) return Device::CUDA;
    return Device::CPU;
}

void print_device_report() {
    const DeviceInfo& d = device_info();
    log_info("---------------- device (portable: cpu + cuda) ----------------");
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
        // PRO-HARDEN: فحص VRAM الحرة قبل cudaMalloc يعطي رسالة عملية
        // (أي B/T تخفض) بدل GAI_FAIL غامض وسط step يضيع ساعة T4.
        const DeviceInfo& di = device_info();
        if (di.cuda_available && di.free_mem > 0 && nbytes > di.free_mem) {
            GAI_FAIL(strfmt("CUDA OOM guard: need %s but only %s free. "
                            "Lower batch_size/seq_len/max_context (see dry-run).",
                            human_bytes(nbytes).c_str(), human_bytes(di.free_mem).c_str()));
        }
        return cuda::malloc_device(nbytes);
#else
        GAI_FAIL("CUDA allocation requested but the build has no CUDA support (CPU-only build)");
#endif
    }
    // CPU: رسالة أوضح من bad_alloc العاري عند RAM ضعيفة.
    try {
        return aligned_alloc_64(nbytes);
    } catch (const std::bad_alloc&) {
        GAI_FAIL(strfmt("CPU OOM: need %s. Close apps, lower batch_size/seq_len, "
                        "or use streaming shards.", human_bytes(nbytes).c_str()));
    }
    return nullptr;  // unreachable
}

void device_free(void* ptr, Device dev) {
    if (!ptr) return;
    if (dev == Device::CUDA) {
#ifdef GAI_CUDA
        cuda::free_device(ptr);
#else
        // PRO-HARDEN: الصمت هنا كان يخفي leak (مؤشر CUDA يضيع بلا تحرير).
        GAI_FAIL("device_free: CUDA pointer freed in a CPU-only build (leak hidden)");
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
#else
        GAI_FAIL("device_memset_zero: CUDA pointer in a CPU-only build (noop hidden)");
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
        // PERF telemetry: direction-tagged transfer bytes for the P2-2 report.
        if (dst_dev == Device::CUDA && src_dev == Device::CPU) ops::perf_note_h2d(nbytes);
        else if (dst_dev == Device::CPU && src_dev == Device::CUDA) ops::perf_note_d2h(nbytes);
        cuda::copy(dst, dst_dev == Device::CUDA, src, src_dev == Device::CUDA, nbytes);
        return;
    }
#endif
    GAI_FAIL("device copy involving unsupported device combination (only cpu/cuda exist)");
}

void device_synchronize(Device dev) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) cuda::synchronize();
#else
    (void)dev;
#endif
}

} // namespace gai
