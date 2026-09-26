#include "core/device.h"
#include "core/ops.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
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
        // FIX P2-1: old guard compared against stale startup free_mem
        // (device_info() is cached once). Query live VRAM so the guard
        // stays correct after GBs of weights/workspaces.
        // Only guard huge allocs (>64MB) to avoid a MemGetInfo per tiny tensor.
        if (nbytes > (64ull << 20)) {
            size_t live_free = cuda::free_bytes_live();
            if (live_free > 0 && nbytes > live_free) {
                GAI_FAIL(strfmt("CUDA OOM guard: need %s but only %s free live. "
                                "Lower batch_size/seq_len/max_context (see dry-run).",
                                human_bytes(nbytes).c_str(), human_bytes(live_free).c_str()));
            }
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
        // A memset on memory that is not device-resident returns the useless
        // "invalid argument". Verify the pointer first so the failure names the
        // real cause (host pointer, or already-freed device memory).
        if (!cuda::is_device_memory(ptr)) {
            GAI_FAIL("device_memset_zero: pointer is not device memory (ptr=" +
                     std::to_string(reinterpret_cast<uintptr_t>(ptr)) +
                     ", nbytes=" + std::to_string(nbytes) +
                     ", storage registered as cpu)");
        }
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

void device_stage_f32_2d(const void* src, Device src_dev, i64 rows, i64 cols,
                          std::vector<float>& out, size_t block_bytes) {
    GAI_CHECK(rows >= 0 && cols >= 0, "device_stage_f32_2d: negative extent");
    GAI_CHECK(cols <= (1LL << 30), "device_stage_f32_2d: column count out of range");
    if (rows == 0 || cols == 0) { out.clear(); return; }
    GAI_CHECK(src != nullptr, "device_stage_f32_2d: null source");
    out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    const size_t cols_sz = static_cast<size_t>(cols);
    const size_t row_bytes = cols_sz * sizeof(float);
    // At least one row per block, even when a single row exceeds the budget:
    // silently truncating the block would drop the tail of every such row.
    const size_t rows_per_block = std::max<size_t>(1, block_bytes / std::max<size_t>(1, row_bytes));
    const char* base = static_cast<const char*>(src);
    for (i64 r = 0; r < rows; r += static_cast<i64>(rows_per_block)) {
        const i64 n = std::min<i64>(static_cast<i64>(rows_per_block), rows - r);
        device_copy(out.data() + static_cast<size_t>(r) * cols_sz, Device::CPU,
                    base + static_cast<size_t>(r) * row_bytes, src_dev,
                    static_cast<size_t>(n) * row_bytes);
    }
}

void device_synchronize(Device dev) {
#ifdef GAI_CUDA
    if (dev == Device::CUDA) {
        cuda::synchronize();
        ops::perf_note_sync();
    }
#else
    (void)dev;
#endif
}

} // namespace gai
