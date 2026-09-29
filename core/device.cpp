#include "core/device.h"
#include "core/ops.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <set>
#include <vector>
#include <algorithm>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

#ifdef GAI_CUDA
#include "cuda/cuda_utils.h"
#endif

namespace gai {
namespace fs = std::filesystem;

bool is_gpu(Device d) { return d == Device::CUDA; }

namespace {

struct FileId {
    unsigned long long hi = 0;
    unsigned long long lo = 0;
    bool operator<(const FileId& o) const {
        return hi != o.hi ? hi < o.hi : lo < o.lo;
    }
};
FileId file_id_of(const std::string& path) {
    FileId id;
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return id;
    BY_HANDLE_FILE_INFORMATION info{};
    if (GetFileInformationByHandle(h, &info)) {
        id.hi = static_cast<unsigned long long>(info.dwVolumeSerialNumber);
        id.lo = (static_cast<unsigned long long>(info.nFileIndexHigh) << 32) |
                static_cast<unsigned long long>(info.nFileIndexLow);
    }
    CloseHandle(h);
#else
    struct stat st {};
    if (::stat(path.c_str(), &st) == 0) {
        id.hi = static_cast<unsigned long long>(st.st_dev);
        id.lo = static_cast<unsigned long long>(st.st_ino);
    }
#endif
    return id;
}
}

size_t physical_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) return static_cast<size_t>(st.ullTotalPhys);
    return 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page  = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page <= 0) return 0;
    return static_cast<size_t>(pages) * static_cast<size_t>(page);
#endif
}

size_t tree_size_bytes(const std::string& path, size_t* unique_files) {
    return tree_size_bytes_excluding(path, {}, unique_files);
}

size_t tree_size_bytes_excluding(const std::string& path,
                                 const std::vector<std::string>& skip_dir_names,
                                 size_t* unique_files) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        if (unique_files) *unique_files = 0;
        return 0;
    }
    if (!fs::is_directory(path, ec)) {
        if (unique_files) *unique_files = 1;
        return static_cast<size_t>(fs::file_size(path, ec));
    }
    size_t total = 0;
    size_t count = 0;
    std::set<FileId> seen;
    for (fs::recursive_directory_iterator it(path, fs::directory_options::skip_permission_denied, ec), end;
         it != end; it.increment(ec)) {

        if (ec) {
            GAI_FAIL("cannot measure tree size under " + path + ": " + ec.message());
        }

        if (!skip_dir_names.empty() && it->is_directory(ec)) {
            const std::string name = it->path().filename().string();
            if (std::find(skip_dir_names.begin(), skip_dir_names.end(), name) != skip_dir_names.end()) {
                it.disable_recursion_pending();
                continue;
            }
        }
        if (!it->is_regular_file(ec)) continue;
        const FileId id = file_id_of(it->path().string());
        if (id.hi || id.lo) {
            if (!seen.insert(id).second) continue;
        }
        total += static_cast<size_t>(it->file_size(ec));
        ++count;
    }
    if (unique_files) *unique_files = count;
    return total;
}

size_t free_disk_bytes(const std::string& path) {

    std::string p = path;
    for (int depth = 0; depth < 8; ++depth) {
#ifdef _WIN32
        const DWORD attr = GetFileAttributesA(p.c_str());
        const bool exists = (attr != INVALID_FILE_ATTRIBUTES);
        if (exists && !(attr & FILE_ATTRIBUTE_DIRECTORY)) return 0;
        if (exists) {
            ULARGE_INTEGER avail{};
            if (GetDiskFreeSpaceExA(p.c_str(), &avail, nullptr, nullptr))
                return static_cast<size_t>(avail.QuadPart);
            return 0;
        }
#else

        struct stat st {};
        if (::stat(p.c_str(), &st) == 0 && !S_ISDIR(st.st_mode)) return 0;
        struct statvfs vfs {};
        if (::statvfs(p.c_str(), &vfs) == 0)
            return static_cast<size_t>(vfs.f_bavail) * static_cast<size_t>(vfs.f_frsize);
#endif
        const size_t slash = p.find_last_of("/\\");
        if (slash == std::string::npos) break;
        p = (slash == 0) ? "/" : p.substr(0, slash);
    }
    return 0;
}

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

static void* aligned_alloc_64(size_t n) {
    if (n == 0) return nullptr;

    GAI_CHECK(n <= (std::numeric_limits<size_t>::max)() - 63, "allocation size overflow");
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

void* device_alloc(size_t nbytes, Device dev, DType ) {
    if (nbytes == 0) return nullptr;
    if (dev == Device::CUDA) {
#ifdef GAI_CUDA

        if (nbytes > (16ull << 20)) {
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

    try {
        return aligned_alloc_64(nbytes);
    } catch (const std::bad_alloc&) {
        GAI_FAIL(strfmt("CPU OOM: need %s. Close apps, lower batch_size/seq_len, "
                        "or use streaming shards.", human_bytes(nbytes).c_str()));
    }
    return nullptr;
}

void device_free(void* ptr, Device dev) {
    if (!ptr) return;
    if (dev == Device::CUDA) {
#ifdef GAI_CUDA
        cuda::free_device(ptr);
#else

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
    GAI_CHECK(rows <= (1LL << 30), "device_stage_f32_2d: row count out of range");
    if (rows == 0 || cols == 0) { out.clear(); return; }
    GAI_CHECK(src != nullptr, "device_stage_f32_2d: null source");

    const u64 total = static_cast<u64>(rows) * static_cast<u64>(cols);
    GAI_CHECK(total <= (static_cast<u64>(1) << 32), "device_stage_f32_2d: staging too large");
    out.resize(static_cast<size_t>(total));
    const size_t cols_sz = static_cast<size_t>(cols);
    const size_t row_bytes = cols_sz * sizeof(float);

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

}
