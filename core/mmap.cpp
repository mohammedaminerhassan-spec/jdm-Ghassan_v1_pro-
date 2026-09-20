#include "core/mmap.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace gai {

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& o) noexcept
    : base_(o.base_), size_(o.size_), path_(std::move(o.path_))
#ifdef _WIN32
    , file_(o.file_), map_(o.map_)
#else
    , fd_(o.fd_)
#endif
{
    o.base_ = nullptr;
    o.size_ = 0;
#ifdef _WIN32
    o.file_ = nullptr;
    o.map_  = nullptr;
#else
    o.fd_ = -1;
#endif
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
    if (this != &o) {
        close();
        base_ = o.base_;
        size_ = o.size_;
        path_ = std::move(o.path_);
#ifdef _WIN32
        file_ = o.file_;
        map_  = o.map_;
        o.file_ = nullptr;
        o.map_  = nullptr;
#else
        fd_   = o.fd_;
        o.fd_ = -1;
#endif
        o.base_ = nullptr;
        o.size_ = 0;
    }
    return *this;
}

bool MappedFile::open(const std::string& path) {
    close();
    if (path.empty()) return false;
#ifdef _WIN32
    // FILE_SHARE_DELETE is essential, not generous: without it, any live
    // mapping (even our own process retraining over the file) makes a later
    // remove/rename fail with Permission denied. The mapping keeps serving
    // the OLD bytes, which is exactly the correct semantics.
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart <= 0) {
        CloseHandle(f);
        return false;
    }
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m == nullptr) {
        CloseHandle(f);
        return false;
    }
    // Whole-file read-only view; OS pages it on demand, shares between
    // processes, and may evict clean pages under pressure (the weak-PC win).
    void* base = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (base == nullptr) {
        CloseHandle(m);
        CloseHandle(f);
        return false;
    }
    file_ = f;
    map_  = m;
    base_ = base;
    size_ = static_cast<u64>(sz.QuadPart);
    path_ = path;
    return true;
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        return false;
    }
    void* base = ::mmap(nullptr, static_cast<size_t>(st.st_size),
                        PROT_READ, MAP_PRIVATE, fd, 0);
    // The fd may be closed right after a successful mmap (POSIX-legal);
    // the mapping itself keeps the pages alive.
    ::close(fd);
    if (base == MAP_FAILED) return false;
    fd_   = -1;
    base_ = base;
    size_ = static_cast<u64>(st.st_size);
    path_ = path;
    return true;
#endif
}

void MappedFile::close() {
    if (base_ == nullptr) {
#ifdef _WIN32
        // Defensive: never leak handles even on a moved-from/partial state.
        if (map_ != nullptr) {
            CloseHandle(static_cast<HANDLE>(map_));
            map_ = nullptr;
        }
        if (file_ != nullptr) {
            CloseHandle(static_cast<HANDLE>(file_));
            file_ = nullptr;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
        size_ = 0;
        return;
    }
#ifdef _WIN32
    UnmapViewOfFile(base_);
    CloseHandle(static_cast<HANDLE>(map_));
    CloseHandle(static_cast<HANDLE>(file_));
    map_  = nullptr;
    file_ = nullptr;
#else
    ::munmap(base_, static_cast<size_t>(size_));
    // fd_ is already -1 here (closed in open()), kept only for clarity.
#endif
    base_ = nullptr;
    size_ = 0;
}

} // namespace gai
