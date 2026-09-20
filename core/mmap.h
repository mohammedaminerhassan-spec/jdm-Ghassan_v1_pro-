#pragma once

#include "core/common.h"
#include <memory>
#include <string>

namespace gai {

// ---------------------------------------------------------------- MappedFile
// Read-only memory-mapped file, portable across the three supported OSes:
//   Windows: CreateFileMapping + MapViewOfFile (read-only view).
//   Linux/macOS: mmap + MAP_PRIVATE (read-only pages, no writes ever).
//
// Purpose (roadmap item 5): weak-PC inference. Model weights stay file-backed
// so they cost ZERO heap commit; the OS pages them on demand, shares the
// pages between processes, and reclaims clean pages under memory pressure.
// Startup is instant (no full-file read) and two chat sessions share RAM.
//
// Ownership: refcounted (shared_ptr). Tensor storages created from a mapping
// hold a shared_ptr copy, so the mapping provably outlives every weight view
// even after the file reader object is destroyed (see Tensor::wrap_external).
// No raw new/delete anywhere; all handles released in close()/destructor.
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept;
    MappedFile& operator=(MappedFile&& o) noexcept;

    // Maps the whole file read-only. Returns false (never throws) when the
    // path is missing, empty, or the OS refuses the mapping.
    bool open(const std::string& path);
    void close();

    bool        valid() const { return base_ != nullptr; }
    const void* data()  const { return base_; }
    u64         size()  const { return size_; }
    const std::string& path() const { return path_; }

private:
    void*       base_ = nullptr;
    u64         size_ = 0;
    std::string path_;
#ifdef _WIN32
    void* file_ = nullptr;   // HANDLE (CreateFile)
    void* map_  = nullptr;   // HANDLE (CreateFileMapping)
#else
    int fd_ = -1;
#endif
};

using MappedFilePtr = std::shared_ptr<MappedFile>;

} // namespace gai
