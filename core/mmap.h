#pragma once

#include "core/common.h"
#include <memory>
#include <string>

namespace gai {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& o) noexcept;
    MappedFile& operator=(MappedFile&& o) noexcept;

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
    void* file_ = nullptr;
    void* map_  = nullptr;
#else
    int fd_ = -1;
#endif
};

using MappedFilePtr = std::shared_ptr<MappedFile>;

}
