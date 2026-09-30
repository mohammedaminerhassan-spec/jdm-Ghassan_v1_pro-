#pragma once

#include "core/common.h"
#include "core/tensor.h"

#include <string>
#include <vector>

namespace gai {

// NOTE: device_name() is declared in core/tensor.h (where Device lives) and
// defined once in core/tensor.cpp. It is NOT redeclared here to avoid a
// duplicate declaration across headers.
bool        is_gpu(Device d);

struct DeviceInfo {
    bool        cuda_available = false;
    int         device_count   = 0;
    int         device_index   = 0;
    std::string name           = "cpu";
    int         cc_major       = 0;
    int         cc_minor       = 0;
    size_t      total_mem      = 0;
    size_t      free_mem       = 0;
    bool        supports_fp16  = false;
    bool        supports_bf16  = false;
    bool        supports_tf32  = false;
};

const DeviceInfo& device_info();
void              print_device_report();
bool              cuda_available();
Device            best_device();

size_t physical_ram_bytes();
size_t free_disk_bytes(const std::string& path);

size_t tree_size_bytes(const std::string& path, size_t* unique_files = nullptr);

size_t tree_size_bytes_excluding(const std::string& path,
                                 const std::vector<std::string>& skip_dir_names,
                                 size_t* unique_files = nullptr);

void* device_alloc(size_t nbytes, Device dev, DType dt = DType::F32);
void  device_free(void* ptr, Device dev);
void  device_memset_zero(void* ptr, size_t nbytes, Device dev);
void  device_copy(void* dst, Device dst_dev, const void* src, Device src_dev, size_t nbytes);
void  device_synchronize(Device dev);

void  device_stage_f32_2d(const void* src, Device src_dev, i64 rows, i64 cols,
                          std::vector<float>& out, size_t block_bytes = (4u << 20));

}
