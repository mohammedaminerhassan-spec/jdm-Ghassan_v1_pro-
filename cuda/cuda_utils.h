#pragma once

#include "core/common.h"

namespace gai {
struct DeviceInfo;

namespace cuda {

// Called by core/device.cpp. Fills the CUDA fields, leaves them false if no
// usable device exists (driver missing, no GPU, etc.) — never throws.
void probe(DeviceInfo& info);

void* malloc_device(size_t nbytes);
void  free_device(void* ptr);
void  memset_zero(void* ptr, size_t nbytes);
void  copy(void* dst, bool dst_is_device, const void* src, bool src_is_device, size_t nbytes);
void  synchronize();

// cuBLAS handle, created lazily on first use.
void* cublas_handle();
void  shutdown();

const char* last_error();

} // namespace cuda
} // namespace gai
