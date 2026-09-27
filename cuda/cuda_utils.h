#pragma once

#include "core/common.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace gai {
struct DeviceInfo;

namespace cuda {

// Called by core/device.cpp. Fills the CUDA fields, leaves them false if no
// usable device exists (driver missing, no GPU, etc.) — never throws.
void probe(DeviceInfo& info);

void* malloc_device(size_t nbytes);
void  free_device(void* ptr);
void  memset_zero(void* ptr, size_t nbytes);
// True only if ptr is live device memory on the current device. Lives here (in
// the .cu) so core/device.cpp never needs the CUDA runtime headers.
bool  is_device_memory(const void* ptr);
void  copy(void* dst, bool dst_is_device, const void* src, bool src_is_device, size_t nbytes);
void  synchronize();
    // Live free-VRAM query for the device.cpp OOM guard. Returns 0 if unavailable.
    size_t free_bytes_live();
    // Check that at least `need` bytes of VRAM are free; GAI_FAIL with a clear
    // message if not. Call before large workspace allocations for better
    // diagnostics than a raw cudaMalloc failure.
    void check_free_vram(size_t need, const char* what);

// cuBLAS handle, created lazily on first use.
void* cublas_handle();
// Device the cuBLAS handle was created on (-1 if not created yet). GEMM
// entry points compare it against the current device so a handle/device
// mismatch fails loudly with both ids instead of a bare cuBLAS error.
int   cublas_device();
void  shutdown();

const char* last_error();

#ifdef __CUDACC__
// ---- RAII wrappers for CUDA resources (leak-safe on exceptions) ----
// Constructors fail loudly on cudaStreamCreate/EventCreate errors instead of
// building objects around invalid handles.
class CudaStream {
    cudaStream_t stream_ = nullptr;
    bool own_ = true;
public:
    CudaStream() {
        if (cudaStreamCreate(&stream_) != cudaSuccess || !stream_)
            GAI_FAIL("CudaStream: cudaStreamCreate failed (OOM or no device)");
    }
    explicit CudaStream(cudaStream_t s, bool own = false) : stream_(s), own_(own) {}
    ~CudaStream() { if (own_ && stream_) cudaStreamDestroy(stream_); }
    CudaStream(CudaStream&& o) noexcept : stream_(o.stream_), own_(o.own_) { o.stream_ = nullptr; o.own_ = false; }
    CudaStream& operator=(CudaStream&& o) noexcept {
        if (this != &o) { if (own_ && stream_) cudaStreamDestroy(stream_); stream_ = o.stream_; own_ = o.own_; o.stream_ = nullptr; o.own_ = false; }
        return *this;
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }
    void synchronize() const { if (stream_) cudaStreamSynchronize(stream_); }
};

class CudaEvent {
    cudaEvent_t event_ = nullptr;
    bool own_ = true;
public:
    CudaEvent(unsigned flags = cudaEventDefault) {
        if (cudaEventCreateWithFlags(&event_, flags) != cudaSuccess || !event_)
            GAI_FAIL("CudaEvent: cudaEventCreate failed (OOM or no device)");
    }
    explicit CudaEvent(cudaEvent_t e, bool own = false) : event_(e), own_(own) {}
    ~CudaEvent() { if (own_ && event_) cudaEventDestroy(event_); }
    CudaEvent(CudaEvent&& o) noexcept : event_(o.event_), own_(o.own_) { o.event_ = nullptr; o.own_ = false; }
    CudaEvent& operator=(CudaEvent&& o) noexcept {
        if (this != &o) { if (own_ && event_) cudaEventDestroy(event_); event_ = o.event_; own_ = o.own_; o.event_ = nullptr; o.own_ = false; }
        return *this;
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }
    void record(cudaStream_t stream = 0) const { if (event_) cudaEventRecord(event_, stream); }
    void synchronize() const { if (event_) cudaEventSynchronize(event_); }
    float elapsed_since(const CudaEvent& start) const {
        float ms = 0; if (event_ && start.event_) cudaEventElapsedTime(&ms, start.get(), event_); return ms;
    }
};

// Scoped device setter (RAII for cudaSetDevice)
class ScopedDevice {
    int prev_ = -1;
public:
    explicit ScopedDevice(int dev) { cudaGetDevice(&prev_); cudaSetDevice(dev); }
    ~ScopedDevice() { if (prev_ >= 0) cudaSetDevice(prev_); }
    ScopedDevice(const ScopedDevice&) = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;
};
#endif // __CUDACC__

} // namespace cuda
} // namespace gai
