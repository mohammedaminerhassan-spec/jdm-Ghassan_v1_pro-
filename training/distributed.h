#pragma once

#include "core/common.h"
#include "core/device.h"

#ifdef GAI_NCCL
#include <nccl.h>
#include <cuda_runtime.h>   // cudaStream_t for the NCCL stream member
#endif

#include <vector>
#include <string>
#include <memory>

namespace gai {

// Multi-GPU distributed training context using NCCL
class DistributedContext {
public:
    struct Config {
        int world_size = 1;           // total GPUs
        int local_rank = 0;           // this process's GPU index
        int global_rank = 0;          // this process's global rank
        std::string master_addr = "localhost";
        int master_port = 29500;
        std::string backend = "nccl";
    };

    DistributedContext() = default;
    ~DistributedContext();

    // Initialize NCCL communicator
    // Must be called once per process, after setting CUDA device
    bool init(const Config& cfg);

    // Clean up
    void finalize();

    // Check if initialized
    bool initialized() const { return initialized_; }

    // Getters
    int world_size() const { return config_.world_size; }
    int local_rank() const { return config_.local_rank; }
    int global_rank() const { return config_.global_rank; }
    bool is_root() const { return global_rank_ == 0; }

    // Gradient synchronization: all-reduce sum across all GPUs
    // `buffer` must be on the current CUDA device, size = numel * sizeof(float)
    void all_reduce_sum(float* buffer, size_t numel);
    void all_reduce_sum(void* buffer, size_t numel, int dtype_size);
    void all_reduce_sum_i64(int64_t* buffer, size_t numel);

    // Broadcast from root to all
    void broadcast(void* buffer, size_t numel, int dtype_size, int root = 0);

    // Barrier
    void barrier();

    // Get NCCL unique ID for initialization
    static std::string get_nccl_unique_id();

private:
    bool initialized_ = false;
    Config config_;
    int global_rank_ = 0;
    int local_rank_ = 0;
    int world_size_ = 1;

#ifdef GAI_NCCL
    ncclComm_t comm_ = nullptr;
    cudaStream_t stream_ = nullptr;
    void* barrier_dev_ = nullptr;   // device-side word for barrier()
    std::string id_file_;           // rendezvous file published by rank 0
#endif
};

// Helper: initialize distributed training from environment variables
// Supports SLURM, torchrun, and manual launch
DistributedContext::Config distributed_config_from_env(int default_world_size = 1);

// Scoped distributed context (RAII)
class ScopedDistributed {
public:
    explicit ScopedDistributed(const DistributedContext::Config& cfg);
    explicit ScopedDistributed(int world_size = 1, int local_rank = 0);
    ~ScopedDistributed();
    DistributedContext& ctx() { return ctx_; }
    operator bool() const { return ctx_.initialized(); }

private:
    DistributedContext ctx_;
};

} // namespace gai