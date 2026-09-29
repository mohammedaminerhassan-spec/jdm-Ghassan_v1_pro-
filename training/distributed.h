#pragma once

#include "core/common.h"
#include "core/device.h"

#ifdef GAI_NCCL
#include <nccl.h>
#include <cuda_runtime.h>
#endif

#include <vector>
#include <string>
#include <memory>

namespace gai {

class DistributedContext {
public:
    struct Config {
        int world_size = 1;
        int local_rank = 0;
        int global_rank = 0;
        std::string master_addr = "localhost";
        int master_port = 29500;
        std::string backend = "nccl";
        bool grad_compression = false;
    };

    DistributedContext() = default;
    ~DistributedContext();

    bool init(const Config& cfg);

    void finalize();

    bool initialized() const { return initialized_; }

    int world_size() const { return config_.world_size; }
    int local_rank() const { return config_.local_rank; }
    int global_rank() const { return config_.global_rank; }
    bool is_root() const { return global_rank_ == 0; }

    void all_reduce_sum(float* buffer, size_t numel);
    void all_reduce_sum(void* buffer, size_t numel, int dtype_size);
    void all_reduce_sum_i64(int64_t* buffer, size_t numel);

    void all_reduce_sum_nosync(void* buffer, size_t numel, int dtype_size);
    void begin_group();
    void end_group();
    void sync_stream();

    void broadcast(void* buffer, size_t numel, int dtype_size, int root = 0);

    void barrier();

    static std::string get_nccl_unique_id();

private:
#ifdef GAI_NCCL

    void* collective_staging(void* p, size_t nbytes, bool* copy_back);

    void wait_for_compute();
#endif

private:
    bool initialized_ = false;
    Config config_;
    int global_rank_ = 0;
    int local_rank_ = 0;
    int world_size_ = 1;

#ifdef GAI_NCCL
    ncclComm_t comm_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t fence_event_ = nullptr;
    void* barrier_dev_ = nullptr;
    void* coll_dev_ = nullptr;
    size_t coll_dev_bytes_ = 0;
    void* f16_stage_ = nullptr;
    size_t f16_stage_bytes_ = 0;
    std::string id_file_;
#endif
};

DistributedContext::Config distributed_config_from_env(int default_world_size = 1);

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

}
