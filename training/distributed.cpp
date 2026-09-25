#include "training/distributed.h"
#include "core/ops.h"   // ops::perf_note_sync (telemetry for blocking collectives)

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef GAI_NCCL
#include <nccl.h>
#include <cuda_runtime.h>

#define NCCL_CHECK(x) do { ncclResult_t _r = (x); if (_r != ncclSuccess) \
    GAI_FAIL(std::string("NCCL ") + #x + ": " + ncclGetErrorString(_r)); } while (0)
#define CU_RT_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) \
    GAI_FAIL(std::string("CUDA ") + #x + ": " + cudaGetErrorString(_e)); } while (0)
#endif

namespace gai {

DistributedContext::~DistributedContext() {
    finalize();
}

bool DistributedContext::init(const Config& cfg) {
    if (initialized_) return true;

    config_ = cfg;
    global_rank_ = cfg.global_rank;
    local_rank_ = cfg.local_rank;
    world_size_ = cfg.world_size;

    if (world_size_ <= 1) {
        initialized_ = true;
        log_info("[dist] Single GPU mode (world_size=1)");
        return true;
    }

#ifdef GAI_NCCL
    // Single-node 4xT4: all ranks share /tmp, so rank 0 publishes the NCCL
    // ID through a file. torchrun users can skip this via GAI_NCCL_ID_FILE.
    CU_RT_CHECK(cudaSetDevice(local_rank_));
    CU_RT_CHECK(cudaStreamCreate(&stream_));
    // Device-side word for barrier (NCCL forbids host pointers).
    CU_RT_CHECK(cudaMalloc(&barrier_dev_, sizeof(int)));
    CU_RT_CHECK(cudaMemset(barrier_dev_, 0, sizeof(int)));

    const char* env_file = std::getenv("GAI_NCCL_ID_FILE");
    std::string id_file = env_file ? env_file
        : "/tmp/gai_nccl_" + std::to_string(config_.master_port) + ".id";

    ncclUniqueId nccl_id;
    std::memset(&nccl_id, 0, sizeof(nccl_id));
    if (global_rank_ == 0) {
        // FIX: a stale ID file from a crashed 4xT4 run has the correct size,
        // so ranks 1..3 could read the OLD id before rank 0 rewrites it ->
        // ncclCommInitRank mismatch / hang. Unlink first so followers only
        // ever see a fresh, complete ID (train_4xt4.sh also removes it).
        {
            std::error_code ec;
            std::filesystem::remove(id_file, ec);
        }
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        std::ofstream f(id_file, std::ios::binary | std::ios::trunc);
        if (!f.good()) GAI_FAIL("NCCL: cannot publish ID file " + id_file);
        f.write(reinterpret_cast<const char*>(&nccl_id), sizeof(nccl_id));
        f.close();
        if (!f.good()) GAI_FAIL("NCCL: failed writing ID file " + id_file);
    } else {
        // Bounded wait: fail loudly instead of hanging forever if rank 0 dies.
        bool seen = false;
        for (int i = 0; i < 3000; ++i) {
            std::error_code ec;
            if (std::filesystem::exists(id_file, ec) &&
                std::filesystem::file_size(id_file, ec) == sizeof(nccl_id)) {
                seen = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!seen) GAI_FAIL("NCCL: rank 0 did not publish " + id_file + " within 30s");
        std::ifstream f(id_file, std::ios::binary);
        if (!f.good()) GAI_FAIL("NCCL: cannot read ID file " + id_file);
        f.read(reinterpret_cast<char*>(&nccl_id), sizeof(nccl_id));
        if (!f.good()) GAI_FAIL("NCCL: truncated ID file " + id_file);
        f.close();
    }

    NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, nccl_id, global_rank_));
    id_file_ = id_file;

    initialized_ = true;
    log_info(strfmt("[dist] NCCL initialized: rank %d/%d on GPU %d",
                    global_rank_, world_size_, local_rank_));
    return true;
#else
    GAI_FAIL("Multi-GPU requested but NCCL not available (build with -DGAI_ENABLE_NCCL=ON)");
    return false;
#endif
}

void DistributedContext::finalize() {
    if (!initialized_) return;

#ifdef GAI_NCCL
    if (comm_) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }
    if (barrier_dev_) {
        cudaFree(barrier_dev_);
        barrier_dev_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    // Rank 0 cleans the rendezvous file (single-node /tmp).
    if (global_rank_ == 0 && !id_file_.empty()) {
        std::error_code ec;
        std::filesystem::remove(id_file_, ec);
        id_file_.clear();
    }
#endif

    initialized_ = false;
    log_info("[dist] Finalized");
}

void DistributedContext::all_reduce_sum(float* buffer, size_t numel) {
    all_reduce_sum(static_cast<void*>(buffer), numel, sizeof(float));
}

void DistributedContext::all_reduce_sum_i64(int64_t* buffer, size_t numel) {
    if (world_size_ <= 1) return;

#ifdef GAI_NCCL
    NCCL_CHECK(ncclAllReduce(buffer, buffer, numel, ncclInt64, ncclSum, comm_, stream_));
    CU_RT_CHECK(cudaStreamSynchronize(stream_));
    ops::perf_note_sync();
#else
    (void)buffer; (void)numel;
#endif
}

void DistributedContext::all_reduce_sum(void* buffer, size_t numel, int dtype_size) {
    if (world_size_ <= 1) return;

#ifdef GAI_NCCL
    ncclDataType_t nccl_dtype = ncclFloat32;
    if (dtype_size == 2) nccl_dtype = ncclFloat16;
    else if (dtype_size == 4) nccl_dtype = ncclFloat32;
    else if (dtype_size == 8) nccl_dtype = ncclFloat64;
    else if (dtype_size == 1) nccl_dtype = ncclInt8;

    NCCL_CHECK(ncclAllReduce(buffer, buffer, numel, nccl_dtype, ncclSum, comm_, stream_));
    CU_RT_CHECK(cudaStreamSynchronize(stream_));
    ops::perf_note_sync();
#else
    (void)buffer; (void)numel; (void)dtype_size;
#endif
}

void DistributedContext::broadcast(void* buffer, size_t numel, int dtype_size, int root) {
    if (world_size_ <= 1) return;

#ifdef GAI_NCCL
    ncclDataType_t nccl_dtype = ncclFloat32;
    if (dtype_size == 2) nccl_dtype = ncclFloat16;
    else if (dtype_size == 4) nccl_dtype = ncclFloat32;
    else if (dtype_size == 8) nccl_dtype = ncclFloat64;
    else if (dtype_size == 1) nccl_dtype = ncclInt8;

    NCCL_CHECK(ncclBroadcast(buffer, buffer, numel, nccl_dtype, root, comm_, stream_));
    CU_RT_CHECK(cudaStreamSynchronize(stream_));
    ops::perf_note_sync();
#else
    (void)buffer; (void)numel; (void)dtype_size; (void)root;
#endif
}

void DistributedContext::barrier() {
    if (world_size_ <= 1) return;

#ifdef GAI_NCCL
    // Dummy all-reduce on DEVICE memory (host pointers are illegal for NCCL).
    if (!barrier_dev_ || !comm_) return;
    NCCL_CHECK(ncclAllReduce(barrier_dev_, barrier_dev_, 1, ncclInt32, ncclSum, comm_, stream_));
    CU_RT_CHECK(cudaStreamSynchronize(stream_));
    ops::perf_note_sync();
#endif
}

std::string DistributedContext::get_nccl_unique_id() {
#ifdef GAI_NCCL
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    std::ostringstream oss;
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(id.internal[i]);
    }
    return oss.str();
#else
    return "";
#endif
}

DistributedContext::Config distributed_config_from_env(int default_world_size) {
    DistributedContext::Config cfg;

    // SLURM environment
    if (const char* slurm_ntasks = std::getenv("SLURM_NTASKS")) {
        cfg.world_size = std::atoi(slurm_ntasks);
    }
    if (const char* slurm_proc_id = std::getenv("SLURM_PROCID")) {
        cfg.global_rank = std::atoi(slurm_proc_id);
    }
    if (const char* slurm_local_id = std::getenv("SLURM_LOCALID")) {
        cfg.local_rank = std::atoi(slurm_local_id);
    }

    // torchrun / torch.distributed.launch environment
    if (const char* world_size = std::getenv("WORLD_SIZE")) {
        cfg.world_size = std::atoi(world_size);
    }
    if (const char* rank = std::getenv("RANK")) {
        cfg.global_rank = std::atoi(rank);
    }
    if (const char* local_rank = std::getenv("LOCAL_RANK")) {
        cfg.local_rank = std::atoi(local_rank);
    }

    // Master address/port
    if (const char* master_addr = std::getenv("MASTER_ADDR")) {
        cfg.master_addr = master_addr;
    }
    if (const char* master_port = std::getenv("MASTER_PORT")) {
        cfg.master_port = std::atoi(master_port);
    }

    // Defaults
    if (cfg.world_size <= 0) cfg.world_size = default_world_size;
    if (cfg.global_rank < 0) cfg.global_rank = 0;
    if (cfg.local_rank < 0) cfg.local_rank = 0;

    return cfg;
}

ScopedDistributed::ScopedDistributed(const DistributedContext::Config& cfg) {
    ctx_.init(cfg);
}

ScopedDistributed::ScopedDistributed(int world_size, int local_rank) {
    DistributedContext::Config cfg;
    cfg.world_size = world_size;
    cfg.local_rank = local_rank;
    cfg.global_rank = local_rank;  // Single node assumption
    ctx_.init(cfg);
}

ScopedDistributed::~ScopedDistributed() {
    ctx_.finalize();
}

} // namespace gai