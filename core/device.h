#pragma once

#include "core/common.h"
#include "core/tensor.h"

namespace gai {

const char* device_name(Device d);
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
Device            best_device();          // T4-ONLY: CUDA > CPU

// ---- host resources (asked for by name, never guessed) ------------------
// physical_ram_bytes(): total installed RAM, 0 if unknown. Budgets derived
// from a hardcoded constant are a lie on a smaller box: the checkpoint writer
// happily queued 32 GB of snapshots on a 30 GB Kaggle session and the OOM
// killer won the race.
// free_disk_bytes(path): free space on the filesystem holding `path` (0 if
// unknown). A checkpoint save needs the previous file AND a .tmp of the same
// size, so the transient peak is ~2x one checkpoint.
size_t physical_ram_bytes();
size_t free_disk_bytes(const std::string& path);
// Recursive size of a directory tree. Hard links are counted ONCE (by file
// id): the checkpoint writer publishes best.ckpt and last.ckpt as two names
// for ONE inode, and a quota projection that counts names twice invents a
// phantom second copy. unique_files (optional) receives the inode count.
size_t tree_size_bytes(const std::string& path, size_t* unique_files = nullptr);

void* device_alloc(size_t nbytes, Device dev, DType dt = DType::F32);
void  device_free(void* ptr, Device dev);
void  device_memset_zero(void* ptr, size_t nbytes, Device dev);
void  device_copy(void* dst, Device dst_dev, const void* src, Device src_dev, size_t nbytes);
void  device_synchronize(Device dev);

// Stage a device-resident [rows, cols] f32 tensor into a host vector, in
// bounded row blocks. Every host-side consumer of Model::forward() output MUST
// go through this: on CUDA the forward result lives in device memory and
// dereferencing it from the host segfaults (that is exactly how `ghassan-ai
// logits` crashed on a T4 while working on CPU). block_bytes caps the staging
// buffer; the block math is what makes it safe, so it is unit-tested.
void  device_stage_f32_2d(const void* src, Device src_dev, i64 rows, i64 cols,
                          std::vector<float>& out, size_t block_bytes = (4u << 20));

} // namespace gai
