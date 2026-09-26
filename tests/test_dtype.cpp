// DType registry regression: every enumerator must survive the name
// round-trip and report the correct element size. Added for I64 (exact DDP
// counters), which broke the CUDA build because the enumerator did not exist
// while the CPU-only build never compiled the #ifdef GAI_CUDA user.
#include "core/dtype.h"

#include <iostream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    CHECK(dtype_nbytes(DType::F32, 10) == 40, "f32 element size");
    CHECK(dtype_nbytes(DType::F16, 10) == 20, "f16 element size");
    CHECK(dtype_nbytes(DType::BF16, 10) == 20, "bf16 element size");
    CHECK(dtype_nbytes(DType::I32, 10) == 40, "i32 element size");
    CHECK(dtype_nbytes(DType::I8, 10) == 10, "i8 element size");
    CHECK(dtype_nbytes(DType::U16, 10) == 20, "u16 element size");
    CHECK(dtype_nbytes(DType::I64, 10) == 80, "i64 element size (NCCL counters)");
    CHECK(dtype_nbytes(DType::I64, 1) == 8, "single i64 staging buffer is 8 bytes");

    CHECK(dtype_from_name(dtype_name(DType::I64)) == DType::I64, "i64 name round-trip");
    CHECK(dtype_from_name(dtype_name(DType::F32)) == DType::F32, "f32 name round-trip");
    CHECK(dtype_from_name(dtype_name(DType::I32)) == DType::I32, "i32 name round-trip");
    CHECK(std::string(dtype_name(DType::I64)) == "i64", "i64 prints as i64");

    CHECK(!dtype_is_quantized(DType::I64), "i64 is not quantized");
    CHECK(dtype_block_size(DType::I64) == 1, "i64 block size is 1");

    if (failures == 0) {
        std::cout << "test_dtype: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_dtype: " << failures << " FAILURES\n";
    return 1;
}
