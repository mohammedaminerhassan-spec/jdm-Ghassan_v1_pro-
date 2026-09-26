#include "core/dtype.h"

namespace gai {

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32:  return "f32";
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::I32:  return "i32";
        case DType::I8:   return "i8";
        case DType::Q8_0: return "q8_0";
        case DType::Q4_0: return "q4_0";
        case DType::Q4_1: return "q4_1";
        case DType::U16:  return "u16";
        case DType::I64:  return "i64";
    }
    return "?";
}

DType dtype_from_name(const std::string& s) {
    if (s == "f32"  || s == "fp32")  return DType::F32;
    if (s == "f16"  || s == "fp16")  return DType::F16;
    if (s == "bf16")                 return DType::BF16;
    if (s == "i32")                  return DType::I32;
    if (s == "i8"   || s == "int8")  return DType::I8;
    if (s == "q8_0" || s == "q8")    return DType::Q8_0;
    if (s == "q4_0" || s == "q4")    return DType::Q4_0;
    if (s == "q4_1")                 return DType::Q4_1;
    if (s == "u16")                  return DType::U16;
    if (s == "i64")                  return DType::I64;
    GAI_FAIL("unknown dtype: " + s);
}

bool dtype_is_quantized(DType t) {
    return t == DType::Q8_0 || t == DType::Q4_0 || t == DType::Q4_1;
}

int dtype_block_size(DType t) {
    switch (t) {
        case DType::Q8_0: return Q8_BLOCK;
        case DType::Q4_0:
        case DType::Q4_1: return Q4_BLOCK;
        default:          return 1;
    }
}

size_t dtype_block_bytes(DType t) {
    switch (t) {
        case DType::F32:  return 4;
        case DType::I32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::U16:  return 2;
        case DType::I8:   return 1;
        case DType::I64:  return 8;
        case DType::Q8_0: return sizeof(BlockQ8_0);
        case DType::Q4_0: return sizeof(BlockQ4_0);
        case DType::Q4_1: return sizeof(BlockQ4_1);
    }
    return 0;
}

size_t dtype_nbytes(DType t, size_t numel) {
    int bs = dtype_block_size(t);
    if (bs == 1) return numel * dtype_block_bytes(t);
    // Quantized storage holds whole blocks; a partial tail block is zero-padded
    // (see quantize_q8_0/q4_0/q4_1). Round up instead of crashing so odd-sized
    // tensors quantize; the logical element count still comes from the shape.
    const size_t bsu = static_cast<size_t>(bs);
    return ((numel + bsu - 1) / bsu) * dtype_block_bytes(t);
}

} // namespace gai
