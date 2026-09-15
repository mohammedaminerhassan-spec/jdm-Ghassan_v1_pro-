#pragma once

#include "core/tensor.h"

namespace gai {
namespace quant {

// ---------------------------------------------------------------- conversions
// All conversions are block-wise and symmetric unless the type says otherwise.
// Block scales keep outlier damage local, which is what makes INT4 usable at 200M.

Tensor quantize(const Tensor& src_f32, DType target);      // f32 -> any
Tensor dequantize(const Tensor& src, DType target = DType::F32);

// low-level, operate on raw buffers.
// n may be any non-negative count: a partial tail block is zero-padded on
// quantize (stats over the real elements) and truncated on dequantize.
// dst/src must hold ceil(n / block) blocks (see dtype_nbytes).
void quantize_q8_0(const float* src, void* dst, i64 n);
void dequantize_q8_0(const void* src, float* dst, i64 n);
void quantize_q4_0(const float* src, void* dst, i64 n);
void dequantize_q4_0(const void* src, float* dst, i64 n);
void quantize_q4_1(const float* src, void* dst, i64 n);
void dequantize_q4_1(const void* src, float* dst, i64 n);

void f32_to_f16(const float* src, u16* dst, i64 n);
void f16_to_f32(const u16* src, float* dst, i64 n);
void f32_to_bf16(const float* src, u16* dst, i64 n);
void bf16_to_f32(const u16* src, float* dst, i64 n);

// ---------------------------------------------------------------- analysis
struct QuantError {
    double rmse = 0;
    double max_abs = 0;
    double rel_rmse = 0;      // rmse / rms(original)
    double cosine = 0;        // cosine similarity to the original
    i64    numel = 0;
};

QuantError measure_error(const Tensor& original_f32, DType target);

// Pads a tensor length to the block size requirement of a dtype (returns true if
// the tensor can be quantized as-is).
bool is_quantizable(i64 numel, DType t);

const char* profile_description(const std::string& profile);

} // namespace quant
} // namespace gai
