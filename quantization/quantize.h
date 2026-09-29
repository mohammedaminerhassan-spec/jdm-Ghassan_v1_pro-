#pragma once

#include "core/tensor.h"

namespace gai {
namespace quant {

Tensor quantize(const Tensor& src_f32, DType target);
Tensor dequantize(const Tensor& src, DType target = DType::F32);

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

struct QuantError {
    double rmse = 0;
    double max_abs = 0;
    double rel_rmse = 0;
    double cosine = 0;
    i64    numel = 0;
};

QuantError measure_error(const Tensor& original_f32, DType target);

bool is_quantizable(i64 numel, DType t);

const char* profile_description(const std::string& profile);

}
}
