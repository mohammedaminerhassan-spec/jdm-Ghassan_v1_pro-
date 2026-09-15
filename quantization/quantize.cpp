#include "quantization/quantize.h"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace gai {
namespace quant {

// ================================================================ float casts
void f32_to_f16(const float* s, u16* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = fp32_to_fp16(s[i]); }
void f16_to_f32(const u16* s, float* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = fp16_to_fp32(s[i]); }
void f32_to_bf16(const float* s, u16* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = fp32_to_bf16(s[i]); }
void bf16_to_f32(const u16* s, float* d, i64 n) { for (i64 i = 0; i < n; ++i) d[i] = bf16_to_fp32(s[i]); }

// ================================================================ Q8_0
// symmetric, per-64 block, int8 range [-127, 127]
// n may be any non-negative count: a partial tail block is zero-padded
// (stats over the real elements only). dst must hold ceil(n / 64) blocks.
void quantize_q8_0(const float* src, void* dst, i64 n) {
    GAI_CHECK(n >= 0, "q8_0 element count must be non-negative");
    BlockQ8_0* out = static_cast<BlockQ8_0*>(dst);
    const i64 full = n / Q8_BLOCK;
    for (i64 b = 0; b < full; ++b) {
        const float* x = src + b * Q8_BLOCK;
        float amax = 0.0f;
        for (int i = 0; i < Q8_BLOCK; ++i) amax = std::max(amax, std::fabs(x[i]));
        float scale = amax / 127.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[b].scale = scale;
        for (int i = 0; i < Q8_BLOCK; ++i) {
            int q = static_cast<int>(std::lround(x[i] * inv));
            out[b].q[i] = static_cast<i8>(std::clamp(q, -127, 127));
        }
    }
    const i64 rem = n % Q8_BLOCK;
    if (rem > 0) {
        const float* x = src + full * Q8_BLOCK;
        float amax = 0.0f;
        for (i64 i = 0; i < rem; ++i) amax = std::max(amax, std::fabs(x[i]));
        float scale = amax / 127.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[full].scale = scale;
        for (i64 i = 0; i < rem; ++i) {
            int q = static_cast<int>(std::lround(x[i] * inv));
            out[full].q[i] = static_cast<i8>(std::clamp(q, -127, 127));
        }
        for (i64 i = rem; i < Q8_BLOCK; ++i) out[full].q[i] = 0;
    }
}

void dequantize_q8_0(const void* src, float* dst, i64 n) {
    GAI_CHECK(n >= 0, "q8_0 element count must be non-negative");
    const BlockQ8_0* in = static_cast<const BlockQ8_0*>(src);
    const i64 full = n / Q8_BLOCK;
    for (i64 b = 0; b < full; ++b) {
        float s = in[b].scale;
        float* o = dst + b * Q8_BLOCK;
        for (int i = 0; i < Q8_BLOCK; ++i) o[i] = static_cast<float>(in[b].q[i]) * s;
    }
    const i64 rem = n % Q8_BLOCK;
    if (rem > 0) {
        float tmp[Q8_BLOCK];
        float s = in[full].scale;
        for (int i = 0; i < Q8_BLOCK; ++i) tmp[i] = static_cast<float>(in[full].q[i]) * s;
        for (i64 i = 0; i < rem; ++i) dst[full * Q8_BLOCK + i] = tmp[i];
    }
}

// ================================================================ Q4_0
// symmetric, per-32 block, int4 range [-8, 7] stored as nibble+8 (GGML-compatible)
// Partial tail block: stats over the real elements, padding nibbles are 8
// (= value 0). dst must hold ceil(n / 32) blocks.
void quantize_q4_0(const float* src, void* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_0 element count must be non-negative");
    BlockQ4_0* out = static_cast<BlockQ4_0*>(dst);
    const i64 full = n / Q4_BLOCK;
    for (i64 b = 0; b < full; ++b) {
        const float* x = src + b * Q4_BLOCK;
        float amax = 0.0f;
        for (int i = 0; i < Q4_BLOCK; ++i) amax = std::max(amax, std::fabs(x[i]));
        float scale = amax / 8.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[b].scale = fp32_to_fp16(scale);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            int q0 = std::clamp(static_cast<int>(std::lround(x[2 * i]     * inv)), -8, 7) + 8;
            int q1 = std::clamp(static_cast<int>(std::lround(x[2 * i + 1] * inv)), -8, 7) + 8;
            out[b].q[i] = static_cast<u8>((q1 << 4) | q0);
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        const float* x = src + full * Q4_BLOCK;
        float amax = 0.0f;
        for (i64 i = 0; i < rem; ++i) amax = std::max(amax, std::fabs(x[i]));
        float scale = amax / 8.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[full].scale = fp32_to_fp16(scale);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            const i64 i0 = static_cast<i64>(2 * i), i1 = i0 + 1;
            int q0 = (i0 < rem ? std::clamp(static_cast<int>(std::lround(x[i0] * inv)), -8, 7) + 8 : 8);
            int q1 = (i1 < rem ? std::clamp(static_cast<int>(std::lround(x[i1] * inv)), -8, 7) + 8 : 8);
            out[full].q[i] = static_cast<u8>((q1 << 4) | q0);
        }
    }
}

void dequantize_q4_0(const void* src, float* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_0 element count must be non-negative");
    const BlockQ4_0* in = static_cast<const BlockQ4_0*>(src);
    const i64 full = n / Q4_BLOCK;
    for (i64 b = 0; b < full; ++b) {
        float s = fp16_to_fp32(in[b].scale);
        float* o = dst + b * Q4_BLOCK;
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            u8 byte = in[b].q[i];
            o[2 * i]     = (static_cast<float>(byte & 0x0F) - 8.0f) * s;
            o[2 * i + 1] = (static_cast<float>(byte >> 4)   - 8.0f) * s;
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        float tmp[Q4_BLOCK];
        float s = fp16_to_fp32(in[full].scale);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            u8 byte = in[full].q[i];
            tmp[2 * i]     = (static_cast<float>(byte & 0x0F) - 8.0f) * s;
            tmp[2 * i + 1] = (static_cast<float>(byte >> 4)   - 8.0f) * s;
        }
        for (i64 i = 0; i < rem; ++i) dst[full * Q4_BLOCK + i] = tmp[i];
    }
}

// ================================================================ Q4_1
// asymmetric (scale + min), per-32 block, uint4 range [0, 15]
// Partial tail block: stats over the real elements, padding nibbles encode
// value 0 under the block scale. dst must hold ceil(n / 32) blocks.
void quantize_q4_1(const float* src, void* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_1 element count must be non-negative");
    BlockQ4_1* out = static_cast<BlockQ4_1*>(dst);
    const i64 full = n / Q4_BLOCK;
    for (i64 b = 0; b < full; ++b) {
        const float* x = src + b * Q4_BLOCK;
        float mn = x[0], mx = x[0];
        for (int i = 1; i < Q4_BLOCK; ++i) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); }
        float scale = (mx - mn) / 15.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[b].scale = fp32_to_fp16(scale);
        out[b].min   = fp32_to_fp16(mn);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            int q0 = std::clamp(static_cast<int>(std::lround((x[2 * i]     - mn) * inv)), 0, 15);
            int q1 = std::clamp(static_cast<int>(std::lround((x[2 * i + 1] - mn) * inv)), 0, 15);
            out[b].q[i] = static_cast<u8>((q1 << 4) | q0);
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        const float* x = src + full * Q4_BLOCK;
        float mn = x[0], mx = x[0];
        for (i64 i = 1; i < rem; ++i) { mn = std::min(mn, x[i]); mx = std::max(mx, x[i]); }
        float scale = (mx - mn) / 15.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
        out[full].scale = fp32_to_fp16(scale);
        out[full].min   = fp32_to_fp16(mn);
        const int qz = std::clamp(static_cast<int>(std::lround((0.0f - mn) * inv)), 0, 15);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            const i64 i0 = static_cast<i64>(2 * i), i1 = i0 + 1;
            int q0 = (i0 < rem ? std::clamp(static_cast<int>(std::lround((x[i0] - mn) * inv)), 0, 15) : qz);
            int q1 = (i1 < rem ? std::clamp(static_cast<int>(std::lround((x[i1] - mn) * inv)), 0, 15) : qz);
            out[full].q[i] = static_cast<u8>((q1 << 4) | q0);
        }
    }
}

void dequantize_q4_1(const void* src, float* dst, i64 n) {
    GAI_CHECK(n >= 0, "q4_1 element count must be non-negative");
    const BlockQ4_1* in = static_cast<const BlockQ4_1*>(src);
    const i64 full = n / Q4_BLOCK;
    for (i64 b = 0; b < full; ++b) {
        float s = fp16_to_fp32(in[b].scale);
        float m = fp16_to_fp32(in[b].min);
        float* o = dst + b * Q4_BLOCK;
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            u8 byte = in[b].q[i];
            o[2 * i]     = static_cast<float>(byte & 0x0F) * s + m;
            o[2 * i + 1] = static_cast<float>(byte >> 4)   * s + m;
        }
    }
    const i64 rem = n % Q4_BLOCK;
    if (rem > 0) {
        float tmp[Q4_BLOCK];
        float s = fp16_to_fp32(in[full].scale);
        float m = fp16_to_fp32(in[full].min);
        for (int i = 0; i < Q4_BLOCK / 2; ++i) {
            u8 byte = in[full].q[i];
            tmp[2 * i]     = static_cast<float>(byte & 0x0F) * s + m;
            tmp[2 * i + 1] = static_cast<float>(byte >> 4)   * s + m;
        }
        for (i64 i = 0; i < rem; ++i) dst[full * Q4_BLOCK + i] = tmp[i];
    }
}

// ================================================================ tensor api
bool is_quantizable(i64 numel, DType t) {
    int bs = dtype_block_size(t);
    return bs == 1 || (numel % bs == 0);
}

Tensor quantize(const Tensor& src, DType target) {
    GAI_CHECK(src.dtype() == DType::F32, "quantize expects an f32 source");
    GAI_CHECK(src.device() == Device::CPU, "quantize runs on CPU tensors");
    if (target == DType::F32) return src.clone();

    // Non-multiples of the block size are zero-padded into the tail block
    // (see the kernels above) instead of crashing; is_quantizable() remains
    // for callers that prefer an f16 fallback for odd sizes (export paths).
    Tensor out(src.shape(), target, Device::CPU);
    const float* s = src.f32();
    const i64 n = src.numel();

    switch (target) {
        case DType::F16:  f32_to_f16(s, out.ptr<u16>(), n); break;
        case DType::BF16: f32_to_bf16(s, out.ptr<u16>(), n); break;
        case DType::Q8_0: quantize_q8_0(s, out.data_ptr(), n); break;
        case DType::Q4_0: quantize_q4_0(s, out.data_ptr(), n); break;
        case DType::Q4_1: quantize_q4_1(s, out.data_ptr(), n); break;
        default: GAI_FAIL(std::string("unsupported quantization target: ") + dtype_name(target));
    }
    out.set_name(src.name());
    return out;
}

Tensor dequantize(const Tensor& src, DType target) {
    GAI_CHECK(target == DType::F32, "dequantize currently targets f32 only");
    GAI_CHECK(src.device() == Device::CPU, "dequantize runs on CPU tensors");
    if (src.dtype() == DType::F32) return src.clone();

    Tensor out(src.shape(), DType::F32, Device::CPU);
    float* d = out.f32();
    const i64 n = src.numel();

    switch (src.dtype()) {
        case DType::F16:  f16_to_f32(src.ptr<u16>(), d, n); break;
        case DType::BF16: bf16_to_f32(src.ptr<u16>(), d, n); break;
        case DType::Q8_0: dequantize_q8_0(src.data_ptr(), d, n); break;
        case DType::Q4_0: dequantize_q4_0(src.data_ptr(), d, n); break;
        case DType::Q4_1: dequantize_q4_1(src.data_ptr(), d, n); break;
        default: GAI_FAIL(std::string("unsupported dequantization source: ") + dtype_name(src.dtype()));
    }
    out.set_name(src.name());
    return out;
}

QuantError measure_error(const Tensor& original, DType target) {
    QuantError e;
    e.numel = original.numel();
    if (e.numel == 0) return e;

    Tensor q = quantize(original, target);
    Tensor r = dequantize(q, DType::F32);

    const float* a = original.f32();
    const float* b = r.f32();
    double se = 0.0, sa = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (i64 i = 0; i < e.numel; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        se += d * d;
        sa += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        e.max_abs = std::max(e.max_abs, std::fabs(d));
    }
    e.rmse = std::sqrt(se / static_cast<double>(e.numel));
    double rms_a = std::sqrt(sa / static_cast<double>(e.numel));
    e.rel_rmse = rms_a > 0 ? e.rmse / rms_a : 0.0;
    e.cosine = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    return e;
}

const char* profile_description(const std::string& p) {
    if (p == "fp32") return "full precision, 803 MB, reference quality";
    if (p == "fp16") return "half precision everywhere, ~401 MB, no measurable quality loss";
    if (p == "int8") return "Q8_0 weights, F32 norms, ~215 MB, negligible quality loss";
    if (p == "int4") return "Q4_0 weights, Q8_0 embeddings, F32 norms, ~130 MB, small quality loss";
    return "unknown profile";
}

} // namespace quant
} // namespace gai
