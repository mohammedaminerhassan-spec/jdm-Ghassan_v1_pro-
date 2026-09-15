#pragma once

#include "core/common.h"
#include <cmath>
#include <cstring>

namespace gai {

enum class DType : u32 {
    F32  = 0,
    F16  = 1,
    BF16 = 2,
    I32  = 3,
    I8   = 4,
    Q8_0 = 5,   // block of 64 int8 + 1 fp32 scale
    Q4_0 = 6,   // block of 32 int4 (symmetric) + 1 fp16 scale
    Q4_1 = 7,   // block of 32 int4 + fp16 scale + fp16 min
    U16  = 8,
};

// ---- quantization block layout (packed, no padding assumptions violated) ----
constexpr int Q8_BLOCK = 64;
constexpr int Q4_BLOCK = 32;

#pragma pack(push, 1)
struct BlockQ8_0 {
    float scale;
    i8    q[Q8_BLOCK];
};
struct BlockQ4_0 {
    u16 scale;              // fp16 bits
    u8  q[Q4_BLOCK / 2];    // two nibbles per byte
};
struct BlockQ4_1 {
    u16 scale;              // fp16 bits
    u16 min;                // fp16 bits
    u8  q[Q4_BLOCK / 2];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ8_0) == 4 + Q8_BLOCK, "BlockQ8_0 layout");
static_assert(sizeof(BlockQ4_0) == 2 + Q4_BLOCK / 2, "BlockQ4_0 layout");
static_assert(sizeof(BlockQ4_1) == 4 + Q4_BLOCK / 2, "BlockQ4_1 layout");

const char* dtype_name(DType t);
DType       dtype_from_name(const std::string& s);
bool        dtype_is_quantized(DType t);
int         dtype_block_size(DType t);      // elements per storage block (1 for dense)
size_t      dtype_block_bytes(DType t);     // bytes per storage block
size_t      dtype_nbytes(DType t, size_t numel);

// ------------------------------------------------------------------ fp16
inline u16 fp32_to_fp16(float f) {
    u32 x;
    std::memcpy(&x, &f, 4);
    u32 sign = (x >> 16) & 0x8000u;
    i32 exp  = static_cast<i32>((x >> 23) & 0xFF) - 127 + 15;
    u32 mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFF) == 0xFF) {                    // inf / nan
        return static_cast<u16>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) return static_cast<u16>(sign | 0x7C00u);   // overflow -> inf
    if (exp <= 0) {                                             // subnormal / zero
        if (exp < -10) return static_cast<u16>(sign);
        mant |= 0x800000u;
        u32 shift = static_cast<u32>(14 - exp);
        u32 sub   = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) sub += 1;               // round to nearest
        return static_cast<u16>(sign | sub);
    }
    u16 h = static_cast<u16>(sign | (static_cast<u32>(exp) << 10) | (mant >> 13));
    if ((mant & 0x1FFFu) > 0x1000u || ((mant & 0x1FFFu) == 0x1000u && (h & 1u))) h += 1;
    return h;
}

inline float fp16_to_fp32(u16 h) {
    u32 sign = static_cast<u32>(h & 0x8000u) << 16;
    u32 exp  = (h >> 10) & 0x1Fu;
    u32 mant = h & 0x3FFu;
    u32 out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            // subnormal -> normalize
            int e = -1;
            do { mant <<= 1; ++e; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            out = sign | (static_cast<u32>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// ------------------------------------------------------------------ bf16
inline u16 fp32_to_bf16(float f) {
    u32 x;
    std::memcpy(&x, &f, 4);
    if (((x >> 23) & 0xFF) == 0xFF) return static_cast<u16>(x >> 16);   // inf/nan
    u32 rounded = x + 0x7FFFu + ((x >> 16) & 1u);
    return static_cast<u16>(rounded >> 16);
}

inline float bf16_to_fp32(u16 b) {
    u32 x = static_cast<u32>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

} // namespace gai
