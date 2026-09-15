#pragma once

#include "core/common.h"
#include <cmath>

namespace gai {

// xoshiro256** — fast, small state, easy to checkpoint.
class Rng {
public:
    explicit Rng(u64 seed = 1234567891011ULL) { seed_with(seed); }

    void seed_with(u64 seed) {
        // splitmix64 expansion
        for (int i = 0; i < 4; ++i) {
            seed += 0x9E3779B97F4A7C15ULL;
            u64 z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s_[i] = z ^ (z >> 31);
        }
        has_spare_ = false;
    }

    u64 next_u64() {
        const u64 r = rotl(s_[1] * 5, 7) * 9;
        const u64 t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return r;
    }

    u32 next_u32() { return static_cast<u32>(next_u64() >> 32); }

    // uniform in [0,1)
    float uniform() {
        return static_cast<float>((next_u64() >> 40) * (1.0 / 16777216.0));
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * uniform(); }

    // uniform integer in [0, n)
    u64 below(u64 n) {
        GAI_CHECK(n > 0, "below(0)");
        return next_u64() % n;
    }

    float normal(float mean = 0.0f, float stddev = 1.0f) {
        if (has_spare_) { has_spare_ = false; return mean + stddev * spare_; }
        float u, v, s;
        do {
            u = uniform() * 2.0f - 1.0f;
            v = uniform() * 2.0f - 1.0f;
            s = u * u + v * v;
        } while (s >= 1.0f || s == 0.0f);
        float m = std::sqrt(-2.0f * std::log(s) / s);
        spare_     = v * m;
        has_spare_ = true;
        return mean + stddev * (u * m);
    }

    // truncated normal in [-2σ, 2σ]  (standard init for transformer weights)
    float truncated_normal(float stddev) {
        for (int i = 0; i < 8; ++i) {
            float x = normal(0.0f, stddev);
            if (std::fabs(x) <= 2.0f * stddev) return x;
        }
        return 0.0f;
    }

    // checkpointable state (full: includes Box-Muller spare for bit-exact resume)
    void  get_state(u64 out[4]) const { for (int i = 0; i < 4; ++i) out[i] = s_[i]; }
    void  set_state(const u64 in[4]) { for (int i = 0; i < 4; ++i) s_[i] = in[i]; has_spare_ = false; spare_ = 0.0f; }
    void  get_full_state(u64 out[4], float& spare, bool& has_spare) const {
        for (int i = 0; i < 4; ++i) out[i] = s_[i];
        spare = spare_;
        has_spare = has_spare_;
    }
    void  set_full_state(const u64 in[4], float spare, bool has_spare) {
        for (int i = 0; i < 4; ++i) s_[i] = in[i];
        spare_ = spare;
        has_spare_ = has_spare;
    }

private:
    static u64 rotl(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
    u64   s_[4]{};
    float spare_     = 0.0f;
    bool  has_spare_ = false;
};

// 64-bit FNV-1a, used for dedup / eval-contamination hashing
inline u64 fnv1a64(const void* data, size_t n) {
    const u8* p = static_cast<const u8*>(data);
    u64 h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

inline u64 hash_string(const std::string& s) { return fnv1a64(s.data(), s.size()); }

inline u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace gai
