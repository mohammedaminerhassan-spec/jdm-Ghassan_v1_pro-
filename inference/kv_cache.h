#pragma once

#include "core/tensor.h"
#include "core/device.h"
#include "model/model.h"
#include <utility>
#include <vector>

namespace gai {

// Contiguous per-layer KV cache. Layout per layer: [max_len, num_kv_heads, head_dim]
// which is exactly what the attention kernels expect, so no reshaping at decode time.
class KVCache {
public:
    KVCache() = default;
    KVCache(const ModelConfig& cfg, int max_len, Device dev);
    KVCache(KVCache&& o) noexcept
        : k_(std::move(o.k_)), v_(std::move(o.v_)),
          max_len_(o.max_len_), kv_dim_(o.kv_dim_),
          layers_(o.layers_), len_(o.len_),
          pinned_(o.pinned_), start_(o.start_) {
        o.len_ = 0;
        o.pinned_ = 0;
        o.start_ = 0;
    }
    KVCache& operator=(KVCache&& o) noexcept {
        if (this != &o) {
            k_ = std::move(o.k_);
            v_ = std::move(o.v_);
            max_len_ = o.max_len_;
            kv_dim_ = o.kv_dim_;
            layers_ = o.layers_;
            len_ = o.len_;
            pinned_ = o.pinned_;
            start_ = o.start_;
            o.len_ = 0;
            o.pinned_ = 0;
            o.start_ = 0;
        }
        return *this;
    }
    KVCache(const KVCache&)            = delete;
    KVCache& operator=(const KVCache&) = delete;

    void reset() {
        len_ = 0;
        start_ = 0;
    }
    int  length() const { return len_; }
    int  capacity() const { return max_len_; }
    int  pinned_prefix() const { return pinned_; }
    int  ring_start() const { return start_; }
    void set_pinned_prefix(int n) {
        GAI_CHECK(n >= 0 && n <= max_len_, "KVCache::set_pinned_prefix out of range");
        pinned_ = n;
    }
    void set_length(int n) {
        GAI_CHECK(n >= 0 && n <= max_len_, "KVCache::set_length out of range");
        len_ = n;
        start_ = 0;
    }
    int physical_slot(int logical) const {
        GAI_CHECK(logical >= 0 && logical < len_, "KVCache::physical_slot out of range");
        if (logical < pinned_) return logical;
        const int ring_capacity = max_len_ - pinned_;
        GAI_CHECK(ring_capacity > 0, "KVCache has no rolling slots");
        GAI_CHECK(start_ >= 0 && start_ < ring_capacity, "KVCache ring start out of range");
        return pinned_ + ((start_ + (logical - pinned_)) % ring_capacity);
    }
    int allocate_slot() {
        GAI_CHECK(pinned_ >= 0 && pinned_ <= max_len_, "KVCache pinned prefix out of range");
        if (len_ < max_len_) {
            int slot = len_;
            if (len_ >= pinned_) {
                const int ring_capacity = max_len_ - pinned_;
                GAI_CHECK(ring_capacity > 0, "KVCache has no rolling slots");
                slot = pinned_ + ((start_ + (len_ - pinned_)) % ring_capacity);
            }
            ++len_;
            return slot;
        }
        GAI_CHECK(pinned_ < max_len_, "KVCache is full and has no rolling slots");
        const int ring_capacity = max_len_ - pinned_;
        const int slot = pinned_ + (start_ % ring_capacity);
        start_ = (start_ + 1) % ring_capacity;
        return slot;
    }

    float* k(int layer) { return k_[static_cast<size_t>(layer)].f32(); }
    float* v(int layer) { return v_[static_cast<size_t>(layer)].f32(); }
    const float* k(int layer) const { return k_[static_cast<size_t>(layer)].f32(); }
    const float* v(int layer) const { return v_[static_cast<size_t>(layer)].f32(); }

    size_t bytes() const;

private:
    std::vector<Tensor> k_, v_;
    int max_len_ = 0;
    int kv_dim_  = 0;
    int layers_  = 0;
    int len_     = 0;
    int pinned_  = 0;
    int start_   = 0;
};

} // namespace gai
