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
    ~KVCache() {
        // P1-07 FIX: free the per-instance eviction temp (was a process-lifetime
        // static that was never freed and could corrupt multi-instance setups).
        if (tmp_) { device_free(tmp_, k_.empty() ? Device::CPU : k_[0].device()); tmp_ = nullptr; }
    }
    // Move-only (raw pointer ownership of tmp_)
    KVCache(KVCache&& o) noexcept
        : k_(std::move(o.k_)), v_(std::move(o.v_)),
          max_len_(o.max_len_), kv_dim_(o.kv_dim_),
          layers_(o.layers_), len_(o.len_),
          tmp_(o.tmp_), tmp_cap_(o.tmp_cap_) {
        o.tmp_ = nullptr;
        o.tmp_cap_ = 0;
        o.len_ = 0;
    }
    KVCache& operator=(KVCache&& o) noexcept {
        if (this != &o) {
            if (tmp_) device_free(tmp_, k_.empty() ? Device::CPU : k_[0].device());
            k_ = std::move(o.k_);
            v_ = std::move(o.v_);
            max_len_ = o.max_len_;
            kv_dim_ = o.kv_dim_;
            layers_ = o.layers_;
            len_ = o.len_;
            tmp_ = o.tmp_;
            tmp_cap_ = o.tmp_cap_;
            o.tmp_ = nullptr;
            o.tmp_cap_ = 0;
            o.len_ = 0;
        }
        return *this;
    }
    KVCache(const KVCache&)            = delete;
    KVCache& operator=(const KVCache&) = delete;

    void reset() { len_ = 0; }
    int  length() const { return len_; }
    int  capacity() const { return max_len_; }
    // FIX: unchecked set/advance allowed len_ > max_len_ -> OOB memmove /
    // D2D copy in evict_front/decode_step (inference crash, heap corruption).
    // Fail fast instead of corrupting memory (also protects low-PC long chats).
    void set_length(int n) {
        GAI_CHECK(n >= 0 && n <= max_len_, "KVCache::set_length out of range");
        len_ = n;
    }
    void advance(int n) {
        GAI_CHECK(n >= 0 && len_ + n <= max_len_, "KVCache::advance overflow");
        len_ += n;
    }

    float* k(int layer) { return k_[static_cast<size_t>(layer)].f32(); }
    float* v(int layer) { return v_[static_cast<size_t>(layer)].f32(); }
    const float* k(int layer) const { return k_[static_cast<size_t>(layer)].f32(); }
    const float* v(int layer) const { return v_[static_cast<size_t>(layer)].f32(); }

    size_t bytes() const;

    // Drops the oldest `n` positions, shifting the rest down. Used when a long
    // chat exceeds the context window (with the system prompt kept via `keep`).
    void evict_front(int n, int keep = 0);

private:
    std::vector<Tensor> k_, v_;
    int max_len_ = 0;
    int kv_dim_  = 0;
    int layers_  = 0;
    int len_     = 0;
    // P1-07: per-instance CUDA eviction temp (replaces the old static g_tmp).
    void*  tmp_     = nullptr;
    size_t tmp_cap_ = 0;
};

} // namespace gai
