#pragma once

#include "core/tensor.h"
#include "model/model.h"

namespace gai {

// Contiguous per-layer KV cache. Layout per layer: [max_len, num_kv_heads, head_dim]
// which is exactly what the attention kernels expect, so no reshaping at decode time.
class KVCache {
public:
    KVCache() = default;
    KVCache(const ModelConfig& cfg, int max_len, Device dev);

    void reset() { len_ = 0; }
    int  length() const { return len_; }
    int  capacity() const { return max_len_; }
    void set_length(int n) { len_ = n; }
    void advance(int n) { len_ += n; }

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
};

} // namespace gai
