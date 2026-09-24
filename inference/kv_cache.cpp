#include "inference/kv_cache.h"
#include "core/device.h"

#include <cstring>
#include <vector>

namespace gai {

KVCache::KVCache(const ModelConfig& cfg, int max_len, Device dev)
    : max_len_(max_len), kv_dim_(cfg.kv_dim()), layers_(cfg.num_layers), len_(0),
      pinned_(0), start_(0) {
    k_.resize(static_cast<size_t>(layers_));
    v_.resize(static_cast<size_t>(layers_));
    for (int i = 0; i < layers_; ++i) {
        k_[static_cast<size_t>(i)] = Tensor::zeros({max_len_, kv_dim_}, DType::F32, dev);
        v_[static_cast<size_t>(i)] = Tensor::zeros({max_len_, kv_dim_}, DType::F32, dev);
    }
}

size_t KVCache::bytes() const {
    size_t n = 0;
    for (const auto& t : k_) n += t.nbytes();
    for (const auto& t : v_) n += t.nbytes();
    return n;
}

} // namespace gai
