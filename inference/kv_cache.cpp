#include "inference/kv_cache.h"
#include "core/device.h"

#include <cstring>
#include <vector>

namespace gai {

KVCache::KVCache(const ModelConfig& cfg, int max_len, Device dev)
    : max_len_(max_len), kv_dim_(cfg.kv_dim()), layers_(cfg.num_layers), len_(0) {
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

void KVCache::evict_front(int n, int keep) {
    if (n <= 0 || len_ <= keep) return;
    n = std::min(n, len_ - keep);
    const size_t row = static_cast<size_t>(kv_dim_) * sizeof(float);
    const size_t move_rows = static_cast<size_t>(len_ - keep - n);
    if (move_rows == 0) { len_ -= n; return; }
    Device dev = k_.empty() ? Device::CPU : k_[0].device();
    if (dev == Device::CPU) {
        for (int l = 0; l < layers_; ++l) {
            char* kp = reinterpret_cast<char*>(k_[static_cast<size_t>(l)].data_ptr());
            char* vp = reinterpret_cast<char*>(v_[static_cast<size_t>(l)].data_ptr());
            std::memmove(kp + static_cast<size_t>(keep) * row,
                         kp + static_cast<size_t>(keep + n) * row, move_rows * row);
            std::memmove(vp + static_cast<size_t>(keep) * row,
                         vp + static_cast<size_t>(keep + n) * row, move_rows * row);
        }
    } else {
        // CUDA: overlapping D2D is undefined. Use per-instance GPU-side temp
        // (allocated on-demand, freed in ~KVCache) to avoid CPU roundtrips and static data races.
        size_t bytes = move_rows * row;
        if (bytes > tmp_cap_) {
            if (tmp_) device_free(tmp_, dev);
            tmp_ = device_alloc(bytes, dev);
            tmp_cap_ = bytes;
        }
        for (int l = 0; l < layers_; ++l) {
            char* kb = static_cast<char*>(k_[static_cast<size_t>(l)].data_ptr());
            char* vb = static_cast<char*>(v_[static_cast<size_t>(l)].data_ptr());
            device_copy(tmp_, dev, kb + static_cast<size_t>(keep + n) * row, dev, bytes);
            device_copy(kb + static_cast<size_t>(keep) * row, dev, tmp_, dev, bytes);
            device_copy(tmp_, dev, vb + static_cast<size_t>(keep + n) * row, dev, bytes);
            device_copy(vb + static_cast<size_t>(keep) * row, dev, tmp_, dev, bytes);
        }
    }
    len_ -= n;
}

} // namespace gai
