#include "core/ops.h"
#include "inference/kv_cache.h"
#include "model/model.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    ModelConfig cfg;
    cfg.vocab_size = 8;
    cfg.hidden_size = 8;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 16;
    cfg.max_seq_len = 16;
    cfg.use_moe = false;

    const int capacity = 4;
    const int pinned = 1;
    const int heads = cfg.num_heads;
    const int kv_heads = cfg.num_kv_heads;
    const int head_dim = cfg.head_dim();
    const int kv_dim = cfg.kv_dim();
    KVCache cache(cfg, capacity, Device::CPU);
    cache.set_pinned_prefix(pinned);

    std::vector<int> slots;
    for (int logical = 0; logical < 6; ++logical) {
        const int slot = cache.allocate_slot();
        slots.push_back(slot);
        float* k = cache.k(0) + static_cast<size_t>(slot) * kv_dim;
        float* v = cache.v(0) + static_cast<size_t>(slot) * kv_dim;
        for (int i = 0; i < kv_dim; ++i) {
            k[i] = 1.0f;
            v[i] = static_cast<float>(logical);
        }
    }
    const std::vector<int> expected_slots = {0, 1, 2, 3, 1, 2};
    CHECK(slots == expected_slots, "ring overwrites oldest rolling slots");
    CHECK(cache.length() == capacity, "ring length stays capped");
    CHECK(cache.ring_start() == 2, "ring start advances");
    const std::vector<int> expected_physical = {0, 3, 1, 2};
    for (int logical = 0; logical < capacity; ++logical) {
        CHECK(cache.physical_slot(logical) == expected_physical[static_cast<size_t>(logical)],
              "logical order survives wrap");
    }

    std::vector<float> q(static_cast<size_t>(heads) * head_dim, 1.0f);
    std::vector<float> out(static_cast<size_t>(heads) * head_dim, 0.0f);
    std::vector<float> scratch(static_cast<size_t>(heads) * cache.length(), 0.0f);
    ops::attention_decode_ring(Device::CPU, q.data(), cache.k(0), cache.v(0), out.data(),
                               heads, kv_heads, head_dim, cache.ring_start(),
                               cache.pinned_prefix(), cache.length(), cache.capacity(),
                               1.0f, scratch.data(), 0);
    for (float value : out) CHECK(std::fabs(value - 3.0f) < 1e-5f, "ring attention reads window");

    if (failures == 0) {
        std::cout << "test_kv_ring: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_kv_ring: " << failures << " FAILURES\n";
    return 1;
}
