#include "core/dtype.h"
#include "core/ops.h"
#include "model/model.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static void test_model_cache_parity() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 32;
    cfg.use_moe = false;
    Model model(cfg, Device::CPU);
    model.init_weights(5);
    const std::vector<i32> ids = {1, 2, 3, 4};
    Activations reference = model.make_activations(1, 4, false);
    Tensor& a = model.forward(ids.data(), 1, 4, reference);
    std::vector<float> expected(a.f32(), a.f32() + a.numel());
    model.enable_fp16_weight_cache(true);
    Activations cached = model.make_activations(1, 4, false);
    Tensor& b = model.forward(ids.data(), 1, 4, cached);
    for (i64 i = 0; i < b.numel(); ++i) {
        float tolerance = 8e-3f * std::max(1.0f, std::fabs(expected[static_cast<size_t>(i)]));
        CHECK(std::fabs(expected[static_cast<size_t>(i)] - b.f32()[i]) <= tolerance,
              "model fp16 cache parity");
    }
}

int main() {
    const int M = 2;
    const int K = 4;
    const int N = 3;
    const std::vector<float> x = {0.5f, -1.0f, 2.0f, 0.25f,
                                  1.5f, 0.75f, -0.5f, 3.0f};
    const std::vector<float> w = {0.1f, 0.2f, -0.3f, 0.4f,
                                  -0.5f, 0.6f, 0.7f, -0.8f,
                                  0.9f, -1.0f, 1.1f, 0.2f};
    std::vector<u16> half(w.size());
    for (size_t i = 0; i < w.size(); ++i) half[i] = fp32_to_fp16(w[i]);
    std::vector<float> f32(M * N, 0.0f);
    std::vector<float> f16(M * N, 0.0f);
    ops::linear_forward(Device::CPU, x.data(), w.data(), f32.data(), M, K, N);
    ops::linear_forward_fp16(Device::CPU, x.data(), half.data(), f16.data(), M, K, N);
    for (size_t i = 0; i < f32.size(); ++i) {
        float tolerance = 2e-3f * std::max(1.0f, std::fabs(f32[i]));
        CHECK(std::fabs(f32[i] - f16[i]) <= tolerance, "fp16 linear parity");
    }
    test_model_cache_parity();
    if (failures == 0) {
        std::cout << "test_fp16_gemm_parity: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_fp16_gemm_parity: " << failures << " FAILURES\n";
    return 1;
}
