#include "model/model.h"
#include "training/checkpoint.h"
#include "training/optimizer.h"

#include <filesystem>
#include <iostream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static ModelConfig test_config() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 64;
    cfg.use_moe = true;
    cfg.num_experts = 2;
    cfg.moe_top_k = 1;
    cfg.moe_expert_dim = 16;
    cfg.moe_shared = false;
    cfg.moe_aux_scale = 0.0f;
    cfg.moe_aux_free = true;
    return cfg;
}

int main() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "gai_ema_bias_persist_test.ckpt";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ModelConfig cfg = test_config();
    Model source(cfg, Device::CPU);
    source.init_weights(3);
    const float expected0[2] = {0.125f, -0.25f};
    const float expected1[2] = {-0.375f, 0.5f};
    source.set_moe_bias(0, expected0, 2);
    source.set_moe_bias(1, expected1, 2);
    source.enable_grad(true);
    AdamWConfig opt_cfg;
    AdamW source_opt(source, opt_cfg);
    TrainState state;
    state.step = 7;
    state.tokens_seen = 70;
    state.tok_vocab = cfg.vocab_size;
    Checkpoint::save(path.string(), source, source_opt, state);

    Model loaded_model(cfg, Device::CPU);
    loaded_model.init_weights(99);
    loaded_model.enable_grad(true);
    AdamW loaded_opt(loaded_model, opt_cfg);
    TrainState loaded_state;
    bool moments = false;
    CHECK(Checkpoint::load(path.string(), loaded_model, &loaded_opt, loaded_state, &moments),
          "v10 checkpoint loads");
    CHECK(moments, "optimizer moments restored");
    CHECK(loaded_state.step == 7, "training step restored");
    const auto& bias = loaded_model.moe_bias_all();
    CHECK(bias.size() == 2 && bias[0].size() == 2 && bias[1].size() == 2,
          "router bias shape restored");
    CHECK(bias[0][0] == expected0[0] && bias[0][1] == expected0[1] &&
          bias[1][0] == expected1[0] && bias[1][1] == expected1[1],
          "router bias values round-trip");
    std::filesystem::remove(path, ec);
    if (failures == 0) {
        std::cout << "test_ema_bias_persist: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_ema_bias_persist: " << failures << " FAILURES\n";
    return 1;
}
