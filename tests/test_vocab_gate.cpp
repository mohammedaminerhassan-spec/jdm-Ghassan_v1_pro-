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
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 32;
    cfg.use_moe = false;
    return cfg;
}

int main() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "gai_vocab_gate_test.ckpt";
    std::error_code ec;
    std::filesystem::remove(path, ec);

    ModelConfig cfg = test_config();
    Model model(cfg, Device::CPU);
    model.init_weights(7);
    model.enable_grad(true);
    AdamWConfig opt_cfg;
    AdamW opt(model, opt_cfg);

    TrainState mismatched;
    mismatched.step = 1;
    mismatched.tok_vocab = 32;
    Checkpoint::save(path.string(), model, opt, mismatched);
    TrainState loaded_state;
    CHECK(!Checkpoint::load(path.string(), model, &opt, loaded_state),
          "checkpoint with mismatched vocab is rejected");

    std::filesystem::remove(path, ec);
    if (failures == 0) {
        std::cout << "test_vocab_gate: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_vocab_gate: " << failures << " FAILURES\n";
    return 1;
}
