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
    cfg.vocab_size = 32;
    cfg.hidden_size = 8;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 16;
    cfg.max_seq_len = 32;
    cfg.use_moe = false;
    return cfg;
}

int main() {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "gai_save_barrier_test";
    const std::filesystem::path good = dir / "last.ckpt";
    const std::filesystem::path blocked = dir / "blocked";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);

    ModelConfig cfg = test_config();
    Model model(cfg, Device::CPU);
    model.init_weights(13);
    model.enable_grad(true);
    AdamWConfig opt_cfg;
    AdamW opt(model, opt_cfg);
    TrainState state;
    state.step = 3;
    state.tok_vocab = cfg.vocab_size;
    Checkpoint::save(good.string(), model, opt, state);

    TrainState loaded;
    CHECK(Checkpoint::load(good.string(), model, &opt, loaded), "baseline checkpoint loads");

    std::filesystem::create_directories(blocked, ec);
    const std::string blocked_tmp = blocked.string() + ".tmp";
    bool threw = false;
    try {
        Checkpoint::save(Checkpoint::capture(model, opt, state), blocked.string());
    } catch (...) {
        threw = true;
    }
    CHECK(threw, "invalid checkpoint destination fails before publishing");
    CHECK(!std::filesystem::exists(blocked_tmp), "TmpGuard: no stale .tmp file remains after save failure");
    CHECK(std::filesystem::is_directory(blocked), "blocked directory target remains a directory");

    TrainState reloaded;
    CHECK(Checkpoint::load(good.string(), model, &opt, reloaded), "last checkpoint survives failed save");
    CHECK(reloaded.step == 3, "surviving checkpoint state is intact");
    CHECK(!std::filesystem::exists(good.string() + ".tmp"), "successful save cleans up .tmp file");
    CHECK(!std::filesystem::exists(good.string() + ".bak"), "successful save cleans up .bak file");

    // Replacement save: the previous checkpoint is temporarily staged as .bak,
    // but a successful publish must remove that recovery artifact so a long
    // Kaggle run does not retain an extra full checkpoint on disk.
    state.step = 4;
    Checkpoint::save(good.string(), model, opt, state);
    CHECK(std::filesystem::exists(good), "replacement checkpoint exists after publish");
    CHECK(!std::filesystem::exists(good.string() + ".bak"),
          "successful replacement removes retained .bak checkpoint");
    TrainState replaced;
    CHECK(Checkpoint::load(good.string(), model, &opt, replaced),
          "replacement checkpoint loads");
    CHECK(replaced.step == 4, "replacement checkpoint state is current");

    const std::filesystem::path backup = good.string() + ".bak";
    std::filesystem::rename(good, backup, ec);
    CHECK(!ec, "interrupted publish fixture staged");
    CHECK(Checkpoint::latest_in(dir.string()) == backup.string(),
          "interrupted publish falls back to staged backup");

    std::filesystem::remove_all(dir, ec);
    if (failures == 0) {
        std::cout << "test_save_barrier: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_save_barrier: " << failures << " FAILURES\n";
    return 1;
}
