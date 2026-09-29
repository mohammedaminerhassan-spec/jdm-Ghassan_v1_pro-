#include "training/trainer.h"
#include "training/checkpoint.h"
#include "core/config.h"
#include "core/ops.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static ModelConfig tiny_config() {
    ModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 32;
    return cfg;
}

static void write_shard(const std::filesystem::path& path, u32 vocab) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ShardWriter writer(path.string(), vocab, false);
    std::vector<i32> doc;
    for (int i = 0; i < 512; ++i) doc.push_back(static_cast<i32>((i * 7) % vocab));
    for (int d = 0; d < 8; ++d) writer.add_document(doc);
    writer.close();
}

static TrainerConfig scaler_cfg(const std::string& data_dir, const std::string& ckpt_dir,
                                double max_scale) {
    TrainerConfig c;
    c.data_dir = data_dir;
    c.train_prefix = "train";
    c.val_prefix = "val";
    c.batch_size = 1;
    c.seq_len = 8;
    c.grad_accum = 1;
    c.max_steps = 2;
    c.epochs = 0;
    c.warmup_steps = 1;
    c.learning_rate = 1e-3f;
    c.min_lr_ratio = 0.1f;
    c.scheduler = "cosine";
    c.optimizer = "adamw";
    c.log_every = 0;
    c.eval_every = 0;
    c.eval_batches = 0;
    c.save_every = 1;
    c.checkpoint_dir = ckpt_dir;
    c.device = "cpu";
    c.seed = 5;
    c.gemm_fp16 = true;
    c.loss_scale_init = 100.0;
    c.loss_scale_window = 1;
    c.loss_scale_max = max_scale;
    c.stage = "pretrain";
    return c;
}

static double run_and_read(const std::string& data_dir, const std::string& ckpt_dir,
                           double max_scale) {
    Model model(tiny_config(), Device::CPU);
    model.init_weights(5);
    model.enable_grad(true);
    TrainerConfig cfg = scaler_cfg(data_dir, ckpt_dir, max_scale);
    Trainer trainer(model, cfg);
    ops::set_gemm_fp16(true);
    trainer.run();
    ops::set_gemm_fp16(false);
    ModelConfig ckpt_cfg;
    TrainState st;
    if (!Checkpoint::peek(ckpt_dir + "/last.ckpt", ckpt_cfg, st)) return -1.0;
    return st.loss_scale;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "gai_scaler_cap";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "data", ec);
    fs::create_directories(root / "ckpt_free", ec);
    fs::create_directories(root / "ckpt_cap", ec);

    const u32 vocab = 32;
    write_shard(root / "data" / "train_0000.gbin", vocab);
    write_shard(root / "data" / "val_0000.gbin", vocab);
    const std::string data_dir = (root / "data").string();

    const double free_scale = run_and_read(data_dir, (root / "ckpt_free").string(), 0.0);
    CHECK(free_scale == 400.0, "uncapped scaler grows 100->200->400 (window=1, 2 steps)");
    const double capped_scale = run_and_read(data_dir, (root / "ckpt_cap").string(), 150.0);
    CHECK(capped_scale == 150.0, "loss_scale_max=150 stops growth at 150");

    fs::remove_all(root, ec);
    if (failures == 0) { std::cout << "test_scaler_cap: ALL PASS\n"; return 0; }
    std::cerr << "test_scaler_cap: " << failures << " FAILURES\n";
    return 1;
}
