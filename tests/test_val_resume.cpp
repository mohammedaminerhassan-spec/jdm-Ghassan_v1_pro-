#include "training/trainer.h"
#include "training/checkpoint.h"

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

static TrainerConfig val_cfg(const std::string& data_dir, const std::string& ckpt_dir,
                             i64 max_steps) {
    TrainerConfig c;
    c.data_dir = data_dir;
    c.train_prefix = "train";
    c.val_prefix = "val";
    c.batch_size = 1;
    c.seq_len = 8;
    c.grad_accum = 1;
    c.max_steps = max_steps;
    c.epochs = 0;
    c.warmup_steps = 1;
    c.learning_rate = 1e-3f;
    c.min_lr_ratio = 0.1f;
    c.scheduler = "cosine";
    c.optimizer = "adamw";
    c.log_every = 0;
    c.eval_every = 1;
    c.eval_batches = 2;
    c.save_every = 2;
    c.checkpoint_dir = ckpt_dir;
    c.device = "cpu";
    c.seed = 5;
    c.gemm_fp16 = false;
    c.loss_scale_init = 0.0;
    c.stage = "pretrain";
    return c;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "gai_val_resume";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "data", ec);
    fs::create_directories(root / "ckptA", ec);
    fs::create_directories(root / "ckptB", ec);

    const u32 vocab = 32;
    write_shard(root / "data" / "train_0000.gbin", vocab);
    write_shard(root / "data" / "val_0000.gbin", vocab);
    const std::string data_dir = (root / "data").string();

    double best_a = 0.0;
    {
        Model model(tiny_config(), Device::CPU);
        model.init_weights(5);
        model.enable_grad(true);
        TrainerConfig cfg = val_cfg(data_dir, (root / "ckptA").string(), 4);
        Trainer trainer(model, cfg);
        trainer.run();
        CHECK(trainer.state().step == 4, "run A completed 4 steps");
        best_a = trainer.state().best_val;
        ModelConfig ckpt_cfg;
        TrainState st;
        CHECK(Checkpoint::peek((root / "ckptA" / "last.ckpt").string(), ckpt_cfg, st),
              "run A last.ckpt readable");
        CHECK(st.val_loader.batches == 8,
              "run A checkpoint carries the val cursor (8 batches)");
    }

    {
        Model model(tiny_config(), Device::CPU);
        model.init_weights(5);
        model.enable_grad(true);
        TrainerConfig cfg = val_cfg(data_dir, (root / "ckptB").string(), 2);
        Trainer trainer(model, cfg);
        trainer.run();
        CHECK(trainer.state().step == 2, "run B completed 2 steps");
        ModelConfig ckpt_cfg;
        TrainState st;
        CHECK(Checkpoint::peek((root / "ckptB" / "last.ckpt").string(), ckpt_cfg, st),
              "run B last.ckpt readable");
        CHECK(st.val_loader.batches == 4,
              "run B checkpoint carries the val cursor (4 batches)");
    }

    {
        Model model(tiny_config(), Device::CPU);
        model.init_weights(1);
        model.enable_grad(true);
        TrainerConfig cfg = val_cfg(data_dir, (root / "ckptB").string(), 4);
        cfg.resume = "auto";
        Trainer trainer(model, cfg);
        trainer.run();
        CHECK(trainer.state().step == 4, "resumed run reached step 4");
        CHECK(trainer.state().best_val == best_a,
              "resumed best_val equals the uninterrupted run (val stream continued)");
        ModelConfig ckpt_cfg;
        TrainState st;
        CHECK(Checkpoint::peek((root / "ckptB" / "last.ckpt").string(), ckpt_cfg, st),
              "resumed last.ckpt readable");
        CHECK(st.val_loader.batches == 8,
              "resumed checkpoint val cursor continued (8 batches, not restarted)");
    }

    fs::remove_all(root, ec);
    if (failures == 0) { std::cout << "test_val_resume: ALL PASS\n"; return 0; }
    std::cerr << "test_val_resume: " << failures << " FAILURES\n";
    return 1;
}
