// P0-2 regression: exact resume of a genuine LION continuation must be
// ACCEPTED. The scheduler gate used to compare the saved (effective Lion)
// peak against raw learning_rate, so every exact Lion resume failed with a
// bogus "scheduler recipe differs" mismatch (3e-5 vs 3e-4).
//
// Mirrors tests/test_trainer_ckpt_flow.cpp on CPU with a tiny model:
// run 3 Lion steps -> exact-resume for 2 more -> the schedule continues.
// Before the effective_peak_lr() fix, step 2 aborted here.
#include "training/trainer.h"
#include "training/checkpoint.h"

#include <filesystem>
#include <fstream>
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

static TrainerConfig lion_cfg(const std::string& data_dir, const std::string& ckpt_dir) {
    TrainerConfig c;
    c.data_dir = data_dir;
    c.train_prefix = "train";
    c.val_prefix = "val";
    c.batch_size = 1;
    c.seq_len = 8;
    c.grad_accum = 1;
    c.max_steps = 3;
    c.epochs = 0;
    c.warmup_steps = 1;
    c.learning_rate = 3e-4f;   // Lion effective peak must be 3e-5, not 3e-4
    c.min_lr_ratio = 0.1f;
    c.scheduler = "wsd";
    c.optimizer = "lion";
    c.beta2 = 0.99f;
    c.log_every = 0;
    c.eval_every = 1;
    c.eval_batches = 1;
    c.save_every = 1;
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
    const fs::path root = fs::temp_directory_path() / "gai_lion_exact_resume";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "data", ec);
    fs::create_directories(root / "ckpt", ec);

    const u32 vocab = 32;
    write_shard(root / "data" / "train_0000.gbin", vocab);
    write_shard(root / "data" / "val_0000.gbin", vocab);

    const std::string data_dir = (root / "data").string();
    const std::string ckpt_dir = (root / "ckpt").string();

    i64 final_step = 0;
    {
        Model model(tiny_config(), Device::CPU);
        model.init_weights(5);
        model.enable_grad(true);
        TrainerConfig cfg = lion_cfg(data_dir, ckpt_dir);
        Trainer trainer(model, cfg);
        trainer.run();
        final_step = trainer.state().step;
        CHECK(final_step == 3, "lion run completed the planned steps");
    }
    CHECK(fs::exists(root / "ckpt" / "last.ckpt", ec), "last.ckpt was published");

    // The saved scheduler peak must be the EFFECTIVE Lion peak (0.1x).
    {
        ModelConfig ckpt_cfg;
        TrainState st;
        CHECK(Checkpoint::peek((root / "ckpt" / "last.ckpt").string(), ckpt_cfg, st),
              "last.ckpt header is readable");
        const double rel = std::fabs((double)st.sched_peak - 3e-5) / 3e-5;
        CHECK(rel < 1e-4, "saved sched_peak is the Lion effective peak (3e-5)");
    }

    // Exact resume of the genuine Lion continuation is accepted.
    {
        Model model(tiny_config(), Device::CPU);
        model.init_weights(1);
        model.enable_grad(true);
        TrainerConfig cfg = lion_cfg(data_dir, ckpt_dir);
        cfg.resume_mode = "exact";
        cfg.resume = "auto";
        cfg.max_steps = final_step + 2;
        Trainer trainer(model, cfg);
        trainer.run();
        CHECK(trainer.state().step == final_step + 2,
              "exact Lion resume continued the schedule (P0-2 fixed)");
    }

    fs::remove_all(root, ec);
    if (failures == 0) { std::cout << "test_lion_exact_resume: ALL PASS\n"; return 0; }
    std::cerr << "test_lion_exact_resume: " << failures << " FAILURES\n";
    return 1;
}
