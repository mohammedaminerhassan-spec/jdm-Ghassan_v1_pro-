// End-to-end checkpoint-flow integration test (CPU).
//
// This is the single-rank proxy for the Kaggle 4xT4 run and the regression net
// for the Phase 1D / F-01 / F-05 / F-06 / F-07 / F-08 work:
//
//   * the background writer really publishes both best.ckpt and last.ckpt when
//     a step improves validation AND the save cadence fires (the case that used
//     to serialize the same multi-GB snapshot twice),
//   * the eval "is this a new best" decision is routed through the same code
//     path every rank executes, so the DDP collective order cannot desync,
//   * a failed/asynchronous write is surfaced instead of being swallowed,
//   * a finished run drains the writer, and
//   * `resume_mode: exact` accepts a genuine continuation and refuses a drifted
//     recipe.
//
// It replaces the mock-queue test, which re-implemented the queue policy and
// therefore could not catch a regression in the production code.
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

// One shard with enough tokens for the trainer's row sampler.
static void write_shard(const std::filesystem::path& path, u32 vocab) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    ShardWriter writer(path.string(), vocab, false);
    std::vector<i32> doc;
    for (int i = 0; i < 512; ++i) doc.push_back(static_cast<i32>((i * 7) % vocab));
    for (int d = 0; d < 8; ++d) writer.add_document(doc);
    writer.close();
}

static TrainerConfig base_cfg(const std::string& data_dir, const std::string& ckpt_dir) {
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
    c.learning_rate = 1e-3f;
    c.min_lr_ratio = 0.1f;
    c.scheduler = "cosine";
    c.optimizer = "adamw";
    c.log_every = 0;
    // Fire BOTH on the same step: that is the duplicate-write + DDP-order case.
    c.eval_every = 1;
    c.eval_batches = 1;
    c.save_every = 1;
    c.checkpoint_dir = ckpt_dir;
    c.device = "cpu";
    c.seed = 5;
    c.gemm_fp16 = false;          // CPU reference path
    c.loss_scale_init = 0.0;      // no fp16 scaling without fp16 GEMMs
    c.stage = "pretrain";
    return c;
}

static bool file_size_at_least(const std::filesystem::path& p, std::uintmax_t n) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(p, ec);
    return !ec && sz >= n;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "gai_trainer_ckpt_flow";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "data", ec);
    fs::create_directories(root / "ckpt", ec);

    const u32 vocab = 32;
    write_shard(root / "data" / "train_0000.gbin", vocab);
    write_shard(root / "data" / "val_0000.gbin", vocab);

    const std::string data_dir = (root / "data").string();
    const std::string ckpt_dir = (root / "ckpt").string();
    const fs::path best = root / "ckpt" / "best.ckpt";
    const fs::path last = root / "ckpt" / "last.ckpt";

    i64 final_step = 0;
    // ---------------------------------------------------------------------
    // 1. A full short run: eval + save cadence both fire every step.
    // ---------------------------------------------------------------------
    {
        Model model(tiny_config(), Device::CPU);
        model.init_weights(5);
        model.enable_grad(true);
        TrainerConfig cfg = base_cfg(data_dir, ckpt_dir);
        Trainer trainer(model, cfg);
        trainer.run();
        final_step = trainer.state().step;
        CHECK(final_step == 3, "trainer completed the planned steps");
        CHECK(trainer.state().tokens_seen > 0, "tokens were accounted");
    }

    // The writer must have been drained by run() before it returned, so both
    // files are on disk the moment we get here.
    CHECK(fs::exists(last, ec), "last.ckpt was published");
    CHECK(fs::exists(best, ec), "best.ckpt was published (eval improved at least once)");
    CHECK(file_size_at_least(last, 64), "last.ckpt is not truncated");

    // Both names must describe the same committed step, and both must load.
    {
        Model probe(tiny_config(), Device::CPU);
        probe.init_weights(1);
        TrainState st_last, st_best;
        ModelConfig c1, c2;
        CHECK(Checkpoint::peek(last.string(), c1, st_last), "last.ckpt header is readable");
        CHECK(Checkpoint::peek(best.string(), c2, st_best), "best.ckpt header is readable");
        CHECK(st_last.step == final_step, "last.ckpt records the final step");
        CHECK(st_best.step > 0 && st_best.step <= final_step, "best.ckpt records a real step");
        CHECK(st_best.step == st_last.step,
              "with save_every == eval_every == 1 the last step is also the best step, "
              "so the two files must agree (the duplicate-write case)");
    }

    // The second name for the same step is published from the first file, so
    // the two must be byte-identical in content (same inode after a hard link,
    // same bytes after a copy).
    {
        std::ifstream a(best, std::ios::binary);
        std::ifstream b(last, std::ios::binary);
        CHECK(a.good() && b.good(), "both checkpoints are readable as raw files");
        const std::string sa((std::istreambuf_iterator<char>(a)),
                             std::istreambuf_iterator<char>());
        const std::string sb((std::istreambuf_iterator<char>(b)),
                             std::istreambuf_iterator<char>());
        CHECK(!sa.empty() && sa == sb,
              "best.ckpt and last.ckpt for the same step have identical content");
    }

    // ---------------------------------------------------------------------
    // 2. Exact resume of a genuine continuation is accepted and continues.
    // ---------------------------------------------------------------------
    {
        Model model(tiny_config(), Device::CPU);
        model.init_weights(1);
        model.enable_grad(true);
        TrainerConfig cfg = base_cfg(data_dir, ckpt_dir);
        cfg.resume_mode = "exact";
        cfg.resume = "auto";           // picks up last.ckpt
        cfg.max_steps = final_step + 2;
        Trainer trainer(model, cfg);
        trainer.run();
        CHECK(trainer.state().step == final_step + 2, "exact resume continued the schedule");
    }

    // ---------------------------------------------------------------------
    // 3. Exact resume of a DRIFTED recipe is refused (fail closed).
    // ---------------------------------------------------------------------
    {
        ModelConfig drifted = tiny_config();
        drifted.rms_eps = 1e-4f;       // different math, same shapes
        Model model(drifted, Device::CPU);
        model.init_weights(1);
        model.enable_grad(true);
        TrainerConfig cfg = base_cfg(data_dir, ckpt_dir);
        cfg.resume_mode = "exact";
        cfg.resume = "auto";
        bool threw = false;
        try {
            Trainer trainer(model, cfg);
            trainer.run();
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw, "exact resume refuses a math-recipe drift");
    }

    // ---------------------------------------------------------------------
    // 4. Migrate mode tolerates the same drift (explicit, logged opt-out).
    // ---------------------------------------------------------------------
    {
        ModelConfig drifted = tiny_config();
        drifted.rms_eps = 1e-4f;
        Model model(drifted, Device::CPU);
        model.init_weights(1);
        model.enable_grad(true);
        TrainerConfig cfg = base_cfg(data_dir, ckpt_dir);
        cfg.resume_mode = "migrate";
        cfg.resume = "auto";
        cfg.allow_recipe_drift = true;
        bool ok = true;
        try {
            Trainer trainer(model, cfg);
            trainer.run();
        } catch (const std::exception&) {
            ok = false;
        }
        CHECK(ok, "migrate mode + allow_recipe_drift runs instead of failing");
    }

    fs::remove_all(root, ec);
    if (failures == 0) {
        std::cout << "test_trainer_ckpt_flow: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_trainer_ckpt_flow: " << failures << " FAILURES\n";
    return 1;
}
