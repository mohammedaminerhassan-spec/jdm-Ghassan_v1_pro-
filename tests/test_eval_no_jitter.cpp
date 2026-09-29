// Router jitter is TRAIN-ONLY, and evaluate() must leave it that way.
//
// The contract has two halves and both are easy to break:
//
//   1. Jitter reaches only the training forward. Both routers gate it on the
//      training path (cuda/moe.cu k_route: `train = probs != nullptr`;
//      core/ops_cpu_moe.cpp: `jj > 0 && probs_cache`), so an eval forward must be
//      bit-identical no matter what jitter seed is installed.
//   2. Trainer::evaluate() silences the global and RESTORES it. A restore that
//      is missing (or an evaluate() that zeroes without restoring) leaves every
//      step after the first validation training with a dead router — a silent,
//      permanent loss of exploration that no log line would ever report.
//
// Half 1 is asserted on the real forward/forward_backward pair: the same input
// and weights give different gradients with different jitter seeds on the
// training path, and identical logits on the eval path. Half 2 is asserted
// against a real Trainer.
#include "training/trainer.h"
#include "core/ops.h"

#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static ModelConfig tiny_moe() {
    ModelConfig cfg;
    cfg.vocab_size        = 32;
    cfg.hidden_size       = 32;
    cfg.num_layers        = 2;
    cfg.num_heads         = 4;
    cfg.num_kv_heads      = 2;
    cfg.intermediate_size = 64;
    cfg.max_seq_len       = 32;
    cfg.use_moe           = true;
    cfg.num_experts       = 4;
    cfg.moe_top_k         = 2;
    cfg.moe_expert_dim    = 16;
    cfg.moe_shared        = true;
    cfg.use_qk_norm       = true;
    cfg.moe_jitter        = 0.01f;   // the value every shipped MoE recipe uses
    return cfg;
}

static void write_shards(const fs::path& dir, u32 vocab) {
    fs::create_directories(dir);
    std::vector<i32> doc;
    for (int i = 0; i < 256; ++i) doc.push_back(static_cast<i32>((i * 7) % vocab));
    for (const char* prefix : {"train", "val"}) {
        const std::string p = (dir / (std::string(prefix) + "_dom_0000.gbin")).string();
        std::error_code ec;
        fs::remove(p, ec);
        ShardWriter w(p, static_cast<int>(vocab), /*with_loss_mask=*/false);
        for (int d = 0; d < 8; ++d) w.add_document(doc);
        w.close();
    }
}

static TrainerConfig cfg_for(const fs::path& dir) {
    TrainerConfig c;
    c.data_dir       = dir.string();
    c.train_prefix   = "train";
    c.val_prefix     = "val";
    c.mix["dom"]     = 1.0;
    c.batch_size     = 1;
    c.seq_len        = 16;
    c.grad_accum     = 1;
    c.max_steps      = 1;
    c.epochs         = 0;
    c.warmup_steps   = 0;
    c.learning_rate  = 1e-4f;
    c.min_lr_ratio   = 0.1f;
    c.scheduler      = "cosine";
    c.optimizer      = "adamw";
    c.device         = "cpu";
    c.ddp            = false;
    c.eval_every     = 0;          // evaluate() is called explicitly below
    c.save_every     = 0;
    c.log_every      = 0;
    c.checkpoint_dir = (dir / "ckpt").string();
    return c;
}

int main() {
    const fs::path root = fs::temp_directory_path() / "gai_test_eval_no_jitter";
    std::error_code ec;
    fs::remove_all(root, ec);
    write_shards(root, 32);

    // ---- half 1: the TRAINING path is jitter-sensitive, the EVAL path is not.
    {
        const ModelConfig m = tiny_moe();
        Model model(m, Device::CPU);
        model.init_weights(7);
        model.enable_grad(true);
        ops::set_moe_jitter(m.moe_jitter);

        std::vector<i32> ids(16, 3);
        std::vector<i32> tgt(16, 4);
        Activations tr = model.make_activations(1, 16, true, 1);
        model.zero_grad();
        ops::set_moe_jitter_seed(1);
        model.forward_backward(ids.data(), tgt.data(), 1, 16, tr);
        const float g1 = model.find_parameter("layers.0.moe_gate")->g.f32()[0];
        model.zero_grad();
        ops::set_moe_jitter_seed(2);          // same weights, different noise
        model.forward_backward(ids.data(), tgt.data(), 1, 16, tr);
        const float g2 = model.find_parameter("layers.0.moe_gate")->g.f32()[0];
        CHECK(g1 != g2,
              "control: the training path IS jitter-sensitive (two seeds differ) "
              "— otherwise this test cannot observe a leak");

        Activations ev = model.make_activations(1, 16, false);
        ops::set_moe_jitter_seed(1);
        Tensor& a = model.forward(ids.data(), 1, 16, ev);
        const float l1 = a.f32()[0];
        ops::set_moe_jitter_seed(2);
        Tensor& b = model.forward(ids.data(), 1, 16, ev);
        CHECK(l1 == b.f32()[0],
              "the eval path (no probs cache) is never jittered: logits are "
              "bit-identical across jitter seeds");
    }

    // ---- half 2: Trainer::evaluate() must restore the configured jitter.
    {
        const ModelConfig m = tiny_moe();
        Model model(m, Device::CPU);
        model.init_weights(11);
        model.enable_grad(true);
        Trainer trainer(model, cfg_for(root));
        trainer.run();                        // wires loaders + eval arena
        CHECK(ops::moe_jitter() == m.moe_jitter,
              "the run installed the recipe's moe_jitter (" +
                  std::to_string(ops::moe_jitter()) + ")");
        const double v = trainer.evaluate(4);
        CHECK(v > 0.0, "evaluate() returned a real loss on the tiny shard");
        CHECK(ops::moe_jitter() == m.moe_jitter,
              "evaluate() restores the configured moe_jitter — a missing restore "
              "would silently kill router exploration for the rest of the run "
              "(got " + std::to_string(ops::moe_jitter()) + ")");
    }
    ops::set_moe_jitter(0.0f);

    fs::remove_all(root, ec);
    if (failures == 0) { std::cout << "test_eval_no_jitter: ALL PASS\n"; return 0; }
    std::cerr << "test_eval_no_jitter: " << failures << " FAILURES\n";
    return 1;
}
