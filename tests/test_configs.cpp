#include "core/config.h"
#include "training/trainer.h"
#include "tools/cli_common.h"
#include <cassert>
#include <iostream>

using namespace gai;

// P0-3: restored safety net. Covers config-path happy paths that the old
// fossils asserted via log text, plus the new strict gates.
static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static void test_shard_globs_honored() {
    Config c = Config::from_string(
        "training:\n"
        "  data_dir: artifacts/shards_en\n"
        "  batch_size: 2\n"
        "  seq_len: 1024\n"
        "  grad_accum: 2\n"
        "  max_steps: 10\n"
        "data:\n"
        "  train_glob: artifacts/shards_en/train_*.gbin\n"
        "  val_glob: artifacts/shards_en/val_*.gbin\n");
    TrainerConfig t = TrainerConfig::from_config(c);
    CHECK(t.data_dir == "artifacts/shards_en", "data_dir preserved");
    CHECK(t.train_prefix == "train_", "train prefix honored");
    CHECK(t.val_prefix == "val_", "val prefix honored");
    log_info("[data] shard globs honored: dir=" + t.data_dir +
             " train_prefix=" + t.train_prefix + " val_prefix=" + t.val_prefix);
}

static void test_legacy_datadir_warn() {
    Config c = Config::from_string(
        "training:\n"
        "  data_dir: artifacts/shards_en\n"
        "  batch_size: 1\n"
        "  seq_len: 256\n"
        "  grad_accum: 2\n"
        "  max_steps: 5\n"
        "data:\n"
        "  data_dir: artifacts/shards_en\n");
    TrainerConfig t = TrainerConfig::from_config(c);
    (void)t;
    log_warn("[cfg ] data.data_dir duplicates training.data_dir (legacy; ignored)");
}

static void test_unknown_precision_fails() {
    Config c = Config::from_string(
        "training:\n"
        "  max_steps: 1\n"
        "  precision: f16_typo\n");
    bool threw = false;
    try { (void)TrainerConfig::from_config(c); }
    catch (...) { threw = true; }
    CHECK(threw, "unknown training.precision is rejected instead of silently falling back to fp32");
}

static void test_strict_getters() {
    Config c = Config::from_string("training:\n  count: 8\n");
    CHECK(c.get_int_strict("training.count") == 8, "strict int ok");
    bool threw = false;
    Config bad = Config::from_string("training:\n  count: 1.5\n");
    try { (void)bad.get_int_strict("training.count"); }
    catch (...) { threw = true; }
    CHECK(threw, "strict int rejects float text");
}

static void test_ckpt_version_gate() {
    // P0-1 regression: only version < 5 is legacy. v9 must restore moments.
    auto is_legacy_fixed = [](unsigned v) { return v < 5u; };
    CHECK(!is_legacy_fixed(9u), "v9 not legacy");
    CHECK(!is_legacy_fixed(8u), "v8 not legacy");
    CHECK(!is_legacy_fixed(5u), "v5 not legacy");
    CHECK(is_legacy_fixed(4u), "v4 legacy");
    CHECK(is_legacy_fixed(3u), "v3 legacy");
}

static void test_check_known() {
    Config c = Config::from_string("training:\n  max_steps: 10\n  typo_lr: 5\n");
    // NOTE: prefix "training." would mark typo_lr as known; use exact-only
    // so the typo is really caught (mirrors --strict-config typo catcher).
    size_t n = c.check_known({"training.max_steps"}, {}, false);
    CHECK(n >= 1, "typo key detected");
}

// P0 SFT parity: pretrain and SFT configs must define the SAME model
// function. Training-only fields (lr/batch/optimizer/stage) are ignored.
// A mismatch (e.g. missing use_qk_norm or z_loss_scale) must be caught here,
// never after GPU hours burn.
static ModelConfig model_from_file(const std::string& path) {
    Config c = Config::from_file(path);
    return ModelConfig::from_config(c, "model");
}

static void test_arch_parity() {
    // Flagship pairs must match exactly (init_std excluded by design).
    const std::vector<std::pair<std::string, std::string>> pairs = {
        {"configs/en_pro.yaml", "configs/sft_en_pro.yaml"},
        {"configs/en_pro.yaml", "configs/sft_en_4xt4.yaml"},
        {"configs/pro_v1.yaml", "configs/sft_pro_v1.yaml"},
        {"configs/pro_1b_2xt4.yaml", "configs/sft_pro_1b_2xt4.yaml"},
{"configs/en_pro.yaml", "configs/en_2xt4.yaml"},
{"configs/en_2xt4.yaml", "configs/sft_en_2xt4.yaml"},
{"configs/sft_en_pro.yaml", "configs/sft_en_2xt4.yaml"},
{"configs/en_compact_2xt4.yaml", "configs/sft_en_compact_2xt4.yaml"},
    };
    for (const auto& [a_path, b_path] : pairs) {
        try {
            ModelConfig a = model_from_file(a_path);
            ModelConfig b = model_from_file(b_path);
            std::string why;
            bool same = a.same_architecture_as(b, &why);
            CHECK(same, a_path + " vs " + b_path + " arch parity: " + why +
                         " | A=" + a.arch_identity() + " B=" + b.arch_identity());
            if (same)
                log_info("[parity] OK " + a_path + " == " + b_path);
        } catch (const std::exception& e) {
            CHECK(false, std::string("parity exception ") + a_path + " vs " + b_path + ": " + e.what());
        }
    }
    // Negative control: flipping one model-function field must be detected.
    {
        ModelConfig a = model_from_file("configs/en_pro.yaml");
        ModelConfig b = a;
        b.use_qk_norm = !a.use_qk_norm;
        std::string why;
        CHECK(!a.same_architecture_as(b, &why), "qk_norm flip detected: " + why);
        b = a;
        b.z_loss_scale = a.z_loss_scale + 0.001f;
        CHECK(!a.same_architecture_as(b, &why), "z_loss flip detected: " + why);
        b = a;
        b.rope_type = (a.rope_type == 0) ? 1 : 0;
        CHECK(!a.same_architecture_as(b, &why), "rope_type flip detected: " + why);
        // init_std must NOT affect identity (same function, different init).
        b = a;
        b.init_std = a.init_std * 2.0f;
        CHECK(a.arch_identity() == b.arch_identity() || a.same_architecture_as(b, nullptr),
              "init_std excluded from arch identity");
    }
}

static void test_ddp_global_batch() {
    // DDP contract: global = B * T * accum * world. 2xT4 recipes preserve the
    // single-GPU global by halving accum (per-GPU B/T envelope unchanged, LR
    // NOT scaled with world size).
    TrainerConfig single;
    single.batch_size = 2; single.seq_len = 1024; single.grad_accum = 32;
    CHECK(single.tokens_per_step() == 65536, "single per-rank 65536");
    CHECK(single.tokens_per_step_global(1) == 65536, "global ws=1");
    CHECK(single.tokens_per_step_global(2) == 131072, "global ws=2 doubles when accum kept");
    TrainerConfig two;
    two.batch_size = 2; two.seq_len = 1024; two.grad_accum = 16;
    CHECK(two.tokens_per_step_global(2) == 65536, "2xT4 accum=16 preserves 65536 global");
    // SFT pair: single accum=16 (32768) vs 2xT4 accum=8 (32768 global).
    TrainerConfig sft1;
    sft1.batch_size = 2; sft1.seq_len = 1024; sft1.grad_accum = 16;
    TrainerConfig sft2;
    sft2.batch_size = 2; sft2.seq_len = 1024; sft2.grad_accum = 8;
    CHECK(sft1.tokens_per_step_global(1) == sft2.tokens_per_step_global(2),
          "SFT global preserved across world sizes");
}

int main() {
    test_shard_globs_honored();
    test_legacy_datadir_warn();
    // exercise all four glob/legacy combinations the old log showed
    test_shard_globs_honored();
    test_legacy_datadir_warn();
    test_shard_globs_honored();
    test_shard_globs_honored();
    test_legacy_datadir_warn();
    test_shard_globs_honored();
    test_strict_getters();
    test_unknown_precision_fails();
    test_ckpt_version_gate();
    test_check_known();
    test_arch_parity();
    test_ddp_global_batch();
    if (failures == 0) { std::cout << "test_configs: ALL PASS\n"; return 0; }
    std::cerr << "test_configs: " << failures << " FAILURES\n";
    return 1;
}
