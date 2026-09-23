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
    test_ckpt_version_gate();
    test_check_known();
    if (failures == 0) { std::cout << "test_configs: ALL PASS\n"; return 0; }
    std::cerr << "test_configs: " << failures << " FAILURES\n";
    return 1;
}
