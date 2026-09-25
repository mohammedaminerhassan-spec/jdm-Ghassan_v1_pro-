// F-08 regression: `resume_mode: exact | migrate`.
//
//   migrate (historical): every recovery path warns and continues — a missing
//           parameter keeps its fresh init, an optimizer-kind switch restarts
//           moments at t_=0, a reshaped schedule only warns.
//   exact: the same situations are hard failures, so a production run can never
//           silently continue with a different model or a reset optimizer.
//
// This exercises the Checkpoint::load(strict=...) contract directly (the
// trainer-level gate is a thin wrapper on top of it) plus the config parsing.
#include "training/checkpoint.h"
#include "training/trainer.h"

#include <filesystem>
#include <iostream>
#include <string>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static ModelConfig base_config() {
    ModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 32;
    cfg.use_qk_norm = true;          // gives the model extra, easily-missed params
    return cfg;
}

static TrainState make_state(int step) {
    TrainState st;
    st.step = step;
    st.tokens_seen = step * 100;
    st.best_val = 2.5;
    st.last_loss = 3.0;
    st.seed = 42;
    st.tok_vocab = 32;
    st.sched_total = 1000;
    st.sched_warmup = 10;
    st.sched_peak = 3e-4f;
    st.sched_kind = 0;
    st.ddp_world = 1;
    return st;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "gai_resume_mode_test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    // ---------------------------------------------------------------------
    // 1. Baseline: an identical recipe resumes in BOTH modes.
    // ---------------------------------------------------------------------
    {
        const std::string path = (dir / "match.ckpt").string();
        const ModelConfig cfg = base_config();
        Model src(cfg, Device::CPU);
        src.init_weights(5);
        src.enable_grad(true);
        AdamWConfig oc;
        AdamW src_opt(src, oc);
        Checkpoint::save(path, src, src_opt, make_state(42));

        for (int strict = 0; strict <= 1; ++strict) {
            Model dst(cfg, Device::CPU);
            dst.init_weights(999);
            dst.enable_grad(true);
            AdamW dst_opt(dst, oc);
            TrainState st;
            bool moments = false;
            const bool ok = Checkpoint::load(path, dst, &dst_opt, st, &moments, strict != 0);
            CHECK(ok, std::string("matching checkpoint resumes with strict=") + (strict ? "1" : "0"));
            CHECK(moments, std::string("moments restored with strict=") + (strict ? "1" : "0"));
            CHECK(st.step == 42, "step restored");
            CHECK(st.sched_total == 1000 && st.sched_peak > 0.0f, "scheduler snapshot restored");
        }
    }

    // ---------------------------------------------------------------------
    // 2. Optimizer kind mismatch: adamw checkpoint + lion trainer.
    // ---------------------------------------------------------------------
    {
        const std::string path = (dir / "kind.ckpt").string();
        const ModelConfig cfg = base_config();
        Model src(cfg, Device::CPU);
        src.init_weights(5);
        src.enable_grad(true);
        AdamWConfig oc;
        AdamW src_opt(src, oc);
        Checkpoint::save(path, src, src_opt, make_state(11));

        Model dst(cfg, Device::CPU);
        dst.init_weights(999);
        dst.enable_grad(true);
        LionConfig lc;
        Lion dst_opt(dst, lc);

        TrainState st;
        bool moments = true;
        const bool ok_migrate = Checkpoint::load(path, dst, &dst_opt, st, &moments, false);
        CHECK(ok_migrate, "migrate: an optimizer-kind switch still loads the weights");
        CHECK(!moments, "migrate: moments are reported as NOT restored (caller must reset t_)");

        TrainState st2;
        bool moments2 = true;
        const bool ok_exact = Checkpoint::load(path, dst, &dst_opt, st2, &moments2, true);
        CHECK(!ok_exact, "exact: an optimizer-kind switch is refused");
    }

    // ---------------------------------------------------------------------
    // 3. Parameter-set mismatch.
    //
    //    Every knob that adds parameters (use_qk_norm, moe_*) is also an
    //    arch_match field, so in practice a parameter-set change is already a
    //    hard failure in BOTH modes. The `strict` missing-parameter branch is
    //    defense-in-depth for parameters a future version adds without a
    //    matching arch_match field; what is testable here is that neither mode
    //    silently accepts a mismatched parameter set.
    // ---------------------------------------------------------------------
    {
        const std::string path = (dir / "missing_param.ckpt").string();
        const ModelConfig cfg = base_config();
        ModelConfig no_qk = cfg;
        no_qk.use_qk_norm = false;      // fewer parameters than cfg

        Model smaller(no_qk, Device::CPU);
        smaller.init_weights(5);
        smaller.enable_grad(true);
        AdamWConfig oc;
        AdamW smaller_opt(smaller, oc);
        Checkpoint::save(path, smaller, smaller_opt, make_state(3));

        {
            // Matching recipe: both modes accept it.
            Model dst(no_qk, Device::CPU);
            dst.init_weights(1);
            dst.enable_grad(true);
            AdamW dst_opt(dst, oc);
            TrainState st;
            bool moments = false;
            CHECK(Checkpoint::load(path, dst, &dst_opt, st, &moments, false),
                  "migrate: a matching checkpoint loads");
            CHECK(Checkpoint::load(path, dst, &dst_opt, st, &moments, true),
                  "exact: a matching checkpoint loads");
            CHECK(moments, "moments restored for the matching architecture");
        }
        {
            // Destination has parameters the file does not carry.
            Model dst(cfg, Device::CPU);
            dst.init_weights(1);
            dst.enable_grad(true);
            AdamW dst_opt(dst, oc);
            TrainState st;
            bool moments = true;
            CHECK(!Checkpoint::load(path, dst, &dst_opt, st, &moments, false),
                  "migrate: a parameter-set mismatch is still a hard failure");
            CHECK(!Checkpoint::load(path, dst, &dst_opt, st, &moments, true),
                  "exact: a parameter-set mismatch is a hard failure");
        }
    }

    // ---------------------------------------------------------------------
    // 4. Tokenizer/vocab mismatch is refused in BOTH modes (pre-existing hard
    //    gate) — exact must not weaken it.
    // ---------------------------------------------------------------------
    {
        const std::string path = (dir / "vocab.ckpt").string();
        const ModelConfig cfg = base_config();
        Model src(cfg, Device::CPU);
        src.init_weights(5);
        src.enable_grad(true);
        AdamWConfig oc;
        AdamW src_opt(src, oc);
        TrainState st = make_state(7);
        st.tok_vocab = 32;
        Checkpoint::save(path, src, src_opt, st);

        ModelConfig bigger = cfg;
        bigger.vocab_size = 64;
        Model dst(bigger, Device::CPU);
        dst.init_weights(1);
        dst.enable_grad(true);
        AdamW dst_opt(dst, oc);
        TrainState out;
        bool moments = false;
        CHECK(!Checkpoint::load(path, dst, &dst_opt, out, &moments, false),
              "vocab mismatch refused in migrate mode");
        CHECK(!Checkpoint::load(path, dst, &dst_opt, out, &moments, true),
              "vocab mismatch refused in exact mode");
    }

    // ---------------------------------------------------------------------
    // 5. Config parsing: resume_mode is validated, unknown values fail.
    // ---------------------------------------------------------------------
    // 5. Config parsing: resume_mode is validated, unknown values fail.
    // ---------------------------------------------------------------------
    {
        static const char* kBase =
            "training:\n"
            "  max_steps: 10\n"
            "  epochs: 0\n"
            "  batch_size: 1\n"
            "  seq_len: 32\n";
        const std::string exact_yaml = std::string(kBase) + "  resume_mode: exact\n";
        const Config c1 = Config::from_string(exact_yaml);
        const TrainerConfig t1 = TrainerConfig::from_config(c1, false);
        CHECK(t1.resume_mode == "exact", "training.resume_mode: exact is honoured");
        CHECK(t1.resume_exact(), "resume_exact() mirrors resume_mode");

        const Config c2 = Config::from_string(std::string(kBase) + "  resume_mode: wat\n");
        bool threw = false;
        try {
            (void)TrainerConfig::from_config(c2, false);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw, "an unknown training.resume_mode fails fast");

        const Config c3 = Config::from_string(kBase);
        const TrainerConfig t3 = TrainerConfig::from_config(c3, false);
        CHECK(t3.resume_mode == "migrate",
              "the default resume contract is migrate (no behavior change)");
    }

    fs::remove_all(dir, ec);
    if (failures == 0) {
        std::cout << "test_resume_mode: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_resume_mode: " << failures << " FAILURES\n";
    return 1;
}
