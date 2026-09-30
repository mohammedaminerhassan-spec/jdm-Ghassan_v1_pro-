// Exact-resume contract + data/tokenizer strictness regression tests.
//
// Covers the P0/P1/P2 fixes:
//  - P0-02: optimizer exact hyperparam validation (AdamW/Lion/Muon)
//  - P0-03: optimizer step counter (t_) round-trips through save/load
//           (the trainer must NOT overwrite it with state_.step)
//  - P1-08: unknown JSON roles are quarantined, never mapped to User
//  - P2-03: overlong JSON strings fail instead of silent truncation
//  - P2-02: non-binary loss masks rejected (write-time + load-time)
//  - P2-23: shard trailing garbage rejected (exact size match)
//  - P1-09: tokenizer with shuffled special IDs fails to load
//  - P0-01 (schedule lock) is covered by test_trainer_ckpt_flow /
//           test_lion_exact_resume; checkpoint trailing-garbage gate here.
#include "model/model.h"
#include "training/optimizer.h"
#include "training/checkpoint.h"
#include "training/dataloader.h"
#include "dataset/json_reader.h"
#include "tokenizer/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
    else { std::cout << "ok: " << msg << "\n"; } \
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

// ---------------------------------------------------------------- AdamW ---
static void test_adamw_exact_hyperparams() {
    Model m(tiny_config(), Device::CPU);
    m.init_weights(7);
    m.enable_grad(true);

    AdamWConfig oc;
    oc.weight_decay = 0.1f;
    oc.grad_clip = 1.0f;
    AdamW opt(m, oc);
    opt.set_step_count(7);
    std::stringstream ss;
    opt.save_state(ss);
    const std::string blob = ss.str();

    // Same recipe, exact: loads and preserves t_.
    {
        AdamW dst(m, oc);
        std::stringstream in(blob);
        CHECK(dst.load_state(in, OPT_STATE_CURRENT, true), "adamw exact: same recipe loads");
        CHECK(dst.step_count() == 7, "adamw exact: step counter (t_) preserved (P0-03)");
    }
    // Drifted weight_decay, exact: refused.
    {
        AdamWConfig drifted = oc;
        drifted.weight_decay = 0.5f;
        AdamW dst(m, drifted);
        std::stringstream in(blob);
        CHECK(!dst.load_state(in, OPT_STATE_CURRENT, true),
              "adamw exact: weight_decay drift refused (P0-02)");
    }
    // Drifted grad_clip, exact: refused.
    {
        AdamWConfig drifted = oc;
        drifted.grad_clip = 0.25f;
        AdamW dst(m, drifted);
        std::stringstream in(blob);
        CHECK(!dst.load_state(in, OPT_STATE_CURRENT, true),
              "adamw exact: grad_clip drift refused (P0-02)");
    }
    // Same drift, migrate (exact=false): still loads (legacy behavior kept).
    {
        AdamWConfig drifted = oc;
        drifted.weight_decay = 0.5f;
        AdamW dst(m, drifted);
        std::stringstream in(blob);
        CHECK(dst.load_state(in, OPT_STATE_CURRENT, false),
              "adamw migrate: hyperparam drift still loads moments");
    }
}

// ---------------------------------------------------------------- Lion ----
static void test_lion_exact_hyperparams() {
    Model m(tiny_config(), Device::CPU);
    m.init_weights(7);
    m.enable_grad(true);

    LionConfig lc;
    lc.weight_decay = 0.1f;
    Lion opt(m, lc);
    opt.set_step_count(4);
    std::stringstream ss;
    opt.save_state(ss);
    const std::string blob = ss.str();

    {
        Lion dst(m, lc);
        std::stringstream in(blob);
        CHECK(dst.load_state(in, OPT_STATE_CURRENT, true), "lion exact: same recipe loads");
        CHECK(dst.step_count() == 4, "lion exact: step counter (t_) preserved (P0-03)");
    }
    {
        LionConfig drifted = lc;
        drifted.weight_decay = 0.9f;
        Lion dst(m, drifted);
        std::stringstream in(blob);
        CHECK(!dst.load_state(in, OPT_STATE_CURRENT, true),
              "lion exact: weight_decay drift refused (P0-02)");
    }
    {
        LionConfig drifted = lc;
        drifted.beta1 = 0.5f;
        Lion dst(m, drifted);
        std::stringstream in(blob);
        CHECK(!dst.load_state(in, OPT_STATE_CURRENT, true),
              "lion exact: beta1 drift refused (P0-02)");
    }
}

// ---------------------------------------------------------------- Muon ----
static void test_muon_exact_hyperparams() {
    Model m(tiny_config(), Device::CPU);
    m.init_weights(7);
    m.enable_grad(true);

    MuonConfig mc;
    mc.vec_lr_ratio = 0.1f;
    mc.ns_steps = 5;
    mc.min_ns_dim = 0;
    Muon opt(m, mc);
    opt.set_step_count(9);
    std::stringstream ss;
    opt.save_state(ss);
    const std::string blob = ss.str();

    {
        Muon dst(m, mc);
        std::stringstream in(blob);
        CHECK(dst.load_state(in, OPT_STATE_CURRENT, true), "muon exact: same recipe loads (fmt=2)");
        CHECK(dst.step_count() == 9, "muon exact: step counter (t_) preserved (P0-03)");
    }
    {
        MuonConfig drifted = mc;
        drifted.vec_lr_ratio = 0.5f;
        Muon dst(m, drifted);
        std::stringstream in(blob);
        CHECK(!dst.load_state(in, OPT_STATE_CURRENT, true),
              "muon exact: vec_lr_ratio drift refused (P0-02)");
    }
    {
        MuonConfig drifted = mc;
        drifted.ns_steps = 3;
        Muon dst(m, drifted);
        std::stringstream in(blob);
        CHECK(!dst.load_state(in, OPT_STATE_CURRENT, true),
              "muon exact: ns_steps drift refused (P0-02)");
    }
    {
        // min_ns_dim changes the NS/vec branch assignment: exact must refuse.
        MuonConfig drifted = mc;
        drifted.min_ns_dim = 64;
        Muon dst(m, drifted);
        std::stringstream in(blob);
        CHECK(!dst.load_state(in, OPT_STATE_CURRENT, true),
              "muon exact: min_ns_dim drift refused (P0-02)");
    }
}

// ------------------------------------------------------- checkpoint tail --
static void test_checkpoint_trailing_garbage(const std::filesystem::path& dir) {
    const std::string path = (dir / "tail.ckpt").string();
    Model src(tiny_config(), Device::CPU);
    src.init_weights(5);
    src.enable_grad(true);
    AdamWConfig oc;
    AdamW src_opt(src, oc);
    TrainState st;
    st.step = 3;
    Checkpoint::save(path, src, src_opt, st);

    // Append one garbage byte (concatenation / partial overwrite).
    {
        std::ofstream f(path, std::ios::binary | std::ios::app);
        const char x = static_cast<char>(0xAB);
        f.write(&x, 1);
    }

    {
        Model dst(tiny_config(), Device::CPU);
        dst.init_weights(1);
        dst.enable_grad(true);
        AdamW dst_opt(dst, oc);
        TrainState out;
        bool moments = false;
        CHECK(!Checkpoint::load(path, dst, &dst_opt, out, &moments, true),
              "exact: checkpoint with trailing garbage refused");
    }
    {
        Model dst(tiny_config(), Device::CPU);
        dst.init_weights(1);
        dst.enable_grad(true);
        AdamW dst_opt(dst, oc);
        TrainState out;
        bool moments = false;
        CHECK(Checkpoint::load(path, dst, &dst_opt, out, &moments, false),
              "migrate: checkpoint with trailing garbage still loads (warn)");
    }
}

// ------------------------------------------------------------- JSON roles --
static void test_json_unknown_role() {
    // A "tool" message must NOT become a User turn.
    {
        JsonDoc doc;
        const std::string text =
            "{\"messages\":["
            "{\"role\":\"tool\",\"content\":\"do x\"},"
            "{\"role\":\"user\",\"content\":\"hi\"}]}";
        const bool ok = doc_from_json_text(text, doc);
        CHECK(ok, "json: conversation with tool message still parses");
        if (ok) {
            CHECK(doc.messages.size() == 1, "json: unknown 'tool' role quarantined (not User)");
            if (!doc.messages.empty())
                CHECK(doc.messages[0].role == Role::User && doc.messages[0].content == "hi",
                      "json: surviving turn keeps its role/content");
        }
    }
    // A conversation of ONLY unknown roles has no valid turns.
    {
        JsonDoc doc;
        const std::string text = "{\"messages\":[{\"role\":\"bot\",\"content\":\"x\"}]}";
        CHECK(!doc_from_json_text(text, doc), "json: all-unknown roles yield no document");
    }
    // Known roles still map (incl. common aliases).
    {
        JsonDoc doc;
        const std::string text =
            "{\"messages\":["
            "{\"role\":\"system\",\"content\":\"s\"},"
            "{\"role\":\"human\",\"content\":\"q\"},"
            "{\"role\":\"gpt\",\"content\":\"a\"}]}";
        const bool ok = doc_from_json_text(text, doc);
        CHECK(ok, "json: known roles/aliases still parse");
        if (ok) {
            CHECK(doc.messages.size() == 3, "json: three known turns kept");
            if (doc.messages.size() == 3) {
                CHECK(doc.messages[0].role == Role::System, "json: system mapped");
                CHECK(doc.messages[1].role == Role::User, "json: human alias mapped to User");
                CHECK(doc.messages[2].role == Role::Assistant, "json: gpt alias mapped to Assistant");
            }
        }
    }
}

// -------------------------------------------------------- JSON truncation --
static void test_json_truncation() {
    JsonReaderOptions opts;
    opts.max_value_bytes = 16;
    JsonDoc doc;
    const std::string text = "{\"text\":\"abcdefghijklmnopqrstuvwxyz\"}";
    CHECK(!doc_from_json_text(text, doc, opts),
          "json: overlong string fails instead of silent truncation (P2-03)");
    const std::string short_text = "{\"text\":\"hi\"}";
    CHECK(doc_from_json_text(short_text, doc, opts), "json: short string still parses");
}

// ------------------------------------------------------------ shard masks --
static void test_shard_mask_validation(const std::filesystem::path& dir) {
    // Write-time: mask value 2 must throw.
    {
        bool threw = false;
        try {
            ShardWriter w((dir / "badmask.gbin").string(), 32, true);
            const std::vector<i32> toks = {1, 2, 3, 4};
            const std::vector<u8> mask = {1, 2, 0, 1};
            w.add_document(toks, &mask);
            w.close();
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw, "shard writer: non-binary mask (2) rejected (P2-02)");
    }
    // Load-time: a corrupted mask byte (patched to 5) must fail.
    {
        const std::string path = (dir / "patched.gbin").string();
        {
            ShardWriter w(path, 32, true);
            const std::vector<i32> toks = {1, 2, 3, 4, 5, 6, 7, 8};
            const std::vector<u8> mask = {1, 1, 1, 1, 0, 0, 0, 0};
            w.add_document(toks, &mask);
            w.close();
        }
        Shard good;
        CHECK(good.load(path), "shard: valid masked shard loads");
        // Layout: header(32) + tokens(n*2 u16) + offsets(1*8) + mask(n).
        // Patch the first mask byte to 5.
        {
            std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
            const std::streamoff mask_off =
                static_cast<std::streamoff>(32 + 8 * 2 + 8);
            f.seekp(mask_off);
            const char v = static_cast<char>(5);
            f.write(&v, 1);
        }
        Shard bad;
        CHECK(!bad.load(path), "shard: non-binary mask byte rejected at load (P2-02)");
    }
    // Trailing garbage: exact size match required.
    {
        const std::string path = (dir / "trailing.gbin").string();
        {
            ShardWriter w(path, 32, false);
            const std::vector<i32> toks = {1, 2, 3, 4};
            w.add_document(toks);
            w.close();
        }
        Shard good;
        CHECK(good.load(path), "shard: exact-size file loads");
        {
            std::ofstream f(path, std::ios::binary | std::ios::app);
            const char x = 0;
            f.write(&x, 1);
        }
        Shard bad;
        CHECK(!bad.load(path), "shard: trailing garbage rejected (P2-23)");
    }
}

// ------------------------------------------------------- tokenizer specials
static void test_tokenizer_special_ids(const std::filesystem::path& dir) {
    const std::string path = (dir / "specials.gtok").string();
    {
        Tokenizer tk;
        tk.init_empty(NormalizerConfig{});
        tk.save(path);
    }
    {
        Tokenizer tk;
        CHECK(tk.load(path), "tokenizer: clean file loads");
    }
    // Patch "<pad>" (first vocab string) to same-length junk.
    {
        std::ifstream f(path, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        const std::string from = "<pad>";
        const std::string to = "<XXX>";
        const size_t pos = bytes.find(from);
        CHECK(pos != std::string::npos, "tokenizer test: patch anchor found");
        if (pos != std::string::npos) {
            bytes.replace(pos, from.size(), to);
            std::ofstream o(path, std::ios::binary | std::ios::trunc);
            o.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
    }
    {
        Tokenizer tk;
        CHECK(!tk.load(path), "tokenizer: shuffled special IDs rejected (P1-09)");
    }
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "gai_exact_contract_test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    test_adamw_exact_hyperparams();
    test_lion_exact_hyperparams();
    test_muon_exact_hyperparams();
    test_checkpoint_trailing_garbage(dir);
    test_json_unknown_role();
    test_json_truncation();
    test_shard_mask_validation(dir);
    test_tokenizer_special_ids(dir);

    fs::remove_all(dir, ec);
    if (failures == 0) {
        std::cout << "test_exact_contract: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_exact_contract: " << failures << " FAILURES\n";
    return 1;
}
