#include "training/checkpoint.h"
#include "core/ops.h"
#include "core/common.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static ModelConfig small_cfg() {
    ModelConfig c;
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.num_layers = 2;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.intermediate_size = 64;
    c.max_seq_len = 64;
    c.tie_embeddings = true;
    c.use_moe = false;
    return c;
}

static u64 fingerprint_blob(const std::string& s) {
    u64 h = 1469598103934665603ULL;
    for (unsigned char ch : s) { h ^= ch; h *= 1099511628211ULL; }
    return h ? h : 1ULL;
}

int main() {
    const ModelConfig cfg = small_cfg();
    Model model(cfg, Device::CPU);
    model.init_weights(7);
    model.enable_grad(true);

    const std::string path = "test_tok_fp.ckpt";

    {
        AdamW opt(model, AdamWConfig{});
        TrainState st;
        st.step = 5;
        st.tokens_seen = 5 * 64;
        st.loss_scale = 8192.0;
        st.tok_vocab = cfg.vocab_size;
        st.tok_fingerprint = fingerprint_blob("tokenizer-A-gtok-bytes");
        Checkpoint::save(path, model, opt, st);
    }

    {
        TrainState st;
        bool restored = false;
        const bool ok = Checkpoint::load(path, model, static_cast<AdamW*>(nullptr), st,
                                         &restored, false);
        CHECK(ok, "resume with the SAME tokenizer is accepted");
        CHECK(st.tok_fingerprint == fingerprint_blob("tokenizer-A-gtok-bytes"),
              "fingerprint round-trips through the checkpoint");
    }

    {
        TrainState st;
        const bool ok = Checkpoint::load(path, model, static_cast<AdamW*>(nullptr), st,
                                         nullptr, false);
        CHECK(ok, "load itself succeeds (the trainer is what refuses)");
        CHECK(st.tok_vocab == cfg.vocab_size, "vocab_size is identical (size cannot catch it)");
        const u64 fp_b = fingerprint_blob("tokenizer-B-different-merges");
        CHECK(fp_b != fingerprint_blob("tokenizer-A-gtok-bytes"),
              "the two tokenizers really differ in content");

        const bool trainer_would_refuse = (st.tok_fingerprint != 0 && fp_b != 0 &&
                                           st.tok_fingerprint != fp_b);
        CHECK(trainer_would_refuse, "trainer gate refuses a different tokenizer");
    }

    {
        TrainState st;
        st.tok_fingerprint = 0;
        const u64 fp_b = fingerprint_blob("tokenizer-B-different-merges");
        const bool trainer_blocks = (st.tok_fingerprint != 0 && fp_b != 0 &&
                                     st.tok_fingerprint != fp_b);
        CHECK(!trainer_blocks, "an unknown (pre-v11) fingerprint never blocks a resume");
    }

    {
        const std::string tok = "test_tok_fp_fixture.gtok";
        const std::string bytes = "tokenizer-A-gtok-bytes";
        {
            std::ofstream f(tok, std::ios::binary);
            f << bytes;
        }
        CHECK(fingerprint_file(tok) == fingerprint_blob(bytes),
              "fingerprint_file matches the reference FNV-1a hash");
        CHECK(fingerprint_file("does_not_exist.gtok") == 0,
              "a missing file yields 0 (unknown, never a false mismatch)");
        std::filesystem::remove(tok);
    }

    std::filesystem::remove(path);

    if (failures == 0) { std::cout << "test_tok_fingerprint: ALL PASS\n"; return 0; }
    std::cerr << "test_tok_fingerprint: " << failures << " FAILURES\n";
    return 1;
}
