// test_t4_fixes.cpp — regression coverage for the T4 fixes:
//   1. Lion optimizer learns (loss decreases, grads finite, state roundtrips)
//   2. WSD scheduler shape (warmup -> stable -> decay to min)
//   3. Shard streaming (header-only open + read_window matches full load)
//   4. Retrieval BM25 ranking (exact top-1 + fuzzy paraphrase still matches)
//   5. JSON reader (chat/instruction/text schemas + malformed-line safety)

#include "model/model.h"
#include "training/optimizer.h"
#include "training/scheduler.h"
#include "training/dataloader.h"
#include "dataset/retrieval.h"
#include "dataset/json_reader.h"
#include "core/ops.h"
#include "core/rng.h"
#include "core/common.h"

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>

using namespace gai;
namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond, name) do { \
    if (cond) { std::cout << "  [ok]   " << name << "\n"; } \
    else      { std::cout << "  [FAIL] " << name << "\n"; ++g_fail; } } while(0)

static ModelConfig tiny_dense_cfg() {
    ModelConfig c;
    c.vocab_size        = 128;
    c.hidden_size       = 32;
    c.num_layers        = 1;
    c.num_heads         = 4;
    c.num_kv_heads      = 2;
    c.intermediate_size = 64;
    c.max_seq_len       = 32;
    c.tie_embeddings    = true;
    c.init_std          = 0.02f;
    c.use_moe           = false;
    return c;
}

int main() {
    std::cout << "== T4 fixes regression tests ==\n";
    set_log_level(LogLevel::Warn);

    // ---- 1. Lion learns on a tiny dense model ----
    {
        auto cfg = tiny_dense_cfg();
        Model m(cfg);
        m.init_weights(11);
        m.enable_grad(true);
        LionConfig lc;
        lc.lr = 3e-4f;
        lc.grad_clip = 1.0f;
        Lion opt(m, lc);
        const int B = 2, T = 16;
        std::vector<i32> ids(static_cast<size_t>(B * T)), tgt(static_cast<size_t>(B * T));
        u64 rng = 0x1234;
        for (auto& v : ids) { rng = splitmix64(rng); v = static_cast<i32>(rng % 100); }
        for (size_t i = 0; i < tgt.size(); ++i) tgt[i] = ids[i];
        Activations act = m.make_activations(B, T, true);
        double l0 = 0, l1 = 0;
        bool finite = true;
        for (int s = 0; s < 100; ++s) {
            m.zero_grad();
            i64 ntok = 0;
            double loss = m.forward_backward(ids.data(), tgt.data(), B, T, act, &ntok);
            double gn = opt.step(lc.lr);
            if (!std::isfinite(loss) || !std::isfinite(gn)) { finite = false; break; }
            if (s == 0) l0 = loss;
            l1 = loss;
        }
        std::cout << "    lion loss " << l0 << " -> " << l1 << "\n";
        CHECK(finite, "lion grads/loss stay finite");
        CHECK(l1 < l0 * 0.95, "lion drives the loss down");
        CHECK(opt.state_bytes() > 0, "lion state non-empty");
        // state roundtrip preserves the step counter
        std::stringstream ss;
        opt.save_state(ss);
        i64 t_before = opt.step_count();
        Lion opt2(m, lc);
        CHECK(opt2.load_state(ss), "lion state roundtrips");
        CHECK(opt2.step_count() == t_before, "lion step count restored");
        // Lion uses ~half the memory of AdamW on the same model
        AdamWConfig ac;
        Model m2(cfg);
        m2.init_weights(11);
        m2.enable_grad(true);
        AdamW adam(m2, ac);
        std::cout << "    lion bytes=" << opt.state_bytes()
                  << " adamw bytes=" << adam.state_bytes() << "\n";
        CHECK(opt.state_bytes() * 2 == adam.state_bytes(), "lion is half the size of adamw");
    }

    // ---- 2. WSD schedule ----
    {
        LrScheduler wsd(1e-3f, 10, 100, 0.1f, "wsd", 0.2f);
        float l0 = wsd.lr_at(0);
        float lstab = wsd.lr_at(50);
        float ltail = wsd.lr_at(99);
        float lend = wsd.lr_at(100000);
        std::cout << "    wsd lr: step0=" << l0 << " stable=" << lstab
                  << " tail=" << ltail << " end=" << lend << "\n";
        CHECK(l0 < 1e-3f && l0 > 0, "wsd warms up from below peak");
        CHECK(std::fabs(lstab - 1e-3f) < 1e-7f, "wsd holds peak in the stable phase");
        CHECK(ltail < 1e-3f && ltail > 1e-4f, "wsd decays in the final window");
        CHECK(std::fabs(lend - 1e-4f) < 1e-7f, "wsd clamps at min_ratio*peak");
        LrScheduler cos(1e-3f, 10, 100, 0.1f, "cosine");
        CHECK(cos.lr_at(50) < 1e-3f, "cosine already decays mid-run (unlike wsd)");
    }

    // ---- 3. Shard streaming ----
    {
        fs::path dir = fs::temp_directory_path() / "gai_t4_shard_test";
        fs::create_directories(dir);
        std::string path = (dir / "train_demo_00000.gbin").string();
        {
            ShardWriter w(path, 256, true);
            w.add_document({1, 5, 9, 2}, nullptr);
            std::vector<u8> mask = {1, 1, 0, 1};
            w.add_document({7, 8, 2}, &mask);
            w.close();
        }
        Shard full;
        CHECK(full.load(path), "shard full load works");
        Shard hdr;
        CHECK(hdr.load_header(path), "shard header-only open works");
        CHECK(hdr.is_streaming(), "header shard reports streaming");
        CHECK(hdr.n_tokens() == full.n_tokens(), "streaming token count matches");
        CHECK(hdr.n_docs() == full.n_docs(), "streaming doc count matches");
        CHECK(hdr.has_mask() == full.has_mask(), "streaming mask flag matches");
        std::vector<u32> toks;
        std::vector<u8> masks;
        CHECK(hdr.read_window(0, hdr.n_tokens(), toks, masks), "read_window succeeds");
        bool match = (toks.size() == full.n_tokens());
        for (size_t i = 0; match && i < toks.size(); ++i)
            match = (static_cast<i32>(toks[i]) == full.token(i));
        CHECK(match, "streaming tokens match full load");
        bool mmatch = (masks.size() == full.n_tokens());
        for (size_t i = 0; mmatch && i < masks.size(); ++i)
            mmatch = (masks[i] == full.mask(i));
        CHECK(mmatch, "streaming masks match full load");
        // DataLoader header path still samples (no crash, supervised tokens found)
        DataLoader dl;
        BatchSpec spec{2, 4};
        CHECK(dl.open({path}, spec, 42), "dataloader opens header shard");
        Batch b;
        CHECK(dl.next(b), "dataloader samples a batch");
        CHECK(b.tokens_supervised > 0, "streamed batch has supervised tokens");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    // ---- 4. Retrieval BM25 ----
    {
        std::vector<QaEntry> docs = {
            {1, "salam bikhir kidayer", "labass hamdollah"},
            {2, "chno smitk", "ana smiti Ghassan"},
            {3, "kifach nsayb tagine djaj", "9ta3 djaj w khodra"},
            {4, "what is the capital of france", "paris"},
        };
        RetrievalIndex idx;
        idx.build(docs);
        auto h1 = idx.query("salam labas", 1, 0.0);
        CHECK(!h1.empty() && idx.doc(h1[0].doc).id == 1, "bm25 ranks the salam doc top-1");
        auto h2 = idx.query("salam bokhir", 1, 0.0);   // typo -> fuzzy bridge
        CHECK(!h2.empty() && idx.doc(h2[0].doc).id == 1, "fuzzy paraphrase still matches");
        auto h3 = idx.query("capital of france", 1, 0.0);
        CHECK(!h3.empty() && idx.doc(h3[0].doc).id == 4, "english query finds its doc");
        // trigram rescue: every token misses (edit distance > 2) but the
        // character structure still matches -> still returns the right doc.
        std::vector<QaEntry> docs2 = {{10, "abcdefghij", "rescue_answer"}};
        RetrievalIndex idx2;
        idx2.build(docs2);
        auto h4 = idx2.query("abcdexghxy", 1, 0.0);
        CHECK(!h4.empty() && idx2.doc(h4[0].doc).id == 10, "trigram rescue fires on heavy paraphrase");
        // guard: a >15-char token must not overflow the lev table (no crash).
        // 30 z's share no trigrams with the doc -> empty, but must not crash.
        auto h5 = idx2.query("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", 1, 0.0);
        CHECK(h5.empty(), "oversize token safely yields no hit");
    }

    // ---- 5. JSON reader: schemas + malformed-line safety ----
    {
        fs::path dir = fs::temp_directory_path() / "gai_t4_json_test";
        fs::create_directories(dir);
        std::string jl = (dir / "docs.jsonl").string();
        {
            std::ofstream f(jl, std::ios::binary);
            f << "{\"messages\": [{\"role\": \"user\", \"content\": \"salam\"}, {\"role\": \"assistant\", \"content\": \"labass\"}]}\n";
            f << "{\"instruction\": \"add\", \"input\": \"2+2\", \"output\": \"4\"}\n";
            f << "{\"prompt\": \"capital of france?\", \"completion\": \"paris\"}\n";
            f << "{\"text\": \"plain pretraining sentence here\"}\n";
            f << "this is not json\n";
            f << "{\"unknown\": 123}\n";
        }
        std::string arr = (dir / "arr.json").string();
        {
            std::ofstream f(arr, std::ios::binary);
            f << "[{\"text\": \"array doc one\"}, {\"messages\": [{\"role\": \"human\", \"content\": \"hi\"}]}]";
        }
        size_t nchat = 0, ntext = 0;
        JsonReaderOptions ro;
        ro.verbose = false;
        size_t n1 = read_json_docs(jl, [&](const JsonDoc& d) {
            if (d.is_chat) {
                ++nchat;
                CHECK(!d.messages.empty(), "chat doc has messages");
            } else {
                ++ntext;
                CHECK(!d.text.empty(), "text doc is non-empty");
            }
        }, ro);
        CHECK(n1 == 4, "jsonl yields 4 docs (2 bad lines skipped)");
        CHECK(nchat == 3 && ntext == 1, "jsonl schema mix is 3 chat / 1 text");
        size_t n2 = read_json_dir(dir.string(), [&](const JsonDoc&) {}, ro);
        CHECK(n2 == 6, "json dir scan finds both files (4 + 2 docs)");
        // instruction/input/output must map to user(+input)/assistant turns
        bool saw_instr = false;
        read_json_docs(jl, [&](const JsonDoc& d) {
            if (!d.is_chat || d.messages.size() != 2) return;
            if (d.messages[0].content == "add\n2+2" && d.messages[1].content == "4")
                saw_instr = true;
        }, ro);
        CHECK(saw_instr, "instruction schema maps to user/assistant turns");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    std::cout << (g_fail == 0 ? "== ALL T4 FIX TESTS PASSED ==\n"
                               : "== FAILURES: " + std::to_string(g_fail) + " ==\n");
    return g_fail == 0 ? 0 : 1;
}
