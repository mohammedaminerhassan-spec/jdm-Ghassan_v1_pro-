// data_pipeline - the full ingestion path:
//   raw text / synthetic conversations
//     -> clean -> normalise -> PII filter -> quality -> toxicity -> langid
//     -> anti-robotic style filter -> dedup -> tokenize -> shard (train/val)

#include "tools/cli_common.h"
#include "dataset/cleaner.h"
#include "dataset/dedup.h"
#include "dataset/english_logic.h"
#include "dataset/langid.h"
#include "dataset/synth.h"
#include "dataset/corpus_stats.h"
#include "dataset/json_reader.h"
#include "dataset/parquet_reader.h"
#include "dataset/retrieval.h"
#include "training/dataloader.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/normalizer.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <sstream>
#include <unordered_set>

namespace fs = std::filesystem;
using namespace gai;

static void usage() {
    std::cout <<
    "data_pipeline - dataset construction for Ghassan AI (PARQUET-ONLY)\n\n"
    "The prebuilt Hermes Parquet lake is the only training input:\n"
    "  data_pipeline parquet --lake english_parquet --tokenizer tok.gtok --out artifacts/shards_en\n\n"
    "usage:\n"
    "  data_pipeline parquet  --lake <parquet-dir> --tokenizer tok.gtok --out artifacts/shards\n"
    "  data_pipeline synth    --out data/synth.jsonl --n 50000\n"
    "  data_pipeline build    --tokenizer tok.gtok --out artifacts/shards \\\n"
    "                         [--text <file|dir>] [--chat <file.jsonl>] [--synth <n>]\n"
    "  data_pipeline inspect  --shards artifacts/shards --tokenizer tok.gtok\n"
    "  data_pipeline tok-info --tokenizer tok.gtok\n\n"
  "  data_pipeline csvs --dir <csv-root> --out corpus_csv.txt\n"
    "  data_pipeline parquet --lake dataset/parquet/by_domain --tokenizer tok.gtok\n"
    "                         --out artifacts/shards --domain darija_qa\n"
    "  data_pipeline parquet-corpus --lake english_parquet --out corpus_en.txt [--limit 400000]\n"
  "  data_pipeline csv2json --dir <csv-root> --out json_shards\n"
    "  data_pipeline jsons --dir <json-root> --out corpus_json.txt\n"
    "  data_pipeline retrieve --index <json-file|dir> --query \"salam labas\" [--top 3]\n\n"
    "csvs options (Darija vocabulary/grammar tables -> corpus lines):\n"
    "  --dir <dir>            root scanned RECURSIVELY for *.csv\n"
    "  --out <file>           output corpus                 [corpus_csv.txt]\n"
    "  --min-chars <n>        minimum chars per line        [2]\n"
    "  --with-english         also emit english glosses (off: Darija signal only)\n"
    "  --min-keep <n>         fail if fewer lines kept      [1]\n\n"
    "csv2json options (CSVs -> id/question/answer JSON shards, English layout):\n"
    "  --dir <dir>            root scanned RECURSIVELY for *.csv\n"
    "  --out <dir>            output directory              [json_shards]\n"
    "  --docs-per-file <n>    QA pairs per train-*.json     [20000]\n"
    "  --id-start <n>         first global id               [1000000]\n"
    "  --min-chars <n>        minimum chars per side        [2]\n"
    "  --min-keep <n>         fail if fewer pairs kept      [1]\n\n"
    "retrieve options (keyword search over id/question/answer JSON):\n"
    "  --index <path>         train-*.json file or directory\n"
    "  --query <text>         user question (paraphrases match by shared words)\n"
    "  --top <n>              hits to show                  [3]\n"
    "  --min-score <f>        hide hits below this score    [0]\n\n"
  "parquet options (THE training-data path: .parquet lake -> .gbin shards):\n"
  "  Hermes lake: english_parquet/english_chat_part*.parquet +\n"
  "               english_parquet/english_instruction_part*.parquet\n"
  "               (supplied lake format)\n"
  "  --lake <dir>           lake dir or single .parquet file [dataset/parquet/by_domain]\n"
  "  --dir <dir>            alias for --lake\n"
  "  --match <substr>       only files whose NAME contains substr (per-domain\n"
  "                       builds from one mixed lake dir, no copies)\n"
  "  --tokenizer <path>     .gtok tokenizer (required)\n"
  "  --out <dir>            shard output directory        [artifacts/shards]\n"
  "  --val-ratio <f>        validation fraction           [0.005]\n"
  "  --shard-tokens <n>     tokens per shard              [50000000]\n"
  "  --seq-len <n>          max tokens per window (long docs split, not cut) [4096]\n"
  "  --domain <name>        domain label -> train_<domain>_*.gbin (for data.mix)\n"
  "  --min-keep <n>         fail if fewer docs kept       [1]\n"
  "  --keep-robotic       do not apply the anti-boilerplate filter\n"
  "  --style-mode <m>     darija|en|off                   [darija]\n"
  "  --keep-case          preserve Latin case (REQUIRED with a --keep-case\n"
  "                       tokenizer; default lowercases Latin for Darija)\n"
  "  --no-dedup           skip deduplication\n"
  "  --expect-vocab <n>   fail unless tokenizer vocab == n (0 = skip check)\n"
  "  --mode qa|text|chat|auto  qa: only Q&A rows (refuse if none); text: all rows\n"
  "                       as pretraining lines; chat: messages_json (or user +\n"
  "                       assistant cols) -> multi-turn SFT docs; auto: QA->chat,\n"
  "                       messages->chat, rest->text [auto]\n"
  "  --probe              exit 0 iff native parquet is compiled in (for scripts)\n"
  "  QA tables (question/answer cols, e.g. qa_all.parquet) -> chat docs;\n"
  "  other string tables -> flattened pretraining text lines (like csvs).\n"
  "  Needs -DGAI_ENABLE_PARQUET=ON + Arrow (setup.sh --with-parquet);\n"
  "  without it the command fails loudly (no JSON fallback: parquet-only).\n\n"
    "jsons options (id/question/answer JSON -> corpus lines for BPE):\n"
    "  --dir <dir>            root scanned RECURSIVELY for *.json (or one file)\n"
    "  --out <file>           output corpus                 [corpus_json.txt]\n"
    "  --min-chars <n>        minimum chars per line        [2]\n"
    "  --min-keep <n>         fail if fewer lines kept      [1]\n\n"
    "build options:\n"
    "  --tokenizer <path>   .gtok tokenizer (required)\n"
    "  --out <dir>          shard output directory        [artifacts/shards]\n"
    "  --text <path>        plain text corpus (one doc per line)\n"
    "  --json <path>        QA json file|dir (each pair -> one Q\\nA doc)\n"
    "  --chat <path>        conversations jsonl\n"
    "  --synth <n>          generate n synthetic conversations and include them\n"
    "  --val-ratio <f>      validation fraction           [0.005]\n"
    "  --shard-tokens <n>   tokens per shard              [50000000]\n"
    "  --seq-len <n>        max tokens per document       [4096]\n"
    "  --domain <name>      domain label -> train_<domain>_*.gbin (for data.mix)\n"
    "  --eval-blocklist <p> eval prompts to exclude from training\n"
    "  --no-dedup           skip deduplication\n"
    "  --keep-robotic       do not apply the anti-boilerplate filter\n"
    "  --style-mode <m>     darija|en|off                   [darija]\n"
    "  --keep-case          preserve Latin case (REQUIRED with a --keep-case\n"
    "                       tokenizer; default lowercases Latin for Darija)\n"
    "  --expect-vocab <n>   fail unless tokenizer vocab == n (0 = skip check)\n"
    "  --report <path>      write the pipeline report here\n";
}

struct PipelineCounters {
    u64 raw = 0, cleaned = 0, quality_dropped = 0, toxic_dropped = 0;
    u64 robotic_dropped = 0, dup_dropped = 0, blocked = 0, kept = 0;
    u64 tokens_train = 0, tokens_val = 0, docs_train = 0, docs_val = 0;
    u64 split_docs = 0;  // PRO-EN: docs split into >1 seq_cap windows
    std::map<std::string, u64> lang_kept;
};

static int cmd_synth(const Args& args) {
    SynthConfig cfg;
    cfg.seed = args.num_u64("seed", 1234);
    cfg.num_conversations = args.num_int("n", 20000);
    cfg.max_template_uses = args.num_int("max-template-uses", 40);
    GAI_CHECK(cfg.num_conversations >= 0, "--n must be >= 0");
    GAI_CHECK(cfg.max_template_uses > 0, "--max-template-uses must be > 0");

    std::string out = args.str("out", "data/synth.jsonl");
    fs::create_directories(fs::path(out).has_parent_path() ? fs::path(out).parent_path() : ".");

    log_info(strfmt("[synth] target %d conversations, seed %llu",
                    cfg.num_conversations, (unsigned long long)cfg.seed));
    Timer t;
    SynthGenerator gen(cfg);
    auto convs = gen.generate_many(cfg.num_conversations);
    write_conversations_jsonl(out, convs);

    log_info("[synth] " + gen.stats().summary());
    log_info(strfmt("[synth] wrote %s conversations to %s in %s",
                    human_count(convs.size()).c_str(), out.c_str(),
                    human_duration(t.seconds()).c_str()));

    log_info("  -- by domain --");
    for (const auto& [d, n] : gen.stats().by_domain)
        log_info(strfmt("    %-20s %s", d.c_str(), human_count(n).c_str()));
    log_info("  -- by script --");
    for (const auto& [s, n] : gen.stats().by_script)
        log_info(strfmt("    %-20s %s", s.c_str(), human_count(n).c_str()));

    // show a couple of examples so a human can eyeball the style
    log_info("\n  -- sample conversation --");
    if (!convs.empty()) {
        for (const auto& m : convs[0].messages) {
            const char* r = m.role == Role::System ? "system" :
                            m.role == Role::User ? "user" : "assistant";
            log_info(strfmt("    %-9s : %s", r, m.content.c_str()));
        }
    }
    return 0;
}

// ================================================================ json command
// THE training-data path: all *.json / *.jsonl under --dir (recursive, or a
// single file) directly to .gbin training shards. Schemas are auto-detected
// per object (see dataset/json_reader.h): chat/instruction/prompt docs are
// encoded with the chat template so the loss mask supervises ASSISTANT
// tokens only (SFT-ready); plain-text docs get full masks (pretraining).
// Same quality stack as `build`: clean -> toxicity -> PII -> style -> dedup.
// ---------------------------------------------------------------- shard build core
// Shared by `json` and `parquet`: identical tokenizer gate, writers, cleaning,
// split, masks and report — one implementation so the two input routes can
// never drift apart (the old json/build duplication already caused one leak).
struct ShardBuild {
    Tokenizer tok;
    double val_ratio = 0.005;
    u64 shard_tokens = 50000000;
    int seq_cap = 4096;
    bool do_dedup = true;
    bool style_filter = true;
    // PRO-EN style modes: "darija" = current anti-robotic list (default, unchanged),
    // "en" = English-appropriate (only hard AI-disclosure boilerplate + bullets;
    //   "certainly!"/"of course!"/"i hope this helps" are NORMAL English and kept),
    // "off" = no style filtering at all. Set via --style-mode.
    std::string style_mode = "darija";
    std::string domain;
    std::string outdir;

    int train_idx = 0, val_idx = 0;
    std::unique_ptr<ShardWriter> train_w, val_w;
    u64 train_shard_tok = 0, val_shard_tok = 0;

    Cleaner cleaner;
    Deduplicator dedup;
    LangId lid;
    PipelineCounters ctr;
    std::map<std::string, u64> act_counts;
    u64 total_tokens = 0;
    u64 chat_docs = 0, text_docs = 0;
    CleanStats cs;  // text-path cleaning stats (chat turns use a local one)

    void init_tokenizer(const Args& args) {
        std::string tokpath = args.str("tokenizer");
        GAI_CHECK(!tokpath.empty(), "--tokenizer is required");
        GAI_CHECK(tok.load(tokpath), "cannot load tokenizer: " + tokpath);
        log_info(strfmt("[tok ] vocab=%d", tok.vocab_size()));
        // Fail-loud gate: a 16k legacy file must never silently encode shards
        // for a 32k model (half the embedding rows would train on nothing).
        const int expect_vocab = args.num_int("expect-vocab", 0);
        GAI_CHECK(expect_vocab >= 0, "--expect-vocab must be >= 0");
        GAI_CHECK(expect_vocab <= 0 || tok.vocab_size() == expect_vocab,
                  strfmt("tokenizer vocab %d != expected %d (%s): refusing to encode shards",
                         tok.vocab_size(), expect_vocab, tokpath.c_str()));
    }

    void init_options(const Args& args) {
        val_ratio    = args.real("val-ratio", 0.005);
        shard_tokens = args.num_u64("shard-tokens", 50000000);
        GAI_CHECK(std::isfinite(val_ratio) && val_ratio >= 0.0 && val_ratio <= 1.0,
                  "--val-ratio must be finite and in [0,1]");
        GAI_CHECK(shard_tokens > 0, "--shard-tokens must be > 0");
        seq_cap      = args.num_int("seq-len", 4096);
        // FIX P2 (silent misconfig): --seq-len<=0 previously disabled
        // splitting silently; empty --out wrote to a root path.
        GAI_CHECK(seq_cap > 0, "--seq-len must be > 0 (splitting is mandatory)");
        {
            std::string out0 = args.str("out", "artifacts/shards");
            GAI_CHECK(!out0.empty(), "--out must be non-empty");
        }
        do_dedup     = !args.flag("no-dedup");
        style_filter = !args.flag("keep-robotic");
        // PRO-EN: --style-mode darija|en|off (default darija = legacy behavior).
        style_mode = args.str("style-mode", "darija");
        for (char& ch : style_mode) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
        if (style_mode != "darija" && style_mode != "en" && style_mode != "off")
            GAI_FAIL("unknown --style-mode '" + style_mode + "' (darija|en|off)");
        if (style_mode == "off") style_filter = false;
        // PRO-EN: --keep-case preserves Latin case in cleaning (default
        // lowercases for Darija). MUST match the tokenizer's --keep-case or
        // shards/train/infer disagree on casing. No-op for Arabic script.
        CleanConfig cc;
        cc.drop_pii_lines = style_mode != "en";
        cc.redact_instead_drop = style_mode == "en";
        if (args.flag("keep-case", false)) cc.normalizer.lowercase_latin = false;
        if (cc.redact_instead_drop || args.flag("keep-case", false)) cleaner = Cleaner(cc);
        if (args.flag("keep-case", false))
            log_info("[clean] keep-case: Latin capitalization preserved (English corpus)");
        // Optional domain label: shards become train_<domain>_*.gbin so the
        // trainer's data.mix weights can sample domains (empty = legacy names).
        domain = args.str("domain", "");
        for (char& ch : domain) {
            if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
        }
        outdir = args.str("out", "artifacts/shards");
        GAI_CHECK(!outdir.empty(), "--out must be non-empty");
        fs::create_directories(outdir);
        // FIX P2 (empty shards): writers were opened before any input was
        // validated, leaving empty train_*.gbin that passed later preflight.
        // Now lazy: first emit opens the writer (see ensure_train/ensure_val).
        train_w.reset(); val_w.reset();
        // DeepSeek eval-hygiene: every training-data path honors the eval
        // blocklist — otherwise eval prompts leak into train shards silently.
        if (args.has("eval-blocklist")) dedup.load_blocklist(args.str("eval-blocklist"));
    }

    std::string shard_name(const char* split, int idx) {
        if (domain.empty()) return strfmt("%s/%s_%04d.gbin", outdir.c_str(), split, idx);
        return strfmt("%s/%s_%s_%04d.gbin", outdir.c_str(), split, domain.c_str(), idx);
    }
    void open_train() {
        std::string p = shard_name("train", train_idx++);
        train_w = std::make_unique<ShardWriter>(p, tok.vocab_size(), true);
        train_shard_tok = 0;
        log_info("[shard] opened " + p);
    }
    void open_val() {
        std::string p = shard_name("val", val_idx++);
        val_w = std::make_unique<ShardWriter>(p, tok.vocab_size(), true);
        val_shard_tok = 0;
        log_info("[shard] opened " + p);
    }
    void ensure_train() { if (!train_w) open_train(); }
    void ensure_val() { if (!val_w) open_val(); }

    // DeepSeek split rule: hash CANONICAL text, not raw. Near-dups
    // (paraphrases, same canonical, different raw hash) previously landed
    // independently in train/val → leakage. Canonical keeps them together.
    bool is_val(const std::string& key) {
        const std::string ck = Normalizer::canonical(key);
        const std::string& hkey = ck.empty() ? key : ck;
        return (hash_string(hkey) % 10000ull) < static_cast<u64>(val_ratio * 10000.0);
    }
    void emit(const std::vector<i32>& ids, const std::vector<u8>& mask,
              const std::string& key, const std::string& lang) {
        if (ids.empty()) return;
        emit_split(ids, mask, key, lang, is_val(key));
    }
    // PRO-EN: long docs (chat_if convos average ~3-4k tokens) were TRUNCATED to
    // seq_cap, silently discarding ~85% of assistant signal. Split into
    // non-overlapping seq_cap windows instead: every token trains, the
    // train/val split decision is computed ONCE from the base key so all
    // windows of one document stay on the same side (no leakage), and each
    // window is an independent training doc with its own correct loss mask.
    void emit_split(const std::vector<i32>& ids, const std::vector<u8>& mask,
                    const std::string& key, const std::string& lang, bool val) {
        if (ids.empty()) return;
        if (val) {
            ensure_val();
            if (val_shard_tok >= shard_tokens) { val_w->close(); open_val(); }
            val_w->add_document(ids, &mask);
            val_shard_tok += ids.size();
            ctr.tokens_val += ids.size();
            ++ctr.docs_val;
        } else {
            ensure_train();
            if (train_shard_tok >= shard_tokens) { train_w->close(); open_train(); }
            train_w->add_document(ids, &mask);
            train_shard_tok += ids.size();
            ctr.tokens_train += ids.size();
            ++ctr.docs_train;
        }
        ++ctr.kept;
        ctr.lang_kept[lang]++;
    }
    void emit_windowed(const std::vector<i32>& ids, const std::vector<u8>& mask,
                       const std::string& key, const std::string& lang) {
        if (ids.empty()) return;
        if (seq_cap <= 0 || static_cast<int>(ids.size()) <= seq_cap) {
            emit(ids, mask, key, lang);
            return;
        }
        const bool val = is_val(key);
        const size_t step = static_cast<size_t>(seq_cap);
        for (size_t off = 0; off < ids.size(); off += step) {
            const size_t n = std::min(step, ids.size() - off);
            std::vector<i32> wid(ids.begin() + off, ids.begin() + off + n);
            std::vector<u8> wmask(mask.begin() + off, mask.begin() + off + n);
            emit_split(wid, wmask, key, lang, val);
        }
        ++ctr.split_docs;
    }

    // One JsonDoc (chat or text) through the full clean -> filter -> dedup ->
    // tokenize -> shard path. PARQUET-ONLY route (Hermes lake).
    void on_doc(const JsonDoc& doc) {
        ++ctr.raw;
        const bool is_en = (style_mode == "en");
        if (doc.is_chat) {
            if (doc.messages.empty()) return;
            // DeepSeek data rule: chat turns get the SAME clean+quality
            // pipeline as text (raw HTML/mojibake in SFT skews train/infer).
            std::vector<Message> cleaned_msgs;
            cleaned_msgs.reserve(doc.messages.size());
            {
                CleanStats cs_chat;
                for (const auto& m : doc.messages) {
                    std::string cl;
                    if (!cleaner.clean_line(m.content, cl, cs_chat)) { ++ctr.quality_dropped; return; }
                    if (cl.size() < 2) { ++ctr.quality_dropped; return; }
                    // EN profile keeps "A." multiple-choice + code/math.
                    bool qok = is_en ? quality_check_english(cl).accept
                                     : quality_check(cl).accept;
                    if (!qok) { ++ctr.quality_dropped; return; }
                    cleaned_msgs.push_back(Message{m.role, cl});
                }
            }
            if (is_en) {
                std::string first_user;
                for (const auto& m : cleaned_msgs) {
                    if (m.role == Role::User) { first_user = m.content; break; }
                }
                if (first_user.empty()) first_user = cleaned_msgs.front().content;
                ++act_counts[english_logic::dialog_act_name(
                    english_logic::classify_dialog_act(first_user))];
            }
            std::string key;
            for (const auto& m : cleaned_msgs) key += m.content + "\n";
            std::string last_user;
            for (size_t i = 0; i < cleaned_msgs.size(); ++i) {
                auto& m = cleaned_msgs[i];
                if (m.role == Role::User) last_user = m.content;
                if (check_toxicity(m.content).toxic) { ++ctr.toxic_dropped; return; }
                // EN: redact PII (keep the math/code doc) instead of dropping
                // the whole conversation; Darija keeps the strict drop.
                if (is_en) {
                    PiiReport r = scan_pii(m.content);
                    if (r.any()) m.content = redact_pii(m.content, nullptr);
                } else if (scan_pii(m.content).any()) { ++ctr.blocked; return; }
                // PRO-EN: style gate per --style-mode. darija = legacy full
                // list; en = hard AI-disclosure only (normal English
                // politeness like "Of course!"/"Certainly!" is KEPT).
                if (style_filter && m.role == Role::Assistant) {
                    bool drop = false;
                    if (style_mode == "en") {
                        drop = has_hard_ai_boilerplate(m.content);
                        if (!drop) {
                            const auto rep = english_logic::check_english_reply(last_user, m.content);
                            drop = !rep.disciplined && rep.reason == "bad-mc-format";
                        }
                    } else {
                        std::string prev = i > 0 ? cleaned_msgs[i - 1].content : "";
                        drop = check_assistant_style(m.content, prev, lid).robotic();
                    }
                    if (drop) { ++ctr.robotic_dropped; return; }
                }
            }
            ++ctr.cleaned;
            if (do_dedup && !dedup.add(key)) { ++ctr.dup_dropped; return; }
            std::vector<u8> mask;
            std::vector<i32> ids = ChatTemplate::encode(tok, cleaned_msgs, false, &mask);
            if (ids.empty()) return;
            // PRO-EN: split long convos into windows (was: truncate, losing ~85%
            // of assistant signal on chat_if-style corpora).
            total_tokens += ids.size();
            LangScore ls = lid.classify(key);
            emit_windowed(ids, mask, key, lang_name(ls.tag));
            ++chat_docs;
        } else {
            if (doc.text.size() < 10) return;  // skip very short strings
            std::string cleaned;
            if (!cleaner.clean_line(doc.text, cleaned, cs)) return;
            ++ctr.cleaned;
            if (!(is_en ? quality_check_english(cleaned).accept : quality_check(cleaned).accept)) {
                ++ctr.quality_dropped;
                return;
            }
            if (check_toxicity(cleaned).toxic) { ++ctr.toxic_dropped; return; }
            if (do_dedup && !dedup.add(cleaned)) { ++ctr.dup_dropped; return; }
            if (is_en) {
                ++act_counts[english_logic::dialog_act_name(
                    english_logic::classify_dialog_act(cleaned))];
            }
            LangScore ls = lid.classify(cleaned);
            std::vector<i32> ids = tok.encode(cleaned, true, true);
            if (ids.empty()) return;
            // PRO-EN: same window split for long text docs (was: truncate).
            total_tokens += ids.size();
            std::vector<u8> mask(ids.size(), 1);  // pretraining: supervise all
            emit_windowed(ids, mask, cleaned, lang_name(ls.tag));
            ++text_docs;
        }
    }

    // Close writers, print the standard report, enforce the quality gate.
    int finish(const Args& args, const char* title) {
        if (train_w) train_w->close();
        if (val_w) val_w->close();
        std::ostringstream rep;
        rep << "\n==================== " << title << " ====================\n";
        rep << strfmt("  total docs        : %s  (chat %s / text %s)\n",
                      human_count(ctr.raw).c_str(), human_count(chat_docs).c_str(),
                      human_count(text_docs).c_str());
        rep << strfmt("  cleaned           : %s\n", human_count(ctr.cleaned).c_str());
        rep << strfmt("  quality dropped   : %s\n", human_count(ctr.quality_dropped).c_str());
        rep << strfmt("  toxic dropped     : %s\n", human_count(ctr.toxic_dropped).c_str());
        rep << strfmt("  pii blocked       : %s\n", human_count(ctr.blocked).c_str());
        rep << strfmt("  robotic dropped   : %s\n", human_count(ctr.robotic_dropped).c_str());
        rep << strfmt("  duplicates        : %s\n", human_count(ctr.dup_dropped).c_str());
        rep << strfmt("  split windows     : %s source docs exceeded seq_cap (not truncated)\n",
                      human_count(ctr.split_docs).c_str());
        rep << strfmt("  total tokens      : %s\n", human_count(total_tokens).c_str());
        rep << strfmt("  train windows/tokens: %s / %s  (%d shards)\n",
                      human_count(ctr.docs_train).c_str(), human_count(ctr.tokens_train).c_str(), train_idx);
        rep << strfmt("  val   windows/tokens: %s / %s  (%d shards)\n",
                      human_count(ctr.docs_val).c_str(), human_count(ctr.tokens_val).c_str(), val_idx);
        rep << "  -- by language --\n";
        for (const auto& [l, n] : ctr.lang_kept)
            rep << strfmt("    %-12s %s\n", l.c_str(), human_count(n).c_str());
        if (!act_counts.empty()) {
            rep << "  -- English dialog acts --\n";
            for (const auto& [a, n] : act_counts)
                rep << strfmt("    %-12s %s\n", a.c_str(), human_count(n).c_str());
        }
        rep << "================================================================\n";
        log_info(rep.str());
        if (args.has("report")) {
            std::ofstream rf(args.str("report"), std::ios::binary);
            GAI_CHECK(rf.good(), "cannot write report: " + args.str("report"));
            rf << rep.str();
            rf.close();
            GAI_CHECK(rf.good(), "report write failed: " + args.str("report"));
            log_info("[report] wrote " + args.str("report"));
        }
        // ---- quality gate: never let an empty or fully-duplicate corpus through
        const u64 kept = ctr.docs_train + ctr.docs_val;
        const u64 min_keep = args.num_u64("min-keep", 1);
        if (kept < min_keep) {
            log_error(strfmt("quality gate FAILED: kept %s windows (< min-keep %s)",
                             human_count(kept).c_str(), human_count(min_keep).c_str()));
            return 1;
        }
        if (ctr.raw > 0 && ctr.dup_dropped * 2 > ctr.raw)
            log_warn("quality gate WARNING: >50% exact duplicates — corpus is template-dominated");
        return 0;
    }
};

// PARQUET-ONLY: `data_pipeline json` was REMOVED. The supplied lake is read
// with data_pipeline parquet --lake english_parquet --mode chat ...
static int cmd_json_removed(const Args& /*args*/) {
    log_error("REMOVED: `data_pipeline json` no longer exists (parquet-only project). "
              "Use the supplied English Parquet lake with data_pipeline parquet --lake english_parquet --mode chat --style-mode en --keep-case ...");
    return 1;
}

static int cmd_parquet(const Args& args) {
    // Route probe for scripts (build_billion_data.sh picks the fastest
    // AVAILABLE input: native parquet when compiled in, else JSON).
    if (args.flag("probe")) {
        if (parquet_available()) {
            std::cout << "parquet=native\n";
            return 0;
        }
        std::cout << "parquet=unavailable (rebuild with -DGAI_ENABLE_PARQUET=ON; kaggle/setup.sh --with-parquet)\n";
        return 2;
    }
    // Fail BEFORE opening shard writers: without Arrow the row stream below
    // throws, and unwinding would flush header-only empty shards into the
    // output dir (they would then masquerade as a built domain).
    if (!parquet_available()) {
        log_error("parquet input needs Apache Arrow: rebuild with -DGAI_ENABLE_PARQUET=ON "
                  "(kaggle/setup.sh --with-parquet). JSON file route was REMOVED (parquet-only).");
        return 1;
    }
    ShardBuild b;
    b.init_tokenizer(args);
    b.init_options(args);
    // --lake preferred, --dir accepted as an alias (mirrors json).
    std::string lake = args.str("lake", args.str("dir", "dataset/parquet/by_domain"));
    std::vector<std::string> files = list_parquet_files(lake);
    // PRO-EN: --match <substr> keeps only files whose NAME contains substr,
    // so one lake dir holding english_chat_part*.parquet +
    // english_instruction_part*.parquet builds each --domain separately
    // without copying gigabytes on a 19.5GB Kaggle HDD.
    if (args.has("match")) {
        const std::string m = args.str("match");
        std::vector<std::string> kept;
        for (const auto& f : files)
            if (fs::path(f).filename().string().find(m) != std::string::npos)
                kept.push_back(f);
        files.swap(kept);
    }
    if (files.empty()) {
        log_error("no .parquet files found in: " + lake);
        return 1;
    }
    log_info(strfmt("[parquet] found %zu file(s) in %s (native=%s)",
                    files.size(), lake.c_str(), parquet_available() ? "yes" : "no"));
    // --mode selects the contract (default auto):
    //   qa   : only question/answer rows become chat docs; anything else is
    //          skipped. Refuses loudly when no QA table exists (protects the
    //          darija_qa mix weight from silently filling with text docs).
    //   text : every row becomes one pretraining text line (the csvs
    //          equivalent for the typed lake; used for darija_vocab).
    //   chat : messages_json (canonical [{"role","content"}...], as emitted by
    //          the chat_if converter) or user+assistant columns become
    //          multi-turn SFT docs. Refuses loudly when no chat column exists.
    //   auto : QA rows -> chat docs, messages_* rows -> chat docs,
    //          everything else -> text lines.
    std::string mode = args.str("mode", "auto");
    if (mode != "auto" && mode != "qa" && mode != "text" && mode != "chat")
        GAI_FAIL("parquet: unknown --mode '" + mode + "' (qa|text|chat|auto)");
    ParquetOptions popts;
    popts.verbose = true;
    u64 qa_rows = 0, skipped_rows = 0, qa_files = 0, chat_rows = 0;
    JsonReaderOptions jopts;  // shared caps for messages_json parsing
    jopts.verbose = false;
    size_t n = read_parquet_docs(files, [&](const std::map<std::string, std::string>& row) {
        auto get = [&](const char* k) -> std::string {
            auto it = row.find(k);
            return it != row.end() ? it->second : "";
        };
        // QA schema (what build_qa.py emits): question/answer -> chat doc
        // with assistant-only loss, exactly like the JSON prompt schema.
        const std::string q = get("question");
        const std::string a = get("answer");
        if (!q.empty() && !a.empty()) {
            if (mode == "text") {
                // Fall through to the text flattening below (vocab use).
            } else {
                JsonDoc doc;
                doc.is_chat = true;
                doc.messages = {Message{Role::User, q}, Message{Role::Assistant, a}};
                b.on_doc(doc);
                ++qa_rows;
                return;
            }
        } else if (mode == "qa") {
            ++skipped_rows;  // non-QA row under --mode qa: skip loudly-counted
            return;
        }
        // PRO-EN chat schema: messages_json canonical array, or user/assistant
        // (+optional system) columns. Same JsonDoc/cleaning/masks as JSON chat.
        if (mode == "chat" || mode == "auto") {
            const std::string mj = get("messages_json");
            if (!mj.empty()) {
                JsonDoc doc;
                if (doc_from_json_text("{\"messages\":" + mj + "}", doc, jopts)) {
                    b.on_doc(doc);
                    ++chat_rows;
                    return;
                }
                // malformed messages_json falls through to skip counting below
            } else {
                const std::string u = get("user");
                const std::string as = get("assistant");
                if (!u.empty() && !as.empty()) {
                    JsonDoc doc;
                    doc.is_chat = true;
                    const std::string sys = get("system");
                    if (!sys.empty()) doc.messages.push_back({Role::System, sys});
                    doc.messages.push_back({Role::User, u});
                    doc.messages.push_back({Role::Assistant, as});
                    b.on_doc(doc);
                    ++chat_rows;
                    return;
                }
            }
            if (mode == "chat") { ++skipped_rows; return; }
            // auto: no chat columns -> fall through to text flattening below
        }
        // Lexicon/text schema (variant cols + eng, like the csvs path):
        // flatten the row's informative cells into one pretraining text line.
        // (chat_if metadata cols are prefixed _ so they never leak into text.)
        std::string line;
        for (const auto& [k, v] : row) {
            if (v.empty() || k.empty() || k[0] == '_') continue;
            if (k == "messages_json" || k == "uuid") continue;
            if (!line.empty()) line += " / ";
            line += v;
        }
        if (!line.empty()) {
            JsonDoc doc;
            doc.is_chat = false;
            doc.text = line;
            b.on_doc(doc);
            return;
        }
        ++skipped_rows;
    }, popts);
    if (mode == "qa") {
        for (const auto& f : files) {
            ParquetTableInfo ti = inspect_parquet(f);
            bool has_q = false, has_a = false;
            for (const auto& c : ti.columns) {
                if (c == "question") has_q = true;
                if (c == "answer") has_a = true;
            }
            if (has_q && has_a) ++qa_files;
        }
    }
    u64 chat_files = 0;
    if (mode == "chat") {
        for (const auto& f : files) {
            ParquetTableInfo ti = inspect_parquet(f);
            bool has_mj = false, has_u = false, has_as = false;
            for (const auto& c : ti.columns) {
                if (c == "messages_json") has_mj = true;
                if (c == "user") has_u = true;
                if (c == "assistant") has_as = true;
            }
            if (has_mj || (has_u && has_as)) ++chat_files;
        }
    }
    log_info(strfmt("[parquet] mode=%s streamed %s rows (%s QA rows, %s chat rows, %s skipped)",
                    mode.c_str(), human_count(n).c_str(), human_count(qa_rows).c_str(),
                    human_count(chat_rows).c_str(), human_count(skipped_rows).c_str()));
    if (mode == "qa" && qa_files == 0) {
        log_error("no question/answer table found under " + lake +
                  " (lexicon-only input with --mode qa would starve the mix weight)");
        return 1;
    }
    if (mode == "chat" && chat_files == 0) {
        log_error("no messages_json (or user+assistant) table found under " + lake +
                  " (wrong lake for --mode chat would starve the mix weight)");
        return 1;
    }
    return b.finish(args, "parquet pipeline report");
}

static int cmd_parquet_corpus(const Args& args) {
    if (!parquet_available()) {
        log_error("parquet-corpus needs Apache Arrow: rebuild with -DGAI_ENABLE_PARQUET=ON "
                  "(kaggle/setup.sh --with-parquet)");
        return 1;
    }
    const std::string lake = args.str("lake", args.str("dir", "english_parquet"));
    std::vector<std::string> files = list_parquet_files(lake);
    if (args.has("match")) {
        const std::string match = args.str("match");
        std::vector<std::string> kept;
        for (const auto& file : files)
            if (fs::path(file).filename().string().find(match) != std::string::npos)
                kept.push_back(file);
        files.swap(kept);
    }
    if (files.empty()) {
        log_error("no .parquet files found in: " + lake);
        return 1;
    }

    const i64 requested_limit = args.num("limit", 0);
    GAI_CHECK(requested_limit >= 0, "parquet-corpus --limit must be >= 0");
    const size_t limit = static_cast<size_t>(requested_limit);
    const std::string outpath = args.str("out", "artifacts/corpus/corpus_en.txt");
    if (fs::path(outpath).has_parent_path())
        fs::create_directories(fs::path(outpath).parent_path());
    std::ofstream out(outpath, std::ios::binary);
    GAI_CHECK(out.good(), "cannot write corpus: " + outpath);

    ParquetOptions popts;
    popts.max_rows = limit;
    popts.verbose = true;
    JsonReaderOptions jopts;
    jopts.verbose = false;
    u64 rows = 0;
    u64 lines = 0;
    auto emit = [&](const std::string& value) {
        std::string line;
        line.reserve(value.size());
        for (char ch : value) line.push_back((ch == '\r' || ch == '\n') ? ' ' : ch);
        if (line.find_first_not_of(" \t") == std::string::npos) return;
        out << line << '\n';
        ++lines;
    };

    read_parquet_docs(files, [&](const std::map<std::string, std::string>& row) {
        ++rows;
        auto get = [&](const char* key) {
            const auto it = row.find(key);
            return it == row.end() ? std::string() : it->second;
        };
        const std::string messages = get("messages_json");
        if (!messages.empty()) {
            JsonDoc doc;
            if (doc_from_json_text("{\"messages\":" + messages + "}", doc, jopts)) {
                for (const auto& message : doc.messages) emit(message.content);
                return;
            }
        }
        const std::string question = get("question");
        const std::string answer = get("answer");
        if (!question.empty() || !answer.empty()) {
            emit(question + "\n" + answer);
            return;
        }
        const std::string user = get("user");
        const std::string assistant = get("assistant");
        if (!user.empty() || !assistant.empty()) {
            emit(user + "\n" + assistant);
            return;
        }
        std::string line;
        for (const auto& [key, value] : row) {
            if (value.empty() || key.empty() || key[0] == '_' ||
                key == "messages_json" || key == "uuid" || key == "id" ||
                key == "source" || key == "category") continue;
            if (!line.empty()) line += " / ";
            line += value;
        }
        emit(line);
    }, popts);

    GAI_CHECK(out.good(), "corpus write failed: " + outpath);
    if (rows == 0 || lines == 0) {
        log_error("parquet-corpus produced no corpus lines: " + outpath);
        return 1;
    }
    log_info(strfmt("[parquet-corpus] %s rows -> %s lines: %s",
                    human_count(rows).c_str(), outpath.c_str(), human_count(lines).c_str()));
    return 0;
}

// PARQUET-ONLY: `json-inspect` REMOVED (no JSON file route).
// Inspect the lake instead: data_pipeline parquet --lake english_parquet ...
// (row counts + schema via the parquet report) or python pq.read_table.

// Prints the ACTUAL vocabulary size stored in a .gtok file (parseable line
// first, for scripts). This is the fail-loud gate against silently building
// 32k-model shards with the legacy 16k tokenizer: compare the printed number
// with the model vocab before encoding a single document.
static int cmd_tok_info(const Args& args) {
    std::string tokpath = args.str("tokenizer");
    GAI_CHECK(!tokpath.empty(), "--tokenizer is required");
    Tokenizer tok;
    GAI_CHECK(tok.load(tokpath), "cannot load tokenizer: " + tokpath);
    std::cout << "vocab_size=" << tok.vocab_size() << "\n";
    log_info(strfmt("[tok ] %s vocab=%d", tokpath.c_str(), tok.vocab_size()));
    return 0;
}

// ================================================================ csvs command
// Turns Darija vocabulary/grammar CSV tables into corpus lines.
// Why this matters: tables like (klb // kalb // kelb // كلب // dog) teach the
// model that spelling variants are the SAME word — this is the typo/paraphrase
// robustness mechanism (a 1-letter change stays in the same neighborhood).
// Per row we emit: every variant as its own line, the Arabic-script line, and
// ONE alignment line ("a / b / ج") so equivalence is learned explicitly.
namespace {

std::string csv_trim(std::string s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

// Minimal robust CSV row splitter: quotes, "" escapes, commas, \r\n.
std::vector<std::string> csv_split_row(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_q = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_q) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
                else in_q = false;
            } else cur += c;
        } else if (c == '"') {
            in_q = true;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

std::string csv_lower(std::string s) {
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

bool csv_is_meta_col(const std::string& name) {
    const std::string n = csv_lower(name);
    return n == "id" || n == "source" || n == "dataset_source" || n == "metadata" ||
           n == "token_count" || n == "word_count" || n == "count" || n == "freq";
}

bool csv_is_arabic_col(const std::string& name) {
    const std::string n = csv_lower(name);
    return n == "arabic" || n == "ar" || n == "darija_ar" ||
           (n.size() > 3 && n.compare(n.size() - 3, 3, "_ar") == 0);
}

bool csv_is_english_col(const std::string& name) {
    const std::string n = csv_lower(name);
    return n == "eng" || n == "english" || n.rfind("eng", 0) == 0;
}

bool csv_looks_like_header(const std::vector<std::string>& cells) {
    for (const auto& c : cells) {
        const std::string n = csv_lower(csv_trim(c));
        if (n == "darija" || n == "darija_ar" || n == "eng" || n == "english" ||
            n == "root" || n == "arabic" || n.rfind("n1", 0) == 0) return true;
    }
    return false;
}

std::string csv_flat(std::string s) {
    for (char& ch : s)
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    return csv_trim(s);
}

} // namespace

static int cmd_csvs(const Args& args) {
    std::string root    = args.str("dir", ".");
    std::string outpath = args.str("out", "corpus_csv.txt");
    const int min_chars = args.num_int("min-chars", 2);
    GAI_CHECK(min_chars >= 0, "--min-chars must be >= 0");
    const bool with_eng = args.flag("with-english", false);

    std::vector<std::string> csv_files;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        std::error_code file_ec;
        if (!e.is_regular_file(file_ec) || file_ec) continue;
        if (e.path().extension().string() == ".csv")
            csv_files.push_back(e.path().string());
    }
    std::sort(csv_files.begin(), csv_files.end());
    if (csv_files.empty()) {
        log_error("no .csv files found under: " + root);
        return 1;
    }
    log_info(strfmt("[csvs] found %zu file(s) under %s", csv_files.size(), root.c_str()));

    if (auto pp = fs::path(outpath).parent_path(); !pp.empty())
        fs::create_directories(pp);
    std::ofstream out(outpath, std::ios::binary);
    GAI_CHECK(out.good(), "cannot write: " + outpath);

    std::unordered_set<u64> seen;
    seen.reserve(1 << 20);
    u64 files_ok = 0, rows_in = 0, lines_out = 0, align_out = 0;

    std::vector<std::string> raw_lines;
    for (const auto& path : csv_files) {
        std::ifstream f(path, std::ios::binary);
        if (!f.good()) { log_warn("  cannot open " + path); continue; }
        raw_lines.clear();
        std::string line;
        bool first = true;
        while (std::getline(f, line)) {
            if (first && line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF)
                line.erase(0, 3);   // strip UTF-8 BOM
            first = false;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            raw_lines.push_back(line);
        }
        if (raw_lines.empty()) continue;

        // header detection
        size_t data_from = 0;
        std::vector<std::string> header;
        {
            auto cells0 = csv_split_row(raw_lines[0]);
            if (csv_looks_like_header(cells0)) {
                header = cells0;
                data_from = 1;
            }
        }
        // column roles
        std::vector<int> role;  // 0=skip, 1=variant, 2=arabic, 3=english
        if (!header.empty()) {
            for (const auto& h : header) {
                const std::string n = csv_trim(h);
                if (csv_is_meta_col(n)) role.push_back(0);
                else if (csv_is_arabic_col(n)) role.push_back(2);
                else if (csv_is_english_col(n)) role.push_back(3);
                else role.push_back(1);
            }
        }

        u64 file_lines = 0;
        auto emit = [&](const std::string& s) {
            std::string t = csv_flat(s);
            if (static_cast<int>(t.size()) < min_chars) return;
            if (!seen.insert(hash_string(t)).second) return;
            out << t << '\n';
            ++file_lines;
        };

        for (size_t li = data_from; li < raw_lines.size(); ++li) {
            if (raw_lines[li].empty()) continue;
            auto cells = csv_split_row(raw_lines[li]);
            ++rows_in;
            std::string first_variant, arabic;
            std::vector<std::string> variants;
            for (size_t ci = 0; ci < cells.size(); ++ci) {
                int r = header.empty() ? 1 : (ci < role.size() ? role[ci] : 1);
                std::string t = csv_flat(cells[ci]);
                if (static_cast<int>(t.size()) < min_chars) continue;
                if (r == 0) continue;
                if (r == 2) {
                    emit(t);
                    if (arabic.empty()) arabic = t;
                } else if (r == 3) {
                    if (with_eng) emit(t);
                } else {
                    emit(t);
                    variants.push_back(t);
                    if (first_variant.empty()) first_variant = t;
                }
            }
            // alignment line: ties variants + arabic script together
            if (!variants.empty() && !arabic.empty() && variants[0] != arabic) {
                std::string a = variants[0] + " / " + arabic;
                if (variants.size() > 1 && variants[1] != variants[0] &&
                    variants[1] != arabic)
                    a = variants[0] + " / " + variants[1] + " / " + arabic;
                std::string t = csv_flat(a);
                if (seen.insert(hash_string(t)).second) {
                    out << t << '\n';
                    ++file_lines;
                    ++align_out;
                }
            }
        }
        lines_out += file_lines;
        ++files_ok;
    }
    out.close();
    GAI_CHECK(out.good(), "corpus write failed");

    log_info("\n==================== csvs pipeline report ====================");
    log_info(strfmt("  files processed : %s / %s", human_count(files_ok).c_str(),
                    human_count(csv_files.size()).c_str()));
    log_info(strfmt("  csv rows in     : %s", human_count(rows_in).c_str()));
    log_info(strfmt("  corpus lines    : %s (align: %s)", human_count(lines_out).c_str(),
                    human_count(align_out).c_str()));
    log_info(strfmt("  output          : %s", outpath.c_str()));
    log_info("===============================================================");
    const u64 min_keep = args.num_u64("min-keep", 1);
    if (lines_out < min_keep) {
        log_error("quality gate FAILED: kept too few lines");
        return 1;
    }
    return 0;
}

// ================================================================ csv2json command
// Converts Darija CSV tables into sharded id/question/answer JSON files with
// the same layout as the reference English data:
//   [ { "id": 1000000, "question": "...", "answer": "..." }, ... ]
// Mapping per row (same column roles as `csvs`):
//   - latin variants + arabic script  -> Q=variant, A=arabic (one pair per
//     variant, so spelling paraphrases like "salam bikhir"/"salam labas"
//     become separate retrievable/training pairs instead of one line);
//   - latin variant + english only     -> Q=variant, A=english;
//   - rows where Q==A (or either side empty) are skipped and counted.
// Shards hold --docs-per-file pairs (default 20000) named train-%05d.json.
// IDs are global and sequential from --id-start (default 1000000, far above
// the English ids) so JSON shards can be merged without collisions.
namespace {

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"':  o += "\\\""; ++i; break;
            case '\\': o += "\\\\"; ++i; break;
            case '\n': o += "\\n"; ++i; break;
            case '\r': o += "\\r"; ++i; break;
            case '\t': o += "\\t"; ++i; break;
            case '\b': o += "\\b"; ++i; break;
            case '\f': o += "\\f"; ++i; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                    ++i;
                } else {
                    o += s[i];
                    ++i;
                }
        }
    }
    return o;
}

} // namespace

static int cmd_csv2json(const Args& args) {
    std::string root     = args.str("dir", ".");
    std::string outdir   = args.str("out", "json_shards");
    const int per_file   = args.num_int("docs-per-file", 20000);
    long long next_id    = args.num("id-start", 1000000);
    GAI_CHECK(next_id >= 0, "--id-start must be >= 0");
    const int min_chars  = args.num_int("min-chars", 2);
    GAI_CHECK(min_chars >= 0, "--min-chars must be >= 0");
    GAI_CHECK(per_file > 0, "--docs-per-file must be > 0");

    std::vector<std::string> csv_files;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        std::error_code file_ec;
        if (!e.is_regular_file(file_ec) || file_ec) continue;
        if (e.path().extension().string() == ".csv")
            csv_files.push_back(e.path().string());
    }
    std::sort(csv_files.begin(), csv_files.end());
    if (csv_files.empty()) {
        log_error("no .csv files found under: " + root);
        return 1;
    }
    fs::create_directories(outdir);

    std::unordered_set<u64> seen_q;
    seen_q.reserve(1 << 20);
    u64 rows_in = 0, pairs_out = 0, skipped_same = 0, skipped_empty = 0, skipped_dup = 0;
    int file_idx = 0, in_file = 0;
    std::unique_ptr<std::ofstream> jf;

    auto open_shard = [&]() {
        if (jf) { *jf << "\n]\n"; jf->close(); jf.reset(); }
        std::string p = strfmt("%s/train-%05d.json", outdir.c_str(), file_idx++);
        jf = std::make_unique<std::ofstream>(p, std::ios::binary);
        GAI_CHECK(jf->good(), "cannot write: " + p);
        *jf << "[\n";
        in_file = 0;
        log_info("[json] opened " + p);
    };
    auto emit_pair = [&](const std::string& q, const std::string& a) {
        std::string qq = csv_flat(q), aa = csv_flat(a);
        if (static_cast<int>(qq.size()) < min_chars || static_cast<int>(aa.size()) < min_chars) {
            ++skipped_empty;
            return;
        }
        if (qq == aa) { ++skipped_same; return; }
        if (!seen_q.insert(hash_string(qq)).second) { ++skipped_dup; return; }
        if (!jf || in_file >= per_file) open_shard();
        if (in_file > 0) *jf << ",\n";
        *jf << "{\n\"id\": " << next_id++ << ",\n"
            << "\"question\": \"" << json_escape(qq) << "\",\n"
            << "\"answer\": \"" << json_escape(aa) << "\"\n}";
        ++in_file;
        ++pairs_out;
    };

    std::vector<std::string> raw_lines;
    for (const auto& path : csv_files) {
        std::ifstream f(path, std::ios::binary);
        if (!f.good()) { log_warn("  cannot open " + path); continue; }
        raw_lines.clear();
        std::string line;
        bool first = true;
        while (std::getline(f, line)) {
            if (first && line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF)
                line.erase(0, 3);
            first = false;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            raw_lines.push_back(line);
        }
        if (raw_lines.empty()) continue;

        size_t data_from = 0;
        std::vector<std::string> header;
        {
            auto cells0 = csv_split_row(raw_lines[0]);
            if (csv_looks_like_header(cells0)) { header = cells0; data_from = 1; }
        }
        std::vector<int> role;
        if (!header.empty()) {
            for (const auto& h : header) {
                const std::string n = csv_trim(h);
                if (csv_is_meta_col(n)) role.push_back(0);
                else if (csv_is_arabic_col(n)) role.push_back(2);
                else if (csv_is_english_col(n)) role.push_back(3);
                else role.push_back(1);
            }
        }

        for (size_t li = data_from; li < raw_lines.size(); ++li) {
            if (raw_lines[li].empty()) continue;
            auto cells = csv_split_row(raw_lines[li]);
            ++rows_in;
            std::string arabic, english;
            std::vector<std::string> variants;
            for (size_t ci = 0; ci < cells.size(); ++ci) {
                int r = header.empty() ? 1 : (ci < role.size() ? role[ci] : 1);
                std::string t = csv_flat(cells[ci]);
                if (static_cast<int>(t.size()) < min_chars) continue;
                if (r == 0) continue;
                if (r == 2) { if (arabic.empty()) arabic = t; }
                else if (r == 3) { if (english.empty()) english = t; }
                else variants.push_back(t);
            }
            if (!variants.empty() && !arabic.empty()) {
                for (const auto& v : variants) emit_pair(v, arabic);
            } else if (!variants.empty() && !english.empty()) {
                emit_pair(variants[0], english);
            } else if (variants.size() >= 2 && arabic.empty() && english.empty()) {
                // header-less variant tables (e.g. n1..n4 + arabic missed):
                // first cell is the question, rest are accepted answers.
                for (size_t vi = 1; vi < variants.size(); ++vi) emit_pair(variants[0], variants[vi]);
            } else if (!variants.empty() && variants.size() == 1 && arabic.empty() && english.empty()) {
                ++skipped_empty;
            }
        }
    }
    if (jf) { *jf << "\n]\n"; jf->close(); jf.reset(); }

    log_info("\n==================== csv2json report ====================");
    log_info(strfmt("  csv files       : %s", human_count(csv_files.size()).c_str()));
    log_info(strfmt("  csv rows in     : %s", human_count(rows_in).c_str()));
    log_info(strfmt("  QA pairs out    : %s (%d files)", human_count(pairs_out).c_str(), file_idx));
    log_info(strfmt("  skipped same    : %s (Q==A)", human_count(skipped_same).c_str()));
    log_info(strfmt("  skipped empty   : %s", human_count(skipped_empty).c_str()));
    log_info(strfmt("  skipped dup Q   : %s", human_count(skipped_dup).c_str()));
    log_info(strfmt("  output dir      : %s", outdir.c_str()));
    log_info("==========================================================");
    const u64 min_keep = args.num_u64("min-keep", 1);
    if (pairs_out < min_keep) {
        log_error("quality gate FAILED: kept too few pairs");
        return 1;
    }
    return 0;
}

// ================================================================ jsons command
// Dumps id/question/answer JSON shards to a plain one-line-per-side corpus,
// ready for train_tokenizer (BPE needs raw text). Each pair contributes its
// question line and its answer line, so the tokenizer sees greetings,
// paraphrases and both Darija scripts.
static int cmd_jsons(const Args& args) {
    std::string root    = args.str("dir", ".");
    std::string outpath = args.str("out", "corpus_json.txt");
    const int min_chars = args.num_int("min-chars", 2);
    GAI_CHECK(min_chars >= 0, "--min-chars must be >= 0");

    std::vector<std::string> json_files;
    std::error_code ec;
    if (fs::is_regular_file(root, ec) && fs::path(root).extension() == ".json") {
        json_files.push_back(root);
    } else {
        for (const auto& e : fs::recursive_directory_iterator(root, ec)) {
            if (ec) break;
            std::error_code file_ec;
            if (!e.is_regular_file(file_ec) || file_ec) continue;
            std::string ext = e.path().extension().string();
            for (char& ch : ext) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
            if (ext == ".json" || ext == ".jsonl")
                json_files.push_back(e.path().string());
        }
    }
    std::sort(json_files.begin(), json_files.end());
    if (json_files.empty()) {
        log_error("no .json files found under: " + root);
        return 1;
    }
    log_info(strfmt("[jsons] found %zu file(s) under %s", json_files.size(), root.c_str()));

    if (auto pp = fs::path(outpath).parent_path(); !pp.empty())
        fs::create_directories(pp);
    std::ofstream out(outpath, std::ios::binary);
    GAI_CHECK(out.good(), "cannot write: " + outpath);

    std::unordered_set<u64> seen;
    seen.reserve(1 << 20);
    u64 pairs_in = 0, lines_out = 0;
    auto emit = [&](const std::string& s) {
        std::string t = csv_flat(s);
        if (static_cast<int>(t.size()) < min_chars) return;
        if (!seen.insert(hash_string(t)).second) return;
        out << t << '\n';
        ++lines_out;
    };

    for (const auto& path : json_files) {
        std::vector<QaEntry> pairs;
        if (!load_qa_json(path, pairs)) { log_warn("  cannot parse " + path); continue; }
        pairs_in += pairs.size();
        for (const auto& pr : pairs) {
            emit(pr.question);
            emit(pr.answer);
        }
    }
    out.close();
    GAI_CHECK(out.good(), "corpus write failed");

    log_info("\n==================== jsons pipeline report ====================");
    log_info(strfmt("  files processed : %s", human_count(json_files.size()).c_str()));
    log_info(strfmt("  QA pairs in     : %s", human_count(pairs_in).c_str()));
    log_info(strfmt("  corpus lines    : %s", human_count(lines_out).c_str()));
    log_info(strfmt("  output          : %s", outpath.c_str()));
    log_info("===============================================================");
    const u64 min_keep = args.num_u64("min-keep", 1);
    if (lines_out < min_keep) {
        log_error("quality gate FAILED: kept too few lines");
        return 1;
    }
    return 0;
}

// PARQUET-ONLY: `dump-text` (JSON->corpus) REMOVED.
// BPE corpus comes from the lake instead:
//   data_pipeline parquet-corpus --lake english_parquet --out corpus_en.txt
// (or data_pipeline csvs for Darija tables). No JSON file route.

static std::vector<std::string> collect_files(const std::string& path) {
    std::vector<std::string> files;
    std::error_code ec;
    if (path.empty()) return files;
    if (fs::is_directory(path, ec)) {
        for (const auto& e : fs::recursive_directory_iterator(path, ec)) {
            if (ec) break;
            std::error_code file_ec;
            if (e.is_regular_file(file_ec) && !file_ec) files.push_back(e.path().string());
        }
    } else if (fs::exists(path, ec)) {
        files.push_back(path);
    }
    std::sort(files.begin(), files.end());
    return files;
}

static int cmd_build(const Args& args) {
    std::string tokpath = args.str("tokenizer");
    GAI_CHECK(!tokpath.empty(), "--tokenizer is required");
    Tokenizer tok;
    GAI_CHECK(tok.load(tokpath), "cannot load tokenizer: " + tokpath);
    log_info(strfmt("[tok] vocab=%d", tok.vocab_size()));
    {
        const int expect_vocab = args.num_int("expect-vocab", 0);
        GAI_CHECK(expect_vocab >= 0, "--expect-vocab must be >= 0");
        GAI_CHECK(expect_vocab <= 0 || tok.vocab_size() == expect_vocab,
                  strfmt("tokenizer vocab %d != expected %d (%s): refusing to encode shards",
                         tok.vocab_size(), expect_vocab, tokpath.c_str()));
    }

    std::string outdir = args.str("out", "artifacts/shards");
    GAI_CHECK(!outdir.empty(), "--out must be non-empty");
    fs::create_directories(outdir);

    const double val_ratio  = args.real("val-ratio", 0.005);
    const u64 shard_tokens  = args.num_u64("shard-tokens", 50000000);
    GAI_CHECK(std::isfinite(val_ratio) && val_ratio >= 0.0 && val_ratio <= 1.0,
              "--val-ratio must be finite and in [0,1]");
    GAI_CHECK(shard_tokens > 0, "--shard-tokens must be > 0");
    const int  seq_cap      = args.num_int("seq-len", 4096);
    GAI_CHECK(seq_cap > 0, "--seq-len must be > 0 (splitting is mandatory)");
    const bool do_dedup     = !args.flag("no-dedup");
    const bool style_filter_in = !args.flag("keep-robotic");
    // PRO-EN: same --style-mode/--keep-case contract as the ShardBuild route
    // (json/parquet). This legacy build route must not drift from it.
    std::string style_mode = args.str("style-mode", "darija");
    for (char& ch : style_mode) ch = static_cast<char>(::tolower(static_cast<unsigned char>(ch)));
    if (style_mode != "darija" && style_mode != "en" && style_mode != "off")
        GAI_FAIL("unknown --style-mode '" + style_mode + "' (darija|en|off)");
    const bool style_filter = style_filter_in && style_mode != "off";
    // Optional domain label for data.mix sampling (empty = legacy names).
    std::string domain = args.str("domain", "");
    for (char& ch : domain) {
        if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
    }

    CleanConfig cc;
    cc.drop_pii_lines = style_mode != "en";
    cc.redact_instead_drop = style_mode == "en";
    if (args.flag("keep-case", false)) cc.normalizer.lowercase_latin = false;
    Cleaner cleaner(cc);
    Deduplicator dedup;
    LangId lid;
    PipelineCounters ctr;

    if (args.has("eval-blocklist")) dedup.load_blocklist(args.str("eval-blocklist"));

    // shard writers, rotated when they reach shard_tokens
    int train_idx = 0, val_idx = 0;
    std::unique_ptr<ShardWriter> train_w, val_w;
    u64 train_shard_tokens = 0, val_shard_tokens = 0;

    auto shard_name = [&](const char* split, int idx) {
        if (domain.empty()) return strfmt("%s/%s_%04d.gbin", outdir.c_str(), split, idx);
        return strfmt("%s/%s_%s_%04d.gbin", outdir.c_str(), split, domain.c_str(), idx);
    };
    auto open_train = [&]() {
        std::string p = shard_name("train", train_idx++);
        train_w = std::make_unique<ShardWriter>(p, tok.vocab_size(), true);
        train_shard_tokens = 0;
        log_info("[shard] opened " + p);
    };
    auto open_val = [&]() {
        std::string p = shard_name("val", val_idx++);
        val_w = std::make_unique<ShardWriter>(p, tok.vocab_size(), true);
        val_shard_tokens = 0;
        log_info("[shard] opened " + p);
    };
    auto ensure_train = [&]() { if (!train_w) open_train(); };
    auto ensure_val = [&]() { if (!val_w) open_val(); };
    // FIX P2: lazy writers (was: empty shards written before input validation).

    // Deterministic split by CANONICAL hash: the same document always lands
    // on the same side, and near-dups (same canonical) stay together —
    // otherwise paraphrases leak across train/val silently.
    auto is_val = [&](const std::string& key) {
        const std::string ck = Normalizer::canonical(key);
        const std::string& hkey = ck.empty() ? key : ck;
        return (hash_string(hkey) % 10000ull) < static_cast<u64>(val_ratio * 10000.0);
    };

    // Local window splitter (mirrors ShardBuild::emit_windowed): long docs
    // become non-overlapping seq_cap windows on ONE split side (was: truncate).
    auto emit_windowed_impl = [&](const std::vector<i32>& ids, const std::vector<u8>& mask,
                    const std::string& key, const std::string& lang, bool val) {
        if (ids.empty()) return;
        if (val) {
            ensure_val();
            if (val_shard_tokens >= shard_tokens) { val_w->close(); open_val(); }
            val_w->add_document(ids, &mask);
            val_shard_tokens += ids.size();
            ctr.tokens_val += ids.size();
            ++ctr.docs_val;
        } else {
            ensure_train();
            if (train_shard_tokens >= shard_tokens) { train_w->close(); open_train(); }
            train_w->add_document(ids, &mask);
            train_shard_tokens += ids.size();
            ctr.tokens_train += ids.size();
            ++ctr.docs_train;
        }
        ++ctr.kept;
        ctr.lang_kept[lang]++;
    };
    auto emit = [&](const std::vector<i32>& ids, const std::vector<u8>& mask,
                    const std::string& key, const std::string& lang) {
        if (ids.empty()) return;
        emit_windowed_impl(ids, mask, key, lang, is_val(key));
    };
    auto emit_windowed = [&](const std::vector<i32>& ids, const std::vector<u8>& mask,
                    const std::string& key, const std::string& lang) {
        if (ids.empty()) return;
        if (seq_cap <= 0 || static_cast<int>(ids.size()) <= seq_cap) {
            emit(ids, mask, key, lang);
            return;
        }
        ++ctr.split_docs;
        const bool val = is_val(key);
        const size_t step = static_cast<size_t>(seq_cap);
        for (size_t off = 0; off < ids.size(); off += step) {
            const size_t n = std::min(step, ids.size() - off);
            emit_windowed_impl(
                std::vector<i32>(ids.begin() + off, ids.begin() + off + n),
                std::vector<u8>(mask.begin() + off, mask.begin() + off + n),
                key, lang, val);
        }
    };

    // ---------------- plain text ----------------
    for (const auto& path : collect_files(args.str("text"))) {
        log_info("[text] " + path);
        std::ifstream f(path, std::ios::binary);
        std::string line, cleaned;
        CleanStats cs;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            ++ctr.raw;
            if (!cleaner.clean_line(line, cleaned, cs)) continue;
            ++ctr.cleaned;

            if (!(style_mode == "en" ? quality_check_english(cleaned).accept
                                    : quality_check(cleaned).accept)) {
                ++ctr.quality_dropped;
                continue;
            }
            if (check_toxicity(cleaned).toxic)  { ++ctr.toxic_dropped; continue; }
            if (do_dedup && !dedup.add(cleaned)) { ++ctr.dup_dropped; continue; }

            LangScore ls = lid.classify(cleaned);
            std::vector<i32> ids = tok.encode(cleaned, true, true);
            if (ids.empty()) continue;
            std::vector<u8> mask(ids.size(), 1);   // pretraining supervises everything
            emit_windowed(ids, mask, cleaned, lang_name(ls.tag));
        }
        log_info("  clean: " + cs.summary());
    }

    // ---------------- QA json (id/question/answer -> one "Q\nA" doc each) ------
    // Links the json_seed / csv2json shards to training: every pair becomes a
    // pretraining document (mask = supervise everything), so greetings and
    // paraphrases ("salam labas" next to "salam bikhir") are learned as text.
    // collect_files() already expands a --json directory recursively, so both
    // a single train-*.json file and a whole directory work here.
    for (const auto& path : collect_files(args.str("json"))) {
        if (fs::path(path).extension() != ".json") continue;
        std::vector<QaEntry> pairs;
        if (!load_qa_json(path, pairs)) { log_warn("  cannot parse " + path); continue; }
        log_info(strfmt("[json] %s: %s pairs", path.c_str(), human_count(pairs.size()).c_str()));
        CleanStats cs;
        for (const auto& pr : pairs) {
            ++ctr.raw;
            std::string q, a;
            if (!cleaner.clean_line(pr.question, q, cs)) continue;
            if (!cleaner.clean_line(pr.answer, a, cs)) continue;
            std::string doc = q + "\n" + a;
            ++ctr.cleaned;
            if (!(style_mode == "en" ? quality_check_english(doc).accept
                                    : quality_check(doc).accept)) {
                ++ctr.quality_dropped;
                continue;
            }
            if (check_toxicity(doc).toxic)  { ++ctr.toxic_dropped; continue; }
            if (do_dedup && !dedup.add(doc)) { ++ctr.dup_dropped; continue; }
            LangScore ls = lid.classify(doc);
            std::vector<i32> ids = tok.encode(doc, true, true);
            if (ids.empty()) continue;
            std::vector<u8> mask(ids.size(), 1);
            emit_windowed(ids, mask, doc, lang_name(ls.tag));
        }
    }

    // ---------------- conversations ----------------
    std::vector<Conversation> convs;
    for (const auto& path : collect_files(args.str("chat"))) {
        auto c = read_conversations_jsonl(path);
        log_info(strfmt("[chat] %s: %s conversations", path.c_str(), human_count(c.size()).c_str()));
        convs.insert(convs.end(), c.begin(), c.end());
    }
    const int nsynth = args.num_int("synth", 0);
    GAI_CHECK(nsynth >= 0, "--synth must be >= 0");
    if (nsynth > 0) {
        SynthConfig sc;
        sc.seed = args.num_u64("seed", 1234);
        sc.num_conversations = nsynth;
        SynthGenerator gen(sc);
        auto c = gen.generate_many(nsynth);
        log_info("[synth] " + gen.stats().summary());
        convs.insert(convs.end(), c.begin(), c.end());
    }

    for (const auto& c : convs) {
        ++ctr.raw;
        if (c.messages.empty()) continue;

        // Same clean+quality rule as cmd_json: chat turns are cleaned per
        // turn so SFT never learns raw HTML/mojibake the text path strips.
        std::vector<Message> cleaned_msgs;
        cleaned_msgs.reserve(c.messages.size());
        {
            CleanStats cs_chat;
            bool bad = false;
            for (const auto& m : c.messages) {
                std::string cl;
                if (!cleaner.clean_line(m.content, cl, cs_chat)) { ++ctr.quality_dropped; bad = true; break; }
                if (cl.size() < 2 ||
                    !(style_mode == "en" ? quality_check_english(cl).accept
                                        : quality_check(cl).accept)) {
                    ++ctr.quality_dropped;
                    bad = true;
                    break;
                }
                cleaned_msgs.push_back(Message{m.role, cl});
            }
            if (bad) continue;
        }
        // build a joining key for dedup / split
        std::string key;
        for (const auto& m : cleaned_msgs) key += m.content + "\n";

        // filters on assistant turns (style gate honors --style-mode)
        bool drop = false;
        for (size_t i = 0; i < cleaned_msgs.size(); ++i) {
            const auto& m = cleaned_msgs[i];
            if (check_toxicity(m.content).toxic) { ++ctr.toxic_dropped; drop = true; break; }
            if (scan_pii(m.content).any())       { ++ctr.blocked; drop = true; break; }
            if (style_filter && m.role == Role::Assistant) {
                bool bad_style = false;
                if (style_mode == "en") {
                    bad_style = has_hard_ai_boilerplate(m.content);
                } else {
                    std::string prev = i > 0 ? cleaned_msgs[i - 1].content : "";
                    bad_style = check_assistant_style(m.content, prev, lid).robotic();
                }
                if (bad_style) { ++ctr.robotic_dropped; drop = true; break; }
            }
        }
        if (drop) continue;
        ++ctr.cleaned;

        if (do_dedup && !dedup.add(key)) { ++ctr.dup_dropped; continue; }

        std::vector<u8> mask;
        std::vector<i32> ids = ChatTemplate::encode(tok, cleaned_msgs, false, &mask);
        if (ids.empty()) continue;
        LangScore ls = lid.classify(key);
        emit_windowed(ids, mask, key, lang_name(ls.tag));
    }

    if (train_w) train_w->close();
    if (val_w) val_w->close();

    // ---------------- report ----------------
    std::ostringstream rep;
    rep << "\n==================== data pipeline report ====================\n";
    rep << strfmt("  raw documents      : %s\n", human_count(ctr.raw).c_str());
    rep << strfmt("  passed cleaning    : %s\n", human_count(ctr.cleaned).c_str());
    rep << strfmt("  dropped quality    : %s\n", human_count(ctr.quality_dropped).c_str());
    rep << strfmt("  dropped toxicity   : %s  (rule lists only, NOT a safety guarantee)\n", human_count(ctr.toxic_dropped).c_str());
    rep << strfmt("  dropped robotic    : %s\n", human_count(ctr.robotic_dropped).c_str());
    rep << strfmt("  dropped pii        : %s\n", human_count(ctr.blocked).c_str());
    rep << strfmt("  dropped duplicate  : %s\n", human_count(ctr.dup_dropped).c_str());
    rep << strfmt("  split windows      : %s source docs exceeded seq_cap\n", human_count(ctr.split_docs).c_str());
    rep << strfmt("  kept windows       : %s\n", human_count(ctr.kept).c_str());
    rep << "\n";
    rep << strfmt("  train windows/tokens: %s / %s  (%d shards)\n",
                  human_count(ctr.docs_train).c_str(), human_count(ctr.tokens_train).c_str(), train_idx);
    rep << strfmt("  val   windows/tokens: %s / %s  (%d shards)\n",
                  human_count(ctr.docs_val).c_str(), human_count(ctr.tokens_val).c_str(), val_idx);
    rep << "\n  -- language mix of kept windows --\n";
    u64 tot = ctr.kept ? ctr.kept : 1;
    for (const auto& [l, n] : ctr.lang_kept)
        rep << strfmt("    %-14s %8s  %6.2f%%\n", l.c_str(), human_count(n).c_str(),
                      100.0 * double(n) / double(tot));
    if (do_dedup) rep << "\n  dedup: " << dedup.summary() << "\n";
    rep << "==============================================================\n";

    std::cout << rep.str();
    if (args.has("report")) {
        std::ofstream rf(args.str("report"));
        rf << rep.str();
        log_info("[report] wrote " + args.str("report"));
    }
    // ---- quality gate
    const u64 min_keep = args.num_u64("min-keep", 1);
    if (ctr.kept < min_keep) {
        log_error(strfmt("quality gate FAILED: kept %s windows (< min-keep %s)",
                         human_count(ctr.kept).c_str(), human_count(min_keep).c_str()));
        return 1;
    }
    return 0;
}

static int cmd_retrieve(const Args& args) {
    std::string index = args.str("index");
    std::string query = args.str("query");
    if (query.empty() && args.positional().size() > 1) query = args.positional()[1];
    GAI_CHECK(!index.empty(), "--index <train-*.json file|dir> is required");
    GAI_CHECK(!query.empty(), "--query <text> is required");
    const int top = args.num_int("top", 3);
    const double min_score = args.real("min-score", 0.0);

    std::vector<QaEntry> docs;
    size_t n = load_qa_dir(index, docs);
    if (n == 0) {
        log_error("no QA pairs loaded from: " + index);
        return 1;
    }
    log_info(strfmt("[retrieve] index: %s pairs", human_count(n).c_str()));
    RetrievalIndex idx;
    idx.build(docs);
    auto hits = idx.query(query, top > 0 ? top : 3, min_score);
    if (hits.empty()) {
        log_info("[retrieve] no hit above min-score (answer: spontaneous generation)");
        return 0;
    }
    for (size_t i = 0; i < hits.size(); ++i) {
        const QaEntry& e = idx.doc(hits[i].doc);
        std::cout << strfmt("--- hit %zu  id=%lld  score=%.3f ---\n", i + 1, e.id, hits[i].score);
        std::cout << "Q: " << e.question << "\n";
        std::cout << "A: " << e.answer << "\n";
    }
    return 0;
}

static int cmd_inspect(const Args& args) {
    std::string dir = args.str("shards", "artifacts/shards");
    Tokenizer tok;
    bool have_tok = args.has("tokenizer") && tok.load(args.str("tokenizer"));

    for (const std::string& prefix : {std::string("train"), std::string("val")}) {
        auto files = list_shards(dir, prefix);
        u64 total = 0;
        log_info(strfmt("---- %s: %zu shards ----", prefix.c_str(), files.size()));
        for (const auto& p : files) {
            Shard s;
            if (!s.load_header(p) && !s.load(p)) { log_warn("  cannot read " + p); continue; }
            total += s.n_tokens();
            log_info(strfmt("  %-40s %10s tokens  %8s docs  mask=%s",
                            fs::path(p).filename().string().c_str(),
                            human_count(s.n_tokens()).c_str(),
                            human_count(s.n_docs()).c_str(),
                            s.has_mask() ? "yes" : "no"));
        }
        log_info(strfmt("  total: %s tokens", human_count(total).c_str()));
    }

    if (have_tok) {
        auto files = list_shards(dir, "train");
        if (!files.empty()) {
            Shard s;
            if (s.load(files[0])) {
                u64 n = std::min<u64>(s.n_tokens(), 120);
                std::vector<i32> ids;
                for (u64 i = 0; i < n; ++i) ids.push_back(s.token(i));
                log_info("---- first tokens decoded ----");
                log_info("  " + tok.decode(ids));
            }
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);

    std::string cmd = args.command();
    if (cmd.empty() || args.flag("help")) { usage(); return cmd.empty() ? 1 : 0; }

    try {
        if (cmd == "json" || cmd == "json-inspect" || cmd == "dump-text")
            return cmd_json_removed(args);
        if (cmd == "parquet")         return cmd_parquet(args);
        if (cmd == "parquet-corpus")  return cmd_parquet_corpus(args);
        if (cmd == "tok-info")        return cmd_tok_info(args);
        if (cmd == "csvs")            return cmd_csvs(args);
        if (cmd == "csv2json")        return cmd_csv2json(args);
        if (cmd == "jsons")           return cmd_jsons(args);
        if (cmd == "retrieve")        return cmd_retrieve(args);
        if (cmd == "synth")           return cmd_synth(args);
        if (cmd == "build")           return cmd_build(args);
        if (cmd == "inspect")         return cmd_inspect(args);
        std::cerr << "unknown command: " << cmd << "\n";
        usage();
        return 1;
    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
