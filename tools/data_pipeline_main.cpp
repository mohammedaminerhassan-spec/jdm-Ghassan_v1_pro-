// data_pipeline - the full ingestion path:
//   raw text / synthetic conversations
//     -> clean -> normalise -> PII filter -> quality -> toxicity -> langid
//     -> anti-robotic style filter -> dedup -> tokenize -> shard (train/val)

#include "tools/cli_common.h"
#include "dataset/cleaner.h"
#include "dataset/dedup.h"
#include "dataset/langid.h"
#include "dataset/synth.h"
#include "dataset/corpus_stats.h"
#include "dataset/json_reader.h"
#include "dataset/retrieval.h"
#include "training/dataloader.h"
#include "tokenizer/chat_template.h"

#include <cctype>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_set>

namespace fs = std::filesystem;
using namespace gai;

static void usage() {
    std::cout <<
    "data_pipeline - dataset construction for Ghassan AI\n\n"
    "usage:\n"
    "  data_pipeline json     --dir <json-dir> --tokenizer tok.gtok --out artifacts/shards\n"
    "  data_pipeline synth    --out data/synth.jsonl --n 50000\n"
    "  data_pipeline build    --tokenizer tok.gtok --out artifacts/shards \\\n"
    "                         [--text <file|dir>] [--chat <file.jsonl>] [--synth <n>]\n"
    "  data_pipeline inspect  --shards artifacts/shards --tokenizer tok.gtok\n"
    "  data_pipeline json-inspect --file <file.json|jsonl>\n"
    "  data_pipeline dump-text --dir <json-dir> --out corpus.txt\n\n"
    "  data_pipeline csvs --dir <csv-root> --out corpus_csv.txt\n"
    "  data_pipeline csv2json --dir <csv-root> --out json_shards\n"
    "  data_pipeline jsons --dir <json-root> --out corpus_json.txt\n"
    "  data_pipeline retrieve --index <json-file|dir> --query \"salam labas\" [--top 3]\n\n"
    "dump-text options:\n"
    "  --dir <dir>            directory with *.json / *.jsonl files (recursive)\n"
    "  --out <file>           output text corpus (one doc per line) [corpus.txt]\n"
  "  --min-chars <n>        minimum cleaned chars         [10]\n\n"
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
  "json options (THE training-data path: JSON/JSONL -> .gbin shards):\n"
  "  --dir <dir>            directory with *.json / *.jsonl (recursive, or one file)\n"
  "  --tokenizer <path>     .gtok tokenizer (required)\n"
  "  --out <dir>            shard output directory        [artifacts/shards]\n"
  "  --val-ratio <f>        validation fraction           [0.005]\n"
  "  --shard-tokens <n>     tokens per shard              [50000000]\n"
  "  --seq-len <n>          max tokens per document       [4096]\n"
  "  --domain <name>        domain label -> train_<domain>_*.gbin (for data.mix)\n"
  "  --min-keep <n>         fail if fewer docs kept       [1]\n"
  "  --keep-robotic       do not apply the anti-boilerplate filter\n"
  "  --no-dedup           skip deduplication\n\n"
  "  accepted doc schemas (auto-detected per object):\n"
  "    chat:        {\"messages\":[{\"role\":\"user\",\"content\":\"...\"}, ...]}\n"
  "    instruction: {\"instruction\":\"...\",\"input\":\"...\",\"output\":\"...\"}\n"
  "    prompt:      {\"prompt\":\"...\",\"completion\":\"...\"} (also question/answer)\n"
  "    text:        {\"text\":\"...\"} (also content/sentence/document/...)\n"
  "  chat/instruction/prompt docs get SFT loss masks (assistant only);\n"
  "  plain-text docs get full masks (pretraining).\n\n"
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
    "  --report <path>      write the pipeline report here\n";
}

struct PipelineCounters {
    u64 raw = 0, cleaned = 0, quality_dropped = 0, toxic_dropped = 0;
    u64 robotic_dropped = 0, dup_dropped = 0, blocked = 0, kept = 0;
    u64 tokens_train = 0, tokens_val = 0, docs_train = 0, docs_val = 0;
    std::map<std::string, u64> lang_kept;
};

static int cmd_synth(const Args& args) {
    SynthConfig cfg;
    cfg.seed = static_cast<u64>(args.num("seed", 1234));
    cfg.num_conversations = static_cast<int>(args.num("n", 20000));
    cfg.max_template_uses = static_cast<int>(args.num("max-template-uses", 40));

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
static int cmd_json(const Args& args) {
    // ---- tokenizer
    std::string tokpath = args.str("tokenizer");
    GAI_CHECK(!tokpath.empty(), "--tokenizer is required");
    Tokenizer tok;
    GAI_CHECK(tok.load(tokpath), "cannot load tokenizer: " + tokpath);
    log_info(strfmt("[tok ] vocab=%d", tok.vocab_size()));

    // ---- options
    std::string json_dir  = args.str("dir", ".");
    std::string outdir        = args.str("out", "artifacts/shards");
    const double val_ratio    = args.real("val-ratio", 0.005);
    const u64 shard_tokens    = static_cast<u64>(args.num("shard-tokens", 50000000));
    const int seq_cap         = static_cast<int>(args.num("seq-len", 4096));
    const bool do_dedup       = !args.flag("no-dedup");
    const bool style_filter   = !args.flag("keep-robotic");
    // Optional domain label: shards become train_<domain>_*.gbin so the
    // trainer's data.mix weights can sample domains (empty = legacy names).
    std::string domain = args.str("domain", "");
    for (char& ch : domain) {
        if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
    }

    fs::create_directories(outdir);

    // ---- shard writers (WITH masks: chat docs need SFT supervision)
    int train_idx = 0, val_idx = 0;
    std::unique_ptr<ShardWriter> train_w, val_w;
    u64 train_shard_tok = 0, val_shard_tok = 0;

    auto shard_name = [&](const char* split, int idx) {
        if (domain.empty()) return strfmt("%s/%s_%04d.gbin", outdir.c_str(), split, idx);
        return strfmt("%s/%s_%s_%04d.gbin", outdir.c_str(), split, domain.c_str(), idx);
    };
    auto open_train = [&]() {
        std::string p = shard_name("train", train_idx++);
        train_w = std::make_unique<ShardWriter>(p, tok.vocab_size(), true);
        train_shard_tok = 0;
        log_info("[shard] opened " + p);
    };
    auto open_val = [&]() {
        std::string p = shard_name("val", val_idx++);
        val_w = std::make_unique<ShardWriter>(p, tok.vocab_size(), true);
        val_shard_tok = 0;
        log_info("[shard] opened " + p);
    };
    open_train();
    open_val();

    Cleaner cleaner;
    Deduplicator dedup;
    LangId lid;
    PipelineCounters ctr;
    u64 total_tokens = 0;
    u64 chat_docs = 0, text_docs = 0;

    auto is_val = [&](const std::string& key) {
        return (hash_string(key) % 10000ull) < static_cast<u64>(val_ratio * 10000.0);
    };
    auto emit = [&](const std::vector<i32>& ids, const std::vector<u8>& mask,
                    const std::string& key, const std::string& lang) {
        if (ids.empty()) return;
        if (is_val(key)) {
            if (val_shard_tok >= shard_tokens) { val_w->close(); open_val(); }
            val_w->add_document(ids, &mask);
            val_shard_tok += ids.size();
            ctr.tokens_val += ids.size();
            ++ctr.docs_val;
        } else {
            if (train_shard_tok >= shard_tokens) { train_w->close(); open_train(); }
            train_w->add_document(ids, &mask);
            train_shard_tok += ids.size();
            ctr.tokens_train += ids.size();
            ++ctr.docs_train;
        }
        ++ctr.kept;
        ctr.lang_kept[lang]++;
    };

    // ---- collect json files (RECURSIVE: datasets may be nested in subfolders)
    std::vector<std::string> json_files;
    std::error_code ec;
    if (fs::is_regular_file(json_dir, ec)) {
        json_files.push_back(json_dir);
    } else {
        for (const auto& e : fs::recursive_directory_iterator(json_dir, ec)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            for (char& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (ext == ".json" || ext == ".jsonl") json_files.push_back(e.path().string());
        }
    }
    std::sort(json_files.begin(), json_files.end());
    if (json_files.empty()) {
        log_error("no .json/.jsonl files found in: " + json_dir);
        return 1;
    }
    log_info(strfmt("[json] found %zu file(s) in %s", json_files.size(), json_dir.c_str()));

    JsonReaderOptions ropts;
    ropts.verbose = false;
    CleanStats cs;

    for (const auto& jfile : json_files) {
        log_info("[json] processing: " + jfile);
        size_t file_docs = read_json_docs(jfile, [&](const JsonDoc& doc) {
            ++ctr.raw;
            if (doc.is_chat) {
                if (doc.messages.empty()) return;
                std::string key;
                for (const auto& m : doc.messages) key += m.content + "\n";
                // filters on every turn (same policy as `build --chat`)
                for (size_t i = 0; i < doc.messages.size(); ++i) {
                    const auto& m = doc.messages[i];
                    if (m.content.size() < 2) { ++ctr.quality_dropped; return; }
                    if (check_toxicity(m.content).toxic) { ++ctr.toxic_dropped; return; }
                    if (scan_pii(m.content).any()) { ++ctr.blocked; return; }
                    if (style_filter && m.role == Role::Assistant) {
                        std::string prev = i > 0 ? doc.messages[i - 1].content : "";
                        if (check_assistant_style(m.content, prev, lid).robotic()) {
                            ++ctr.robotic_dropped;
                            return;
                        }
                    }
                }
                ++ctr.cleaned;
                if (do_dedup && !dedup.add(key)) { ++ctr.dup_dropped; return; }
                std::vector<u8> mask;
                std::vector<i32> ids = ChatTemplate::encode(tok, doc.messages, false, &mask);
                if (ids.empty()) return;
                if (static_cast<int>(ids.size()) > seq_cap) {
                    ids.resize(static_cast<size_t>(seq_cap));
                    mask.resize(static_cast<size_t>(seq_cap));
                }
                total_tokens += ids.size();
                LangScore ls = lid.classify(key);
                emit(ids, mask, key, lang_name(ls.tag));
                ++chat_docs;
            } else {
                if (doc.text.size() < 10) return;  // skip very short strings
                std::string cleaned;
                if (!cleaner.clean_line(doc.text, cleaned, cs)) return;
                ++ctr.cleaned;
                if (!quality_check(cleaned).accept) { ++ctr.quality_dropped; return; }
                if (check_toxicity(cleaned).toxic) { ++ctr.toxic_dropped; return; }
                if (do_dedup && !dedup.add(cleaned)) { ++ctr.dup_dropped; return; }
                LangScore ls = lid.classify(cleaned);
                std::vector<i32> ids = tok.encode(cleaned, true, true);
                if (ids.empty()) return;
                if (static_cast<int>(ids.size()) > seq_cap)
                    ids.resize(static_cast<size_t>(seq_cap));
                total_tokens += ids.size();
                std::vector<u8> mask(ids.size(), 1);  // pretraining: supervise all
                emit(ids, mask, cleaned, lang_name(ls.tag));
                ++text_docs;
            }
        }, ropts);
        log_info(strfmt("  -> %s docs so far (this file: %zu)",
                        human_count(ctr.raw).c_str(), file_docs));
    }

    train_w->close();
    val_w->close();

    log_info("\n==================== json pipeline report ====================");
    log_info(strfmt("  total docs        : %s  (chat %s / text %s)",
                    human_count(ctr.raw).c_str(), human_count(chat_docs).c_str(),
                    human_count(text_docs).c_str()));
    log_info(strfmt("  cleaned           : %s", human_count(ctr.cleaned).c_str()));
    log_info(strfmt("  quality dropped   : %s", human_count(ctr.quality_dropped).c_str()));
    log_info(strfmt("  toxic dropped     : %s", human_count(ctr.toxic_dropped).c_str()));
    log_info(strfmt("  pii blocked       : %s", human_count(ctr.blocked).c_str()));
    log_info(strfmt("  robotic dropped   : %s", human_count(ctr.robotic_dropped).c_str()));
    log_info(strfmt("  duplicates        : %s", human_count(ctr.dup_dropped).c_str()));
    log_info(strfmt("  total tokens      : %s", human_count(total_tokens).c_str()));
    log_info(strfmt("  train docs/tokens : %s / %s  (%d shards)",
                    human_count(ctr.docs_train).c_str(), human_count(ctr.tokens_train).c_str(), train_idx));
    log_info(strfmt("  val   docs/tokens : %s / %s  (%d shards)",
                    human_count(ctr.docs_val).c_str(), human_count(ctr.tokens_val).c_str(), val_idx));
    log_info("  -- by language --");
    for (const auto& [l, n] : ctr.lang_kept)
        log_info(strfmt("    %-12s %s", l.c_str(), human_count(n).c_str()));
    log_info("================================================================");
    // ---- quality gate: never let an empty or fully-duplicate corpus through
    const u64 kept = ctr.docs_train + ctr.docs_val;
    const u64 min_keep = static_cast<u64>(args.num("min-keep", 1));
    if (kept < min_keep) {
        log_error(strfmt("quality gate FAILED: kept %s docs (< min-keep %s)",
                         human_count(kept).c_str(), human_count(min_keep).c_str()));
        return 1;
    }
    if (ctr.raw > 0 && ctr.dup_dropped * 2 > ctr.raw)
        log_warn("quality gate WARNING: >50% exact duplicates — corpus is template-dominated");
    return 0;
}

static int cmd_json_inspect(const Args& args) {
    std::string file = args.str("file");
    GAI_CHECK(!file.empty(), "--file is required");
    inspect_json(file);
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
    const int min_chars = static_cast<int>(args.num("min-chars", 2));
    const bool with_eng = args.flag("with-english", false);

    std::vector<std::string> csv_files;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(root, ec)) {
        if (!e.is_regular_file()) continue;
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
    const u64 min_keep = static_cast<u64>(args.num("min-keep", 1));
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
    const int per_file   = static_cast<int>(args.num("docs-per-file", 20000));
    long long next_id    = args.num("id-start", 1000000);
    const int min_chars  = static_cast<int>(args.num("min-chars", 2));
    GAI_CHECK(per_file > 0, "--docs-per-file must be > 0");

    std::vector<std::string> csv_files;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(root, ec)) {
        if (!e.is_regular_file()) continue;
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
    const u64 min_keep = static_cast<u64>(args.num("min-keep", 1));
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
    const int min_chars = static_cast<int>(args.num("min-chars", 2));

    std::vector<std::string> json_files;
    std::error_code ec;
    if (fs::is_regular_file(root, ec) && fs::path(root).extension() == ".json") {
        json_files.push_back(root);
    } else {
        for (const auto& e : fs::recursive_directory_iterator(root, ec)) {
            if (!e.is_regular_file()) continue;
            if (e.path().extension().string() == ".json")
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
    const u64 min_keep = static_cast<u64>(args.num("min-keep", 1));
    if (lines_out < min_keep) {
        log_error("quality gate FAILED: kept too few lines");
        return 1;
    }
    return 0;
}

// ================================================================ dump-text command
// Dumps cleaned JSON/JSONL text to a plain one-doc-per-line corpus, ready
// for train_tokenizer (BPE needs raw text, not .gbin shards).
static int cmd_dump_text(const Args& args) {
    std::string json_dir = args.str("dir", ".");
    std::string outpath     = args.str("out", "corpus.txt");
    const int min_chars     = static_cast<int>(args.num("min-chars", 10));

    if (auto pp = fs::path(outpath).parent_path(); !pp.empty())
        fs::create_directories(pp);
    std::ofstream out(outpath, std::ios::binary);
    GAI_CHECK(out.good(), "cannot write: " + outpath);

    Cleaner cleaner;
    CleanStats cs;
    JsonReaderOptions opts;
    opts.verbose = true;

    u64 raw = 0, kept = 0;
    auto on_text = [&](const std::string& text) {
        ++raw;
        std::string cleaned;
        if (!cleaner.clean_line(text, cleaned, cs)) return;
        if (static_cast<int>(cleaned.size()) < min_chars) return;
        for (char& ch : cleaned)
            if (ch == '\n' || ch == '\r') ch = ' ';
        out << cleaned << '\n';
        ++kept;
    };
    if (fs::is_regular_file(json_dir)) {
        log_info("[dump] processing: " + json_dir);
        read_json_docs(json_dir, [&](const JsonDoc& d) {
            if (d.is_chat) {
                for (const auto& m : d.messages) on_text(m.content);
            } else {
                on_text(d.text);
            }
        }, opts);
    } else {
        std::vector<std::string> json_files;
        std::error_code ec;
        for (const auto& e : fs::recursive_directory_iterator(json_dir, ec)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            for (char& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (ext == ".json" || ext == ".jsonl") json_files.push_back(e.path().string());
        }
        std::sort(json_files.begin(), json_files.end());
        if (json_files.empty()) {
            log_error("no .json/.jsonl files found in: " + json_dir);
            return 1;
        }
        log_info(strfmt("[dump] found %zu file(s) in %s",
                        json_files.size(), json_dir.c_str()));
        for (const auto& jfile : json_files) {
            log_info("[dump] processing: " + jfile);
            read_json_docs(jfile, [&](const JsonDoc& d) {
                if (d.is_chat) {
                    for (const auto& m : d.messages) on_text(m.content);
                } else {
                    on_text(d.text);
                }
            }, opts);
        }
    }
    out.close();
    GAI_CHECK(out.good(), "corpus write failed");

    log_info("\n==================== dump-text report ====================");
    log_info(strfmt("  raw strings  : %s", human_count(raw).c_str()));
    log_info(strfmt("  kept lines   : %s", human_count(kept).c_str()));
    log_info(strfmt("  output       : %s", outpath.c_str()));
    log_info("  clean: " + cs.summary());
    log_info("===========================================================");
    log_info("  next: train_tokenizer --input " + outpath +
             " --vocab 16000 --output artifacts/tokenizer/darija.gtok");
    return 0;
}

static std::vector<std::string> collect_files(const std::string& path) {
    std::vector<std::string> files;
    std::error_code ec;
    if (path.empty()) return files;
    if (fs::is_directory(path, ec)) {
        for (const auto& e : fs::recursive_directory_iterator(path, ec))
            if (e.is_regular_file()) files.push_back(e.path().string());
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

    std::string outdir = args.str("out", "artifacts/shards");
    fs::create_directories(outdir);

    const double val_ratio  = args.real("val-ratio", 0.005);
    const u64 shard_tokens  = static_cast<u64>(args.num("shard-tokens", 50000000));
    const int  seq_cap      = static_cast<int>(args.num("seq-len", 4096));
    const bool do_dedup     = !args.flag("no-dedup");
    const bool style_filter = !args.flag("keep-robotic");
    // Optional domain label for data.mix sampling (empty = legacy names).
    std::string domain = args.str("domain", "");
    for (char& ch : domain) {
        if (ch == ' ' || ch == '/' || ch == '\\') ch = '_';
    }

    Cleaner cleaner;
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
    open_train();
    open_val();

    // Deterministic split by document hash: the same document always lands on the
    // same side, so re-running the pipeline never leaks val into train.
    auto is_val = [&](const std::string& key) {
        return (hash_string(key) % 10000ull) < static_cast<u64>(val_ratio * 10000.0);
    };

    auto emit = [&](const std::vector<i32>& ids, const std::vector<u8>& mask,
                    const std::string& key, const std::string& lang) {
        if (ids.empty()) return;
        if (is_val(key)) {
            if (val_shard_tokens >= shard_tokens) { val_w->close(); open_val(); }
            val_w->add_document(ids, &mask);
            val_shard_tokens += ids.size();
            ctr.tokens_val += ids.size();
            ++ctr.docs_val;
        } else {
            if (train_shard_tokens >= shard_tokens) { train_w->close(); open_train(); }
            train_w->add_document(ids, &mask);
            train_shard_tokens += ids.size();
            ctr.tokens_train += ids.size();
            ++ctr.docs_train;
        }
        ++ctr.kept;
        ctr.lang_kept[lang]++;
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

            if (!quality_check(cleaned).accept) { ++ctr.quality_dropped; continue; }
            if (check_toxicity(cleaned).toxic)  { ++ctr.toxic_dropped; continue; }
            if (do_dedup && !dedup.add(cleaned)) { ++ctr.dup_dropped; continue; }

            LangScore ls = lid.classify(cleaned);
            std::vector<i32> ids = tok.encode(cleaned, true, true);
            if (static_cast<int>(ids.size()) > seq_cap) ids.resize(static_cast<size_t>(seq_cap));
            std::vector<u8> mask(ids.size(), 1);   // pretraining supervises everything
            emit(ids, mask, cleaned, lang_name(ls.tag));
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
            if (!quality_check(doc).accept) { ++ctr.quality_dropped; continue; }
            if (check_toxicity(doc).toxic)  { ++ctr.toxic_dropped; continue; }
            if (do_dedup && !dedup.add(doc)) { ++ctr.dup_dropped; continue; }
            LangScore ls = lid.classify(doc);
            std::vector<i32> ids = tok.encode(doc, true, true);
            if (static_cast<int>(ids.size()) > seq_cap) ids.resize(static_cast<size_t>(seq_cap));
            std::vector<u8> mask(ids.size(), 1);
            emit(ids, mask, doc, lang_name(ls.tag));
        }
    }

    // ---------------- conversations ----------------
    std::vector<Conversation> convs;
    for (const auto& path : collect_files(args.str("chat"))) {
        auto c = read_conversations_jsonl(path);
        log_info(strfmt("[chat] %s: %s conversations", path.c_str(), human_count(c.size()).c_str()));
        convs.insert(convs.end(), c.begin(), c.end());
    }
    i64 nsynth = args.num("synth", 0);
    if (nsynth > 0) {
        SynthConfig sc;
        sc.seed = static_cast<u64>(args.num("seed", 1234));
        sc.num_conversations = static_cast<int>(nsynth);
        SynthGenerator gen(sc);
        auto c = gen.generate_many(static_cast<int>(nsynth));
        log_info("[synth] " + gen.stats().summary());
        convs.insert(convs.end(), c.begin(), c.end());
    }

    for (const auto& c : convs) {
        ++ctr.raw;
        if (c.messages.empty()) continue;

        // build a joining key for dedup / split
        std::string key;
        for (const auto& m : c.messages) key += m.content + "\n";

        // filters on assistant turns
        bool drop = false;
        for (size_t i = 0; i < c.messages.size(); ++i) {
            const auto& m = c.messages[i];
            if (check_toxicity(m.content).toxic) { ++ctr.toxic_dropped; drop = true; break; }
            if (scan_pii(m.content).any())       { ++ctr.blocked; drop = true; break; }
            if (style_filter && m.role == Role::Assistant) {
                std::string prev = i > 0 ? c.messages[i - 1].content : "";
                if (check_assistant_style(m.content, prev, lid).robotic()) {
                    ++ctr.robotic_dropped;
                    drop = true;
                    break;
                }
            }
        }
        if (drop) continue;
        ++ctr.cleaned;

        if (do_dedup && !dedup.add(key)) { ++ctr.dup_dropped; continue; }

        std::vector<u8> mask;
        std::vector<i32> ids = ChatTemplate::encode(tok, c.messages, false, &mask);
        if (static_cast<int>(ids.size()) > seq_cap) {
            ids.resize(static_cast<size_t>(seq_cap));
            mask.resize(static_cast<size_t>(seq_cap));
        }
        LangScore ls = lid.classify(key);
        emit(ids, mask, key, lang_name(ls.tag));
    }

    train_w->close();
    val_w->close();

    // ---------------- report ----------------
    std::ostringstream rep;
    rep << "\n==================== data pipeline report ====================\n";
    rep << strfmt("  raw documents      : %s\n", human_count(ctr.raw).c_str());
    rep << strfmt("  passed cleaning    : %s\n", human_count(ctr.cleaned).c_str());
    rep << strfmt("  dropped quality    : %s\n", human_count(ctr.quality_dropped).c_str());
    rep << strfmt("  dropped toxicity   : %s\n", human_count(ctr.toxic_dropped).c_str());
    rep << strfmt("  dropped robotic    : %s\n", human_count(ctr.robotic_dropped).c_str());
    rep << strfmt("  dropped pii        : %s\n", human_count(ctr.blocked).c_str());
    rep << strfmt("  dropped duplicate  : %s\n", human_count(ctr.dup_dropped).c_str());
    rep << strfmt("  kept               : %s\n", human_count(ctr.kept).c_str());
    rep << "\n";
    rep << strfmt("  train docs/tokens  : %s / %s  (%d shards)\n",
                  human_count(ctr.docs_train).c_str(), human_count(ctr.tokens_train).c_str(), train_idx);
    rep << strfmt("  val   docs/tokens  : %s / %s  (%d shards)\n",
                  human_count(ctr.docs_val).c_str(), human_count(ctr.tokens_val).c_str(), val_idx);
    rep << "\n  -- language mix of kept documents --\n";
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
    const u64 min_keep = static_cast<u64>(args.num("min-keep", 1));
    if (ctr.kept < min_keep) {
        log_error(strfmt("quality gate FAILED: kept %s docs (< min-keep %s)",
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
    const int top = static_cast<int>(args.num("top", 3));
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
            if (!s.load(p)) { log_warn("  cannot read " + p); continue; }
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
        if (cmd == "json")            return cmd_json(args);
        if (cmd == "json-inspect")    return cmd_json_inspect(args);
        if (cmd == "dump-text")       return cmd_dump_text(args);
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
