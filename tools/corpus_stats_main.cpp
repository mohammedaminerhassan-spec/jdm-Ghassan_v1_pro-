// corpus_stats - measures everything about a corpus that the design doc promises:
// Arabic/Darija/MSA/Arabizi/French shares, duplicates, lengths, vocab coverage.

#include "tools/cli_common.h"
#include "dataset/corpus_stats.h"
#include "dataset/dedup.h"
#include "dataset/cleaner.h"

#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace gai;

static void usage() {
    std::cout <<
    "corpus_stats - corpus measurement tool\n\n"
    "usage:\n"
    "  corpus_stats --input <file|dir> [--tokenizer tok.gtok] [--dedup] [--pii]\n\n"
    "options:\n"
    "  --input <path>       file or directory (one document per line)\n"
    "  --tokenizer <path>   .gtok file, enables token statistics\n"
    "  --dedup              also compute the duplicate ratio (slower)\n"
    "  --pii                scan for personal data and report counts\n"
    "  --quality            report how many documents fail the quality filter\n"
    "  --limit <n>          stop after n documents\n"
    "  --sample <n>         print n example documents per language tag\n";
}

int main(int argc, char** argv) {
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);
    if (args.flag("help") || argc == 1) { usage(); return 0; }

    try {
        std::string input = args.str("input");
        GAI_CHECK(!input.empty(), "--input is required");

        Tokenizer tok;
        bool have_tok = false;
        if (args.has("tokenizer")) {
            have_tok = tok.load(args.str("tokenizer"));
            GAI_CHECK(have_tok, "cannot load tokenizer: " + args.str("tokenizer"));
            log_info(strfmt("[tok] loaded vocab=%d", tok.vocab_size()));
        }

        std::vector<std::string> files;
        std::error_code ec;
        if (fs::is_directory(input, ec)) {
            for (const auto& e : fs::recursive_directory_iterator(input, ec)) {
                if (ec) break;
                std::error_code file_ec;
                if (e.is_regular_file(file_ec) && !file_ec) files.push_back(e.path().string());
            }
        } else {
            files.push_back(input);
        }
        std::sort(files.begin(), files.end());
        GAI_CHECK(!files.empty(), "no input files found");

        CorpusAnalyzer an(have_tok ? &tok : nullptr);
        Deduplicator dedup;
        LangId lid;
        PiiReport pii_total;
        u64 quality_fail = 0, docs = 0;
        std::map<std::string, u64> quality_reasons;
        std::map<std::string, std::vector<std::string>> samples;
        const i64 limit = args.num("limit", 0);
        const int nsample = args.num_int("sample", 0);
        GAI_CHECK(limit >= 0, "--limit must be >= 0");
        GAI_CHECK(nsample >= 0, "--sample must be >= 0");

        Timer t;
        for (const auto& path : files) {
            std::ifstream f(path, std::ios::binary);
            if (!f.good()) continue;
            std::string line;
            while (std::getline(f, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                ++docs;

                an.add_document(line);
                if (args.flag("dedup")) dedup.add(line);
                if (args.flag("pii")) {
                    PiiReport r = scan_pii(line);
                    pii_total.emails += r.emails;
                    pii_total.phones += r.phones;
                    pii_total.urls_with_creds += r.urls_with_creds;
                    pii_total.ibans += r.ibans;
                    pii_total.cards += r.cards;
                    pii_total.api_keys += r.api_keys;
                    pii_total.ids += r.ids;
                    pii_total.ips += r.ips;
                }
                if (args.flag("quality")) {
                    QualityVerdict v = quality_check(line);
                    if (!v.accept) { ++quality_fail; quality_reasons[v.reason]++; }
                }
                if (nsample > 0) {
                    std::string tag = lang_name(lid.classify(line).tag);
                    auto& s = samples[tag];
                    if (static_cast<int>(s.size()) < nsample) s.push_back(line);
                }
                if (limit && static_cast<i64>(docs) >= limit) break;
            }
            if (limit && static_cast<i64>(docs) >= limit) break;
        }

        CorpusStats st = an.finish();
        if (args.flag("dedup")) st.dup_ratio = dedup.dup_ratio();

        std::cout << st.report(have_tok ? tok.vocab_size() : 0);

        if (args.flag("dedup")) std::cout << "  dedup: " << dedup.summary() << "\n";
        if (args.flag("pii"))   std::cout << "  pii  : " << pii_total.summary() << "\n";
        if (args.flag("quality")) {
            std::cout << strfmt("  quality: %s failed (%.2f%%)\n",
                                human_count(quality_fail).c_str(),
                                docs ? 100.0 * double(quality_fail) / double(docs) : 0.0);
            for (const auto& [r, n] : quality_reasons)
                std::cout << strfmt("    %-20s %s\n", r.c_str(), human_count(n).c_str());
        }
        for (const auto& [tag, list] : samples) {
            std::cout << "\n  -- samples: " << tag << " --\n";
            for (const auto& s : list) std::cout << "    " << s.substr(0, 160) << "\n";
        }
        std::cout << strfmt("\n  analysed in %s\n", human_duration(t.seconds()).c_str());
        return 0;

    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
