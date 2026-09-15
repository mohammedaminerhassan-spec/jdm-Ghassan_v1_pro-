// train_tokenizer - builds the Arabic/Darija BPE tokenizer and (optionally) runs
// the vocabulary-size study described in docs/DESIGN.md section 4.

#include "tools/cli_common.h"
#include "tokenizer/bpe_trainer.h"
#include "dataset/synth.h"
#include "core/config.h"

#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace gai;

static void usage() {
    std::cout <<
    "train_tokenizer - Arabic/Darija BPE tokenizer trainer\n\n"
    "usage:\n"
    "  train_tokenizer --input <file|dir> [--output tok.gtok] [--vocab 32000]\n"
    "  train_tokenizer --study --input <file|dir>        # compare 16k/24k/32k\n"
    "  train_tokenizer --synth 20000 --output tok.gtok   # train on generated darija\n\n"
    "options:\n"
    "  --input <path>      corpus file or directory of .txt/.jsonl (one doc per line)\n"
    "  --output <path>     output .gtok file            [artifacts/tokenizer/darija32k.gtok]\n"
    "  --vocab <n>         vocabulary size              [32000]\n"
    "  --min-freq <n>      minimum pair frequency       [2]\n"
    "  --synth <n>         add n synthetic darija conversations to the corpus\n"
    "  --study             train 16k/24k/32k and report fertility per slice\n"
    "  --eval <file>       held-out file for the fertility report\n"
    "  --keep-diacritics   do not strip harakat\n"
    "  --fold-letters      normalise أإآ->ا etc (not recommended for the LM)\n";
}

static std::vector<std::string> collect_inputs(const std::string& path) {
    std::vector<std::string> files;
    std::error_code ec;
    if (path.empty()) return files;
    if (fs::is_directory(path, ec)) {
        for (const auto& e : fs::recursive_directory_iterator(path, ec)) {
            if (!e.is_regular_file()) continue;
            auto ext = e.path().extension().string();
            if (ext == ".txt" || ext == ".jsonl" || ext == ".text") files.push_back(e.path().string());
        }
    } else if (fs::exists(path, ec)) {
        files.push_back(path);
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<std::string> read_lines(const std::string& path, size_t limit = 0) {
    std::vector<std::string> out;
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
        if (limit && out.size() >= limit) break;
    }
    return out;
}

// Representative evaluation slices, used to pick the vocabulary size.
struct Slice { const char* name; std::vector<std::string> lines; };

static std::vector<Slice> default_slices() {
    return {
        {"darija-arab", {
            "شنو خبارك اليوم أخويا واش كلشي مزيان",
            "بغيت نمشي للسوق نشري شي خضرة ولحم",
            "دابا غادي نشوف شنو خاصني ندير فهاد المشكل",
            "واخا صافي غادي نتلاقاو غدا فالصباح إن شاء الله",
            "ماكاينش مشكل خويا غير قولي فوقاش باغي تجي",
            "الله يعطيك الصحة على هاد الخدمة الزوينة",
            "كنت كنقلب عليك من الصباح وماجاوبتيش",
            "هاد الطاجين بنين بزاف شكون طيبو",
        }},
        {"darija-latn", {
            "chno khbark lyoum a khoya wach kolchi mzyan",
            "bghit nemchi l souk nechri chi khodra w l7em",
            "daba ghadi nchouf chno khasni ndir f had lmochkil",
            "wakha safi ghadi ntla9aw ghedda f sba7 inchallah",
            "makaynch mochkil khoya ghir 9oli fo9ach baghi tji",
            "allah y3tik sa7a 3la had lkhedma zwina",
        }},
        {"arabizi", {
            "3lach ma jiti lbare7 kona kanstanaw fik",
            "kif7alk a sa7bi labas 3lik hamdollah",
            "7it ma3andich lwa9t ghadi n2ajel l ghedda",
            "9olto lih bli 5asso yji b7al 9bel",
            "wach 3ndk chi 7aja n9ol lik 3liha",
        }},
        {"msa", {
            "السلام عليكم ورحمة الله وبركاته أهلا بكم",
            "إن التعليم هو أساس تقدم الأمم والشعوب",
            "يجب على الطالب أن يجتهد في دراسته لينجح",
            "المغرب بلد يقع في شمال غرب القارة الأفريقية",
            "تعتبر اللغة العربية من أقدم اللغات في العالم",
        }},
        {"fr-mixed", {
            "بغيت ندير واحد rendez-vous مع الطبيب",
            "الخدمة ديالي فواحد entreprise فالدار البيضاء",
            "khasni ncharji l portable dyali 7it l batterie sala",
            "عندي واحد problème مع الconnexion ديال الأنترنت",
        }},
        {"chat", {
            "<|user|>سلام كيداير<|end|><|assistant|>الحمد لله بخير نتا كيف داير<|end|>",
            "<|user|>واش نقدر نسولك شي حاجة<|end|>><|assistant|>إييه سول براحتك<|end|>",
        }},
    };
}

int main(int argc, char** argv) {
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);

    if (args.flag("help") || args.flag("h") || (argc == 1)) { usage(); return 0; }

    try {
        NormalizerConfig ncfg;
        if (args.flag("keep-diacritics")) ncfg.strip_diacritics = false;
        if (args.flag("fold-letters")) ncfg.fold_letters = true;

        // ---------------- gather corpus
        std::vector<std::string> corpus_lines;
        for (const auto& f : collect_inputs(args.str("input"))) {
            auto l = read_lines(f);
            log_info(strfmt("[corpus] %s: %s lines", f.c_str(), human_count(l.size()).c_str()));
            corpus_lines.insert(corpus_lines.end(), l.begin(), l.end());
        }

        i64 nsynth = args.num("synth", 0);
        if (nsynth > 0) {
            log_info(strfmt("[synth] generating %lld darija conversations for tokenizer training",
                            static_cast<long long>(nsynth)));
            SynthConfig sc;
            sc.num_conversations = static_cast<int>(nsynth);
            sc.max_template_uses = 100000;   // tokenizer wants coverage, not novelty
            SynthGenerator gen(sc);
            auto convs = gen.generate_many(static_cast<int>(nsynth));
            for (const auto& c : convs)
                for (const auto& m : c.messages) corpus_lines.push_back(m.content);
            log_info("[synth] " + gen.stats().summary());
        }

        GAI_CHECK(!corpus_lines.empty(),
                  "no corpus data. use --input <path> and/or --synth <n>");
        log_info(strfmt("[corpus] total %s lines", human_count(corpus_lines.size()).c_str()));

        // ---------------- evaluation slices
        std::vector<Slice> slices = default_slices();
        if (args.has("eval")) {
            slices.push_back({"held-out", read_lines(args.str("eval"), 20000)});
        }

        auto report_fertility = [&](const Tokenizer& tk, const std::string& label) {
            log_info("  ---- fertility: " + label + " ----");
            for (const auto& s : slices) {
                if (s.lines.empty()) continue;
                auto f = tk.measure(s.lines);
                log_info(strfmt("    %-12s  %.3f tok/word   %.2f bytes/tok",
                                s.name, f.tokens_per_word, f.bytes_per_token));
            }
            auto all = tk.measure(corpus_lines);
            log_info(strfmt("    %-12s  %.3f tok/word   %.2f bytes/tok",
                            "corpus", all.tokens_per_word, all.bytes_per_token));
        };

        // ---------------- study mode
        if (args.flag("study")) {
            log_info("================ vocabulary size study ================");
            struct Result { int vocab; double darija_ar; double darija_lt; double arabizi; double msa; };
            std::vector<Result> results;

            for (int v : {16000, 24000, 32000}) {
                log_info(strfmt("\n[study] training vocab=%d ...", v));
                BpeTrainerConfig cfg;
                cfg.vocab_size = v;
                cfg.min_frequency = static_cast<int>(args.num("min-freq", 2));
                cfg.normalizer = ncfg;
                cfg.verbose = false;
                BpeTrainer tr(cfg);
                for (const auto& l : corpus_lines) tr.add_text(l);
                Tokenizer tk = tr.train();
                report_fertility(tk, std::to_string(v));

                Result r{v, 0, 0, 0, 0};
                for (const auto& s : slices) {
                    if (s.lines.empty()) continue;
                    auto f = tk.measure(s.lines);
                    std::string n = s.name;
                    if (n == "darija-arab") r.darija_ar = f.tokens_per_word;
                    else if (n == "darija-latn") r.darija_lt = f.tokens_per_word;
                    else if (n == "arabizi") r.arabizi = f.tokens_per_word;
                    else if (n == "msa") r.msa = f.tokens_per_word;
                }
                results.push_back(r);

                std::string out = strfmt("artifacts/tokenizer/study_%dk.gtok", v / 1000);
                fs::create_directories(fs::path(out).parent_path());
                tk.save(out);
                log_info("  saved " + out);
            }

            log_info("\n================ study summary (tokens per word, lower is better) ====");
            log_info("  vocab   darija-ar  darija-lt  arabizi     msa    embedding params");
            for (const auto& r : results) {
                log_info(strfmt("  %5dk   %8.3f   %8.3f  %7.3f  %6.3f  %14s",
                                r.vocab / 1000, r.darija_ar, r.darija_lt, r.arabizi, r.msa,
                                human_count(static_cast<u64>(r.vocab) * 768).c_str()));
            }
            // selection rule from the design doc
            double best = 1e9;
            for (const auto& r : results) best = std::min(best, r.darija_ar);
            int chosen = results.empty() ? 16000 : results.back().vocab;
            for (const auto& r : results) {
                if (r.darija_ar <= best * 1.03) { chosen = r.vocab; break; }
            }
            log_info(strfmt("\n  selection rule: smallest vocab within 3%% of the best darija "
                            "fertility -> %d", chosen));
            return 0;
        }

        // ---------------- normal training
        BpeTrainerConfig cfg;
        cfg.vocab_size = static_cast<int>(args.num("vocab", 32000));
        cfg.min_frequency = static_cast<int>(args.num("min-freq", 2));
        cfg.normalizer = ncfg;
        cfg.verbose = true;

        log_info(strfmt("[bpe] training vocab=%d min_freq=%d", cfg.vocab_size, cfg.min_frequency));
        Timer t;
        BpeTrainer tr(cfg);
        for (const auto& l : corpus_lines) tr.add_text(l);
        Tokenizer tk = tr.train();
        log_info(strfmt("[bpe] trained in %s", human_duration(t.seconds()).c_str()));

        report_fertility(tk, "final");

        std::string out = args.str("output", "artifacts/tokenizer/darija.gtok");
        fs::create_directories(fs::path(out).has_parent_path()
                               ? fs::path(out).parent_path() : fs::path("."));
        tk.save(out);
        log_info("[bpe] saved " + out);

        // round-trip sanity on real Darija
        const char* probes[] = {
            "شنو خبارك؟", "chno khbark a sahbi", "3lach ma jiti",
            "بغيت نdownloadi هاد الفيديو", "السلام عليكم ورحمة الله",
        };
        log_info("  ---- sample encodings ----");
        for (const char* p : probes) {
            auto ids = tk.encode(p);
            std::string back = tk.decode(ids);
            log_info(strfmt("    %-40s -> %2zu tokens", p, ids.size()));
            std::string pieces;
            for (i32 id : ids) {
                if (!pieces.empty()) pieces += "|";
                pieces += tk.token_text(id);
            }
            log_info("      " + pieces);
        }
        return 0;

    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
