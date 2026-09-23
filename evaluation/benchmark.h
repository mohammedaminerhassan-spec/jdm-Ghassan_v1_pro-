#pragma once

#include "inference/generator.h"
#include "dataset/langid.h"
#include <map>
#include <string>
#include <vector>

namespace gai {

// The 14 evaluation categories from the design doc.
enum class EvalCategory : u8 {
    BasicConversation = 0,
    DarijaComprehension,
    ArabiziComprehension,
    ArabicComprehension,
    CodeSwitching,
    ContextRetention,
    MultiTurn,
    CommonSense,
    CulturalContext,
    InstructionFollowing,
    Repetition,
    Hallucination,
    Toxicity,
    Naturalness,
    Count
};

const char* category_name(EvalCategory c);
EvalCategory category_from_name(const std::string& s);

struct EvalItem {
    EvalCategory              category = EvalCategory::BasicConversation;
    std::string               id;
    std::vector<Message>      context;      // multi-turn prompt
    std::string               prompt;       // final user turn
    std::vector<std::string>  expect_any;   // acceptable substrings (fuzzy match)
    std::vector<std::string>  forbid;       // substrings that must NOT appear
    std::string               expect_lang;  // "ar-MA-arab" | "ar-MA-latn" | "ar-MSA" | ""
    int                       max_words = 0;// 0 = no constraint
    std::string               note;
};

struct ItemResult {
    std::string  id;
    EvalCategory category;
    std::string  prompt;
    std::string  response;
    bool         matched = false;
    bool         forbidden_hit = false;
    bool         lang_ok = true;
    bool         length_ok = true;
    bool         robotic = false;
    bool         toxic = false;
    bool         discipline_ok = true;
    double       distinct1 = 0, distinct2 = 0;
    int          max_ngram_repeat = 0;
    double       darija_ratio = 0, msa_ratio = 0;
    int          words = 0;
    double       seconds = 0;
    double       score = 0;   // 0..1 automatic score
    double       tokens_per_sec = 0;  // FIX P2: was never filled; now estimated
};

struct CategoryScore {
    int    items = 0;
    double score_sum = 0;
    int    matched = 0, robotic = 0, toxic = 0, lang_fail = 0, length_fail = 0;
    int    discipline_fail = 0;
    double avg() const { return items ? score_sum / items : 0.0; }
};

struct BenchmarkReport {
    std::vector<ItemResult> results;
    std::map<std::string, CategoryScore> by_category;
    double overall = 0;
    double robotic_rate = 0, toxicity_rate = 0, repetition_rate = 0, discipline_rate = 0;
    double avg_distinct1 = 0, avg_distinct2 = 0;
    double avg_darija_ratio = 0, avg_msa_leak = 0;
    double avg_words = 0, tokens_per_sec = 0;
    int    total = 0;

    std::string to_string() const;
    void        write_jsonl(const std::string& path) const;
    std::string human_eval_sheet() const;   // blank rating form for human raters
};

struct BenchmarkConfig {
    std::string      suite_dir = "evaluation/datasets";
    std::string      categories;         // comma-separated filter, empty = all
    int              max_prompts = 0;    // 0 = all
    bool             verbose = false;
    std::string      output_path;
    GenerationConfig gen;
};

class Benchmark {
public:
    Benchmark(Generator& gen, const Tokenizer& tok, BenchmarkConfig cfg);

    BenchmarkReport run();

    // Loads .jsonl suites from a directory; falls back to the built-in suite.
    // PARQUET-ONLY EN: builtin_suite() is Darija; builtin_suite_en() is the
    // Hermes English suite (40 items: conversation, instruction, reasoning,
    // coding, hallucination, toxicity). Use --suite-en or categories filter.
    static std::vector<EvalItem> load_suite(const std::string& dir);
    static std::vector<EvalItem> builtin_suite();
    static std::vector<EvalItem> builtin_suite_en();
    static void write_suite(const std::string& path, const std::vector<EvalItem>& items);

private:
    ItemResult evaluate_item(const EvalItem& item);

    Generator&      gen_;
    const Tokenizer& tok_;
    BenchmarkConfig cfg_;
    LangId          lid_;
};

// text metrics reused by the report
double distinct_n(const std::string& text, int n);
int    max_ngram_repeat(const std::string& text, int n);
double darija_marker_ratio(const std::string& text, const LangId& lid);

} // namespace gai
