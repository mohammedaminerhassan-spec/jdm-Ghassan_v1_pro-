#pragma once

#include "core/common.h"
#include "dataset/langid.h"
#include "tokenizer/tokenizer.h"
#include <map>
#include <unordered_set>

namespace gai {

// Measures everything the design doc promises to measure about a corpus.
struct CorpusStats {
    u64 documents = 0;
    u64 lines = 0;
    u64 bytes = 0;
    u64 chars = 0;
    u64 words = 0;
    u64 tokens = 0;

    std::map<std::string, u64> by_lang;      // lang tag -> docs
    u64 arabic_script_docs = 0;
    u64 latin_script_docs  = 0;
    u64 arabizi_docs       = 0;
    u64 french_mixed_docs  = 0;
    u64 emoji_docs         = 0;

    double dup_ratio = 0.0;
    double avg_doc_chars = 0.0;
    double avg_doc_words = 0.0;
    double avg_doc_tokens = 0.0;
    double median_doc_tokens = 0.0;
    double tokens_per_word = 0.0;
    double bytes_per_token = 0.0;

    u64 vocab_used = 0;                      // distinct token ids observed
    double vocab_coverage = 0.0;             // vocab_used / vocab_size
    double oov_byte_rate = 0.0;              // fraction of tokens that are raw bytes

    std::vector<u64> doc_token_hist;

    double pct(u64 n) const { return documents ? 100.0 * double(n) / double(documents) : 0.0; }
    std::string report(int vocab_size) const;
};

class CorpusAnalyzer {
public:
    explicit CorpusAnalyzer(const Tokenizer* tok = nullptr) : tok_(tok) {}

    void add_document(const std::string& text);
    void add_file(const std::string& path, bool line_per_doc = true);

    CorpusStats finish();

private:
    const Tokenizer* tok_ = nullptr;
    LangId lid_;
    CorpusStats st_;
    std::vector<u64> doc_tokens_;
    std::unordered_set<i32> seen_tokens_;
};

} // namespace gai
