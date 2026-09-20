#pragma once

#include "tokenizer/tokenizer.h"
#include <functional>

namespace gai {

struct BpeTrainerConfig {
    int  vocab_size      = 32000;
    int  min_frequency   = 2;
    int  max_token_bytes = 32;     // guard against absurdly long merges (codepoints, not bytes — fair for Arabic)
    bool verbose         = true;
    NormalizerConfig normalizer{};
};

// Word-frequency based BPE trainer (the standard efficient formulation:
// merges are counted over a dictionary of unique pre-tokenized chunks weighted
// by their corpus frequency, not over the raw byte stream).
class BpeTrainer {
public:
    explicit BpeTrainer(BpeTrainerConfig cfg) : cfg_(std::move(cfg)) {}

    void add_text(const std::string& text);       // normalizes + pre-tokenizes + counts
    void add_file(const std::string& path);
    void add_chunk_counts(const std::string& chunk, u64 count);

    Tokenizer train();

    u64 unique_chunks() const { return counts_.size(); }
    u64 total_chunks()  const { return total_; }

private:
    BpeTrainerConfig cfg_;
    std::unordered_map<std::string, u64> counts_;
    u64 total_ = 0;
};

} // namespace gai
