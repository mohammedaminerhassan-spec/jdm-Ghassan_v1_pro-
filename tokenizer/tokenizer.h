#pragma once

#include "core/common.h"
#include "tokenizer/normalizer.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace gai {

// ---------------------------------------------------------------- special tokens
namespace special {
constexpr i32 PAD       = 0;
constexpr i32 BOS       = 1;
constexpr i32 EOS       = 2;
constexpr i32 UNK       = 3;
constexpr i32 SYSTEM    = 4;
constexpr i32 USER      = 5;
constexpr i32 ASSISTANT = 6;
constexpr i32 END       = 7;
constexpr i32 NL        = 8;
constexpr i32 FR        = 9;
constexpr i32 AR        = 10;
constexpr i32 DARIJA    = 11;
constexpr i32 LATIN     = 12;
constexpr i32 TOOL      = 13;
constexpr i32 MASK      = 14;
constexpr i32 RESERVED  = 15;
constexpr i32 COUNT     = 16;
}

const std::vector<std::string>& special_token_strings();

// ---------------------------------------------------------------- tokenizer
// Byte-level BPE with an Arabic/Darija-aware pre-tokenizer and byte fallback:
// every input round-trips exactly, <unk> is never produced.
class Tokenizer {
public:
    Tokenizer() = default;

    // ---- construction
    void init_empty(const NormalizerConfig& ncfg);
    // vocab: id -> raw byte string; merges: ordered (left,right) pairs
    void build(const std::vector<std::string>& vocab,
               const std::vector<std::pair<std::string, std::string>>& merges,
               const NormalizerConfig& ncfg);

    // ---- io
    bool load(const std::string& path);
    void save(const std::string& path) const;

    // ---- encoding
    std::vector<i32> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;
    // encodes text that may contain literal special-token markers such as "<|user|>"
    std::vector<i32> encode_with_specials(const std::string& text) const;
    std::string      decode(const std::vector<i32>& ids, bool skip_special = false) const;
    std::string      decode_one(i32 id) const;

    // Streaming-safe decoder: buffers incomplete UTF-8 sequences.
    class Stream {
    public:
        explicit Stream(const Tokenizer& tk) : tk_(tk) {}
        std::string push(i32 id);
        std::string flush();
    private:
        const Tokenizer& tk_;
        std::string buf_;
    };

    // ---- introspection
    int  vocab_size() const { return static_cast<int>(vocab_.size()); }
    const std::string& token_text(i32 id) const;
    i32  token_to_id(const std::string& tok) const;
    // Full vocabulary (id -> raw bytes), for GGUF embedding (self-contained file).
    const std::vector<std::string>& vocab_entries() const { return vocab_; }
    // BPE merges as (left, right) string pairs ordered by rank, for
    // tokenizer.ggml.merges. Gaps (filtered during build) are skipped.
    std::vector<std::pair<std::string, std::string>> merge_pairs_ordered() const;
    bool is_special(i32 id) const { return id >= 0 && id < special::COUNT; }
    const NormalizerConfig& normalizer_config() const { return norm_.config(); }
    const Normalizer& normalizer() const { return norm_; }

    // fertility measurement (tokens per whitespace-word)
    struct Fertility {
        double tokens_per_word = 0;
        double bytes_per_token = 0;
        u64    tokens = 0, words = 0, bytes = 0;
    };
    Fertility measure(const std::vector<std::string>& lines) const;

private:
    void   finalize_index();
    void   bpe_chunk(const std::string& piece, std::vector<i32>& out) const;

    std::vector<std::string>                     vocab_;
    std::unordered_map<std::string, i32>         token_ids_;
    // merge rank keyed by (left_id << 32 | right_id) -> (rank, merged_id)
    std::unordered_map<u64, std::pair<i32, i32>> merges_;
    Normalizer                                   norm_;
    mutable std::unordered_map<std::string, std::vector<i32>> cache_;
    mutable size_t cache_limit_ = 200000;
};

} // namespace gai
