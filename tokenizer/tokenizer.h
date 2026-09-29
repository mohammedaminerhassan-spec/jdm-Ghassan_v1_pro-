#pragma once

#include "core/common.h"
#include "tokenizer/normalizer.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace gai {

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

class Tokenizer {
public:
    Tokenizer() = default;

    void init_empty(const NormalizerConfig& ncfg);

    void build(const std::vector<std::string>& vocab,
               const std::vector<std::pair<std::string, std::string>>& merges,
               const NormalizerConfig& ncfg);

    bool load(const std::string& path);
    void save(const std::string& path) const;

    std::vector<i32> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;

    std::vector<i32> encode_with_specials(const std::string& text) const;
    std::string      decode(const std::vector<i32>& ids, bool skip_special = false) const;
    std::string      decode_one(i32 id) const;

    class Stream {
    public:
        explicit Stream(const Tokenizer& tk) : tk_(tk) {}
        std::string push(i32 id);
        std::string flush();
    private:
        const Tokenizer& tk_;
        std::string buf_;
    };

    int  vocab_size() const { return static_cast<int>(vocab_.size()); }
    const std::string& token_text(i32 id) const;
    i32  token_to_id(const std::string& tok) const;

    const std::vector<std::string>& vocab_entries() const { return vocab_; }

    std::vector<std::pair<std::string, std::string>> merge_pairs_ordered() const;
    bool is_special(i32 id) const { return id >= 0 && id < special::COUNT; }
    const NormalizerConfig& normalizer_config() const { return norm_.config(); }
    const Normalizer& normalizer() const { return norm_; }

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

    std::unordered_map<u64, std::pair<i32, i32>> merges_;
    Normalizer                                   norm_;
    mutable std::unordered_map<std::string, std::vector<i32>> cache_;
    mutable size_t cache_limit_ = 200000;
};

}
