#pragma once

#include "core/common.h"
#include "core/unicode.h"
#include <vector>

namespace gai {

struct NormalizerConfig {
    bool strip_diacritics    = true;
    bool strip_tatweel       = true;
    bool unify_presentation  = true;
    bool arabic_digits_ascii = true;
    bool fold_letters        = false;
    bool lowercase_latin     = true;
    bool collapse_whitespace = true;
    bool strip_control       = true;
    bool strip_zero_width    = true;

    bool collapse_repeats    = false;
    bool fold_fullwidth      = true;
    int  max_repeat          = 3;

    u32 pack() const;
    static NormalizerConfig unpack(u32 bits);
};

class Normalizer {
public:
    Normalizer() = default;
    explicit Normalizer(const NormalizerConfig& cfg) : cfg_(cfg) {}

    std::string normalize(const std::string& text) const;
    const NormalizerConfig& config() const { return cfg_; }
    void set_config(const NormalizerConfig& c) { cfg_ = c; }

    static std::string canonical(const std::string& text);

private:
    NormalizerConfig cfg_;
};

enum class ChunkKind : u8 {
    Arabic = 0,
    Latin  = 1,
    Number = 2,
    Punct  = 3,
    Space  = 4,
    Emoji  = 5,
    Other  = 6,
};

struct Chunk {
    std::string text;
    ChunkKind   kind = ChunkKind::Other;
};

std::vector<Chunk> pre_tokenize(const std::string& text);

}
