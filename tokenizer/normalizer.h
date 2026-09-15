#pragma once

#include "core/common.h"
#include "core/unicode.h"
#include <vector>

namespace gai {

struct NormalizerConfig {
    bool strip_diacritics    = true;   // harakat carry no information in Darija
    bool strip_tatweel       = true;
    bool unify_presentation  = true;   // FB50..FEFF -> base letters
    bool arabic_digits_ascii = true;   // ٠..٩ -> 0..9
    bool fold_letters        = false;  // أإآ->ا etc.  OFF for the LM, ON for langid/dedup
    bool lowercase_latin     = true;
    bool collapse_whitespace = true;
    bool strip_control       = true;
    bool strip_zero_width    = true;   // keeps ZWNJ inside Arabic words
    bool collapse_repeats    = true;   // "سلاااااام" -> "سلااام" (max 3)
    bool fold_fullwidth      = true;
    int  max_repeat          = 3;

    u32 pack() const;
    static NormalizerConfig unpack(u32 bits);
};

// Text normalization used by both the tokenizer and the dataset pipeline.
class Normalizer {
public:
    Normalizer() = default;
    explicit Normalizer(const NormalizerConfig& cfg) : cfg_(cfg) {}

    std::string normalize(const std::string& text) const;
    const NormalizerConfig& config() const { return cfg_; }
    void set_config(const NormalizerConfig& c) { cfg_ = c; }

    // aggressive variant used for hashing / language id / dedup
    static std::string canonical(const std::string& text);

private:
    NormalizerConfig cfg_;
};

// ---------------------------------------------------------------- pre-tokenizer
enum class ChunkKind : u8 {
    Arabic = 0,     // Arabic-script word
    Latin  = 1,     // Latin word, may embed arabizi digits (3, 7, 9 ...)
    Number = 2,
    Punct  = 3,
    Space  = 4,
    Emoji  = 5,
    Other  = 6,
};

struct Chunk {
    std::string text;      // includes the leading space, if any
    ChunkKind   kind = ChunkKind::Other;
};

// Splits text into BPE-mergeable units. A leading space is glued to the following
// word so detokenisation is exactly reversible.
std::vector<Chunk> pre_tokenize(const std::string& text);

} // namespace gai
