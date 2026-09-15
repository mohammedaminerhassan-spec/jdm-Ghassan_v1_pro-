#include "tokenizer/normalizer.h"

namespace gai {

// ---------------------------------------------------------------- config bits
u32 NormalizerConfig::pack() const {
    u32 b = 0;
    b |= strip_diacritics    ? 1u << 0  : 0u;
    b |= strip_tatweel       ? 1u << 1  : 0u;
    b |= unify_presentation  ? 1u << 2  : 0u;
    b |= arabic_digits_ascii ? 1u << 3  : 0u;
    b |= fold_letters        ? 1u << 4  : 0u;
    b |= lowercase_latin     ? 1u << 5  : 0u;
    b |= collapse_whitespace ? 1u << 6  : 0u;
    b |= strip_control       ? 1u << 7  : 0u;
    b |= strip_zero_width    ? 1u << 8  : 0u;
    b |= collapse_repeats    ? 1u << 9  : 0u;
    b |= fold_fullwidth      ? 1u << 10 : 0u;
    b |= (static_cast<u32>(max_repeat) & 0xFu) << 16;
    return b;
}

NormalizerConfig NormalizerConfig::unpack(u32 bits) {
    NormalizerConfig c;
    c.strip_diacritics    = bits & (1u << 0);
    c.strip_tatweel       = bits & (1u << 1);
    c.unify_presentation  = bits & (1u << 2);
    c.arabic_digits_ascii = bits & (1u << 3);
    c.fold_letters        = bits & (1u << 4);
    c.lowercase_latin     = bits & (1u << 5);
    c.collapse_whitespace = bits & (1u << 6);
    c.strip_control       = bits & (1u << 7);
    c.strip_zero_width    = bits & (1u << 8);
    c.collapse_repeats    = bits & (1u << 9);
    c.fold_fullwidth      = bits & (1u << 10);
    c.max_repeat          = static_cast<int>((bits >> 16) & 0xFu);
    if (c.max_repeat <= 0) c.max_repeat = 3;
    return c;
}

// ---------------------------------------------------------------- normalize
std::string Normalizer::normalize(const std::string& text) const {
    std::vector<u32> in = utf8_to_codepoints(text);
    std::vector<u32> out;
    out.reserve(in.size());

    for (size_t i = 0; i < in.size(); ++i) {
        u32 cp = in[i];

        if (cfg_.strip_control && is_control(cp)) continue;

        if (cfg_.strip_zero_width && is_zero_width(cp)) {
            // keep ZWNJ (200C) only when it sits between two Arabic letters
            bool keep = false;
            if (cp == 0x200C && i > 0 && i + 1 < in.size()) {
                keep = is_arabic_letter(in[i - 1]) && is_arabic_letter(in[i + 1]);
            }
            if (!keep) continue;
        }

        if (cfg_.unify_presentation && cp >= 0xFE70 && cp <= 0xFEFF) {
            if (cp >= 0xFEF5 && cp <= 0xFEFC) {          // lam-alef ligature -> lam + alef
                out.push_back(0x0644);
                out.push_back(0x0627);
                continue;
            }
            u32 base = arabic_presentation_to_base(cp);
            if (base) cp = base;
        }

        if (cfg_.strip_tatweel && is_tatweel(cp)) continue;
        if (cfg_.strip_diacritics && is_arabic_diacritic(cp)) continue;
        if (cfg_.arabic_digits_ascii) cp = arabic_digit_to_ascii(cp);
        if (cfg_.fold_fullwidth) cp = fold_fullwidth(cp);
        if (cfg_.fold_letters) cp = fold_arabic_letter(cp);
        if (cfg_.lowercase_latin) cp = lower_ascii(cp);

        if (is_whitespace(cp)) {
            u32 ws = (cp == '\n') ? '\n' : ' ';
            if (cfg_.collapse_whitespace) {
                if (!out.empty() && (out.back() == ' ' || out.back() == '\n')) {
                    // a newline wins over a plain space
                    if (ws == '\n' && out.back() == ' ') out.back() = '\n';
                    continue;
                }
            }
            out.push_back(ws);
            continue;
        }

        if (cfg_.collapse_repeats && out.size() >= static_cast<size_t>(cfg_.max_repeat)) {
            bool run = true;
            for (int r = 1; r <= cfg_.max_repeat; ++r) {
                if (out[out.size() - static_cast<size_t>(r)] != cp) { run = false; break; }
            }
            if (run) continue;
        }

        out.push_back(cp);
    }

    // trim
    size_t a = 0, b = out.size();
    while (a < b && (out[a] == ' ' || out[a] == '\n')) ++a;
    while (b > a && (out[b - 1] == ' ' || out[b - 1] == '\n')) --b;

    std::string res;
    res.reserve((b - a) * 2);
    for (size_t i = a; i < b; ++i) utf8_encode(out[i], res);
    return res;
}

std::string Normalizer::canonical(const std::string& text) {
    NormalizerConfig c;
    c.fold_letters = true;
    c.lowercase_latin = true;
    c.collapse_repeats = true;
    c.max_repeat = 1;
    return Normalizer(c).normalize(text);
}

// ---------------------------------------------------------------- pre-tokenize
static ChunkKind classify(u32 cp) {
    if (is_whitespace(cp))                                    return ChunkKind::Space;
    if (is_arabic_letter(cp) || is_arabic_diacritic(cp))      return ChunkKind::Arabic;
    if (is_latin_letter(cp))                                  return ChunkKind::Latin;
    if (is_ascii_digit(cp) || is_arabic_digit(cp))            return ChunkKind::Number;
    if (is_emoji(cp))                                         return ChunkKind::Emoji;
    if (is_punct(cp))                                         return ChunkKind::Punct;
    return ChunkKind::Other;
}

std::vector<Chunk> pre_tokenize(const std::string& text) {
    std::vector<u32> cps = utf8_to_codepoints(text);
    std::vector<Chunk> out;
    size_t i = 0;
    const size_t n = cps.size();

    while (i < n) {
        // Leading whitespace: a single space is glued to the next word; newlines and
        // longer runs become their own chunk so the model can learn layout.
        std::string prefix;
        if (is_whitespace(cps[i])) {
            size_t start = i;
            bool has_nl = false;
            while (i < n && is_whitespace(cps[i])) { has_nl |= (cps[i] == '\n'); ++i; }
            size_t count = i - start;
            if (has_nl || count > 1 || i >= n) {
                Chunk c;
                for (size_t k = start; k < i; ++k) utf8_encode(cps[k], c.text);
                c.kind = ChunkKind::Space;
                out.push_back(std::move(c));
                continue;
            }
            prefix = " ";
        }
        if (i >= n) {
            if (!prefix.empty()) out.push_back(Chunk{prefix, ChunkKind::Space});
            break;
        }

        ChunkKind kind = classify(cps[i]);

        // Arabizi: a digit that starts a word ("3lach", "7it", "9rib") belongs to the
        // Latin word that follows, not to a numeric chunk.
        if (kind == ChunkKind::Number && is_arabizi_digit(cps[i]) &&
            i + 1 < n && is_latin_letter(cps[i + 1])) {
            kind = ChunkKind::Latin;
        }

        Chunk c;
        c.text = prefix;
        c.kind = kind;

        switch (kind) {
            case ChunkKind::Arabic: {
                while (i < n && (is_arabic_letter(cps[i]) || is_arabic_diacritic(cps[i]) ||
                                 cps[i] == 0x200C)) {
                    utf8_encode(cps[i], c.text);
                    ++i;
                }
                break;
            }
            case ChunkKind::Latin: {
                // Latin word with embedded arabizi digits ("3lach", "kif7alk", "b9it")
                bool started = false;
                while (i < n) {
                    u32 cp = cps[i];
                    bool ok = is_latin_letter(cp);
                    if (!ok && (cp == '\'' || cp == 0x2019)) {
                        // apostrophe only inside a word
                        ok = started && (i + 1 < n) && is_latin_letter(cps[i + 1]);
                    }
                    if (!ok && is_arabizi_digit(cp)) {
                        // a digit belongs to the word if it continues one, or introduces one
                        bool next_word = (i + 1 < n) && (is_latin_letter(cps[i + 1]) ||
                                                         is_arabizi_digit(cps[i + 1]));
                        ok = started || next_word;
                    }
                    if (!ok) break;
                    utf8_encode(cp, c.text);
                    ++i;
                    started = true;
                }
                if (!started) {   // safety: never emit an empty body
                    utf8_encode(cps[i], c.text);
                    ++i;
                }
                break;
            }
            case ChunkKind::Number: {
                while (i < n && (is_ascii_digit(cps[i]) || is_arabic_digit(cps[i]) ||
                                 ((cps[i] == '.' || cps[i] == ',') && i + 1 < n && is_ascii_digit(cps[i + 1])))) {
                    utf8_encode(cps[i], c.text);
                    ++i;
                }
                break;
            }
            case ChunkKind::Punct: {
                u32 first = cps[i];
                while (i < n && cps[i] == first) { utf8_encode(cps[i], c.text); ++i; }
                break;
            }
            case ChunkKind::Emoji: {
                while (i < n && (is_emoji(cps[i]) || cps[i] == 0x200D)) {
                    utf8_encode(cps[i], c.text);
                    ++i;
                }
                break;
            }
            default: {
                utf8_encode(cps[i], c.text);
                ++i;
                break;
            }
        }
        out.push_back(std::move(c));
    }
    return out;
}

} // namespace gai
