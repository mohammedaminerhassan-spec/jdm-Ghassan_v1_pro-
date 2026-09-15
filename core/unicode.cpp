#include "core/unicode.h"

namespace gai {

// ---------------------------------------------------------------- UTF-8
int utf8_seq_len(u8 b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 0;
}

u32 utf8_decode(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0;
    u8 b0 = static_cast<u8>(s[i]);
    int len = utf8_seq_len(b0);
    if (len == 0 || i + static_cast<size_t>(len) > s.size()) { ++i; return 0xFFFD; }

    u32 cp = 0;
    switch (len) {
        case 1: cp = b0; break;
        case 2: cp = b0 & 0x1Fu; break;
        case 3: cp = b0 & 0x0Fu; break;
        case 4: cp = b0 & 0x07u; break;
        default: break;
    }
    for (int k = 1; k < len; ++k) {
        u8 bk = static_cast<u8>(s[i + static_cast<size_t>(k)]);
        if ((bk & 0xC0) != 0x80) { ++i; return 0xFFFD; }
        cp = (cp << 6) | (bk & 0x3Fu);
    }
    // reject overlong / surrogates / out of range
    if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) ||
        (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
        ++i;
        return 0xFFFD;
    }
    i += static_cast<size_t>(len);
    return cp;
}

void utf8_encode(u32 cp, std::string& out) {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = 0xFFFD;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string utf8_from_cp(u32 cp) {
    std::string s;
    utf8_encode(cp, s);
    return s;
}

std::vector<u32> utf8_to_codepoints(const std::string& s) {
    std::vector<u32> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) out.push_back(utf8_decode(s, i));
    return out;
}

std::string codepoints_to_utf8(const std::vector<u32>& cps) {
    std::string out;
    out.reserve(cps.size() * 2);
    for (u32 c : cps) utf8_encode(c, out);
    return out;
}

size_t utf8_length(const std::string& s) {
    size_t i = 0, n = 0;
    while (i < s.size()) { utf8_decode(s, i); ++n; }
    return n;
}

bool utf8_valid(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t before = i;
        u32 cp = utf8_decode(s, i);
        if (cp == 0xFFFD && !(s.size() - before >= 3 &&
                              static_cast<u8>(s[before]) == 0xEF &&
                              static_cast<u8>(s[before + 1]) == 0xBF &&
                              static_cast<u8>(s[before + 2]) == 0xBD)) {
            return false;
        }
    }
    return true;
}

bool utf8_is_complete(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        u8 b = static_cast<u8>(s[i]);
        int len = utf8_seq_len(b);
        if (len == 0) return false;
        if (i + static_cast<size_t>(len) > s.size()) return false;
        i += static_cast<size_t>(len);
    }
    return true;
}

// ---------------------------------------------------------------- classification
bool is_arabic_letter(u32 cp) {
    if (cp >= 0x0621 && cp <= 0x063A) return true;          // hamza .. ghain
    if (cp >= 0x0641 && cp <= 0x064A) return true;          // feh .. yeh
    if (cp >= 0x066E && cp <= 0x06D3) return true;          // extended letters (incl. peh, gaf, veh)
    if (cp == 0x0671 || cp == 0x0672 || cp == 0x0673) return true;
    if (cp >= 0x06FA && cp <= 0x06FF) return true;
    if (cp >= 0x0750 && cp <= 0x077F) return true;          // Arabic Supplement
    if (cp >= 0x08A0 && cp <= 0x08BF) return true;          // Arabic Extended-A letters
    return false;
}

bool is_arabic_diacritic(u32 cp) {
    if (cp >= 0x064B && cp <= 0x065F) return true;          // harakat + extras
    if (cp == 0x0670) return true;                          // superscript alef
    if (cp >= 0x06D6 && cp <= 0x06ED) return true;          // quranic marks
    if (cp >= 0x08D3 && cp <= 0x08FF) return true;          // extended marks
    return false;
}

bool is_arabic_punct(u32 cp) {
    switch (cp) {
        case 0x060C: case 0x061B: case 0x061F: case 0x066A: case 0x066B:
        case 0x066C: case 0x066D: case 0x06D4: case 0x061E: case 0x0600:
            return true;
        default: return false;
    }
}

bool is_arabic_digit(u32 cp) {
    return (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9);
}

bool is_tatweel(u32 cp) { return cp == 0x0640; }

bool is_latin_letter(u32 cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp >= 0x00C0 && cp <= 0x024F) {                     // Latin-1 supp + extended A/B
        if (cp == 0x00D7 || cp == 0x00F7) return false;
        return true;
    }
    return false;
}

bool is_ascii_digit(u32 cp) { return cp >= '0' && cp <= '9'; }

bool is_whitespace(u32 cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_control(u32 cp) {
    return (cp < 0x20 && cp != '\n' && cp != '\t' && cp != '\r') || (cp >= 0x7F && cp <= 0x9F);
}

bool is_zero_width(u32 cp) {
    return cp == 0x200B || cp == 0x200C || cp == 0x200D || cp == 0x200E || cp == 0x200F ||
           (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2066 && cp <= 0x2069) || cp == 0xFEFF;
}

bool is_punct(u32 cp) {
    if (cp < 0x80) {
        return (cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') ||
               (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~');
    }
    if (is_arabic_punct(cp)) return true;
    if (cp >= 0x2010 && cp <= 0x205E) return true;
    if (cp >= 0x00A1 && cp <= 0x00BF) return true;
    return false;
}

bool is_emoji(u32 cp) {
    return (cp >= 0x1F300 && cp <= 0x1FAFF) ||
           (cp >= 0x1F000 && cp <= 0x1F2FF) ||
           (cp >= 0x2600  && cp <= 0x27BF)  ||
           (cp >= 0x2190  && cp <= 0x21FF)  ||
           (cp >= 0xFE00  && cp <= 0xFE0F)  ||
           cp == 0x2764 || cp == 0x203C || cp == 0x2049;
}

bool is_arabizi_digit(u32 cp) {
    return cp == '2' || cp == '3' || cp == '5' || cp == '6' || cp == '7' || cp == '8' || cp == '9';
}

// ---------------------------------------------------------------- transforms
u32 arabic_presentation_to_base(u32 cp) {
    // Arabic Presentation Forms-B (FE70..FEFF): isolated/initial/medial/final glyphs.
    static const struct { u32 lo, hi, base; } fb[] = {
        {0xFE80, 0xFE80, 0x0621}, {0xFE81, 0xFE82, 0x0622}, {0xFE83, 0xFE84, 0x0623},
        {0xFE85, 0xFE86, 0x0624}, {0xFE87, 0xFE88, 0x0625}, {0xFE89, 0xFE8C, 0x0626},
        {0xFE8D, 0xFE8E, 0x0627}, {0xFE8F, 0xFE92, 0x0628}, {0xFE93, 0xFE94, 0x0629},
        {0xFE95, 0xFE98, 0x062A}, {0xFE99, 0xFE9C, 0x062B}, {0xFE9D, 0xFEA0, 0x062C},
        {0xFEA1, 0xFEA4, 0x062D}, {0xFEA5, 0xFEA8, 0x062E}, {0xFEA9, 0xFEAA, 0x062F},
        {0xFEAB, 0xFEAC, 0x0630}, {0xFEAD, 0xFEAE, 0x0631}, {0xFEAF, 0xFEB0, 0x0632},
        {0xFEB1, 0xFEB4, 0x0633}, {0xFEB5, 0xFEB8, 0x0634}, {0xFEB9, 0xFEBC, 0x0635},
        {0xFEBD, 0xFEC0, 0x0636}, {0xFEC1, 0xFEC4, 0x0637}, {0xFEC5, 0xFEC8, 0x0638},
        {0xFEC9, 0xFECC, 0x0639}, {0xFECD, 0xFED0, 0x063A}, {0xFED1, 0xFED4, 0x0641},
        {0xFED5, 0xFED8, 0x0642}, {0xFED9, 0xFEDC, 0x0643}, {0xFEDD, 0xFEE0, 0x0644},
        {0xFEE1, 0xFEE4, 0x0645}, {0xFEE5, 0xFEE8, 0x0646}, {0xFEE9, 0xFEEC, 0x0647},
        {0xFEED, 0xFEEE, 0x0648}, {0xFEEF, 0xFEF0, 0x0649}, {0xFEF1, 0xFEF4, 0x064A},
    };
    if (cp >= 0xFE80 && cp <= 0xFEF4) {
        for (auto& e : fb) if (cp >= e.lo && cp <= e.hi) return e.base;
    }
    if (cp >= 0xFEF5 && cp <= 0xFEFC) return 0x0644;   // lam-alef ligatures -> lam (alef added by caller)
    return 0;
}

u32 arabic_digit_to_ascii(u32 cp) {
    if (cp >= 0x0660 && cp <= 0x0669) return '0' + (cp - 0x0660);
    if (cp >= 0x06F0 && cp <= 0x06F9) return '0' + (cp - 0x06F0);
    return cp;
}

u32 fold_arabic_letter(u32 cp) {
    switch (cp) {
        case 0x0622: case 0x0623: case 0x0625: case 0x0671: return 0x0627;  // آأإٱ -> ا
        case 0x0649: return 0x064A;                                          // ى -> ي
        case 0x0629: return 0x0647;                                          // ة -> ه
        case 0x0624: case 0x0626: return 0x0621;                             // ؤئ -> ء
        default: return cp;
    }
}

u32 lower_ascii(u32 cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32;
    if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7) return cp + 32;
    return cp;
}

u32 fold_fullwidth(u32 cp) {
    if (cp >= 0xFF01 && cp <= 0xFF5E) return cp - 0xFEE0;
    if (cp == 0x3000) return ' ';
    return cp;
}

std::string to_lower_ascii(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        u32 cp = utf8_decode(s, i);
        utf8_encode(lower_ascii(cp), out);
    }
    return out;
}

// ---------------------------------------------------------------- stats
ScriptStats script_stats(const std::string& s) {
    ScriptStats st;
    size_t i = 0;
    while (i < s.size()) {
        u32 cp = utf8_decode(s, i);
        ++st.total;
        if (is_whitespace(cp))                                   ++st.whitespace;
        else if (is_arabic_letter(cp) || is_arabic_diacritic(cp)) ++st.arabic;
        else if (is_latin_letter(cp))                             ++st.latin;
        else if (is_ascii_digit(cp) || is_arabic_digit(cp))       ++st.digits;
        else if (is_emoji(cp))                                    ++st.emoji;
        else if (is_punct(cp))                                    ++st.punct;
        else                                                      ++st.other;
    }
    return st;
}

} // namespace gai
