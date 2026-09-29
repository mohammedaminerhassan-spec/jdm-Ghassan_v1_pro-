#pragma once

#include "core/common.h"
#include <vector>
#include <string>

namespace gai {

u32  utf8_decode(const std::string& s, size_t& i);
void utf8_encode(u32 cp, std::string& out);
std::string utf8_from_cp(u32 cp);
std::vector<u32> utf8_to_codepoints(const std::string& s);
std::string codepoints_to_utf8(const std::vector<u32>& cps);
size_t utf8_length(const std::string& s);
bool   utf8_valid(const std::string& s);

int    utf8_seq_len(u8 b);

bool   utf8_is_complete(const std::string& s);

bool is_arabic_letter(u32 cp);
bool is_arabic_diacritic(u32 cp);
bool is_arabic_punct(u32 cp);
bool is_arabic_digit(u32 cp);
bool is_tatweel(u32 cp);
bool is_latin_letter(u32 cp);
bool is_ascii_digit(u32 cp);
bool is_whitespace(u32 cp);
bool is_punct(u32 cp);
bool is_emoji(u32 cp);
bool is_zero_width(u32 cp);
bool is_arabizi_digit(u32 cp);
bool is_control(u32 cp);

u32  arabic_presentation_to_base(u32 cp);
u32  arabic_digit_to_ascii(u32 cp);
u32  fold_arabic_letter(u32 cp);
u32  lower_ascii(u32 cp);
u32  fold_fullwidth(u32 cp);

std::string to_lower_ascii(const std::string& s);

struct ScriptStats {
    size_t arabic     = 0;
    size_t latin      = 0;
    size_t digits     = 0;
    size_t punct      = 0;
    size_t emoji      = 0;
    size_t whitespace = 0;
    size_t other      = 0;
    size_t total      = 0;

    double arabic_ratio() const { return total ? double(arabic) / double(total) : 0.0; }
    double latin_ratio()  const { return total ? double(latin)  / double(total) : 0.0; }
};

ScriptStats script_stats(const std::string& s);

}
