#pragma once

#include "core/common.h"
#include <memory>
#include <string>
#include <vector>

namespace gai {

enum class LangTag : u8 {
    DarijaArab = 0,
    DarijaLatn = 1,
    MSA        = 2,
    French     = 3,
    English    = 4,
    Mixed      = 5,
    Other      = 6,
};

const char* lang_name(LangTag t);

struct LangScore {
    LangTag tag = LangTag::Other;
    double  confidence = 0.0;
    double  darija_score = 0.0;
    double  msa_score = 0.0;
    double  french_score = 0.0;
    double  english_score = 0.0;
    double  arabizi_score = 0.0;
    double  arabic_ratio = 0.0;
    double  latin_ratio = 0.0;
    bool    is_darija() const { return tag == LangTag::DarijaArab || tag == LangTag::DarijaLatn; }
    bool    is_english() const { return tag == LangTag::English; }
};

class LangId {
public:
    LangId();
    LangScore classify(const std::string& text) const;

    bool is_darija(const std::string& text, double min_conf = 0.35) const;
    double msa_ratio(const std::string& text) const;

private:
    struct Lexicons;
    std::shared_ptr<Lexicons> lex_;
};

struct StyleFlags {
    bool has_boilerplate   = false;
    bool excessive_msa     = false;
    bool too_many_bullets  = false;
    bool too_long          = false;
    int  bullet_count      = 0;
    double msa_ratio       = 0.0;
    std::vector<std::string> matched_phrases;
    bool robotic() const { return has_boilerplate || excessive_msa || too_many_bullets; }
};

StyleFlags check_assistant_style(const std::string& reply, const std::string& user_msg,
                                 const LangId& lid);

const std::vector<std::string>& robotic_phrases();

bool has_hard_ai_boilerplate(const std::string& reply);

}
