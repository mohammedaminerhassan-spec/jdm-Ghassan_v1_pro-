#pragma once

#include "core/common.h"
#include <memory>
#include <string>
#include <vector>

namespace gai {

enum class LangTag : u8 {
    DarijaArab = 0,   // Moroccan Darija in Arabic script
    DarijaLatn = 1,   // Moroccan Darija in Latin script / Arabizi
    MSA        = 2,   // Modern Standard Arabic
    French     = 3,
    English    = 4,
    Mixed      = 5,   // significant code-switching
    Other      = 6,
};

const char* lang_name(LangTag t);

struct LangScore {
    LangTag tag = LangTag::Other;
    double  confidence = 0.0;
    double  darija_score = 0.0;      // darija-marker density
    double  msa_score = 0.0;
    double  french_score = 0.0;
    double  arabizi_score = 0.0;     // digit-as-letter density
    double  arabic_ratio = 0.0;
    double  latin_ratio = 0.0;
    bool    is_darija() const { return tag == LangTag::DarijaArab || tag == LangTag::DarijaLatn; }
};

// Rule + lexicon based identifier tuned for our exact problem: separating Moroccan
// Darija from MSA (which share a script and much vocabulary) and recognising
// Arabizi. A statistical LID would need labelled Darija data we do not have.
class LangId {
public:
    LangId();
    LangScore classify(const std::string& text) const;

    // Convenience helpers used by the filters
    bool is_darija(const std::string& text, double min_conf = 0.35) const;
    double msa_ratio(const std::string& text) const;

private:
    struct Lexicons;
    std::shared_ptr<Lexicons> lex_;
};

// ---------------------------------------------------------------- style checks
// Detects the "robotic assistant" register we explicitly want to keep out of
// the training data.
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

// PRO-EN: English-mode style gate. The full robotic_phrases() list contains
// normal English politeness ("certainly!", "of course!", "i hope this helps")
// that would nuke most of a native English corpus, so --style-mode en only
// rejects hard AI-disclosure boilerplate (model breaking character).
bool has_hard_ai_boilerplate(const std::string& reply);

} // namespace gai
