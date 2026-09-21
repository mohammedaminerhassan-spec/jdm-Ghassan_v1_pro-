#include "dataset/langid.h"
#include "core/unicode.h"
#include "tokenizer/normalizer.h"

#include <unordered_set>
#include <algorithm>
#include <cmath>

namespace gai {

const char* lang_name(LangTag t) {
    switch (t) {
        case LangTag::DarijaArab: return "ar-MA-arab";
        case LangTag::DarijaLatn: return "ar-MA-latn";
        case LangTag::MSA:        return "ar-MSA";
        case LangTag::French:     return "fr";
        case LangTag::English:    return "en";
        case LangTag::Mixed:      return "mixed";
        case LangTag::Other:      return "other";
    }
    return "other";
}

struct LangId::Lexicons {
    // High-precision Darija markers in Arabic script. These words essentially never
    // appear in MSA, so their presence is strong evidence.
    std::unordered_set<std::string> darija_ar = {
        "شنو","شنی","اشنو","فين","فوقاش","علاش","كيفاش","شحال","بزاف","دابا","واخا",
        "غادي","كاين","كاينة","ماكاينش","مزيان","مزيانة","زوين","زوينة","دير","ديري",
        "ديال","ديالي","ديالك","ديالو","ديالنا","نتا","نتي","نتوما","حنا","هوما",
        "بغيت","بغا","بغينا","خاصني","خاصك","خصنا","عندي","عندك","صافي","يالله",
        "بسلامة","تهلا","حيت","حيث","ولا","ماشي","ماشي","والو","شوية","بشوية",
        "دغيا","درتها","درت","مشيت","جيت","كنت","غنمشي","غادية","خدام","خدمة",
        "لبارح","غدا","اليوم","دروك","هاد","هادي","هادو","داكشي","هادشي","ديك",
        "منين","باش","بلاتي","سير","سيري","اجي","اجيو","عافاك","اسمح","سمحلي",
        "الله","بلا","معايا","معاك","معانا","فيا","فيك","ليا","ليك","لينا",
        "كيدير","كتدير","كنديرو","كيهضر","نهضر","هضرة","دوز","نوض","گلس","جلس",
        "شوف","شوفي","شفت","عرفت","كنعرف","معرفتش","مابغيتش","ماعرفتش","مافيهاش",
        "طاجين","كسكس","حرشة","أطاي","اتاي","درهم","سوق","دار","الدار","حومة",
        "خويا","خوتي","صاحبي","سحابي","ولد","بنت","والدي","ميمتي","بابا"
    };
    // Latin/Arabizi Darija markers
    std::unordered_set<std::string> darija_latn = {
        "chno","shno","chnou","fin","fine","fo9ach","foqach","3lach","3lash","kifach",
        "kifash","kifak","chhal","shhal","bzaf","bezzaf","daba","wakha","waxa","ghadi",
        "ghadya","kayn","kayna","makaynch","mzyan","mzyane","zwin","zwina","dir","diri",
        "dyal","dyali","dyalk","dyalo","nta","nti","ntoma","7na","homa","bghit","bgha",
        "khasni","khasek","3andi","3andek","safi","yallah","yalah","bslama","thala",
        "7it","hit","wla","machi","mashi","walou","walo","chwiya","shwiya","daghya",
        "mchit","jit","kont","khdam","lbare7","lbarah","ghedda","lyoum","had","hadi",
        "hadchi","dakchi","mnin","bach","bash","sir","aji","3afak","smhli","smehli",
        "m3aya","m3ak","lia","lik","kidir","katdir","kayhder","nhder","dwi","chouf",
        "chouff","shof","chft","3reft","kan3ref","ma3reftch","mabghitch","khoya",
        "sahbi","s7abi","walid","bnt","tajine","kesksou","atay","derham","dar","houma",
        "labas","lbas","hamdollah","hamdulah","nichan","nishan","bzzaf","bghina",
        "khedma","khdma","tanjia","msakn","3ndna","3ndkom","fhamt","fhemt","mafhemtch"
    };
    // MSA-only markers (formal register). Their density separates MSA from Darija.
    std::unordered_set<std::string> msa = {
        "الذي","التي","الذين","اللذان","هذا","هذه","هؤلاء","ذلك","تلك","أولئك",
        "إن","أن","لكن","لأن","حيث","بينما","عندما","بالتالي","لذلك","إذن",
        "يجب","ينبغي","يمكن","سوف","قد","لقد","كان","كانت","يكون","تكون",
        "أيضا","كذلك","فقط","جدا","كثيرا","دائما","أبدا","ربما","تقريبا",
        "بالتأكيد","بالطبع","طبعا","شكرا","عفوا","مرحبا","أهلا","سلام",
        "المستخدم","المعلومات","الموضوع","الخدمة","المساعدة","النظام","البيانات",
        "الأمر","الشيء","الوقت","اليوم","الحال","السؤال","الجواب","الإجابة",
        "بشكل","بصورة","بطريقة","نحو","خلال","حول","ضمن","عبر","وفق","حسب",
        "ليس","ليست","لست","لسنا","سيتم","يتم","تتم","إلى","على","في","من","عن"
    };
    std::unordered_set<std::string> french = {
        "le","la","les","un","une","des","du","de","et","ou","mais","donc","car",
        "je","tu","il","elle","nous","vous","ils","elles","ce","cette","ces",
        "pour","avec","sans","dans","sur","sous","chez","vers","chaque","tout",
        "est","sont","être","avoir","faire","aller","pouvoir","vouloir","savoir",
        "bonjour","merci","salut","oui","non","peut","très","bien","plus","aussi",
        "problème","travail","école","maison","voiture","argent","temps","chose"
    };
    // PARQUET-ONLY EN: expanded from 30 to ~220 markers so Hermes English
    // (522k chat + 478k instruction) classifies as en, not Other.
    // Covers questions, instructions, connectors, reasoning, coding.
    std::unordered_set<std::string> english = {
        "the","and","is","are","was","were","this","that","with","from","have",
        "has","you","your","they","their","what","when","where","which","would",
        "could","should","about","there","been","will","can","not","for","but",
        "who","whom","whose","why","how","whether","either","neither","each",
        "every","some","any","many","much","more","most","other","such","only",
        "also","very","just","into","over","after","before","between","through",
        "during","because","while","although","though","therefore","however",
        "moreover","furthermore","otherwise","instead","unless","until","again",
        "once","here","there","then","than","then","first","second","third",
        "finally","next","please","write","implement","create","develop",
        "calculate","explain","solve","convert","generate","design","debug",
        "describe","define","list","summarize","translate","example","answer",
        "question","problem","solution","reason","logic","step","result",
        "function","return","code","program","python","javascript","array",
        "string","number","object","class","method","variable","loop",
        "recursion","algorithm","data","model","system","human","assistant",
        "think","know","understand","believe","remember","consider","suggest",
        "mean","include","including","using","used","often","always","never",
        "may","might","must","shall","need","want","like","love","hate",
        "good","bad","great","best","better","worse","big","small","large",
        "long","short","high","low","new","old","young","same","different",
        "own","our","we","us","he","she","him","her","his","its","my","mine",
        "me","i","we","our","ours","your","yours","their","theirs","them",
        "these","those","am","been","being","do","does","did","doing","done",
        "has","having","had","having","will","shall","may","might","must",
        "ought","need","dare","used","very","too","quite","rather","almost",
        "enough","even","still","yet","already","today","yesterday","tomorrow",
        "now","here","there","away","back","off","out","up","down","above",
        "below","under","over","again","further","once","all","both","few",
        "several","various","another","same","different","important","possible",
        "sure","true","false","yes","no","maybe","perhaps","please","thanks",
        "hello","hi","hey",
    };
};

LangId::LangId() : lex_(std::make_shared<Lexicons>()) {}

static std::vector<std::string> word_split(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    size_t i = 0;
    while (i < text.size()) {
        u32 cp = utf8_decode(text, i);
        bool wordy = is_arabic_letter(cp) || is_latin_letter(cp) || is_arabizi_digit(cp);
        if (wordy) utf8_encode(lower_ascii(cp), cur);
        else {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

LangScore LangId::classify(const std::string& raw) const {
    LangScore s;
    if (raw.empty()) return s;

    // fold for matching (أ/إ/آ -> ا etc.) so lexicon lookups are spelling-robust
    NormalizerConfig nc;
    nc.fold_letters = true;
    std::string text = Normalizer(nc).normalize(raw);

    ScriptStats st = script_stats(text);
    size_t letters = st.arabic + st.latin;
    if (letters == 0) return s;

    s.arabic_ratio = static_cast<double>(st.arabic) / static_cast<double>(letters);
    s.latin_ratio  = static_cast<double>(st.latin) / static_cast<double>(letters);

    std::vector<std::string> words = word_split(text);
    if (words.empty()) return s;

    size_t dar_ar = 0, dar_lt = 0, msa = 0, fr = 0, en = 0, arabizi = 0;
    size_t ar_words = 0, lt_words = 0;

    for (const auto& w : words) {
        // classify the word's script from its first codepoint
        size_t p = 0;
        u32 c0 = utf8_decode(w, p);
        bool is_ar = is_arabic_letter(c0);
        if (is_ar) ++ar_words; else ++lt_words;

        if (is_ar) {
            if (lex_->darija_ar.count(w)) ++dar_ar;
            if (lex_->msa.count(w)) ++msa;
            // Darija morphology: verbs prefixed by كا/كي/تا/غا, negation ما...ش
            if (w.size() >= 6) {
                if (w.rfind("كي", 0) == 0 || w.rfind("كا", 0) == 0 ||
                    w.rfind("تا", 0) == 0 || w.rfind("غا", 0) == 0 || w.rfind("ما", 0) == 0) {
                    if (w.size() >= 4 && w.substr(w.size() - 2) == "ش") ++dar_ar;
                    else if (w.rfind("كي", 0) == 0 || w.rfind("كا", 0) == 0) ++dar_ar;
                }
            }
        } else {
            if (lex_->darija_latn.count(w)) ++dar_lt;
            if (lex_->french.count(w)) ++fr;
            if (lex_->english.count(w)) ++en;
            // arabizi: a digit used as a letter inside a Latin word
            bool has_letter = false, has_digit = false;
            size_t q = 0;
            while (q < w.size()) {
                u32 cp = utf8_decode(w, q);
                if (is_latin_letter(cp)) has_letter = true;
                if (is_arabizi_digit(cp)) has_digit = true;
            }
            if (has_letter && has_digit) ++arabizi;
        }
    }

    const double nw = static_cast<double>(words.size());
    s.darija_score  = static_cast<double>(dar_ar + dar_lt) / nw;
    s.msa_score     = static_cast<double>(msa) / nw;
    s.french_score  = static_cast<double>(fr) / nw;
    s.english_score = static_cast<double>(en) / nw;
    s.arabizi_score = static_cast<double>(arabizi) / nw;

    const double arw = std::max<double>(1.0, static_cast<double>(ar_words));
    const double ltw = std::max<double>(1.0, static_cast<double>(lt_words));
    const double dar_ar_density = static_cast<double>(dar_ar) / arw;
    const double dar_lt_density = static_cast<double>(dar_lt + arabizi) / ltw;
    const double msa_density    = static_cast<double>(msa) / arw;
    const double fr_density     = static_cast<double>(fr) / ltw;
    const double en_density     = static_cast<double>(en) / ltw;

    // Decide. Darija wins ties against MSA because Darija markers are high-precision
    // while MSA markers (في, من, على) also occur inside Darija.
    const bool arabic_dominant = s.arabic_ratio > 0.6;
    const bool latin_dominant  = s.latin_ratio > 0.6;
    const bool mixed_script    = !arabic_dominant && !latin_dominant;

    if (arabic_dominant) {
        if (dar_ar_density >= 0.06 || dar_ar >= 2) {
            s.tag = LangTag::DarijaArab;
            s.confidence = std::min(1.0, 0.4 + dar_ar_density * 4.0);
        } else if (msa_density >= 0.10) {
            s.tag = LangTag::MSA;
            s.confidence = std::min(1.0, 0.4 + msa_density * 3.0);
        } else {
            s.tag = LangTag::MSA;
            s.confidence = 0.3;
        }
    } else if (latin_dominant) {
        // Collision guard (Hermes lake): common English words overlap the
        // Darija-Latin lexicon (had/hit/sir/fine...), so 2 stray collisions
        // in clearly-English text used to hijack the tag. Each language wins
        // only against weaker densities; ties keep the legacy Darija tag.
        // EN threshold 0.08 (+count>=2 rescue): Hermes short answers
        // ("A. It compensates...") have few markers but are English.
        const bool dar_hit = (dar_lt_density >= 0.08 || dar_lt + arabizi >= 2);
        const bool en_hit  = (en_density >= 0.08 || en >= 2);
        const bool fr_hit  = (fr_density >= 0.15);
        if (dar_hit && en_density <= dar_lt_density) {
            s.tag = LangTag::DarijaLatn;
            s.confidence = std::min(1.0, 0.4 + dar_lt_density * 4.0);
        } else if (fr_hit && fr_density > en_density && fr_density >= dar_lt_density) {
            s.tag = LangTag::French;
            s.confidence = std::min(1.0, 0.35 + fr_density * 2.0);
        } else if (en_hit && en_density >= dar_lt_density) {
            s.tag = LangTag::English;
            s.confidence = std::min(1.0, 0.35 + en_density * 2.0);
        } else if (dar_hit) {
            s.tag = LangTag::DarijaLatn;
            s.confidence = std::min(1.0, 0.4 + dar_lt_density * 4.0);
        } else {
            s.tag = LangTag::Other;
            s.confidence = 0.2;
        }
    } else {
        // both scripts present in quantity
        if (dar_ar + dar_lt + arabizi >= 1) {
            s.tag = LangTag::Mixed;
            s.confidence = 0.6;
        } else {
            s.tag = LangTag::Mixed;
            s.confidence = 0.35;
        }
    }
    (void)mixed_script;
    return s;
}

bool LangId::is_darija(const std::string& text, double min_conf) const {
    LangScore s = classify(text);
    if (s.is_darija() && s.confidence >= min_conf) return true;
    // mixed content still counts as Darija if the markers are there
    return s.tag == LangTag::Mixed && s.darija_score > 0.05;
}

double LangId::msa_ratio(const std::string& text) const {
    return classify(text).msa_score;
}

// ================================================================ style
const std::vector<std::string>& robotic_phrases() {
    // The exact register we must keep out of the training data.
    static const std::vector<std::string> p = {
        "بالتأكيد",
        "يمكنني مساعدتك",
        "كيف يمكنني مساعدتك",
        "عزيزي المستخدم",
        "عزيزتي المستخدمة",
        "شكرا لسؤالك",
        "شكرا على سؤالك",
        "سأشرح لك",
        "إليك الخطوات",
        "فيما يلي",
        "في الختام",
        "في الخاتمة",
        "خلاصة القول",
        "من المهم أن نلاحظ",
        "تجدر الإشارة إلى",
        "بناء على ما سبق",
        "هل لديك أي أسئلة أخرى",
        "أنا هنا لمساعدتك",
        "يسعدني مساعدتك",
        "كنموذج لغوي",
        "بصفتي مساعدا",
        "as an ai",
        "as a language model",
        "i'm here to help",
        "certainly!",
        "of course!",
        "i hope this helps",
        "let me know if you need",
        "en tant qu'assistant",
        "je suis là pour vous aider",
    };
    return p;
}

static bool contains_ci(const std::string& hay, const std::string& needle) {
    if (needle.empty() || hay.size() < needle.size()) return false;
    std::string h = to_lower_ascii(hay);
    std::string n = to_lower_ascii(needle);
    return h.find(n) != std::string::npos;
}

StyleFlags check_assistant_style(const std::string& reply, const std::string& user_msg,
                                 const LangId& lid) {
    StyleFlags f;

    NormalizerConfig nc;
    nc.fold_letters = true;
    std::string norm = Normalizer(nc).normalize(reply);

    for (const auto& p : robotic_phrases()) {
        std::string pn = Normalizer(nc).normalize(p);
        if (contains_ci(norm, pn)) {
            f.has_boilerplate = true;
            f.matched_phrases.push_back(p);
        }
    }

    // bullet points
    size_t pos = 0;
    while ((pos = reply.find('\n', pos)) != std::string::npos) {
        size_t j = pos + 1;
        while (j < reply.size() && (reply[j] == ' ' || reply[j] == '\t')) ++j;
        if (j < reply.size() && (reply[j] == '-' || reply[j] == '*' ||
                                 (reply[j] >= '1' && reply[j] <= '9' &&
                                  j + 1 < reply.size() && (reply[j + 1] == '.' || reply[j + 1] == ')')))) {
            ++f.bullet_count;
        }
        pos = j;
    }

    LangScore rs = lid.classify(reply);
    f.msa_ratio = rs.msa_score;

    size_t user_words = word_split(user_msg).size();
    size_t reply_words = word_split(reply).size();

    // A short question deserves a short answer; long bulleted essays in reply to
    // "chno khbark" are exactly the robotic failure mode.
    if (user_words <= 20 && f.bullet_count > 3) f.too_many_bullets = true;
    if (user_words <= 12 && reply_words > 80)   f.too_long = true;

    // Darija turn drifting into MSA
    bool user_is_darija = lid.is_darija(user_msg, 0.3);
    if (user_is_darija && rs.tag == LangTag::MSA && rs.msa_score > 0.14) {
        f.excessive_msa = true;
    }
    return f;
}

// PRO-EN: hard disclosure phrases only (see langid.h). Case-insensitive,
// normalized like check_assistant_style so "As an AI" matches "as an ai".
bool has_hard_ai_boilerplate(const std::string& reply) {
    static const char* kHard[] = {
        "as an ai", "as a language model", "as an ai language model",
        "en tant qu'assistant", "je suis une ia",
        "كنموذج لغوي", "بصفتي مساعدا",
    };
    NormalizerConfig nc;
    nc.fold_letters = true;
    std::string norm = Normalizer(nc).normalize(reply);
    for (const char* p : kHard) {
        std::string pn = Normalizer(nc).normalize(p);
        if (contains_ci(norm, pn)) return true;
    }
    return false;
}

} // namespace gai
