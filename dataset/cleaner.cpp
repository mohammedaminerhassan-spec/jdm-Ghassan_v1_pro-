#include "dataset/cleaner.h"
#include "core/unicode.h"

#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <cctype>
#include <cstring>
#include <cmath>

namespace gai {

std::string PiiReport::summary() const {
    return strfmt("emails=%d phones=%d creds=%d iban=%d card=%d apikey=%d id=%d ip=%d",
                  emails, phones, urls_with_creds, ibans, cards, api_keys, ids, ips);
}

std::string CleanStats::summary() const {
    double keep = lines_in ? 100.0 * double(lines_out) / double(lines_in) : 0.0;
    return strfmt("in=%s out=%s (%.1f%% kept) | dropped: empty=%s short=%s long=%s "
                  "pii=%s encoding=%s quality=%s | bytes %s -> %s",
                  human_count(lines_in).c_str(), human_count(lines_out).c_str(), keep,
                  human_count(dropped_empty).c_str(), human_count(dropped_short).c_str(),
                  human_count(dropped_long).c_str(), human_count(dropped_pii).c_str(),
                  human_count(dropped_encoding).c_str(), human_count(dropped_quality).c_str(),
                  human_bytes(bytes_in).c_str(), human_bytes(bytes_out).c_str());
}

// ================================================================ PII scanners
namespace {

inline bool isd(char c) { return c >= '0' && c <= '9'; }
inline bool isal(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
inline bool isaln(char c) { return isd(c) || isal(c); }

struct Span { size_t begin, end; };

// name@domain.tld
void find_emails(const std::string& s, std::vector<Span>& out) {
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '@') continue;
        size_t b = i;
        while (b > 0 && (isaln(s[b - 1]) || s[b - 1] == '.' || s[b - 1] == '_' ||
                         s[b - 1] == '-' || s[b - 1] == '+')) --b;
        if (b == i) continue;
        size_t e = i + 1;
        bool dot = false;
        while (e < s.size() && (isaln(s[e]) || s[e] == '.' || s[e] == '-')) {
            if (s[e] == '.') dot = true;
            ++e;
        }
        while (e > i + 1 && (s[e - 1] == '.' || s[e - 1] == '-')) --e;
        if (dot && e - i >= 4 && i - b >= 1) out.push_back({b, e});
    }
}

// Moroccan and international phone numbers
void find_phones(const std::string& s, std::vector<Span>& out) {
    size_t i = 0;
    while (i < s.size()) {
        if (!isd(s[i]) && s[i] != '+') { ++i; continue; }
        size_t b = i;
        int digits = 0;
        size_t e = i;
        while (e < s.size() && (isd(s[e]) || s[e] == ' ' || s[e] == '-' || s[e] == '.' ||
                                s[e] == '(' || s[e] == ')' || (s[e] == '+' && e == b))) {
            if (isd(s[e])) ++digits;
            if (digits > 15) break;
            ++e;
        }
        while (e > b && !isd(s[e - 1])) --e;
        // Moroccan mobile 06/07 + 8 digits, or +212..., or any 9-15 digit run
        bool ma = (digits == 10 && b + 1 < s.size() && s[b] == '0' && (s[b + 1] == '6' || s[b + 1] == '7'));
        bool intl = (s[b] == '+' && digits >= 9);
        bool longrun = digits >= 9 && digits <= 15;
        if (ma || intl || longrun) {
            // avoid flagging plain long numbers with no separators & no country hint
            if (ma || intl || (e - b) > static_cast<size_t>(digits))
                out.push_back({b, e});
        }
        i = std::max(e, b + 1);
    }
}

// scheme://user:pass@host
void find_url_creds(const std::string& s, std::vector<Span>& out) {
    size_t p = 0;
    while ((p = s.find("://", p)) != std::string::npos) {
        size_t b = p;
        while (b > 0 && isal(s[b - 1])) --b;
        size_t e = p + 3;
        size_t at = std::string::npos;
        while (e < s.size() && !is_whitespace(static_cast<u8>(s[e]))) {
            if (s[e] == '@') at = e;
            if (s[e] == '/' && at == std::string::npos && e > p + 3) break;
            ++e;
        }
        if (at != std::string::npos) {
            bool colon = s.find(':', p + 3) < at;
            if (colon) out.push_back({b, e});
        }
        p += 3;
    }
}

void find_iban(const std::string& s, std::vector<Span>& out) {
    for (size_t i = 0; i + 15 < s.size(); ++i) {
        if (!isal(s[i]) || !isal(s[i + 1]) || !isd(s[i + 2]) || !isd(s[i + 3])) continue;
        if (i > 0 && isaln(s[i - 1])) continue;
        size_t e = i + 4;
        int aln = 4;
        while (e < s.size() && (isaln(s[e]) || s[e] == ' ')) {
            if (isaln(s[e])) ++aln;
            if (aln > 34) break;
            ++e;
        }
        while (e > i && !isaln(s[e - 1])) --e;
        if (aln >= 15) out.push_back({i, e});
    }
}

void find_cards(const std::string& s, std::vector<Span>& out) {
    size_t i = 0;
    while (i < s.size()) {
        if (!isd(s[i])) { ++i; continue; }
        size_t b = i, e = i;
        int digits = 0;
        while (e < s.size() && (isd(s[e]) || s[e] == ' ' || s[e] == '-')) {
            if (isd(s[e])) ++digits;
            if (digits > 19) break;
            ++e;
        }
        while (e > b && !isd(s[e - 1])) --e;
        if (digits >= 13 && digits <= 19) {
            // Luhn check keeps the false-positive rate low
            int sum = 0, parity = digits % 2;
            int idx = 0;
            for (size_t k = b; k < e; ++k) {
                if (!isd(s[k])) continue;
                int d = s[k] - '0';
                if (idx % 2 == parity) { d *= 2; if (d > 9) d -= 9; }
                sum += d;
                ++idx;
            }
            if (sum % 10 == 0) out.push_back({b, e});
        }
        i = std::max(e, b + 1);
    }
}

void find_api_keys(const std::string& s, std::vector<Span>& out) {
    static const char* prefixes[] = {"sk-", "pk-", "ghp_", "gho_", "AKIA", "AIza", "xox",
                                     "Bearer ", "api_key=", "apikey=", "token=", "password=",
                                     "passwd=", "secret="};
    for (const char* pre : prefixes) {
        size_t plen = std::strlen(pre);
        size_t p = 0;
        while ((p = s.find(pre, p)) != std::string::npos) {
            size_t e = p + plen;
            int n = 0;
            while (e < s.size() && (isaln(s[e]) || s[e] == '_' || s[e] == '-' || s[e] == '.')) { ++e; ++n; }
            if (n >= 8) out.push_back({p, e});
            p += plen;
        }
    }
}

// Moroccan CIN: 1-2 letters + 5-6 digits
void find_ids(const std::string& s, std::vector<Span>& out) {
    for (size_t i = 0; i < s.size(); ++i) {
        if (!isal(s[i])) continue;
        if (i > 0 && isaln(s[i - 1])) continue;
        size_t j = i;
        int letters = 0;
        while (j < s.size() && isal(s[j]) && letters < 2) { ++j; ++letters; }
        size_t k = j;
        int digits = 0;
        while (k < s.size() && isd(s[k])) { ++k; ++digits; }
        if (letters >= 1 && digits >= 5 && digits <= 8 && (k >= s.size() || !isaln(s[k]))) {
            out.push_back({i, k});
        }
    }
}

void find_ips(const std::string& s, std::vector<Span>& out) {
    size_t i = 0;
    while (i < s.size()) {
        if (!isd(s[i])) { ++i; continue; }
        if (i > 0 && (isd(s[i - 1]) || s[i - 1] == '.')) { ++i; continue; }
        size_t e = i;
        int parts = 0;
        bool ok = true;
        while (parts < 4 && e < s.size()) {
            int val = 0, nd = 0;
            while (e < s.size() && isd(s[e]) && nd < 3) { val = val * 10 + (s[e] - '0'); ++e; ++nd; }
            if (nd == 0 || val > 255) { ok = false; break; }
            ++parts;
            if (parts < 4) {
                if (e < s.size() && s[e] == '.') ++e;
                else { ok = false; break; }
            }
        }
        if (ok && parts == 4 && (e >= s.size() || !isaln(s[e]))) out.push_back({i, e});
        i = std::max(e, i + 1);
    }
}

} // namespace

PiiReport scan_pii(const std::string& text) {
    PiiReport r;
    std::vector<Span> sp;
    find_emails(text, sp);      r.emails = static_cast<int>(sp.size()); sp.clear();
    find_phones(text, sp);      r.phones = static_cast<int>(sp.size()); sp.clear();
    find_url_creds(text, sp);   r.urls_with_creds = static_cast<int>(sp.size()); sp.clear();
    find_iban(text, sp);        r.ibans = static_cast<int>(sp.size()); sp.clear();
    find_cards(text, sp);       r.cards = static_cast<int>(sp.size()); sp.clear();
    find_api_keys(text, sp);    r.api_keys = static_cast<int>(sp.size()); sp.clear();
    find_ids(text, sp);         r.ids = static_cast<int>(sp.size()); sp.clear();
    find_ips(text, sp);         r.ips = static_cast<int>(sp.size());
    return r;
}

std::string redact_pii(const std::string& text, PiiReport* report) {
    struct Tagged { Span s; const char* tag; };
    std::vector<Tagged> all;
    auto collect = [&](void (*fn)(const std::string&, std::vector<Span>&), const char* tag) {
        std::vector<Span> sp;
        fn(text, sp);
        for (auto& s : sp) all.push_back({s, tag});
        return static_cast<int>(sp.size());
    };
    PiiReport r;
    r.emails          = collect(find_emails, "[EMAIL]");
    r.phones          = collect(find_phones, "[PHONE]");
    r.urls_with_creds = collect(find_url_creds, "[URL]");
    r.ibans           = collect(find_iban, "[IBAN]");
    r.cards           = collect(find_cards, "[CARD]");
    r.api_keys        = collect(find_api_keys, "[KEY]");
    r.ids             = collect(find_ids, "[ID]");
    r.ips             = collect(find_ips, "[IP]");
    if (report) *report = r;
    if (all.empty()) return text;

    std::sort(all.begin(), all.end(), [](const Tagged& a, const Tagged& b) {
        return a.s.begin < b.s.begin;
    });

    std::string out;
    out.reserve(text.size());
    size_t cur = 0;
    for (const auto& t : all) {
        if (t.s.begin < cur) continue;   // overlapping match already handled
        out.append(text, cur, t.s.begin - cur);
        out += t.tag;
        cur = t.s.end;
    }
    out.append(text, cur, std::string::npos);
    return out;
}

// ================================================================ cleaning
std::string strip_html_tags(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_tag = false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '<') {
            // only treat as a tag if it looks like one
            size_t j = i + 1;
            if (j < s.size() && (isal(s[j]) || s[j] == '/' || s[j] == '!')) { in_tag = true; continue; }
        }
        if (in_tag) {
            if (c == '>') in_tag = false;
            continue;
        }
        out.push_back(c);
    }
    // common entities
    static const std::pair<const char*, const char*> ents[] = {
        {"&nbsp;", " "}, {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"},
        {"&quot;", "\""}, {"&#39;", "'"}, {"&apos;", "'"}, {"&mdash;", "-"},
    };
    for (auto& [from, to] : ents) {
        size_t p = 0;
        size_t flen = std::strlen(from);
        while ((p = out.find(from, p)) != std::string::npos) {
            out.replace(p, flen, to);
            p += std::strlen(to);
        }
    }
    return out;
}

std::string fix_mojibake(const std::string& s) {
    // Arabic UTF-8 read as latin-1 then re-encoded produces "Ø§Ù„..." patterns.
    // Detect a high density of C3/C2-prefixed sequences and undo the double encode.
    size_t suspicious = 0, total = 0;
    for (size_t i = 0; i + 1 < s.size(); ++i) {
        u8 a = static_cast<u8>(s[i]);
        if (a == 0xC3 || a == 0xC2) ++suspicious;
        if (a >= 0x80) ++total;
    }
    if (total == 0 || suspicious * 2 < total) return s;

    // decode utf8 -> take low byte of each codepoint -> reinterpret as utf8
    std::string bytes;
    bytes.reserve(s.size());
    size_t i = 0;
    bool ok = true;
    while (i < s.size()) {
        u32 cp = utf8_decode(s, i);
        if (cp > 0xFF) { ok = false; break; }
        bytes.push_back(static_cast<char>(cp));
    }
    if (!ok || !utf8_valid(bytes)) return s;
    return bytes;
}

bool Cleaner::clean_line(const std::string& in, std::string& out, CleanStats& st) const {
    ++st.lines_in;
    st.bytes_in += in.size();

    if (in.empty()) { ++st.dropped_empty; return false; }
    if (static_cast<int>(in.size()) > cfg_.max_chars) { ++st.dropped_long; return false; }

    std::string s = in;
    if (cfg_.fix_mojibake) s = fix_mojibake(s);
    if (!utf8_valid(s)) { ++st.dropped_encoding; return false; }
    if (cfg_.strip_html) s = strip_html_tags(s);

    if (cfg_.drop_pii_lines || cfg_.redact_instead_drop) {
        PiiReport r = scan_pii(s);
        if (r.any()) {
            if (cfg_.redact_instead_drop) s = redact_pii(s);
            else { ++st.dropped_pii; return false; }
        }
    }

    s = norm_.normalize(s);
    if (s.empty()) { ++st.dropped_empty; return false; }
    if (static_cast<int>(utf8_length(s)) < cfg_.min_chars) { ++st.dropped_short; return false; }

    out = std::move(s);
    ++st.lines_out;
    st.bytes_out += out.size();
    return true;
}

// ================================================================ quality
QualityVerdict quality_check(const std::string& text, const QualityConfig& cfg) {
    QualityVerdict v;
    if (text.empty()) { v.accept = false; v.reason = "empty"; return v; }

    if (cfg.reject_ai_disclaimers) {
        std::string low = to_lower_ascii(text);
        static const std::vector<const char*> disclaimers = {
            "as an ai language model", "as a language model",
            "as an artificial intelligence", "i don't have personal opinions",
            "i cannot fulfill this request", "i am an ai", "i'm an ai",
            "openai", "chatgpt", "i cannot provide", "i'm sorry, but i cannot",
            "i am a large language model"
        };
        for (const char* d : disclaimers) {
            if (low.find(d) != std::string::npos) {
                v.accept = false; v.reason = "ai_disclaimer"; return v;
            }
        }
    }

    if (cfg.reject_placeholders) {
        if (text.find("[insert ") != std::string::npos ||
            text.find("[Insert ") != std::string::npos ||
            text.find("<your_") != std::string::npos ||
            text.find("<YOUR_") != std::string::npos ||
            text.find("TODO:") != std::string::npos ||
            text.find("[Your Name]") != std::string::npos ||
            text.find("[your name]") != std::string::npos) {
            v.accept = false; v.reason = "unresolved_placeholder"; return v;
        }
    }

    ScriptStats st = script_stats(text);
    if (st.total == 0) { v.accept = false; v.reason = "no content"; return v; }

    size_t letters = st.arabic + st.latin;
    double letter_ratio = static_cast<double>(letters) / static_cast<double>(st.total);
    double symbol_ratio = static_cast<double>(st.punct + st.other) / static_cast<double>(st.total);
    double digit_ratio  = static_cast<double>(st.digits) / static_cast<double>(st.total);

    if (cfg.require_arabic_or_latin && letters == 0) { v.accept = false; v.reason = "no letters"; return v; }
    if (letter_ratio < cfg.min_letter_ratio) { v.accept = false; v.reason = "low letter ratio"; return v; }
    if (symbol_ratio > cfg.max_symbol_ratio) { v.accept = false; v.reason = "symbol spam"; return v; }
    if (digit_ratio > cfg.max_digit_ratio)   { v.accept = false; v.reason = "digit spam"; return v; }

    // uppercase (latin only)
    size_t upper = 0;
    {
        size_t i = 0;
        while (i < text.size()) {
            u32 cp = utf8_decode(text, i);
            if (cp >= 'A' && cp <= 'Z') ++upper;
        }
    }
    if (st.latin > 20 && static_cast<double>(upper) / static_cast<double>(st.latin) > cfg.max_upper_ratio) {
        v.accept = false; v.reason = "shouting"; return v;
    }

    // word statistics
    std::unordered_map<std::string, int> wc;
    std::string cur;
    size_t nwords = 0, maxlen = 0;
    {
        size_t i = 0;
        while (i <= text.size()) {
            u32 cp = (i < text.size()) ? utf8_decode(text, i) : ' ';
            if (i >= text.size() || is_whitespace(cp)) {
                if (!cur.empty()) {
                    ++nwords;
                    maxlen = std::max(maxlen, utf8_length(cur));
                    wc[cur]++;
                    cur.clear();
                }
                if (i >= text.size()) break;
            } else {
                utf8_encode(cp, cur);
            }
        }
    }
    if (static_cast<int>(nwords) < cfg.min_words) { v.accept = false; v.reason = "too few words"; return v; }
    if (static_cast<int>(maxlen) > cfg.max_word_length) { v.accept = false; v.reason = "absurd word length"; return v; }

    if (nwords >= 8) {
        int top = 0;
        for (const auto& [w, c] : wc) top = std::max(top, c);
        if (static_cast<double>(top) / static_cast<double>(nwords) > cfg.max_word_repeat) {
            v.accept = false; v.reason = "word repetition"; return v;
        }
    }

    // repeated lines inside a document
    {
        std::unordered_map<std::string, int> lc;
        std::istringstream ls(text);
        std::string line;
        int total_lines = 0;
        while (std::getline(ls, line)) {
            if (line.empty()) continue;
            lc[line]++;
            ++total_lines;
        }
        if (total_lines >= 5) {
            int dup = 0;
            for (const auto& [l, c] : lc) if (c > 1) dup += c - 1;
            if (static_cast<double>(dup) / static_cast<double>(total_lines) > cfg.max_repeat_line) {
                v.accept = false; v.reason = "duplicated lines"; return v;
            }
        }
    }
    return v;
}

// ================================================================ english quality
// PARQUET-ONLY EN profile: keeps Hermes multiple-choice ("A."), code and
// math that the Darija defaults drop. Single source of truth for --style-mode en.
QualityConfig english_quality_config() {
    QualityConfig c;
    c.max_symbol_ratio = 0.35;   // code: = * / { } ; are legitimate
    c.max_digit_ratio = 0.50;    // math: numbers dominate short answers
    c.max_upper_ratio = 0.60;    // "A. It compensates..." is uppercase-heavy
    c.max_repeat_line = 0.30;
    c.max_word_repeat = 0.40;    // short answers repeat the prompt words
    c.min_letter_ratio = 0.30;
    c.min_words = 1;             // CRITICAL: keeps "A." multiple-choice answers
    c.max_word_length = 80;      // URLs / code tokens are long
    c.require_arabic_or_latin = true;
    c.reject_ai_disclaimers = false;  // handled by hard list below (not the full Darija list)
    c.reject_placeholders = true;
    return c;
}

static bool has_hard_disclosure_en(const std::string& text) {
    static const char* kHard[] = {
        "as an ai", "as a language model", "as an ai language model",
        "i am a large language model", "i cannot fulfill this request",
    };
    std::string low = to_lower_ascii(text);
    for (const char* p : kHard) {
        if (low.find(p) != std::string::npos) return true;
    }
    return false;
}

QualityVerdict quality_check_english(const std::string& text, const QualityConfig& cfg) {
    QualityConfig c = cfg;
    // defaults are the EN profile when the caller passes a default cfg
    if (c.min_words == 2 && c.min_letter_ratio == 0.45) c = english_quality_config();
    QualityVerdict v = quality_check(text, c);
    if (!v.accept) {
        // single-letter / "A." answers fail word stats: rescue them explicitly.
        std::string t = text;
        size_t a = t.find_first_not_of(" \t\n\r");
        size_t b = t.find_last_not_of(" \t\n\r");
        std::string s = (a == std::string::npos) ? "" : t.substr(a, b - a + 1);
        if ((s.size() == 1 || s.size() == 2) && !s.empty()) {
            v.accept = true; v.reason.clear(); return v;
        }
        if (s.size() <= 4 && (s[0] >= 'A' && s[0] <= 'Z')) {
            v.accept = true; v.reason.clear(); return v;
        }
        return v;
    }
    if (has_hard_disclosure_en(text)) {
        v.accept = false; v.reason = "ai_disclosure_hard"; return v;
    }
    return v;
}

// ================================================================ toxicity
ToxicityResult check_toxicity(const std::string& text) {
    // Conservative rule lists. Word-boundary matching to avoid false positives on
    // innocuous substrings.
    static const std::vector<std::pair<const char*, const char*>> terms = {
        // {term, category} - explicit slurs and sexual content markers
        {"قحبة", "slur"}, {"زامل", "slur"}, {"كلب ابن", "slur"},
        {"نيك", "sexual"}, {"طيز", "sexual"}, {"زب", "sexual"},
        {"7mar wld", "slur"}, {"9a7ba", "slur"}, {"zamel", "slur"},
        {"nik mok", "slur"}, {"tbon", "sexual"},
        {"fuck", "profanity"}, {"bitch", "slur"}, {"nigger", "slur"},
        {"faggot", "slur"}, {"rape", "violence"}, {"kill yourself", "self-harm"},
        {"pute", "slur"}, {"salope", "slur"}, {"enculé", "slur"},
        {"انتحار", "self-harm"}, {"اقتل نفسك", "self-harm"},
    };

    ToxicityResult r;
    std::string low = to_lower_ascii(text);
    for (const auto& [term, cat] : terms) {
        if (low.find(term) != std::string::npos) {
            ++r.hits;
            if (std::find(r.categories.begin(), r.categories.end(), cat) == r.categories.end())
                r.categories.emplace_back(cat);
        }
    }
    r.toxic = r.hits > 0;
    return r;
}

} // namespace gai
