#include "dataset/darija_synthesis_engine.h"
#include "dataset/json_reader.h"
#include "dataset/darija_synthesis_content.h"
#include "dataset/dedup.h"
#include "core/unicode.h"

#include <fstream>
#include <cstring>
#include <sstream>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace gai {

std::string SynthStats::summary() const {
    u64 tried = generated;
    return strfmt("generated=%s accepted=%s (%.1f%%) | rejected dup=%s style=%s entropy=%s",
                  human_count(tried).c_str(), human_count(accepted).c_str(),
                  tried ? 100.0 * double(accepted) / double(tried) : 0.0,
                  human_count(rejected_duplicate).c_str(),
                  human_count(rejected_style).c_str(),
                  human_count(rejected_entropy).c_str());
}

namespace {

struct Translit { const char* ar; const char* lat; const char* alt; };

const std::vector<Translit>& translit_table() {
    static const std::vector<Translit> t = {

        {"الله", "allah", "lah"},
        {"ش",  "ch", "sh"},
        {"خ",  "kh", "5"},
        {"غ",  "gh", "8"},
        {"ث",  "th", "t"},
        {"ذ",  "d",  "dh"},
        {"ظ",  "d",  "dh"},
        {"ص",  "s",  "s"},
        {"ض",  "d",  "d"},
        {"ط",  "t",  "t"},
        {"ح",  "7",  "h"},
        {"ع",  "3",  "3"},
        {"ق",  "9",  "q"},
        {"ء",  "2",  ""},
        {"أ",  "a",  "2a"},
        {"إ",  "i",  "2i"},
        {"آ",  "a",  "aa"},
        {"ا",  "a",  "a"},
        {"ب",  "b",  "b"},
        {"ت",  "t",  "t"},
        {"ج",  "j",  "j"},
        {"د",  "d",  "d"},
        {"ر",  "r",  "r"},
        {"ز",  "z",  "z"},
        {"س",  "s",  "s"},
        {"ف",  "f",  "f"},
        {"ك",  "k",  "k"},
        {"ݣ",  "g",  "g"},
        {"گ",  "g",  "g"},
        {"ل",  "l",  "l"},
        {"م",  "m",  "m"},
        {"ن",  "n",  "n"},
        {"ه",  "h",  "h"},
        {"ة",  "a",  "a"},
        {"و",  "w",  "ou"},
        {"ي",  "i",  "y"},
        {"ى",  "a",  "a"},
        {"پ",  "p",  "p"},
        {"ڤ",  "v",  "v"},
        {"،",  ",",  ","},
        {"؟",  "?",  "?"},
        {"؛",  ";",  ";"},
    };
    return t;
}

const std::unordered_map<std::string, std::vector<std::string>>& word_translit() {
    static const std::unordered_map<std::string, std::vector<std::string>> m = {
        {"شنو",   {"chno", "shnou", "chnou"}},
        {"كيداير",{"kidayr", "kif dayr", "kidir"}},
        {"خبارك", {"khbark", "5bark"}},
        {"لاباس", {"labas", "lbas"}},
        {"بخير",  {"bikhir", "b5ir"}},
        {"الحمد", {"l7amdo", "hamdo"}},
        {"لله",   {"lillah", "lah"}},
        {"نتا",   {"nta", "nta"}},
        {"نتي",   {"nti", "nti"}},
        {"بزاف",  {"bzaf", "bezzaf"}},
        {"دابا",  {"daba", "daba"}},
        {"واخا",  {"wakha", "waxa"}},
        {"مزيان", {"mzyan", "mezyan"}},
        {"غادي",  {"ghadi", "ghadi"}},
        {"كاين",  {"kayn", "kayen"}},
        {"ديال",  {"dyal", "dial"}},
        {"بغيت",  {"bghit", "bghit"}},
        {"عافاك", {"3afak", "3afak"}},
        {"شكرا",  {"chokran", "shukran"}},
        {"سلام",  {"salam", "slam"}},
        {"أهلا",  {"ahlan", "ahla"}},
        {"صافي",  {"safi", "safi"}},
        {"علاش",  {"3lach", "3lash"}},
        {"كيفاش", {"kifach", "kifash"}},
        {"فين",   {"fin", "fine"}},
        {"شحال",  {"chhal", "shhal"}},
        {"خويا",  {"khoya", "5oya"}},
        {"والله", {"wllah", "wallah"}},
        {"معايا", {"m3aya", "m3aya"}},
        {"عندي",  {"3andi", "3ndi"}},
        {"ماشي",  {"machi", "mashi"}},
        {"حيت",   {"7it", "hit"}},
        {"هاد",   {"had", "had"}},
        {"هادشي", {"hadchi", "hadshi"}},
        {"اليوم", {"lyoum", "lyom"}},
        {"غدا",   {"ghedda", "ghda"}},
        {"الدار", {"dar", "ddar"}},
        {"خدمة",  {"khedma", "khdma"}},
        {"مشكل",  {"mochkil", "mushkil"}},
        {"الله",  {"allah", "lah"}},
    };
    return m;
}

static bool is_latin_only(const std::string& text) {
    ScriptStats ss = script_stats(text);
    return ss.arabic == 0 && ss.latin > 0;
}

}

std::string arabic_to_arabizi(const std::string& arabic, Rng& rng, bool heavy_digits) {
    const auto& wt = word_translit();
    const auto& tt = translit_table();

    std::string out;
    size_t i = 0;
    std::string word;

    auto flush_word = [&]() {
        if (word.empty()) return;

        auto digitfold = [&](std::string s) {
            if (heavy_digits) return s;
            for (char& ch : s) {
                if (ch == '3') ch = 'a';
                else if (ch == '7') ch = 'h';
                else if (ch == '9') ch = 'q';
                else if (ch == '2') ch = 'a';
                else if (ch == '5') ch = 'k';
            }
            return s;
        };

        auto it = wt.find(word);
        if (it == wt.end() && word.size() > 4 && word.rfind("ال", 0) == 0) {
            it = wt.find(word.substr(4));
            if (it != wt.end()) {
                const auto& opts = it->second;
                out += "l" + digitfold(opts[rng.below(opts.size())]);
                word.clear();
                return;
            }
        }
        if (it != wt.end()) {
            const auto& opts = it->second;
            out += digitfold(opts[rng.below(opts.size())]);
            word.clear();
            return;
        }

        size_t k = 0;
        while (k < word.size()) {
            bool matched = false;
            for (const auto& e : tt) {
                size_t el = std::strlen(e.ar);
                if (word.compare(k, el, e.ar) == 0) {
                    const char* pick = e.lat;
                    if (e.alt && e.alt[0] && rng.uniform() < 0.3) pick = e.alt;
                    if (!heavy_digits) {

                        if (std::strcmp(e.ar, "ح") == 0) pick = "h";
                        else if (std::strcmp(e.ar, "ق") == 0) pick = "q";
                        else if (std::strcmp(e.ar, "خ") == 0) pick = "kh";
                        else if (std::strcmp(e.ar, "غ") == 0) pick = "gh";
                        else if (std::strcmp(e.ar, "ع") == 0) pick = "a";
                        else if (std::strcmp(e.ar, "ء") == 0) pick = "a";
                    }
                    out += pick;
                    k += el;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                size_t before = k;
                u32 cp = utf8_decode(word, k);
                if (k == before) ++k;
                if (!is_arabic_letter(cp) && !is_arabic_diacritic(cp)) utf8_encode(cp, out);
            }
        }
        word.clear();
    };

    while (i < arabic.size()) {
        size_t before = i;
        u32 cp = utf8_decode(arabic, i);
        if (is_arabic_letter(cp) || is_arabic_diacritic(cp)) {
            out.append(arabic, before, i - before);

            word.append(arabic, before, i - before);
            out.resize(out.size() - (i - before));
        } else {
            flush_word();
            if (cp == 0x060C) out += ",";
            else if (cp == 0x061F) out += "?";
            else if (cp == 0x061B) out += ";";
            else utf8_encode(cp, out);
        }
    }
    flush_word();
    return out;
}

std::string apply_orthographic_noise(const std::string& latin, Rng& rng) {
    std::string out;
    out.reserve(latin.size());
    for (size_t i = 0; i < latin.size(); ++i) {
        char c = latin[i];

        if (c == 'c' && i + 1 < latin.size() && latin[i + 1] == 'h' && rng.uniform() < 0.25) {
            out += "sh";
            ++i;
            continue;
        }
        if (c == 'o' && i + 1 < latin.size() && latin[i + 1] == 'u' && rng.uniform() < 0.2) {
            out += "o";
            ++i;
            continue;
        }
        if (c == '7' && rng.uniform() < 0.12) { out += "h"; continue; }
        if (c == '9' && rng.uniform() < 0.12) { out += "q"; continue; }
        if (c == '3' && rng.uniform() < 0.06) { out += "a"; continue; }
        out.push_back(c);
    }
    return out;
}

struct SynthGenerator::Impl {
    Rng rng;
    Deduplicator dedup;
    LangId lid;
    std::map<u64, int> template_uses;
    std::map<std::string, int> first_token_counts;
    u64 total_accepted = 0;

    explicit Impl(u64 seed) : rng(seed) {
        Deduplicator::Config dc;
        dc.jaccard_threshold = 0.80;
        dedup = Deduplicator(dc);
    }
};

SynthGenerator::SynthGenerator(SynthConfig cfg)
    : impl_(std::make_unique<Impl>(cfg.seed)), cfg_(cfg) {}

SynthGenerator::~SynthGenerator() = default;

int SynthGenerator::domain_count() { return static_cast<int>(synth_data::domains().size()); }
const char* SynthGenerator::domain_name(int i) {
    const auto& d = synth_data::domains();
    return (i >= 0 && i < static_cast<int>(d.size())) ? d[static_cast<size_t>(i)].name : "?";
}

template <typename T>
static const T& pick(const std::vector<T>& v, Rng& rng) {
    return v[rng.below(v.size())];
}

static std::string first_words(const std::string& s, int n) {
    std::istringstream is(s);
    std::string w, out;
    for (int i = 0; i < n && (is >> w); ++i) {
        if (!out.empty()) out += " ";
        out += w;
    }
    return out;
}

bool SynthGenerator::generate(Conversation& out) {
    ++stats_.generated;
    Rng& rng = impl_->rng;

    const auto& domains = synth_data::domains();
    size_t di = rng.below(domains.size());
    const auto& D = domains[di];
    out.domain = D.name;

    double r = rng.uniform();
    Script script;
    bool msa_conv = false;
    if (r < cfg_.p_arabic_script) script = Script::Arabic;
    else if (r < cfg_.p_arabic_script + cfg_.p_latin_script) script = Script::Latin;
    else { script = Script::Arabic; msa_conv = true; }
    out.script = script;

    const bool heavy_digits = rng.uniform() < 0.7;

    auto surface = [&](const std::string& text) -> std::string {
        if (script != Script::Latin) return text;

        ScriptStats ss = script_stats(text);
        if (ss.arabic == 0) return apply_orthographic_noise(text, rng);
        std::string lat = arabic_to_arabizi(text, rng, heavy_digits);
        return apply_orthographic_noise(lat, rng);
    };

    out.messages.clear();

    auto gated = [&](const std::string& u, const std::string& a) {
        return script == Script::Latin || (!is_latin_only(u) && !is_latin_only(a));
    };

    if (cfg_.include_system && rng.uniform() < cfg_.p_system) {
        out.messages.push_back({Role::System, pick(synth_data::system_prompts(), rng)});
    }

    const int turns = cfg_.min_turns +
        static_cast<int>(rng.below(static_cast<u64>(cfg_.max_turns - cfg_.min_turns + 1)));

    u64 tid = di * 1000003ull;

    bool greeted = rng.uniform() < 0.30;
    if (greeted) {
        const auto& gu = synth_data::greetings_user();
        const auto& ga = synth_data::greetings_assistant();
        bool ok = false;
        for (int t = 0; t < 8; ++t) {
            const std::string u = pick(gu, rng);
            const std::string a = pick(ga, rng);
            if (gated(u, a)) {
                out.messages.push_back({Role::User, surface(u)});
                out.messages.push_back({Role::Assistant, surface(a)});
                tid = tid * 31 + hash_string(u);
                ok = true;
                break;
            }
        }
        if (!ok) greeted = false;
    }

    int produced = greeted ? 1 : 0;

    while (produced < turns) {
        double p = rng.uniform();

        if (msa_conv && rng.uniform() < 0.5 && !synth_data::msa_exchanges().empty()) {

            const auto& pool = synth_data::msa_exchanges();
            const synth_data::Exchange* chosen = nullptr;
            for (int t = 0; t < 8; ++t) {
                const auto& cand = pick(pool, rng);
                if (script == Script::Latin || (!is_latin_only(cand.user) &&
                                                !is_latin_only(cand.assistant))) {
                    chosen = &cand;
                    break;
                }
            }
            if (!chosen) continue;
            out.messages.push_back({Role::User, surface(chosen->user)});
            out.messages.push_back({Role::Assistant, surface(chosen->assistant)});
            tid = tid * 31 + hash_string(chosen->user);
        } else if (p < cfg_.p_correction && produced > 0) {
            const auto& e = pick(synth_data::corrections(), rng);
            out.messages.push_back({Role::User, surface(e.user)});
            out.messages.push_back({Role::Assistant, surface(e.assistant)});
            tid = tid * 31 + hash_string(e.user);
        } else if (p < cfg_.p_correction + cfg_.p_misunderstand) {
            const auto& e = pick(synth_data::misunderstandings(), rng);
            out.messages.push_back({Role::User, surface(e.user)});
            out.messages.push_back({Role::Assistant, surface(e.assistant)});
            tid = tid * 31 + hash_string(e.user);
        } else if (p < cfg_.p_correction + cfg_.p_misunderstand + cfg_.p_governor &&
                   !synth_data::governor_exchanges().empty()) {

            const auto& pool = synth_data::governor_exchanges();
            const synth_data::Exchange* chosen = nullptr;
            for (int t = 0; t < 8; ++t) {
                const auto& cand = pick(pool, rng);
                if (script == Script::Latin || (!is_latin_only(cand.user) &&
                                                !is_latin_only(cand.assistant))) {
                    chosen = &cand;
                    break;
                }
            }
            if (!chosen) continue;
            out.messages.push_back({Role::User, surface(chosen->user)});
            out.messages.push_back({Role::Assistant, surface(chosen->assistant)});
            tid = tid * 31 + hash_string(chosen->user);
        } else if (p < cfg_.p_correction + cfg_.p_misunderstand + cfg_.p_governor +
                       cfg_.p_reasoning &&
                   !synth_data::reasoning_exchanges().empty()) {
            const auto& pool = synth_data::reasoning_exchanges();
            const synth_data::Exchange* chosen = nullptr;
            for (int t = 0; t < 8; ++t) {
                const auto& cand = pick(pool, rng);
                if (script == Script::Latin || (!is_latin_only(cand.user) &&
                                                !is_latin_only(cand.assistant))) {
                    chosen = &cand;
                    break;
                }
            }
            if (!chosen) continue;
            out.messages.push_back({Role::User, surface(chosen->user)});
            out.messages.push_back({Role::Assistant, surface(chosen->assistant)});
            tid = tid * 31 + hash_string(chosen->user);
        } else if (rng.uniform() < 0.06) {

            const auto& pool = synth_data::identity_questions();
            const synth_data::Exchange* chosen = nullptr;
            for (int t = 0; t < 8; ++t) {
                const auto& cand = pick(pool, rng);
                if (script == Script::Latin || (!is_latin_only(cand.user) &&
                                                !is_latin_only(cand.assistant))) {
                    chosen = &cand;
                    break;
                }
            }
            if (!chosen) continue;
            out.messages.push_back({Role::User, surface(chosen->user)});
            out.messages.push_back({Role::Assistant, surface(chosen->assistant)});
            tid = tid * 31 + hash_string(chosen->user);
        } else if (produced > 0 && rng.uniform() < cfg_.p_followup && !D.followups.empty()) {
            const synth_data::Exchange* chosen = nullptr;
            for (int t = 0; t < 8 && !chosen; ++t) {
                const auto& cand = pick(D.followups, rng);
                if (gated(cand.user, cand.assistant)) chosen = &cand;
            }
            if (!chosen) continue;
            out.messages.push_back({Role::User, surface(chosen->user)});
            out.messages.push_back({Role::Assistant, surface(chosen->assistant)});
            tid = tid * 31 + hash_string(chosen->user);
        } else {
            const synth_data::Exchange* chosen = nullptr;
            for (int t = 0; t < 8 && !chosen; ++t) {
                const auto& cand = pick(D.openers, rng);
                if (gated(cand.user, cand.assistant)) chosen = &cand;
            }
            if (!chosen) continue;
            const auto& e = *chosen;
            std::string user = e.user;
            std::string asst = e.assistant;

            if (script == Script::Latin && rng.uniform() < cfg_.p_french_switch) {
                const auto& ft = synth_data::french_terms();

                for (int attempt = 0; attempt < 8; ++attempt) {
                    const auto& term = ft[rng.below(ft.size())];
                    size_t pos = asst.find(term.first);
                    if (pos != std::string::npos) {
                        asst.replace(pos, std::strlen(term.first), term.second);
                        break;
                    }
                }
            }

            if (rng.uniform() < 0.22) {
                const std::string f = pick(synth_data::fillers(), rng);
                if (gated(f, f)) asst = f + "، " + asst;
            }
            out.messages.push_back({Role::User, surface(user)});
            out.messages.push_back({Role::Assistant, surface(asst)});
            tid = tid * 31 + hash_string(e.user);
        }

        ++produced;

        if (produced < turns && rng.uniform() < 0.18) {
            const auto& bu = synth_data::backchannels_user();
            bool ok = false;
            for (int t = 0; t < 8 && !ok; ++t) {
                std::string bc = pick(bu, rng);
                const auto& e = D.followups.empty() ? pick(D.openers, rng) : pick(D.followups, rng);
                if (gated(bc, e.assistant)) {
                    out.messages.push_back({Role::User, surface(bc)});
                    out.messages.push_back({Role::Assistant, surface(e.assistant)});
                    ok = true;
                }
            }
            if (ok) ++produced;
        }
    }

    if (rng.uniform() < 0.35) {
        const auto& cu = synth_data::closers_user();
        const auto& ca = synth_data::closers_assistant();
        for (int t = 0; t < 8; ++t) {
            std::string u = pick(cu, rng);
            std::string a = pick(ca, rng);
            if (gated(u, a)) {
                out.messages.push_back({Role::User, surface(u)});
                out.messages.push_back({Role::Assistant, surface(a)});
                break;
            }
        }
    }

    out.template_id = tid;

    if (++impl_->template_uses[tid] > cfg_.max_template_uses) {
        ++stats_.rejected_duplicate;
        return false;
    }

    std::string flat;
    for (const auto& m : out.messages) {
        flat += (m.role == Role::User ? "U:" : m.role == Role::Assistant ? "A:" : "S:");
        flat += m.content;
        flat += "\n";
    }
    if (!impl_->dedup.add(flat)) {
        ++stats_.rejected_duplicate;
        return false;
    }

    for (size_t i = 0; i < out.messages.size(); ++i) {
        if (out.messages[i].role != Role::Assistant) continue;
        std::string user_msg = (i > 0) ? out.messages[i - 1].content : "";
        StyleFlags sf = check_assistant_style(out.messages[i].content, user_msg, impl_->lid);
        if (sf.has_boilerplate || sf.too_many_bullets) {
            ++stats_.rejected_style;
            return false;
        }
    }

    for (const auto& m : out.messages) {
        if (m.role != Role::Assistant) continue;
        std::string head = first_words(m.content, 2);
        int& c = impl_->first_token_counts[head];
        ++c;
        u64 total = impl_->total_accepted + 1;
        if (total > 200 && static_cast<double>(c) / static_cast<double>(total) > 0.06) {
            ++stats_.rejected_entropy;
            return false;
        }
    }

    ++impl_->total_accepted;
    ++stats_.accepted;
    stats_.by_domain[out.domain]++;
    stats_.by_script[script == Script::Arabic ? "arabic" : script == Script::Latin ? "latin" : "arabizi"]++;
    return true;
}

std::vector<Conversation> SynthGenerator::generate_many(int n) {
    GAI_CHECK(n >= 0, "synth conversation count must be >= 0");
    GAI_CHECK(cfg_.max_attempts_multiplier > 0, "synth attempts multiplier must be > 0");
    GAI_CHECK(cfg_.min_turns > 0 && cfg_.max_turns >= cfg_.min_turns &&
                  cfg_.max_turns <= 1024,
              "synth turn range is invalid");
    std::vector<Conversation> out;
    out.reserve(static_cast<size_t>(n));
    i64 attempts = 0;
    const i64 max_attempts =
        static_cast<i64>(n) * static_cast<i64>(cfg_.max_attempts_multiplier) + 5000;
    while (static_cast<i64>(out.size()) < n && attempts < max_attempts) {
        ++attempts;
        Conversation c;
        if (generate(c)) out.push_back(std::move(c));
    }
    if (static_cast<int>(out.size()) < n) {
        log_warn(strfmt("synth: produced %zu/%d conversations before the diversity filters "
                        "saturated (this is the guard working, not a bug)", out.size(), n));
    }
    return out;
}

static std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<u8>(c) < 0x20) o += strfmt("\\u%04x", c);
                else o.push_back(c);
        }
    }
    return o;
}

static std::string json_unescape(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\\' || i + 1 >= s.size()) { o.push_back(s[i]); continue; }
        char n = s[++i];
        switch (n) {
            case 'n': o.push_back('\n'); break;
            case 'r': o.push_back('\r'); break;
            case 't': o.push_back('\t'); break;
            case '"': o.push_back('"'); break;
            case '\\': o.push_back('\\'); break;
            case 'u': {
                auto is_hex = [](char ch) {
                    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
                           (ch >= 'A' && ch <= 'F');
                };
                if (i + 4 < s.size() && is_hex(s[i + 1]) && is_hex(s[i + 2]) &&
                    is_hex(s[i + 3]) && is_hex(s[i + 4])) {
                    u32 cp = 0;
                    for (int k = 1; k <= 4; ++k) {
                        char ch = s[i + static_cast<size_t>(k)];
                        cp <<= 4;
                        if (ch >= '0' && ch <= '9') cp |= static_cast<u32>(ch - '0');
                        else if (ch >= 'a' && ch <= 'f') cp |= static_cast<u32>(ch - 'a' + 10);
                        else cp |= static_cast<u32>(ch - 'A' + 10);
                    }
                    utf8_encode(cp, o);
                    i += 4;
                } else {
                    o.push_back(n);
                }
                break;
            }
            default: o.push_back(n);
        }
    }
    return o;
}

static const char* role_str(Role r) {
    return r == Role::System ? "system" : r == Role::User ? "user" : "assistant";
}

void write_conversations_jsonl(const std::string& path, const std::vector<Conversation>& convs) {
    std::ofstream f(path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot write conversations: " + path);
    for (const auto& c : convs) {
        f << "{\"domain\":\"" << json_escape(c.domain) << "\",\"messages\":[";
        for (size_t i = 0; i < c.messages.size(); ++i) {
            if (i) f << ",";
            f << "{\"role\":\"" << role_str(c.messages[i].role)
              << "\",\"content\":\"" << json_escape(c.messages[i].content) << "\"}";
        }
        f << "]}\n";
    }
}

std::vector<Conversation> read_conversations_jsonl(const std::string& path) {
    std::vector<Conversation> out;
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return out;

    JsonReaderOptions jopts;
    jopts.verbose = false;
    jopts.max_value_bytes = 16u << 20u;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        JsonDoc doc;
        if (!doc_from_json_text(line, doc, jopts) || !doc.is_chat || doc.messages.empty()) continue;
        Conversation c;
        c.messages = std::move(doc.messages);
        size_t dp = line.find("\"domain\":\"");
        if (dp != std::string::npos) {
            size_t b = dp + 10;
            size_t e = b;
            while (e < line.size() && !(line[e] == '"' && line[e - 1] != '\\')) ++e;
            c.domain = json_unescape(line.substr(b, e - b));
        }
        out.push_back(std::move(c));
    }
    return out;
}

}
