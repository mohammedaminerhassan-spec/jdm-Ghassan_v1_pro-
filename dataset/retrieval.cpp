#include "dataset/retrieval.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace gai {

std::string retrieval_normalize(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    bool space = true; // collapse + trim
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 'A' && c <= 'Z') {
            o += static_cast<char>(c - 'A' + 'a');
            space = false;
            ++i;
        } else if ((c >= 'a' && c <= 'z') || c >= 0x80) {
            // a-z and UTF-8 bytes kept (Arabic). Multi-byte UTF-8 passes
            // through byte by byte; we never split or reorder bytes.
            o += s[i];
            space = false;
            ++i;
        } else if (c == 0xD9 && i + 1 < s.size()) {
            // Strip Arabic diacritics (tashkeel U+064B..U+0652, D9 8B..D9 92):
            // "مَرحَبًا" == "مرحبا". Keeps letters, drops marks.
            unsigned char c2 = static_cast<unsigned char>(s[i + 1]);
            if (c2 >= 0x8B && c2 <= 0x92) { i += 2; continue; }
            o += s[i];
            space = false;
            ++i;
        } else if (c >= '0' && c <= '9') {
            // Arabizi digits -> latin letters so "3likom" == "alikom",
            // "7al" == "hal", "9ahwa" == "qahwa". This is the paraphrase
            // bridge the user asked for: different spelling, same intent.
            if (c == '3') o += 'a';
            else if (c == '7') o += 'h';
            else if (c == '9') o += 'q';
            else if (c == '5') o += 'k'; // kh -> k (close enough for match)
            else if (c == '2') o += 'a'; // hamza -> a
            else o += s[i];
            space = false;
            ++i;
        } else {
            if (!space) o += ' ';
            space = true;
            ++i;
        }
    }
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o;
}

std::vector<std::string> retrieval_tokenize(const std::string& s) {
    std::vector<std::string> out;
    std::string norm = retrieval_normalize(s);
    size_t i = 0;
    while (i < norm.size()) {
        while (i < norm.size() && norm[i] == ' ') ++i;
        size_t j = i;
        while (j < norm.size() && norm[j] != ' ') ++j;
        if (j > i) out.push_back(norm.substr(i, j - i));
        i = j;
    }
    return out;
}

namespace {

// ---- minimal JSON reader for our shards: array of flat objects whose values
// are strings or integers. Handles \" \\ \/ \b \f \n \r \t \uXXXX (with
// surrogate pairs). Anything else (nested values) is skipped, never fatal.

struct Cursor {
    const char* p;
    const char* end;
    bool eof() const { return p >= end; }
};

static void skip_ws(Cursor& c) {
    while (!c.eof() && (*c.p == ' ' || *c.p == '\t' || *c.p == '\n' || *c.p == '\r')) ++c.p;
}

static void utf8_emit(std::string& o, unsigned cp) {
    if (cp < 0x80) {
        o += static_cast<char>(cp);
    } else if (cp < 0x800) {
        o += static_cast<char>(0xC0 | (cp >> 6));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        o += static_cast<char>(0xE0 | (cp >> 12));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        o += static_cast<char>(0xF0 | (cp >> 18));
        o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

static unsigned hex4(const char* p) {
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
        else return 0xFFFFFFFFu;
    }
    return v;
}

// Parses a JSON string starting at the opening quote. Returns false on error.
static bool parse_string(Cursor& c, std::string& out) {
    if (c.eof() || *c.p != '"') return false;
    ++c.p;
    out.clear();
    while (!c.eof()) {
        char ch = *c.p++;
        if (ch == '"') return true;
        if (ch == '\\') {
            if (c.eof()) return false;
            char e = *c.p++;
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (c.end - c.p < 4) return false;
                    unsigned cp = hex4(c.p);
                    if (cp == 0xFFFFFFFFu) return false;
                    c.p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && c.end - c.p >= 6 &&
                        c.p[0] == '\\' && c.p[1] == 'u') {
                        unsigned lo = hex4(c.p + 2);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            c.p += 6;
                        }
                    }
                    utf8_emit(out, cp);
                    break;
                }
                default: return false;
            }
        } else {
            out += ch;
        }
    }
    return false;
}

static bool parse_int(Cursor& c, long long& out) {
    skip_ws(c);
    bool neg = false;
    if (!c.eof() && *c.p == '-') { neg = true; ++c.p; }
    if (c.eof() || *c.p < '0' || *c.p > '9') return false;
    long long v = 0;
    while (!c.eof() && *c.p >= '0' && *c.p <= '9') v = v * 10 + (*c.p++ - '0');
    out = neg ? -v : v;
    return true;
}

// Skips one JSON value of any shape (used for unknown keys).
static bool skip_value(Cursor& c) {
    skip_ws(c);
    if (c.eof()) return false;
    if (*c.p == '"') {
        std::string tmp;
        return parse_string(c, tmp);
    }
    if (*c.p == '{') {
        ++c.p;
        skip_ws(c);
        if (!c.eof() && *c.p == '}') { ++c.p; return true; }
        while (true) {
            std::string k;
            skip_ws(c);
            if (!parse_string(c, k)) return false;
            skip_ws(c);
            if (c.eof() || *c.p != ':') return false;
            ++c.p;
            if (!skip_value(c)) return false;
            skip_ws(c);
            if (c.eof()) return false;
            if (*c.p == ',') { ++c.p; continue; }
            if (*c.p == '}') { ++c.p; return true; }
            return false;
        }
    }
    if (*c.p == '[') {
        ++c.p;
        skip_ws(c);
        if (!c.eof() && *c.p == ']') { ++c.p; return true; }
        while (true) {
            if (!skip_value(c)) return false;
            skip_ws(c);
            if (c.eof()) return false;
            if (*c.p == ',') { ++c.p; continue; }
            if (*c.p == ']') { ++c.p; return true; }
            return false;
        }
    }
    // number / true / false / null: consume until delimiter
    while (!c.eof() && *c.p != ',' && *c.p != ']' && *c.p != '}' &&
           *c.p != ' ' && *c.p != '\t' && *c.p != '\n' && *c.p != '\r')
        ++c.p;
    return true;
}

} // namespace

bool load_qa_json(const std::string& path, std::vector<QaEntry>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (text.empty()) return false;
    Cursor c{text.data(), text.data() + text.size()};
    skip_ws(c);
    if (c.eof() || *c.p != '[') return false;
    ++c.p;
    size_t loaded = 0;
    while (true) {
        skip_ws(c);
        if (c.eof()) return false;
        if (*c.p == ']') { ++c.p; break; }
        if (*c.p != '{') return false;
        ++c.p;
        QaEntry e;
        bool has_q = false, has_a = false, has_id = false;
        while (true) {
            skip_ws(c);
            if (c.eof()) return false;
            if (*c.p == '}') { ++c.p; break; }
            std::string key;
            if (!parse_string(c, key)) return false;
            skip_ws(c);
            if (c.eof() || *c.p != ':') return false;
            ++c.p;
            skip_ws(c);
            if (key == "question") {
                if (!parse_string(c, e.question)) return false;
                has_q = true;
            } else if (key == "answer") {
                if (!parse_string(c, e.answer)) return false;
                has_a = true;
            } else if (key == "id") {
                if (!parse_int(c, e.id)) return false;
                has_id = true;
            } else {
                if (!skip_value(c)) return false;
            }
            skip_ws(c);
            if (c.eof()) return false;
            if (*c.p == ',') { ++c.p; continue; }
            if (*c.p == '}') { ++c.p; break; }
            return false;
        }
        if (has_q && has_a) {
            if (!has_id) e.id = static_cast<long long>(out.size() + loaded);
            out.push_back(std::move(e));
            ++loaded;
        }
        skip_ws(c);
        if (c.eof()) return false;
        if (*c.p == ',') { ++c.p; continue; }
        if (*c.p == ']') { ++c.p; break; }
        return false;
    }
    return true;
}

size_t load_qa_dir(const std::string& dir, std::vector<QaEntry>& out) {
    size_t before = out.size();
    std::error_code ec;
    if (!fs::exists(dir, ec)) return 0;
    std::vector<std::string> files;
    auto push_from = [&](const std::string& d) {
        for (const auto& e : fs::recursive_directory_iterator(d, ec)) {
            if (!e.is_regular_file()) continue;
            if (e.path().extension() == ".json") files.push_back(e.path().string());
        }
    };
    if (fs::is_regular_file(dir, ec)) {
        files.push_back(dir);
    } else {
        push_from(dir);
    }
    std::sort(files.begin(), files.end());
    for (const auto& p : files) {
        if (!load_qa_json(p, out)) log_warn("retrieval: cannot parse " + p);
    }
    return out.size() - before;
}

void RetrievalIndex::build(const std::vector<QaEntry>& docs) {
    // Deduplicate by normalized question: the aggregate all_darija file plus
    // the per-category files would otherwise double the index and bias IDF.
    std::map<std::string, size_t> first_seen;
    std::vector<QaEntry> uniq;
    uniq.reserve(docs.size());
    for (const auto& d : docs) {
        std::string nq = retrieval_normalize(d.question);
        if (nq.empty()) continue;
        auto it = first_seen.find(nq);
        if (it != first_seen.end()) continue;
        first_seen[nq] = uniq.size();
        uniq.push_back(d);
    }
    docs_ = std::move(uniq);
    toks_.clear();
    norm_q_.clear();
    doc_len_.clear();
    idf_.clear();
    inv_.clear();
    toks_.reserve(docs_.size());
    norm_q_.reserve(docs_.size());
    std::map<std::string, size_t> df;
    for (const auto& d : docs_) {
        auto t = retrieval_tokenize(d.question);
        toks_.push_back(t);
        norm_q_.push_back(retrieval_normalize(d.question));
        doc_len_.push_back(static_cast<int>(t.size()));
        std::map<std::string, bool> seen;
        for (const auto& w : t) {
            if (!seen[w]) { seen[w] = true; df[w]++; }
        }
    }
    const double N = docs_.empty() ? 1.0 : static_cast<double>(docs_.size());
    for (const auto& [w, n] : df) idf_[w] = std::log((N + 1.0) / (n + 1.0)) + 1.0;
    for (size_t i = 0; i < toks_.size(); ++i) {
        std::map<std::string, bool> seen;
        for (const auto& w : toks_[i]) {
            if (!seen[w]) { seen[w] = true; inv_[w].push_back(i); }
        }
    }
}

static int lev1(const std::string& a, const std::string& b, int maxd) {
    int n = (int)a.size(), m = (int)b.size();
    if (std::abs(n - m) > maxd) return maxd + 1;
    // Hard bound: the DP table is fixed size. Long tokens never match
    // fuzzily (they fall through to the trigram rescue below).
    if (n > 15 || m > 15 || n <= 0 || m <= 0) return maxd + 1;
    int dp[16][16];
    for (int i = 0; i <= n; ++i) dp[i][0] = i;
    for (int j = 0; j <= m; ++j) dp[0][j] = j;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost});
        }
    }
    return dp[n][m];
}

std::vector<RetrievalHit> RetrievalIndex::query(const std::string& text, int top_k,
                                               double min_score) const {
    std::vector<RetrievalHit> out;
    if (docs_.empty() || top_k <= 0) return out;
    auto qt = retrieval_tokenize(text);
    if (qt.empty()) return out;
    std::string nq = retrieval_normalize(text);
    // BM25 lengths (avgdl over the index; N is small, linear scan is fine)
    double avgdl = 1.0;
    {
        double tot = 0.0;
        for (int l : doc_len_) tot += l;
        if (!doc_len_.empty()) avgdl = tot / doc_len_.size();
        if (avgdl < 1.0) avgdl = 1.0;
    }
    constexpr double k1 = 1.2, b = 0.75;
    std::map<size_t, double> acc;
    // query-term frequency (repeated words count, e.g. "salam salam")
    std::map<std::string, int> qtf;
    for (const auto& w : qt) qtf[w]++;
    for (const auto& [w, qf] : qtf) {
        auto itw = idf_.find(w);
        std::string use = w;
        double wpen = 1.0;
        if (itw == idf_.end()) {
            // Fuzzy: bokhir~bikhir, labas~labass, kolxi~kolchi (1-2 edits)
            int maxd = (int)w.size() <= 4 ? 1 : 2;
            double best_idf = 0;
            std::string best;
            for (const auto& kv : idf_) {
                int d = lev1(w, kv.first, maxd);
                if (d <= maxd && kv.second > best_idf) { best_idf = kv.second; best = kv.first; }
            }
            if (best.empty()) continue;
            use = best;
            wpen = 0.75; // penalty for fuzzy
            itw = idf_.find(use);
            if (itw == idf_.end()) continue;
        }
        auto iti = inv_.find(use);
        if (iti == inv_.end()) continue;
        double idf = itw->second;
        for (size_t d : iti->second) {
            // tf of `use` inside doc d (docs are short QA questions)
            int tf = 0;
            for (const auto& dw : toks_[d]) if (dw == use) ++tf;
            if (tf <= 0) continue;
            double dl = static_cast<double>(doc_len_[d]);
            double denom = tf + k1 * (1.0 - b + b * dl / avgdl);
            double s = idf * (tf * (k1 + 1.0) / denom);
            acc[d] += wpen * s * (1.0 + 0.1 * (qf - 1));  // slight qtf boost
        }
    }
    // Trigram rescue: when NO token matched at all (heavy paraphrase or
    // typos in every word), fall back to character-3-gram Dice overlap so
    // "kifach nsayb" still finds "kifach nsawb". Bounded and deterministic.
    if (acc.empty() && !nq.empty()) {
        auto trigrams = [](const std::string& s) {
            std::map<std::string, int> m;
            std::string t = " " + s + " ";
            for (size_t i = 0; i + 3 <= t.size(); ++i) m[t.substr(i, 3)]++;
            return m;
        };
        auto qg = trigrams(nq);
        size_t qn = 0;
        for (const auto& kv : qg) qn += static_cast<size_t>(kv.second);
        if (qn > 0) {
            for (size_t d = 0; d < norm_q_.size(); ++d) {
                auto dg = trigrams(norm_q_[d]);
                size_t dn = 0, inter = 0;
                for (const auto& kv : dg) dn += static_cast<size_t>(kv.second);
                for (const auto& kv : qg) {
                    auto it = dg.find(kv.first);
                    if (it != dg.end()) inter += static_cast<size_t>(std::min(kv.second, it->second));
                }
                if (inter == 0) continue;
                double dice = 2.0 * inter / (qn + dn);
                if (dice >= 0.25) acc[d] = dice * 2.0;  // rescue weight, below real BM25 hits
            }
        }
    }
    std::vector<RetrievalHit> cand;
    cand.reserve(acc.size());
    for (const auto& [d, s] : acc) {
        double score = s;
        // bigram overlap bonus: rewards word order ("chno smitk" vs "smitk chno")
        if (qt.size() >= 2 && toks_[d].size() >= 2) {
            int shared = 0;
            for (size_t i = 0; i + 1 < qt.size(); ++i)
                for (size_t j = 0; j + 1 < toks_[d].size(); ++j)
                    if (qt[i] == toks_[d][j] && qt[i + 1] == toks_[d][j + 1]) { ++shared; break; }
            score += 0.5 * shared;
        }
        if (!nq.empty() && norm_q_[d].find(nq) != std::string::npos) score += 5.0;
        if (score >= min_score) cand.push_back({d, score});
    }
    if (cand.empty()) return out;
    if ((int)cand.size() > top_k) {
        std::nth_element(cand.begin(), cand.begin() + top_k, cand.end(),
                         [](const RetrievalHit& a, const RetrievalHit& b) {
                             return a.score > b.score;
                         });
        cand.resize(static_cast<size_t>(top_k));
    }
    std::sort(cand.begin(), cand.end(),
              [](const RetrievalHit& a, const RetrievalHit& b) { return a.score > b.score; });
    return cand;
}

} // namespace gai
