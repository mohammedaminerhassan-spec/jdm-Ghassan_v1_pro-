// dataset/json_reader.cpp — JSON/JSONL ingestion for training shards.
// See json_reader.h for the format contract. All parsing is bounded:
// nesting depth, member counts, value sizes and file sizes all have caps,
// so a corrupt or hostile file can only cause a skip, never a crash.

#include "dataset/json_reader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace gai {
namespace {

// ---------------------------------------------------------------- DOM
constexpr int   kMaxDepth = 32;
constexpr size_t kMaxMembers = 10000;
constexpr size_t kMaxArray = 10000000;
constexpr size_t kMaxFileBytes = size_t(1) << 31;   // 2 GiB single file cap

struct JVal {
    enum class T { Null, Bool, Num, Str, Arr, Obj } t = T::Null;
    bool b = false;
    double num = 0.0;
    std::string s;
    std::vector<JVal> a;
    std::vector<std::pair<std::string, JVal>> o;
};

struct Cur {
    const char* p;
    const char* end;
    bool eof() const { return p >= end; }
};

static void skip_ws(Cur& c) {
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
    } else if (cp <= 0x10FFFF) {
        o += static_cast<char>(0xF0 | (cp >> 18));
        o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        o += '\xEF'; o += '\xBF'; o += '\xBD';  // U+FFFD
    }
}

static bool hex4(const char* p, const char* end, unsigned& out) {
    if (end - p < 4) return false;
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
        char c = p[i];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
        else return false;
    }
    out = v;
    return true;
}

static bool parse_string(Cur& c, std::string& out, size_t cap) {
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
                    unsigned cp = 0;
                    if (!hex4(c.p, c.end, cp)) return false;
                    c.p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && c.end - c.p >= 6 &&
                        c.p[0] == '\\' && c.p[1] == 'u') {
                        unsigned lo = 0;
                        if (hex4(c.p + 2, c.end, lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            c.p += 6;
                        } else {
                            // Lone high surrogate + malformed \uXXXX: replacement.
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xD800 && cp <= 0xDFFF) {
                        // FIX: lone surrogate fell through to utf8_emit ->
                        // invalid 3-byte CESU-8 (training/inference garbage).
                        cp = 0xFFFD;
                    }
                    if (out.size() < cap) {
                        std::string tmp;
                        utf8_emit(tmp, cp);
                        if (out.size() + tmp.size() <= cap) out += tmp;
                    }
                    break;
                }
                default: return false;
            }
        } else {
            if (out.size() < cap) out += ch;  // over-cap bytes are dropped, parse continues
        }
    }
    return false;  // unterminated
}

static bool parse_value(Cur& c, JVal& out, size_t cap, int depth);

static bool parse_array(Cur& c, JVal& out, size_t cap, int depth) {
    ++c.p;  // [
    out.t = JVal::T::Arr;
    skip_ws(c);
    if (!c.eof() && *c.p == ']') { ++c.p; return true; }
    while (true) {
        if (out.a.size() >= kMaxArray) return false;
        JVal el;
        skip_ws(c);
        if (!parse_value(c, el, cap, depth)) return false;
        out.a.push_back(std::move(el));
        skip_ws(c);
        if (c.eof()) return false;
        if (*c.p == ',') { ++c.p; continue; }
        if (*c.p == ']') { ++c.p; return true; }
        return false;
    }
}

static bool parse_object(Cur& c, JVal& out, size_t cap, int depth) {
    ++c.p;  // {
    out.t = JVal::T::Obj;
    skip_ws(c);
    if (!c.eof() && *c.p == '}') { ++c.p; return true; }
    while (true) {
        if (out.o.size() >= kMaxMembers) return false;
        std::string key;
        skip_ws(c);
        if (!parse_string(c, key, 512)) return false;  // keys are short
        skip_ws(c);
        if (c.eof() || *c.p != ':') return false;
        ++c.p;
        JVal v;
        skip_ws(c);
        if (!parse_value(c, v, cap, depth)) return false;
        out.o.emplace_back(std::move(key), std::move(v));
        skip_ws(c);
        if (c.eof()) return false;
        if (*c.p == ',') { ++c.p; continue; }
        if (*c.p == '}') { ++c.p; return true; }
        return false;
    }
}

static bool parse_value(Cur& c, JVal& out, size_t cap, int depth) {
    if (depth > kMaxDepth) return false;
    skip_ws(c);
    if (c.eof()) return false;
    char ch = *c.p;
    if (ch == '"') { out.t = JVal::T::Str; return parse_string(c, out.s, cap); }
    if (ch == '{') return parse_object(c, out, cap, depth + 1);
    if (ch == '[') return parse_array(c, out, cap, depth + 1);
    if (ch == 't') {
        if (c.end - c.p >= 4 && c.p[1] == 'r' && c.p[2] == 'u' && c.p[3] == 'e') {
            out.t = JVal::T::Bool; out.b = true; c.p += 4; return true;
        }
        return false;
    }
    if (ch == 'f') {
        if (c.end - c.p >= 5 && c.p[1] == 'a' && c.p[2] == 'l' && c.p[3] == 's' && c.p[4] == 'e') {
            out.t = JVal::T::Bool; out.b = false; c.p += 5; return true;
        }
        return false;
    }
    if (ch == 'n') {
        if (c.end - c.p >= 4 && c.p[1] == 'u' && c.p[2] == 'l' && c.p[3] == 'l') {
            out.t = JVal::T::Null; c.p += 4; return true;
        }
        return false;
    }
    // number: validate shape, keep numeric value (ids etc. don't need it)
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
        const char* s = c.p;
        if (*c.p == '-') ++c.p;
        if (c.eof()) return false;
        if (*c.p == '0') { ++c.p; }
        else if (*c.p >= '1' && *c.p <= '9') { while (!c.eof() && *c.p >= '0' && *c.p <= '9') ++c.p; }
        else return false;
        if (!c.eof() && *c.p == '.') {
            ++c.p;
            if (c.eof() || *c.p < '0' || *c.p > '9') return false;
            while (!c.eof() && *c.p >= '0' && *c.p <= '9') ++c.p;
        }
        if (!c.eof() && (*c.p == 'e' || *c.p == 'E')) {
            ++c.p;
            if (!c.eof() && (*c.p == '+' || *c.p == '-')) ++c.p;
            if (c.eof() || *c.p < '0' || *c.p > '9') return false;
            while (!c.eof() && *c.p >= '0' && *c.p <= '9') ++c.p;
        }
        out.t = JVal::T::Num;
        try { out.num = std::stod(std::string(s, c.p)); } catch (...) { out.num = 0.0; }
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- schema mapping
static const JVal* find_key(const JVal& obj, const std::string& key) {
    if (obj.t != JVal::T::Obj) return nullptr;
    for (const auto& kv : obj.o)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

static bool as_str(const JVal* v, std::string& out) {
    if (!v || v->t != JVal::T::Str || v->s.empty()) return false;
    out = v->s;
    return true;
}

static Role map_role(const std::string& r) {
    std::string l = r;
    for (char& ch : l) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (l == "system" || l == "developer") return Role::System;
    if (l == "user" || l == "human" || l == "question" || l == "instruction" ||
        l == "prompt" || l == "problem" || l == "query" || l == "input") return Role::User;
    return Role::Assistant;  // assistant/gpt/ai/answer/response/...
}

static bool extract_message(const JVal& obj, Message& m) {
    const JVal* role = find_key(obj, "role");
    if (!role) role = find_key(obj, "from");
    if (!role) role = find_key(obj, "speaker");
    if (!role) role = find_key(obj, "author");
    const JVal* text = find_key(obj, "content");
    if (!text) text = find_key(obj, "value");
    if (!text) text = find_key(obj, "text");
    if (!text) text = find_key(obj, "message");
    if (!text) text = find_key(obj, "utterance");
    std::string body;
    if (!as_str(text, body)) return false;
    std::string r;
    if (role && role->t == JVal::T::Str) r = role->s;
    m.role = r.empty() ? Role::User : map_role(r);
    m.content = std::move(body);
    return true;
}

// Returns true when `obj` produced a document.
static bool object_to_doc(const JVal& obj, const JsonReaderOptions& opts, JsonDoc& doc) {
    if (obj.t != JVal::T::Obj) return false;

    // 1. chat: messages / conversation / turns (+ "conversations" handled
    //    by the caller: each element is one doc, see array handling)
    for (const char* k : {"messages", "conversation", "turns", "dialogue", "history"}) {
        const JVal* arr = find_key(obj, k);
        if (arr && arr->t == JVal::T::Arr && !arr->a.empty()) {
            JsonDoc d;
            d.is_chat = true;
            for (const auto& el : arr->a) {
                if (el.t == JVal::T::Str) {
                    if (!el.s.empty()) d.messages.push_back({Role::User, el.s});
                } else if (el.t == JVal::T::Obj) {
                    Message m;
                    if (extract_message(el, m) && !m.content.empty())
                        d.messages.push_back(std::move(m));
                }
                if (d.messages.size() >= 128) break;  // absurd turn count cap
            }
            if (d.messages.empty()) return false;
            doc = std::move(d);
            return true;
        }
    }

    // 2. instruction (+ input) / output
    {
        std::string instr, input, output, system;
        bool has_instr = as_str(find_key(obj, "instruction"), instr);
        if (!has_instr) has_instr = as_str(find_key(obj, "prompt"), instr);
        if (!has_instr) has_instr = as_str(find_key(obj, "question"), instr);
        if (!has_instr) has_instr = as_str(find_key(obj, "problem"), instr);
        as_str(find_key(obj, "input"), input);
        bool has_out = as_str(find_key(obj, "output"), output);
        if (!has_out) has_out = as_str(find_key(obj, "completion"), output);
        if (!has_out) has_out = as_str(find_key(obj, "answer"), output);
        if (!has_out) has_out = as_str(find_key(obj, "response"), output);
        if (!has_out) has_out = as_str(find_key(obj, "solution"), output);
        as_str(find_key(obj, "system"), system);
        // NOTE: bare {id,question,answer} retrieval shards also match here;
        // that is intentional: they train fine as user->assistant turns.
        if (has_instr && has_out) {
            JsonDoc d;
            d.is_chat = true;
            if (!system.empty()) d.messages.push_back({Role::System, system});
            std::string u = instr;
            if (!input.empty()) { u += "\n"; u += input; }
            d.messages.push_back({Role::User, std::move(u)});
            d.messages.push_back({Role::Assistant, std::move(output)});
            doc = std::move(d);
            return true;
        }
    }

    // 3. plain text: first matching text key wins
    for (const auto& k : opts.text_keys) {
        std::string t;
        if (as_str(find_key(obj, k), t)) {
            JsonDoc d;
            d.is_chat = false;
            d.text = std::move(t);
            doc = std::move(d);
            return true;
        }
    }
    return false;
}

static std::string strip_bom(const std::string& s) {
    if (s.size() >= 3 && s[0] == '\xEF' && s[1] == '\xBB' && s[2] == '\xBF')
        return s.substr(3);
    return s;
}

// Shared single-text entry: parse one JSON object text -> JsonDoc.
// Used by doc_from_json_text (parquet chat mode) so both routes share one
// schema mapping and one set of caps. Returns false on malformed text.
static bool parse_text_to_doc(const std::string& text, const JsonReaderOptions& opts,
                              JsonDoc& doc) {
    if (text.empty() || text.size() > kMaxFileBytes) return false;
    Cur c{text.data(), text.data() + text.size()};
    JVal v;
    if (!parse_value(c, v, opts.max_value_bytes, 0)) return false;
    skip_ws(c);
    if (!c.eof()) return false;
    if (v.t != JVal::T::Obj) return false;
    return object_to_doc(v, opts, doc);
}

} // namespace

bool doc_from_json_text(const std::string& text, JsonDoc& doc,
                        const JsonReaderOptions& opts) {
    return parse_text_to_doc(text, opts, doc);
}

size_t read_json_docs(const std::string& path, JsonDocCallback cb,
                      const JsonReaderOptions& opts) {
    std::error_code ec;
    size_t fsize = fs::file_size(path, ec);
    if (ec || fsize == 0) {
        if (opts.verbose) log_warn("json: cannot stat: " + path);
        return 0;
    }
    // PRO-EN FIX: البوابة القديمة كانت ترفض أي ملف >2GiB قبل حتى معرفة نوعه،
    // فيرجع ملف chat_if.jsonl (7GB) صفر وثيقة بصمت تام. JSONL يُقرأ سطرا بسطر
    // (ذاكرة ثابتة) فلا يحتاج السقف؛ السقف فقط لمصفوفات JSON الكاملة التي
    // تُحمّل دفعة واحدة. نؤجل الفحص لمسار المصفوفة أدناه.
    const bool over_cap = (fsize > kMaxFileBytes);

    size_t delivered = 0;
    auto emit = [&](const JVal& v) {
        if (opts.max_docs > 0 && delivered >= opts.max_docs) return;
        if (v.t == JVal::T::Obj) {
            JsonDoc d;
            if (object_to_doc(v, opts, d)) { cb(d); ++delivered; }
        } else if (v.t == JVal::T::Arr) {
            // {"conversations":[...]} / {"data":[...]} wrappers
            for (const auto& el : v.a) {
                if (opts.max_docs > 0 && delivered >= opts.max_docs) break;
                if (el.t != JVal::T::Obj) continue;
                JsonDoc d;
                if (object_to_doc(el, opts, d)) { cb(d); ++delivered; }
            }
        }
    };

    // Peek at the first non-ws byte to choose: array vs jsonl/single.
    {
        std::ifstream peek(path, std::ios::binary);
        if (!peek.good()) {
            if (opts.verbose) log_warn("json: cannot open: " + path);
            return 0;
        }
        char ch = 0;
        bool found = false;
        while (peek.get(ch)) {
            if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' &&
                ch != '\xEF' && ch != '\xBB' && ch != '\xBF') { found = true; break; }
        }
        if (!found) return 0;
        if (ch == '[') {
            // Top-level array: load whole file (bounded) and walk elements.
            if (over_cap) {
                log_warn("json: array file over 2GiB, refusing (would need 2GB+ RAM): " + path);
                return 0;
            }
            std::ifstream f(path, std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            text = strip_bom(text);
            Cur c{text.data(), text.data() + text.size()};
            JVal root;
            if (!parse_value(c, root, opts.max_value_bytes, 0) || root.t != JVal::T::Arr) {
                log_warn("json: cannot parse array: " + path);
                return 0;
            }
            for (const auto& el : root.a) {
                if (opts.max_docs > 0 && delivered >= opts.max_docs) break;
                if (el.t == JVal::T::Obj) {
                    JsonDoc d;
                    if (object_to_doc(el, opts, d)) { cb(d); ++delivered; }
                }
            }
            return delivered;
        }
    }

    // JSONL (also covers a single pretty object on one... no: multi-line
    // pretty JSON would split. Handle it: accumulate lines until braces
    // balance, then parse the chunk as one value).
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        if (opts.verbose) log_warn("json: cannot open: " + path);
        return 0;
    }
    std::string line, chunk;
    int depth = 0;
    bool in_str = false, esc = false;
    u64 line_no = 0, bad = 0;
    auto flush_chunk = [&]() {
        if (chunk.empty()) return;
        Cur c{chunk.data(), chunk.data() + chunk.size()};
        JVal v;
        if (parse_value(c, v, opts.max_value_bytes, 0)) {
            skip_ws(c);
            if (c.eof()) { emit(v); return; }
        }
        if (++bad <= 5 && opts.verbose)
            log_warn(strfmt("json: skipping malformed object near line %llu in %s",
                            (unsigned long long)line_no, path.c_str()));
    };
    while (std::getline(f, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (chunk.empty()) {
            std::string t = line;
            size_t i = 0;
            while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
            if (i >= t.size()) continue;  // blank line
            if (t[i] != '{' && t[i] != '[') {
                if (++bad <= 5 && opts.verbose)
                    log_warn(strfmt("json: skipping non-object line %llu in %s",
                                    (unsigned long long)line_no, path.c_str()));
                continue;
            }
        }
        if (!chunk.empty()) chunk += '\n';
        chunk += line;
        if (chunk.size() > kMaxFileBytes) { flush_chunk(); chunk.clear(); depth = 0; in_str = false; continue; }
        for (char ch : line) {
            if (in_str) {
                if (esc) esc = false;
                else if (ch == '\\') esc = true;
                else if (ch == '"') in_str = false;
            } else {
                if (ch == '"') in_str = true;
                else if (ch == '{' || ch == '[') ++depth;
                else if (ch == '}' || ch == ']') --depth;
            }
        }
        if (depth <= 0 && !in_str) { flush_chunk(); chunk.clear(); depth = 0; }
        if (opts.max_docs > 0 && delivered >= opts.max_docs) break;
    }
    if (!chunk.empty() && depth <= 0) flush_chunk();
    else if (!chunk.empty() && opts.verbose) log_warn("json: truncated object at EOF in " + path);
    if (bad > 5 && opts.verbose)
        log_warn(strfmt("json: ... +%llu more bad lines in %s",
                        (unsigned long long)(bad - 5), path.c_str()));
    return delivered;
}

size_t read_json_dir(const std::string& dir, JsonDocCallback cb,
                     const JsonReaderOptions& opts) {
    size_t total = 0;
    std::error_code ec;
    if (fs::is_regular_file(dir, ec)) return read_json_docs(dir, cb, opts);
    if (!fs::exists(dir, ec)) {
        if (opts.verbose) log_warn("json: path not found: " + dir);
        return 0;
    }
    std::vector<std::string> files;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        for (char& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ext == ".json" || ext == ".jsonl") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (opts.verbose)
        log_info(strfmt("[json] found %zu file(s) in %s", files.size(), dir.c_str()));
    for (const auto& p : files) {
        size_t n = read_json_docs(p, cb, opts);
        if (opts.verbose && n > 0)
            log_info(strfmt("  %-60s %s docs", fs::path(p).filename().string().c_str(),
                            human_count(n).c_str()));
        total += n;
        if (opts.max_docs > 0 && total >= opts.max_docs) break;
    }
    return total;
}

std::vector<std::string> load_json_texts(const std::string& path,
                                         const JsonReaderOptions& opts) {
    std::vector<std::string> out;
    read_json_docs(path, [&](const JsonDoc& d) {
        if (d.is_chat) {
            std::string t;
            for (const auto& m : d.messages) {
                if (!t.empty()) t += '\n';
                t += m.content;
            }
            if (!t.empty()) out.push_back(std::move(t));
        } else if (!d.text.empty()) {
            out.push_back(d.text);
        }
    }, opts);
    return out;
}

void inspect_json(const std::string& path) {
    std::error_code ec;
    size_t fsize = fs::exists(path, ec) ? fs::file_size(path, ec) : 0;
    log_info(strfmt("[json inspect] %s (%s)", path.c_str(), human_bytes(fsize).c_str()));
    JsonReaderOptions opts;
    opts.verbose = false;
    size_t chat = 0, text = 0, longest = 0;
    std::string longest_from;
    read_json_docs(path, [&](const JsonDoc& d) {
        if (d.is_chat) {
            ++chat;
            size_t n = 0;
            for (const auto& m : d.messages) n += m.content.size();
            if (n > longest) { longest = n; longest_from = "chat"; }
        } else {
            ++text;
            if (d.text.size() > longest) { longest = d.text.size(); longest_from = "text"; }
        }
    }, opts);
    log_info(strfmt("  chat docs : %s", human_count(chat).c_str()));
    log_info(strfmt("  text docs : %s", human_count(text).c_str()));
    log_info(strfmt("  longest   : %s bytes (%s)", human_count(longest).c_str(), longest_from.c_str()));
}

} // namespace gai
