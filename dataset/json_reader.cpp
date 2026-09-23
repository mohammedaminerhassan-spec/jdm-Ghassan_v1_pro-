// dataset/json_reader.cpp — PARQUET-ONLY: minimal messages_json parser.
//
// File-based JSON ingestion (read_json_docs / read_json_dir / ...) was
// REMOVED. The prebuilt English lake is read directly from
// english_parquet/* via dataset/parquet_reader.h. This file keeps the bounded DOM +
// schema mapping needed to parse ONE {"messages":[...]} text from the
// messages_json parquet column. Malformed rows -> false (skip, never crash).

#include "dataset/json_reader.h"

#include <algorithm>
#include <cctype>

namespace gai {
namespace {

// ---------------------------------------------------------------- DOM
constexpr int   kMaxDepth = 32;
constexpr size_t kMaxMembers = 10000;
constexpr size_t kMaxArray = 10000000;
constexpr size_t kMaxFileBytes = size_t(1) << 31;   // 2 GiB single-text cap

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
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xD800 && cp <= 0xDFFF) {
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
            if (out.size() < cap) out += ch;
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
        if (!parse_string(c, key, 512)) return false;
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
    return Role::Assistant;
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

static bool object_to_doc(const JVal& obj, const JsonReaderOptions& opts, JsonDoc& doc) {
    if (obj.t != JVal::T::Obj) return false;

    // parquet messages_json shape: {"messages":[{role,content},...]}
    // (+ "conversation"/"turns"/"dialogue"/"history" aliases, kept for lake compat)
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
                if (d.messages.size() >= 128) break;
            }
            if (d.messages.empty()) return false;
            doc = std::move(d);
            return true;
        }
    }

    // instruction (+ input) / output (kept: some lake rows use this shape)
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

    // plain text fallback
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

} // namespace gai
