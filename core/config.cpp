#include "core/config.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace gai {

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

static std::string strip_comment(const std::string& line) {
    bool in_s = false, in_d = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '\'' && !in_d) in_s = !in_s;
        else if (c == '"' && !in_s) in_d = !in_d;
        else if (c == '#' && !in_s && !in_d) {
            if (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t') return line.substr(0, i);
        }
    }
    return line;
}

Config Config::from_string(const std::string& text) {
    Config cfg;
    std::istringstream in(text);
    std::string line;
    // stack of (indent, key)
    std::vector<std::pair<int, std::string>> stack;
    std::string list_key;
    int         list_indent = -1;
    int         lineno = 0;

    while (std::getline(in, line)) {
        ++lineno;
        std::string raw = strip_comment(line);
        if (trim(raw).empty()) continue;

        int indent = 0;
        while (indent < static_cast<int>(raw.size()) && (raw[static_cast<size_t>(indent)] == ' ')) ++indent;
        // FIX: tab-indented YAML silently produced indent=0 -> wrong dotted
        // keys -> wrong hparams with no error (training misconfig). Tabs are
        // never valid YAML indentation: fail fast with line number + content.
        if (static_cast<size_t>(indent) < raw.size() && raw[static_cast<size_t>(indent)] == '\t')
            GAI_FAIL(strfmt("config:%d: tab indentation is not allowed (use spaces): %s",
                            lineno, trim(raw).c_str()));
        std::string body = trim(raw);
        // Minimal-subset guard: anchors/aliases/tags/multiline blocks are NOT
        // supported by this parser. Refuse loudly with the line number instead
        // of silently misreading them into wrong hparams (training killer).
        // Covered: `&anchor`, `*alias`, `<<: *merge`, `!tag`, `|`, `>` blocks.
        if (!body.empty() && (body[0] == '&' || body[0] == '*' || body[0] == '!' ||
                              body == "|" || body == ">" || body == "|-" || body == ">-")) {
            GAI_FAIL(strfmt("config:%d: unsupported YAML construct (anchors/aliases/tags/blocks need plain scalars): %s",
                            lineno, body.c_str()));
        }
        if (body.rfind("<<:", 0) == 0) {
            GAI_FAIL(strfmt("config:%d: unsupported YAML merge key '<<:' (use plain maps): %s",
                            lineno, body.c_str()));
        }

        // list item ("- value") belonging to the last key
        if (body.rfind("- ", 0) == 0 || body == "-") {
            if (!list_key.empty()) {
                std::string item = strip_quotes(trim(body.size() > 1 ? body.substr(1) : ""));
                auto it = cfg.kv_.find(list_key);
                if (it == cfg.kv_.end() || it->second.empty()) cfg.kv_[list_key] = item;
                else it->second += "\n" + item;
            }
            continue;
        }

        while (!stack.empty() && stack.back().first >= indent) stack.pop_back();

        size_t colon = std::string::npos;
        {
            bool in_s = false, in_d = false;
            for (size_t i = 0; i < body.size(); ++i) {
                char c = body[i];
                if (c == '\'' && !in_d) in_s = !in_s;
                else if (c == '"' && !in_s) in_d = !in_d;
                else if (c == ':' && !in_s && !in_d) { colon = i; break; }
            }
        }
        // DeepSeek rule: a non-empty, non-list line without ':' is never
        // valid YAML — silently dropping `vocab_size 16000` trains the wrong
        // model. Fail fast with line number.
        if (colon == std::string::npos) {
            if (!body.empty())
                GAI_FAIL(strfmt("config:%d: missing ':' (want 'key: value'): %s",
                                lineno, body.c_str()));
            continue;
        }

        std::string key = trim(body.substr(0, colon));
        std::string val = trim(body.substr(colon + 1));
        // Block scalars (`key: |`) would otherwise be stored as the literal
        // string "|" (silent misconfig). Our configs never need them.
        if (val == "|" || val == ">" || val == "|-" || val == ">-" ||
            val == "|+" || val == ">+") {
            GAI_FAIL(strfmt("config:%d: unsupported block scalar '%s' for key '%s' (use a plain scalar)",
                            lineno, val.c_str(), key.c_str()));
        }
        // Anchors/aliases/tags in the VALUE (`key: &a 5`, `key: *a`) would
        // otherwise be stored literally (silent misconfig). Same loud refusal.
        if (!val.empty() && (val[0] == '&' || val[0] == '*' || val[0] == '!')) {
            GAI_FAIL(strfmt("config:%d: unsupported YAML construct in value for key '%s' (anchors/aliases/tags need plain scalars): %s",
                            lineno, key.c_str(), val.c_str()));
        }

        std::string full;
        for (auto& p : stack) full += p.second + ".";
        full += key;

        if (val.empty()) {
            stack.emplace_back(indent, key);
            list_key    = full;
            list_indent = indent;
            cfg.kv_[full] = "";
        } else if (val.front() == '[' && val.back() == ']') {
            std::string inner = val.substr(1, val.size() - 2);
            std::string joined;
            std::stringstream ss(inner);
            std::string item;
            while (std::getline(ss, item, ',')) {
                item = strip_quotes(trim(item));
                if (item.empty()) continue;
                if (!joined.empty()) joined += "\n";
                joined += item;
            }
            cfg.kv_[full] = joined;
            list_key.clear();
        } else {
            cfg.kv_[full] = strip_quotes(val);
            list_key.clear();
        }
        (void)list_indent;
    }
    return cfg;
}

Config Config::from_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot open config file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return from_string(ss.str());
}

bool Config::has(const std::string& key) const { return kv_.find(key) != kv_.end(); }

std::string Config::get_str(const std::string& key, const std::string& def) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return def;
    return it->second;
}

static bool full_number(const std::string& s, bool allow_float, i64& oi, double& od) {
    try {
        size_t pos = 0;
        if (!allow_float) {
            oi = static_cast<i64>(std::stoll(s, &pos));
            if (pos != s.size()) return false;
            od = static_cast<double>(oi);
            return true;
        }
        od = std::stod(s, &pos);
        if (pos != s.size() || !std::isfinite(od)) return false;
        if (od >= static_cast<double>(std::numeric_limits<i64>::min()) &&
            od <= static_cast<double>(std::numeric_limits<i64>::max()))
            oi = static_cast<i64>(od);
        else
            oi = 0;
        return true;
    } catch (...) { return false; }
}

static bool full_integer_number(const std::string& s, i64& out) {
    try {
        size_t pos = 0;
        const long long v = std::stoll(s, &pos);
        if (pos != s.size()) return false;
        out = static_cast<i64>(v);
        return true;
    } catch (...) {}
    try {
        size_t pos = 0;
        const double v = std::stod(s, &pos);
        if (pos != s.size() || !std::isfinite(v) || std::trunc(v) != v ||
            v < static_cast<double>(std::numeric_limits<i64>::min()) ||
            v > static_cast<double>(std::numeric_limits<i64>::max()))
            return false;
        out = static_cast<i64>(v);
        return true;
    } catch (...) { return false; }
}

i64 Config::get_int(const std::string& key, i64 def) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return def;
    // Strict trailing check: "0.01abc" or "3e-4x" must not parse as 0.01.
    i64 value = 0;
    if (full_integer_number(trim(it->second), value)) return value;
    log_warn("config: ignoring malformed int for '" + key + "': '" + it->second + "'");
    return def;
}

double Config::get_f64(const std::string& key, double def) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return def;
    i64 oi = 0; double od = 0.0;
    if (full_number(trim(it->second), true, oi, od)) return od;
    log_warn("config: ignoring malformed number for '" + key + "': '" + it->second + "'");
    return def;
}

float Config::get_f32(const std::string& key, float def) const {
    return static_cast<float>(get_f64(key, static_cast<double>(def)));
}

bool Config::get_bool(const std::string& key, bool def) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return def;
    std::string v = it->second;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    if (v == "true" || v == "yes" || v == "on" || v == "1") return true;
    if (v == "false" || v == "no" || v == "off" || v == "0") return false;
    log_warn("config: ignoring malformed bool for '" + key + "': '" + it->second + "'");
    return def;
}

i64 Config::get_int_strict(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty())
        GAI_FAIL("config: missing required int '" + key + "'");
    i64 value = 0;
    if (full_integer_number(trim(it->second), value)) return value;
    GAI_FAIL("config: malformed int for '" + key + "': '" + it->second + "' (refusing silent default)");
    return 0;
}

double Config::get_f64_strict(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty())
        GAI_FAIL("config: missing required number '" + key + "'");
    i64 oi = 0; double od = 0.0;
    if (full_number(trim(it->second), true, oi, od)) return od;
    GAI_FAIL("config: malformed number for '" + key + "': '" + it->second + "' (refusing silent default)");
    return 0.0;
}

bool Config::get_bool_strict(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty())
        GAI_FAIL("config: missing required bool '" + key + "'");
    std::string v = it->second;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    if (v == "true" || v == "yes" || v == "on" || v == "1") return true;
    if (v == "false" || v == "no" || v == "off" || v == "0") return false;
    GAI_FAIL("config: malformed bool for '" + key + "': '" + it->second + "' (refusing silent default)");
    return false;
}

std::vector<std::string> Config::get_list(const std::string& key) const {
    std::vector<std::string> out;
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return out;
    std::stringstream ss(it->second);
    std::string item;
    while (std::getline(ss, item, '\n')) if (!item.empty()) out.push_back(item);
    return out;
}

void Config::set(const std::string& key, const std::string& value) { kv_[key] = value; }

size_t Config::check_known(const std::vector<std::string>& exact,
                           const std::vector<std::string>& prefixes,
                           bool strict) const {
    size_t unknown = 0;
    for (const auto& [k, v] : kv_) {
        bool known = false;
        for (const auto& e : exact) {
            if (k == e) { known = true; break; }
        }
        if (!known) {
            for (const auto& p : prefixes) {
                if (k.size() >= p.size() && k.compare(0, p.size(), p) == 0) {
                    known = true;
                    break;
                }
            }
        }
        if (known) continue;
        if (v.empty()) {
            // Structural parent (other keys extend "k.")? Always fine.
            bool has_kids = false;
            const std::string pre = k + ".";
            for (const auto& [k2, v2] : kv_) {
                (void)v2;
                if (k2.size() > pre.size() && k2.compare(0, pre.size(), pre) == 0) {
                    has_kids = true;
                    break;
                }
            }
            if (has_kids) continue;
            if (strict) GAI_FAIL("config: '" + k + "' has no value (typo?)");
            log_warn("[cfg ] '" + k + "' has no value and does nothing (typo?)");
        } else {
            if (strict) GAI_FAIL("config: unknown key '" + k + "' (typo?)");
            log_warn("[cfg ] unknown key '" + k + "' is ignored (typo? use --strict-config to fail)");
        }
        ++unknown;
    }
    return unknown;
}

std::string Config::dump() const {
    std::ostringstream ss;
    for (auto& [k, v] : kv_) {
        std::string sv = v;
        std::replace(sv.begin(), sv.end(), '\n', '|');
        ss << k << " = " << sv << "\n";
    }
    return ss.str();
}

} // namespace gai
