#include "core/config.h"

#include <fstream>
#include <sstream>
#include <algorithm>

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

    while (std::getline(in, line)) {
        std::string raw = strip_comment(line);
        if (trim(raw).empty()) continue;

        int indent = 0;
        while (indent < static_cast<int>(raw.size()) && (raw[static_cast<size_t>(indent)] == ' ')) ++indent;
        std::string body = trim(raw);

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
        if (colon == std::string::npos) continue;

        std::string key = trim(body.substr(0, colon));
        std::string val = trim(body.substr(colon + 1));

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

i64 Config::get_int(const std::string& key, i64 def) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return def;
    try { return static_cast<i64>(std::stoll(it->second)); }
    catch (...) {
        try { return static_cast<i64>(std::stod(it->second)); }
        catch (...) {
            log_warn("config: ignoring malformed int for '" + key + "': '" + it->second + "'");
            return def;
        }
    }
}

double Config::get_f64(const std::string& key, double def) const {
    auto it = kv_.find(key);
    if (it == kv_.end() || it->second.empty()) return def;
    try { return std::stod(it->second); }
    catch (...) {
        log_warn("config: ignoring malformed number for '" + key + "': '" + it->second + "'");
        return def;
    }
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
