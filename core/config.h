#pragma once

#include "core/common.h"
#include <map>
#include <vector>

namespace gai {

class Config {
public:
    Config() = default;

    static Config from_file(const std::string& path);
    static Config from_string(const std::string& text);

    bool has(const std::string& key) const;

    std::string get_str (const std::string& key, const std::string& def = "") const;
    i64         get_int (const std::string& key, i64 def = 0) const;
    double      get_f64 (const std::string& key, double def = 0.0) const;
    float       get_f32 (const std::string& key, float def = 0.0f) const;
    bool        get_bool(const std::string& key, bool def = false) const;

    i64         get_int_strict (const std::string& key) const;
    double      get_f64_strict (const std::string& key) const;
    bool        get_bool_strict(const std::string& key) const;
    std::vector<std::string> get_list(const std::string& key) const;

    void set(const std::string& key, const std::string& value);

    size_t check_known(const std::vector<std::string>& exact,
                       const std::vector<std::string>& prefixes,
                       bool strict) const;

    const std::map<std::string, std::string>& flat() const { return kv_; }
    std::string dump() const;

private:
    std::map<std::string, std::string> kv_;
};

}
