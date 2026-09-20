#pragma once

#include "core/common.h"
#include <map>
#include <vector>

namespace gai {

// Minimal YAML subset parser: nested maps (2-space indent), scalars, inline lists,
// comments, quoted strings. Enough for our config files, no external dependency.
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
    // Strict getters (audit §21): malformed KNOWN scalars fail fast instead
    // of silently falling back to a default that corrupts the recipe
    // (e.g. learning_rate: 3e-4x). Use these for required training
    // hyperparameters; keep the lenient getters for optional metadata.
    i64         get_int_strict (const std::string& key) const;
    double      get_f64_strict (const std::string& key) const;
    bool        get_bool_strict(const std::string& key) const;
    std::vector<std::string> get_list(const std::string& key) const;

    void set(const std::string& key, const std::string& value);

    // P2-5 typo/dead-key catcher: every stored key must either be in `exact`
    // or start with one of `prefixes` (e.g. "data.mix."). Structural parent
    // keys (empty value WITH children, like "data") are always skipped; empty
    // LEAF keys warn too (a valueless setting is almost certainly a mistake).
    // Returns the unknown count. strict=true fails fast on the first hit.
    size_t check_known(const std::vector<std::string>& exact,
                       const std::vector<std::string>& prefixes,
                       bool strict) const;

    const std::map<std::string, std::string>& flat() const { return kv_; }
    std::string dump() const;

private:
    std::map<std::string, std::string> kv_;   // dotted keys -> raw scalar text
};

} // namespace gai
