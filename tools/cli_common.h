#pragma once

#include "core/common.h"
#include <map>
#include <string>
#include <vector>

namespace gai {

// Tiny argument parser shared by all tools: --key value, --flag, positionals.
class Args {
public:
    Args(int argc, char** argv);

    bool        has(const std::string& k) const;
    std::string str(const std::string& k, const std::string& def = "") const;
    i64         num(const std::string& k, i64 def = 0) const;
    int         num_int(const std::string& k, int def = 0) const;
    u64         num_u64(const std::string& k, u64 def = 0) const;
    double      real(const std::string& k, double def = 0.0) const;
    bool        flag(const std::string& k, bool def = false) const;
    // FIX P1-7 (silent GPU-hour burn): strict getters fail fast on malformed
    // values instead of warn+default. Used when --strict-args is passed.
    i64         num_strict(const std::string& k) const;
    int         num_int_strict(const std::string& k) const;
    double      real_strict(const std::string& k) const;

    const std::vector<std::string>& positional() const { return pos_; }
    std::string command() const { return pos_.empty() ? "" : pos_[0]; }
    std::string program() const { return prog_; }

private:
    std::string prog_;
    std::map<std::string, std::string> kv_;
    std::vector<std::string> pos_;
};

// Enables UTF-8 output on Windows consoles so Arabic renders instead of mojibake.
void enable_utf8_console();

void apply_common_flags(const Args& a);   // --quiet --verbose --threads

} // namespace gai
