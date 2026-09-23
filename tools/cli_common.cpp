#include "tools/cli_common.h"

#include <cmath>
#include <iostream>
#include <limits>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

namespace gai {

Args::Args(int argc, char** argv) {
    if (argc > 0) prog_ = argv[0];
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            std::string key = a.substr(2);
            size_t eq = key.find('=');
            if (eq != std::string::npos) {
                kv_[key.substr(0, eq)] = key.substr(eq + 1);
                continue;
            }
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                kv_[key] = argv[++i];
            } else {
                kv_[key] = "1";
            }
        } else if (a.rfind("-", 0) == 0 && a.size() > 1) {
            std::string key = a.substr(1);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0) kv_[key] = argv[++i];
            else kv_[key] = "1";
        } else {
            pos_.push_back(a);
        }
    }
}

bool Args::has(const std::string& k) const { return kv_.find(k) != kv_.end(); }

std::string Args::str(const std::string& k, const std::string& def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
}

static bool parse_cli_i64(const std::string& text, i64& out) {
    try {
        size_t pos = 0;
        const long long v = std::stoll(text, &pos);
        if (pos != text.size()) return false;
        out = static_cast<i64>(v);
        return true;
    } catch (...) {
    }
    try {
        size_t pos = 0;
        const double v = std::stod(text, &pos);
        if (pos != text.size() || !std::isfinite(v) || std::trunc(v) != v ||
            v < static_cast<double>(std::numeric_limits<i64>::min()) ||
            v > static_cast<double>(std::numeric_limits<i64>::max())) {
            return false;
        }
        out = static_cast<i64>(v);
        return true;
    } catch (...) {
        return false;
    }
}

i64 Args::num(const std::string& k, i64 def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    i64 v = def;
    if (parse_cli_i64(it->second, v)) return v;
    log_warn("args: ignoring malformed int for '--" + k + "': '" + it->second + "'");
    return def;
}

int Args::num_int(const std::string& k, int def) const {
    const i64 v = num(k, static_cast<i64>(def));
    if (v < static_cast<i64>(std::numeric_limits<int>::min()) ||
        v > static_cast<i64>(std::numeric_limits<int>::max())) {
        log_warn("args: ignoring out-of-range int for '--" + k + "'");
        return def;
    }
    return static_cast<int>(v);
}

u64 Args::num_u64(const std::string& k, u64 def) const {
    if (!has(k)) return def;
    const i64 v = num(k, 0);
    if (v < 0) {
        log_warn("args: ignoring out-of-range integer for '--" + k + "'");
        return def;
    }
    return static_cast<u64>(v);
}

double Args::real(const std::string& k, double def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    try { return std::stod(it->second); }
    catch (...) {
        log_warn("args: ignoring malformed number for '--" + k + "': '" + it->second + "'");
        return def;
    }
}

bool Args::flag(const std::string& k, bool def) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) return def;
    const std::string& v = it->second;
    return !(v == "0" || v == "false" || v == "no" || v == "off");
}

i64 Args::num_strict(const std::string& k) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) GAI_FAIL("args: missing required --" + k + " (strict-args)");
    i64 v = 0;
    if (!parse_cli_i64(it->second, v))
        GAI_FAIL("args: malformed int for '--" + k + "': '" + it->second + "' (strict-args)");
    return v;
}

int Args::num_int_strict(const std::string& k) const {
    const i64 v = num_strict(k);
    if (v < static_cast<i64>(std::numeric_limits<int>::min()) ||
        v > static_cast<i64>(std::numeric_limits<int>::max()))
        GAI_FAIL("args: out-of-range int for '--" + k + "' (strict-args)");
    return static_cast<int>(v);
}

double Args::real_strict(const std::string& k) const {
    auto it = kv_.find(k);
    if (it == kv_.end()) GAI_FAIL("args: missing required --" + k + " (strict-args)");
    try {
        size_t pos = 0;
        double v = std::stod(it->second, &pos);
        if (pos != it->second.size() || !std::isfinite(v))
            GAI_FAIL("args: malformed number for '--" + k + "'");
        return v;
    } catch (const Error&) { throw; }
    catch (...) { GAI_FAIL("args: malformed number for '--" + k + "': '" + it->second + "' (strict-args)"); }
    return 0.0;  // unreachable
}

void enable_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Larger buffer so Arabic multi-byte writes are not split
    static char buf[1 << 16];
    setvbuf(stdout, buf, _IOFBF, sizeof(buf));
#endif
    std::ios::sync_with_stdio(true);
}

void apply_common_flags(const Args& a) {
    if (a.flag("quiet", false))   set_log_level(LogLevel::Warn);
    if (a.flag("silent", false))  set_log_level(LogLevel::Silent);
    if (a.flag("verbose", false)) set_log_level(LogLevel::Debug);
    i64 t = a.num("threads", 0);
    if (t > 0) {
        set_num_threads(static_cast<int>(t));
    } else {
        // T4-only + weak-PC fast path: pin OpenMP to all hardware threads by
        // default (was lazy: omp used defaults until --threads passed).
        // Explicit pin gives max CPU decode speed on weak local PCs.
        set_num_threads(num_threads());
    }
}

} // namespace gai
