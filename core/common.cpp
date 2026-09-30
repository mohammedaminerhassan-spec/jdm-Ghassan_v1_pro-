#include "core/common.h"

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>

#ifdef GAI_OPENMP
#include <omp.h>
#endif

namespace gai {

// [FIX P2-10] g_threads was a plain static int: concurrent set/num_threads
// from workers is a data race (UB). Atomic with the same lazy-init
// semantics; set once at startup in the normal path.
static std::atomic<int> g_threads{0};
static std::mutex       g_log_mu;

void fail(const std::string& msg, const char* file, int line) {
    std::string full = std::string(file) + ":" + std::to_string(line) + ": " + msg;
    throw Error(full);
}

static LogLevel g_level = LogLevel::Info;
void set_log_level(LogLevel lvl) { g_level = lvl; }
LogLevel log_level() { return g_level; }

void log_raw(LogLevel lvl, const std::string& msg) {
    if (lvl < g_level) return;
    const char* tag = "";
    switch (lvl) {
        case LogLevel::Debug:  tag = "[debug] "; break;
        case LogLevel::Info:   tag = "[info ] "; break;
        case LogLevel::Warn:   tag = "[warn ] "; break;
        case LogLevel::ErrorL: tag = "[error] "; break;
        default: break;
    }
    // Serialize whole lines so main + ckpt-writer threads cannot interleave
    // mid-line (cosmetic, not corruption, but cheap to fix).
    std::lock_guard<std::mutex> lk(g_log_mu);
    std::ostream& os = (lvl >= LogLevel::Warn) ? std::cerr : std::cout;
    os << tag << msg << std::endl;
}

void log_debug(const std::string& m) { log_raw(LogLevel::Debug, m); }
void log_info (const std::string& m) { log_raw(LogLevel::Info,  m); }
void log_warn (const std::string& m) { log_raw(LogLevel::Warn,  m); }
void log_error(const std::string& m) { log_raw(LogLevel::ErrorL, m); }

std::string strfmt(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return {}; }
    std::vector<char> buf(static_cast<size_t>(n) + 1);
    std::vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);
    return std::string(buf.data(), static_cast<size_t>(n));
}

std::string human_bytes(u64 n) {
    const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double v = static_cast<double>(n);
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    return strfmt(i == 0 ? "%.0f %s" : "%.2f %s", v, u[i]);
}

std::string human_count(u64 n) {
    if (n < 1000) return std::to_string(n);
    const char* u[] = {"", "K", "M", "B", "T"};
    double v = static_cast<double>(n);
    int i = 0;
    while (v >= 1000.0 && i < 4) { v /= 1000.0; ++i; }
    return strfmt("%.2f%s", v, u[i]);
}

std::string human_duration(double s) {
    if (s < 1.0)   return strfmt("%.0fms", s * 1000.0);
    if (s < 60.0)  return strfmt("%.1fs", s);
    if (s < 3600)  return strfmt("%dm%02ds", int(s) / 60, int(s) % 60);
    if (s < 86400) return strfmt("%dh%02dm", int(s) / 3600, (int(s) % 3600) / 60);
    return strfmt("%dd%02dh", int(s) / 86400, (int(s) % 86400) / 3600);
}

u64 fingerprint_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return 0;
    u64 h = 1469598103934665603ULL;
    std::vector<char> buf(1 << 16);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        std::streamsize got = f.gcount();
        for (std::streamsize i = 0; i < got; ++i) {
            h ^= static_cast<unsigned char>(buf[static_cast<size_t>(i)]);
            h *= 1099511628211ULL;
        }
        if (got < static_cast<std::streamsize>(buf.size())) break;
    }

    return h ? h : 1ULL;
}

std::string fingerprint_hex(u64 fp) {
    if (fp == 0) return "unknown";
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(fp));
    return std::string(buf);
}

int num_threads() {
    int cur = g_threads.load(std::memory_order_relaxed);
    if (cur > 0) return cur;
    unsigned hc = std::thread::hardware_concurrency();
    int want = hc > 0 ? static_cast<int>(hc) : 1;
    g_threads.compare_exchange_strong(cur, want, std::memory_order_relaxed);
    return g_threads.load(std::memory_order_relaxed);
}
void set_num_threads(int n) {
    int want = n > 0 ? n : 1;
    g_threads.store(want, std::memory_order_relaxed);
#ifdef GAI_OPENMP
    omp_set_num_threads(want);
#endif
}

}
