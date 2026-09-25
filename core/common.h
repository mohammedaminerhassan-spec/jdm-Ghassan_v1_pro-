#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <chrono>

namespace gai {

using i8  = std::int8_t;
using u8  = std::uint8_t;
using i16 = std::int16_t;
using u16 = std::uint16_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;

// ------------------------------------------------------------------ errors
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

[[noreturn]] void fail(const std::string& msg, const char* file, int line);

#define GAI_FAIL(msg) ::gai::fail((msg), __FILE__, __LINE__)
#define GAI_CHECK(cond, msg) \
    do { if (!(cond)) ::gai::fail(std::string("check failed: " #cond " : ") + (msg), __FILE__, __LINE__); } while (0)

// ------------------------------------------------------------------ logging
enum class LogLevel { Debug = 0, Info = 1, Warn = 2, ErrorL = 3, Silent = 4 };

void set_log_level(LogLevel lvl);
LogLevel log_level();
void log_raw(LogLevel lvl, const std::string& msg);

void log_debug(const std::string& m);
void log_info(const std::string& m);
void log_warn(const std::string& m);
void log_error(const std::string& m);

// printf-style helper
std::string strfmt(const char* fmt, ...);

// ------------------------------------------------------------------ timing
class Timer {
public:
    Timer() { reset(); }
    void reset() { t0_ = std::chrono::steady_clock::now(); }
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    }
    double ms() const { return seconds() * 1000.0; }
    // Whole microseconds, for the relaxed-atomic perf counters (ops layer).
    u64 elapsed_us() const {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0_).count());
    }
private:
    std::chrono::steady_clock::time_point t0_;
};

// ------------------------------------------------------------------ misc
std::string human_bytes(u64 n);
std::string human_count(u64 n);
std::string human_duration(double seconds);

int  num_threads();
void set_num_threads(int n);

} // namespace gai
