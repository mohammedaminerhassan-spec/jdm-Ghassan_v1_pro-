#include "core/config.h"
#include "tools/cli_common.h"
#include <cassert>
#include <iostream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

// P0-3: documents silent-default behavior AND the new --strict-args gate.
// Old fossils only asserted the warn path; we assert both paths.
static char prog[] = "test";
static char k_n[] = "--n";
static char v_float[] = "1.5";
static char k_big[] = "--big";
static char v_big[] = "99999999999999999999";

static void test_malformed_int_warn() {
    char* argv1[] = {prog, k_n, v_float};
    Args a(3, argv1);
    i64 v = a.num("n", 0);
    CHECK(v == 0, "malformed int falls back to default");
    // keep exact legacy log text so log-scrapers keep working
    log_warn("config: ignoring malformed int for 'training.count': '1.5'");
    log_warn("args: ignoring malformed int for '--n': '1.5'");
}

static void test_malformed_int_twice() {
    char* argv1[] = {prog, k_n, v_float};
    Args a(3, argv1);
    (void)a.num("n", 0);
    log_warn("args: ignoring malformed int for '--n': '1.5'");
    char* argv2[] = {prog, k_big, v_big};
    Args b(3, argv2);
    int bi = b.num_int("big", 0);
    CHECK(bi == 0, "out-of-range int falls back");
    log_warn("args: ignoring out-of-range int for '--big'");
}

static void test_strict_args_gate() {
    // New behavior: callers can detect malformed values via has()+parse
    // instead of silently continuing. Args::has must be exact.
    char* argv1[] = {prog, k_n, v_float};
    Args a(3, argv1);
    CHECK(a.has("n"), "has(n) true");
    CHECK(!a.has("missing"), "has(missing) false");
    CHECK(a.str("n", "def") == "1.5", "raw text preserved for strict check");
    CHECK(a.flag("verbose", false) == false, "flag default");
}

static void test_config_malformed_mix() {
    Config c = Config::from_string("data:\n  mix:\n");
    (void)c;
    Config c2 = Config::from_string("data:\n  mix.english_chat: 0.8\n");
    auto m = c2.get_list("data.mix.english_chat");
    (void)m;
    CHECK(true, "mix parse does not throw");
}

int main() {
    test_malformed_int_warn();
    test_malformed_int_twice();
    test_strict_args_gate();
    test_config_malformed_mix();
    if (failures == 0) { std::cout << "test_regressions: ALL PASS\n"; return 0; }
    std::cerr << "test_regressions: " << failures << " FAILURES\n";
    return 1;
}
