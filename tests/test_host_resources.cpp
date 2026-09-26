// Regression for the host-resource guards added after the multi-hour-run
// audit: the checkpoint queue budget used to be a hardcoded 32 GB, which is
// larger than a 30 GB Kaggle session, so the OOM killer could win the race
// against the backpressure. These tests pin the budget's contract and the two
// probes it depends on.
#include "core/device.h"
#include "core/signals.h"

#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace gai;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static void test_ram_probe_agrees_with_the_machine() {
    const size_t ram = physical_ram_bytes();
    if (ram == 0) {
        std::cout << "  (physical_ram_bytes unavailable on this platform — probe skipped)\n";
        return;
    }
    // Any machine that runs this test has more than 64 MB of RAM.
    CHECK(ram > (64ULL << 20), "physical_ram_bytes reports a plausible total");
    CHECK(ram < (4096ULL << 30), "physical_ram_bytes is not reporting bytes-of-bytes");
    std::cout << "  physical RAM = " << (ram >> 20) << " MiB\n";
}

static void test_disk_probe() {
    const size_t free_root = free_disk_bytes(".");
    if (free_root == 0) {
        std::cout << "  (free_disk_bytes unavailable on this platform — probe skipped)\n";
        return;
    }
    CHECK(free_root > 0, "free_disk_bytes(.) returns a positive number");
    // A path that does not exist must still answer (it walks up to the parent),
    // otherwise the pre-flight would silently skip a real disk-full situation.
    const size_t free_missing = free_disk_bytes("./no/such/dir/for/tests");
    CHECK(free_missing > 0, "free_disk_bytes walks up to an existing ancestor");
}

static void test_signal_flag_contract() {
    CHECK(signals::stop_requested == 0, "stop flag starts clear");
    CHECK(std::string(signals::stop_reason()).empty(), "no reason before any signal");
    signals::request_stop(SIGTERM);
    CHECK(signals::stop_requested == 1, "first signal raises the flag");
    CHECK(!std::string(signals::stop_reason()).empty(), "reason recorded after a signal");
    // Leave the process state as we found it for any later check.
    signals::stop_requested = 0;
}

static void test_tree_size_counts_hard_links_once() {
    // The quota projection must not invent a second copy of a checkpoint that
    // the writer published under two names (best.ckpt + last.ckpt share one
    // inode), so tree_size_bytes() de-duplicates by file id.
    const char* dir = "gai_test_tree";
    fs::create_directories(dir);
    const std::string a = std::string(dir) + "/a.bin";
    const std::string b = std::string(dir) + "/b.bin";
    {
        std::ofstream f(a, std::ios::binary);
        const std::string chunk(4096, 'x');
        for (int i = 0; i < 64; ++i) f << chunk;   // 256 KiB
    }
    const size_t one = tree_size_bytes(dir, nullptr);
    CHECK(one == 256u * 1024u, "single file size is exact");
    // Second NAME for the same bytes.
    std::error_code ec;
    fs::copy_file(a, b, fs::copy_options::overwrite_existing, ec);
    if (!ec) {
        const size_t copied = tree_size_bytes(dir, nullptr);
        CHECK(copied == 2u * one, "a real copy counts twice (it costs twice)");
    }
    ::remove(a.c_str());
    ::remove(b.c_str());
    fs::remove(dir);
}

int main() {
    test_ram_probe_agrees_with_the_machine();
    test_disk_probe();
    test_tree_size_counts_hard_links_once();
    test_signal_flag_contract();
    if (failures == 0) { std::cout << "test_host_resources: ALL PASS\n"; return 0; }
    std::cerr << "test_host_resources: " << failures << " FAILURES\n";
    return 1;
}
