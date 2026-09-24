#include "dataset/dedup.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "gai_contamination_blocklist.txt";
    std::ostringstream base;
    for (int i = 0; i < 100; ++i) base << "token" << i << ' ';
    std::string eval_text = base.str();
    std::string near_text = eval_text;
    near_text.replace(near_text.find("token50"), 7, "other50");
    {
        std::ofstream out(path);
        out << eval_text << '\n';
    }
    Deduplicator dedup;
    dedup.load_blocklist(path.string());
    CHECK(!dedup.add(eval_text), "exact eval document blocked");
    CHECK(!dedup.add(near_text), "near-duplicate eval document blocked");
    CHECK(dedup.blocked() == 2, "contamination count includes near duplicates");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (failures == 0) {
        std::cout << "test_contamination_blocklist: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_contamination_blocklist: " << failures << " FAILURES\n";
    return 1;
}
