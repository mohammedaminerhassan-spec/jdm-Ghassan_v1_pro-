#include "training/checkpoint.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    auto s1 = std::make_shared<CheckpointSnapshot>();
    s1->weights.push_back(Tensor::zeros({1024, 1024}, DType::F32, Device::CPU)); // 4 MB

    auto s2 = std::make_shared<CheckpointSnapshot>();
    s2->weights.push_back(Tensor::zeros({2048, 1024}, DType::F32, Device::CPU)); // 8 MB

    // Test 1: Single snapshot
    std::vector<std::shared_ptr<CheckpointSnapshot>> refs1 = {s1};
    CHECK(unique_snapshot_bytes(refs1) == s1->bytes(), "Single snapshot size matches s1->bytes()");
    CHECK(s1->bytes() == 4 * 1024 * 1024, "s1 is exactly 4MB");

    // Test 2: Duplicated shared_ptr references (active, queued, cache)
    std::vector<std::shared_ptr<CheckpointSnapshot>> refs2 = {s1, s1, s1, s1};
    CHECK(unique_snapshot_bytes(refs2) == s1->bytes(), "Duplicated shared_ptr refs are counted exactly ONCE");

    // Test 3: Distinct snapshots counted separately
    std::vector<std::shared_ptr<CheckpointSnapshot>> refs3 = {s1, s2, s1, s2, s1};
    CHECK(unique_snapshot_bytes(refs3) == s1->bytes() + s2->bytes(), "Distinct snapshots counted separately without double counting");
    CHECK(unique_snapshot_bytes(refs3) == 12 * 1024 * 1024, "Total unique snapshot bytes is 12MB");

    // Test 4: Backpressure threshold triggering logic
    const size_t budget = 10 * 1024 * 1024; // 10 MB budget
    CHECK(unique_snapshot_bytes(refs1) <= budget, "s1 fits within 10MB budget");
    CHECK(unique_snapshot_bytes(refs3) > budget, "s1 + s2 exceeds 10MB budget (backpressure triggers)");

    if (failures == 0) {
        std::cout << "test_snapshot_memory: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_snapshot_memory: " << failures << " FAILURES\n";
    return 1;
}
