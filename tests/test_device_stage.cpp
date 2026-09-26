// Regression: staging a device-resident [rows, cols] f32 tensor to the host in
// bounded row blocks. `ghassan-ai logits` read Model::forward() output directly
// from the host; on CUDA that memory is on the device and the read segfaulted,
// so the command only ever worked on CPU. This test pins the BLOCK MATH (the
// part that can silently drop the tail of a row or read past the end) by
// forcing many blocks, a partial last block, and a row wider than the budget.
#include "core/device.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static std::vector<float> ramp(i64 rows, i64 cols) {
    std::vector<float> v(static_cast<size_t>(rows * cols));
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i % 9973) * 0.5f - 100.0f;
    return v;
}

static bool identical(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

static void test_single_block() {
    const i64 rows = 3, cols = 5;
    const std::vector<float> src = ramp(rows, cols);
    std::vector<float> out;
    device_stage_f32_2d(src.data(), Device::CPU, rows, cols, out, 1u << 20);
    CHECK(out.size() == static_cast<size_t>(rows * cols), "single block size");
    CHECK(identical(src, out), "single block contents");
}

static void test_many_blocks_exact() {
    // 17 rows x 33 cols with a 256-byte budget -> many partial blocks.
    const i64 rows = 17, cols = 33;
    const std::vector<float> src = ramp(rows, cols);
    std::vector<float> out;
    device_stage_f32_2d(src.data(), Device::CPU, rows, cols, out, 256);
    CHECK(out.size() == src.size(), "many blocks size");
    CHECK(identical(src, out), "many blocks contents (no dropped tail row)");
}

static void test_row_wider_than_budget() {
    // One row of 4096 floats is 16 KiB; budget it at 64 bytes so rows_per_block
    // must clamp to 1 instead of truncating the row.
    const i64 rows = 4, cols = 4096;
    const std::vector<float> src = ramp(rows, cols);
    std::vector<float> out;
    device_stage_f32_2d(src.data(), Device::CPU, rows, cols, out, 64);
    CHECK(out.size() == src.size(), "oversized row size");
    CHECK(identical(src, out), "oversized row contents (row never truncated)");
}

static void test_single_element() {
    const float src = 3.5f;
    std::vector<float> out;
    device_stage_f32_2d(&src, Device::CPU, 1, 1, out, 4);
    CHECK(out.size() == 1, "1x1 size");
    CHECK(out.size() == 1 && out[0] == 3.5f, "1x1 value");
}

static void test_empty_is_noop() {
    std::vector<float> out{1.0f, 2.0f};
    device_stage_f32_2d(nullptr, Device::CPU, 0, 8, out);
    CHECK(out.empty(), "zero rows clears the output and never dereferences src");
    device_stage_f32_2d(nullptr, Device::CPU, 8, 0, out);
    CHECK(out.empty(), "zero cols clears the output");
}

static void test_null_source_rejected() {
    bool threw = false;
    try {
        std::vector<float> out;
        device_stage_f32_2d(nullptr, Device::CPU, 4, 4, out);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "non-empty extent with a null source must fail fast");
}

int main() {
    test_single_block();
    test_many_blocks_exact();
    test_row_wider_than_budget();
    test_single_element();
    test_empty_is_noop();
    test_null_source_rejected();
    if (failures == 0) { std::cout << "test_device_stage: ALL PASS\n"; return 0; }
    std::cerr << "test_device_stage: " << failures << " FAILURES\n";
    return 1;
}
