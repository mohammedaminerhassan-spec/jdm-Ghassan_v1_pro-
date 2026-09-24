#include "core/ops.h"
#include "training/dataloader.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static void test_packed_loader() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "gai_sequence_pack_test.gbin";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    {
        ShardWriter writer(path.string(), 32, false);
        writer.add_document({1, 2, 3});
        writer.add_document({4, 5});
        writer.add_document({6, 7, 8, 9});
        writer.close();
    }
    DataLoader loader;
    BatchSpec spec;
    spec.batch_size = 2;
    spec.seq_len = 6;
    spec.pack_sequences = true;
    CHECK(loader.open({path.string()}, spec, 7), "packed loader opens");
    Batch batch;
    CHECK(loader.next(batch), "packed batch produced");
    CHECK(batch.B == 2 && batch.T == 6, "packed batch shape");
    for (i32 segment : batch.segment_ids) CHECK(segment >= 0, "packed row has no padding");
    int boundaries = 0;
    for (int b = 0; b < batch.B; ++b) {
        for (int t = 1; t < batch.T; ++t) {
            size_t i = static_cast<size_t>(b * batch.T + t);
            if (batch.segment_ids[i] != batch.segment_ids[i - 1]) ++boundaries;
        }
    }
    CHECK(boundaries >= 2, "multiple documents packed per batch");
    CHECK(batch.tokens_supervised > 0, "packed tokens supervised");
    std::filesystem::remove(path, ec);
}

static void test_packed_resume() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "gai_sequence_resume_test.gbin";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    {
        ShardWriter writer(path.string(), 64, false);
        writer.add_document({1, 2, 3, 4, 5});
        writer.add_document({6, 7, 8});
        writer.add_document({9, 10, 11, 12, 13, 14, 15});
        writer.close();
    }
    BatchSpec spec;
    spec.batch_size = 2;
    spec.seq_len = 8;
    spec.pack_sequences = true;
    DataLoader direct;
    DataLoader resumed;
    CHECK(direct.open({path.string()}, spec, 99), "direct loader opens");
    CHECK(resumed.open({path.string()}, spec, 99), "resume loader opens");
    Batch direct0;
    CHECK(direct.next(direct0), "direct first batch");
    CHECK(direct.next(direct0) && direct.next(direct0), "direct next two batches");
    resumed.fast_forward(2);
    Batch resumed2;
    CHECK(resumed.next(resumed2), "resumed second-state batch");
    CHECK(direct0.ids == resumed2.ids && direct0.targets == resumed2.targets &&
          direct0.segment_ids == resumed2.segment_ids, "fast-forward two matches");
    Batch a;
    DataLoader zero;
    CHECK(zero.open({path.string()}, spec, 99), "zero loader opens");
    CHECK(zero.next(a) && zero.next(a) && zero.next(a) && zero.next(a), "direct batches");
    DataLoader three;
    CHECK(three.open({path.string()}, spec, 99), "three-step loader opens");
    three.fast_forward(3);
    Batch b;
    CHECK(three.next(b), "three-step resumed batch");
    CHECK(a.ids == b.ids && a.targets == b.targets && a.segment_ids == b.segment_ids,
          "packed fast-forward preserves exact data stream");
    std::filesystem::remove(path, ec);
}

static void test_segment_attention() {
    const float q[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float k[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float v[4] = {1.0f, 2.0f, 100.0f, 200.0f};
    const i32 segments[4] = {0, 0, 1, 1};
    float out[4] = {};
    ops::attention_forward_ex(Device::CPU, q, k, v, out, nullptr,
                              1, 4, 1, 1, 1, 1.0f, 0, segments);
    CHECK(std::fabs(out[0] - 1.0f) < 1e-6f, "first segment first token");
    CHECK(std::fabs(out[1] - 1.5f) < 1e-6f, "first segment causal mean");
    CHECK(std::fabs(out[2] - 100.0f) < 1e-6f, "second segment isolated");
    CHECK(std::fabs(out[3] - 150.0f) < 1e-6f, "second segment causal mean");
}

int main() {
    test_packed_loader();
    test_packed_resume();
    test_segment_attention();
    if (failures == 0) {
        std::cout << "test_sequence_pack: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_sequence_pack: " << failures << " FAILURES\n";
    return 1;
}
