#include "training/dataloader.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static bool same_batch(const Batch& a, const Batch& b) {
    return a.B == b.B && a.T == b.T && a.ids == b.ids &&
           a.targets == b.targets && a.segment_ids == b.segment_ids &&
           a.tokens_supervised == b.tokens_supervised;
}

int main() {
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "gai_ddp_resume_stream.gbin";
    std::error_code ec;
    fs::remove(path, ec);
    {
        ShardWriter writer(path.string(), 64, false);
        for (int d = 0; d < 16; ++d) {
            std::vector<i32> doc;
            for (int i = 0; i < 64; ++i)
                doc.push_back(static_cast<i32>((d * 131 + i * 7 + 1) % 63 + 1));
            writer.add_document(doc);
        }
        writer.close();
    }

    BatchSpec spec;
    spec.batch_size = 2;
    spec.seq_len = 16;

    const u64 rank1_seed = 42u + 1u * 1000003u;
    DataLoader rank1;
    CHECK(rank1.open({path.string()}, spec, rank1_seed), "rank1 loader opens");
    Batch b0, b1, b2, b3, b4;
    CHECK(rank1.next(b0) && rank1.next(b1) && rank1.next(b2) && rank1.next(b3),
          "rank1 advances 4 batches");
    CHECK(rank1.next(b4), "rank1 batch 4 recorded");

    {
        DataLoader other;
        CHECK(other.open({path.string()}, spec, 777u), "other-seed loader opens");
        Batch ob;
        CHECK(other.next(ob), "other-seed batch");
        CHECK(!same_batch(ob, b0), "different seeds give different streams");
    }

    DataLoader rebuilt;
    CHECK(rebuilt.open({path.string()}, spec, 999u), "rebuilt loader opens");
    rebuilt.reseed(rank1_seed);
    rebuilt.skip_batches(4);
    Batch r4, r5;
    CHECK(rebuilt.next(r4), "rebuilt batch 4");
    CHECK(same_batch(r4, b4), "reseed+skip reproduces rank1 batch 4 exactly");
    CHECK(rebuilt.next(r5), "rebuilt batch 5");
    Batch b5;
    CHECK(rank1.next(b5), "rank1 batch 5 recorded");
    CHECK(same_batch(r5, b5), "rebuilt stream continues identically");

    fs::remove(path, ec);
    if (failures == 0) { std::cout << "test_ddp_resume_stream: ALL PASS\n"; return 0; }
    std::cerr << "test_ddp_resume_stream: " << failures << " FAILURES\n";
    return 1;
}
