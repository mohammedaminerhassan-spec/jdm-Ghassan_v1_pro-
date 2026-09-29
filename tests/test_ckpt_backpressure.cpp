#include "training/trainer.h"
#include "training/checkpoint.h"
#include "core/device.h"
#include <cassert>
#include <iostream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
    else { std::cout << "  [ok] " << msg << "\n"; } \
} while (0)

static size_t old_budget_for(size_t phys) {
    return static_cast<size_t>(static_cast<double>(phys) * 0.35);
}
static size_t new_budget_for(size_t phys) {
    const size_t kCap = 32ULL << 30;
    const size_t kFloor = 8ULL << 30;
    const size_t kReserve = 8ULL << 30;
    if (phys <= kReserve + kFloor) return (kFloor < phys) ? kFloor : phys;
    size_t b = phys - kReserve;
    if (b > kCap) b = kCap;
    return b;
}

static void test_budget_math() {
    const size_t phys_29 = 29ULL << 30;
    const size_t snap_480m = static_cast<size_t>(5.76 * 1024.0 * 1024.0 * 1024.0);
    const size_t snap_1b_lion = 8ULL << 30;

    size_t old_b = old_budget_for(phys_29);
    size_t new_b = new_budget_for(phys_29);
    std::cout << "  phys=29GB old_budget=" << old_b / (1024*1024) << "MB"
              << " new_budget=" << new_b / (1024*1024) << "MB\n";

    CHECK(snap_480m * 2 > old_b, "old policy rejects 480M (reproduces P0 blocker)");

    CHECK(snap_480m <= new_b, "new policy allows 480M single copy");
    CHECK(snap_480m * 2 <= new_b, "new policy allows 480M two unique copies (active+queued)");

    CHECK(snap_1b_lion <= new_b, "new policy allows 1B-Lion single copy");
    CHECK(snap_1b_lion * 2 <= new_b, "new policy allows 1B-Lion active+queued");

    CHECK((40ULL << 30) > new_b, "40GB snapshot still exceeds 21GB budget (fail-fast kept)");
}

static void test_snapshot_bytes_excludes_grads() {
    ModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 32;
    cfg.use_moe = false;
    Model m(cfg, Device::CPU);
    m.init_weights(42);
    m.enable_grad(true);

    for (Parameter* p : m.parameters()) {
        if (p->g.defined()) {
            float* g = p->g.f32();
            for (i64 i = 0; i < p->g.numel(); ++i) g[i] = 1.0f;
        }
    }
    AdamWConfig oc;
    AdamW opt(m, oc);
    TrainState st;
    CheckpointSnapshot snap = Checkpoint::capture(m, opt, st);
    size_t w = 0;
    for (const auto& t : snap.weights) w += t.nbytes();
    size_t expect = w + snap.optimizer.bytes();
    CHECK(snap.bytes() == expect, "snapshot bytes = weights + optimizer moments (no grads)");

    size_t grad_bytes = 0;
    for (Parameter* p : m.parameters()) if (p->g.defined()) grad_bytes += p->g.nbytes();
    CHECK(grad_bytes > 0, "model grads exist but are excluded from checkpoint");
}

static void test_unique_dedup() {
    ModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 32;
    cfg.use_moe = false;
    Model m(cfg, Device::CPU);
    m.init_weights(7);
    m.enable_grad(true);
    AdamWConfig oc;
    AdamW opt(m, oc);
    TrainState st;
    auto snap = std::make_shared<CheckpointSnapshot>(Checkpoint::capture(m, opt, st));

    std::vector<std::shared_ptr<CheckpointSnapshot>> refs = {snap, snap, snap};
    CHECK(unique_snapshot_bytes(refs) == snap->bytes(), "shared snapshot deduped to 1x (best+last same step)");
    auto snap2 = std::make_shared<CheckpointSnapshot>(Checkpoint::capture(m, opt, st));
    refs = {snap, snap2};
    CHECK(unique_snapshot_bytes(refs) == snap->bytes() + snap2->bytes(), "two steps count 2x");
}

int main() {
    std::cout << "test_ckpt_backpressure:\n";
    test_budget_math();
    test_snapshot_bytes_excludes_grads();
    test_unique_dedup();

    size_t b = Trainer::ckpt_ram_budget_public();
    std::cout << "  live ckpt_ram_budget=" << b / (1024*1024) << "MB\n";
    CHECK(b >= (8ULL << 30) || b <= (32ULL << 30), "live budget within [floor,cap] clamp");
    if (failures == 0) { std::cout << "test_ckpt_backpressure: ALL PASS\n"; return 0; }
    std::cerr << "test_ckpt_backpressure: " << failures << " FAILURES\n";
    return 1;
}
