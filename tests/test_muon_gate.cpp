// F-04 regression: Muon's Newton-Schulz gate (`min_ns_dim`) must route small
// decay matrices to the cheap Lion-style branch while large ones still get
// orthogonalized, and the NS telemetry must reflect exactly what ran.
#include "core/ops.h"
#include "model/model.h"
#include "training/optimizer.h"
#include "training/trainer.h"

#include <cmath>
#include <iostream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

// A dense model with one large projection family and one small one.
// hidden=64, heads=8, kv=1: wq [64,64] is NS-eligible at min_ns_dim=64
// (min side 64) while wk/wv [8,64] (min side 8) are gated out. Dense on
// purpose: MoE would add 7 more decay matrices and blur the exact counts.
static ModelConfig cfg_of() {
    ModelConfig c;
    c.vocab_size = 32;
    c.hidden_size = 64;
    c.num_layers = 1;
    c.num_heads = 8;       // qd = 8*8 = 64 -> wq/wo [64,64]
    c.num_kv_heads = 1;    // kvd = 8      -> wk/wv [8,64]
    c.intermediate_size = 256;
    c.max_seq_len = 32;
    c.use_moe = false;     // exact decay-matrix census: wq,wk,wv,wo + gate,up,down
    return c;
}

static void run_one_step(Model& m, MuonConfig oc) {
    Muon opt(m, oc);
    const int B = 1, T = 8;
    Activations act = m.make_activations(B, T, true, 1);
    std::vector<i32> ids(static_cast<size_t>(B) * T), tgt(static_cast<size_t>(B) * T, -100);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<i32>(i % 31) + 1;
    for (size_t i = 0; i + 1 < tgt.size(); ++i) tgt[i] = ids[i + 1];
    m.forward_backward(ids.data(), tgt.data(), B, T, act);
    // feed finite grads: forward_backward already populated .g on CPU
    const double g = opt.step(0.02f, 1.0f);
    CHECK(std::isfinite(g), "muon step produces a finite grad norm");
}

int main() {
    // Gated: only matrices with min(rows,cols) >= 64 run NS.
    {
        ModelConfig cfg = cfg_of();
        Model m(cfg, Device::CPU);
        m.init_weights(3);
        m.enable_grad(true);
        MuonConfig oc;
        oc.ns_steps = 5;
        oc.min_ns_dim = 64;
        ops::perf_reset();
        run_one_step(m, oc);
        const ops::PerfCounters c = ops::perf_counters();
        CHECK(c.muon_ns_calls > 0, "F-04: large matrices still run Newton-Schulz");
        CHECK(c.muon_ns_iters == c.muon_ns_calls * 5,
              "F-04: every NS call runs exactly ns_steps iterations");
        CHECK(c.muon_ns_us > 0, "F-04: NS time is measured");
        // 7 decay matrices total (wq,wk,wv,wo + gate,up,down); wk+wv are gated
        // out, so exactly 5 remain. This is the T4 win: 2 fewer 5-iter NS runs
        // per layer per step, and the gap widens with depth and routers.
        CHECK(c.muon_ns_calls == 5,
              "F-04: the min_ns_dim gate excludes exactly the small decay matrices");
    }
    // Untouched default: min_ns_dim=0 orthogonalizes every decay matrix.
    {
        ModelConfig cfg = cfg_of();
        Model m(cfg, Device::CPU);
        m.init_weights(3);
        m.enable_grad(true);
        MuonConfig oc;
        oc.ns_steps = 5;
        oc.min_ns_dim = 0;
        ops::perf_reset();
        run_one_step(m, oc);
        const ops::PerfCounters c = ops::perf_counters();
        // decay 2D params: wq,wk,wv,wo + w_gate,w_up,w_down = 7
        CHECK(c.muon_ns_calls == 7,
              "F-04: min_ns_dim=0 keeps the historical all-matrix behavior");
    }
    // ns_steps plumbing through the config is clamped, not silent.
    {
        Config c = Config::from_string(
            "training:\n  max_steps: 5\n  epochs: 0\n  batch_size: 1\n  seq_len: 32\n"
            "  ns_steps: 3\n  muon_min_ns_dim: 128\n");
        const TrainerConfig t = TrainerConfig::from_config(c, false);
        CHECK(t.muon_ns_steps == 3, "training.ns_steps is honoured");
        CHECK(t.muon_min_ns_dim == 128, "training.muon_min_ns_dim is honoured");
        Config bad = Config::from_string(
            "training:\n  max_steps: 5\n  epochs: 0\n  batch_size: 1\n  seq_len: 32\n"
            "  ns_steps: 99\n");
        bool threw = false;
        try {
            (void)TrainerConfig::from_config(bad, false);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw, "ns_steps outside 1..10 fails fast");
    }

    if (failures == 0) {
        std::cout << "test_muon_gate: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_muon_gate: " << failures << " FAILURES\n";
    return 1;
}
