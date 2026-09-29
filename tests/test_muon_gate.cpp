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

static ModelConfig cfg_of() {
    ModelConfig c;
    c.vocab_size = 32;
    c.hidden_size = 64;
    c.num_layers = 1;
    c.num_heads = 8;
    c.num_kv_heads = 1;
    c.intermediate_size = 256;
    c.max_seq_len = 32;
    c.use_moe = false;
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

    const double g = opt.step(0.02f, 1.0f);
    CHECK(std::isfinite(g), "muon step produces a finite grad norm");
}

int main() {

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

        CHECK(c.muon_ns_calls == 5,
              "F-04: the min_ns_dim gate excludes exactly the small decay matrices");
    }

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

        CHECK(c.muon_ns_calls == 7,
              "F-04: min_ns_dim=0 keeps the historical all-matrix behavior");
    }

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

    {
        ModelConfig cfg = cfg_of();
        Model m1(cfg, Device::CPU); m1.init_weights(42); m1.enable_grad(true);
        Model m2(cfg, Device::CPU); m2.init_weights(42); m2.enable_grad(true);

        MuonConfig oc;
        oc.min_ns_dim = 0;
        Muon opt1(m1, oc);
        Muon opt2(m2, oc);

        auto& p1 = m1.parameters();
        auto& p2 = m2.parameters();
        for (size_t i = 0; i < p1.size(); ++i) {
            p1[i]->g = Tensor::zeros(p1[i]->shape, DType::F32, Device::CPU);
            p2[i]->g = Tensor::zeros(p2[i]->shape, DType::F32, Device::CPU);
            float* g1 = p1[i]->g.f32();
            float* g2 = p2[i]->g.f32();
            for (i64 k = 0; k < p1[i]->numel(); ++k) {
                float val = std::sin(static_cast<float>(k + 1) * 0.1f);
                g1[k] = val;
                g2[k] = val * 8.0f;
            }
        }

        const double gnorm1 = opt1.step(0.01f, 1.0f);
        const double gnorm2 = opt2.step(0.01f, 1.0f / 8.0f);

        CHECK(std::abs(gnorm1 - gnorm2) < 1e-4, "Muon: unscaled and scaled grad norms match");

        float max_diff = 0.0f;
        for (size_t i = 0; i < p1.size(); ++i) {
            const float* w1 = p1[i]->w.f32();
            const float* w2 = p2[i]->w.f32();
            for (i64 k = 0; k < p1[i]->numel(); ++k) {
                float diff = std::abs(w1[k] - w2[k]);
                if (diff > max_diff) max_diff = diff;
            }
        }
        CHECK(max_diff < 1e-5f, "Muon: weight update is strictly grad_scale invariant");
    }

    if (failures == 0) {
        std::cout << "test_muon_gate: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_muon_gate: " << failures << " FAILURES\n";
    return 1;
}
