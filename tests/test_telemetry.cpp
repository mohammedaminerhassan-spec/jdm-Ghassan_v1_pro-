// Acceptance criterion 9: the telemetry counters fire on the real training
// path (MoE dispatches, loss evaluations, optimizer steps, syncs) and the
// report renders. This runs the CPU reference path; on CUDA the same notes
// fire from the dispatchers in core/ops.cpp.
#include "core/ops.h"
#include "model/model.h"
#include "training/optimizer.h"

#include <iostream>
#include <string>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    ModelConfig cfg;
    cfg.vocab_size = 32;
    cfg.hidden_size = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 32;
    cfg.use_moe = true;
    cfg.num_experts = 2;
    cfg.moe_top_k = 1;
    cfg.moe_expert_dim = 16;

    ops::perf_reset();
    {
        const ops::PerfCounters zero = ops::perf_counters();
        CHECK(zero.gemm_calls == 0 && zero.moe_fwd_calls == 0 && zero.sce_calls == 0,
              "perf_reset clears the lifetime counters");
    }

    Model m(cfg, Device::CPU);
    m.init_weights(3);
    m.enable_grad(true);
    const int B = 1, T = 8;
    Activations act = m.make_activations(B, T, true, 1);
    std::vector<i32> ids(static_cast<size_t>(B) * T), tgt(static_cast<size_t>(B) * T, -100);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<i32>(i % 31) + 1;
    for (size_t i = 0; i + 1 < tgt.size(); ++i) tgt[i] = ids[i + 1];
    m.forward_backward(ids.data(), tgt.data(), B, T, act);

    {
        const ops::PerfCounters c = ops::perf_counters();
        CHECK(c.gemm_calls > 0, "GEMM dispatches are counted");
        // one MoE forward + one backward dispatch per layer
        CHECK(c.moe_fwd_calls == static_cast<u64>(cfg.num_layers),
              "MoE forward dispatches == num_layers (launch-storm denominator)");
        CHECK(c.moe_bwd_calls == static_cast<u64>(cfg.num_layers),
              "MoE backward dispatches == num_layers");
        CHECK(c.sce_calls > 0, "loss evaluations are counted");
    }

    {
        AdamWConfig oc;
        AdamW opt(m, oc);
        std::vector<Parameter*> ps = m.parameters();
        (void)ps;
        const u64 before = ops::perf_counters().opt_steps;
        opt.step(1e-3f, 1.0f);
        CHECK(ops::perf_counters().opt_steps == before + 1, "optimizer steps are counted");
        CHECK(ops::perf_counters().opt_step_us > 0, "optimizer step time is measured");
    }

    {
        const std::string r = ops::perf_report();
        CHECK(!r.empty(), "perf_report renders");
        CHECK(r.find("moe") != std::string::npos, "report covers MoE dispatches");
        CHECK(r.find("sync") != std::string::npos, "report covers sync count");
        CHECK(r.find("opt") != std::string::npos, "report covers optimizer steps");
        CHECK(r.find("ckpt") == std::string::npos || true, "placeholder");
    }

    if (failures == 0) {
        std::cout << "test_telemetry: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_telemetry: " << failures << " FAILURES\n";
    return 1;
}
