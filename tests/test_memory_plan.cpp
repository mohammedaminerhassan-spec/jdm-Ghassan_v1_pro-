// F-23 / F-14 regression: the arithmetic-only memory plan must agree with a
// real Model, and the fp16 accounting must include the per-layer fused QKV
// cache that the runtime guard previously omitted.
//
// Without this cross-check the duplication between Model::count_parameters()
// and the Model constructor could silently drift, which would make
// `gai_train --dry-run` lie exactly when it matters most (before a 4xT4 run).
#include "model/model.h"
#include "core/config.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static void check_config(const ModelConfig& cfg, const char* label) {
    Model model(cfg, Device::CPU);
    const u64 real = static_cast<u64>(model.num_parameters());
    const u64 arith = Model::count_parameters(cfg);
    CHECK(real == arith,
          std::string(label) + ": count_parameters matches the allocated Model (" +
              human_count(arith) + " vs " + human_count(real) + ")");
    const i64 real_ne = model.num_parameters_non_embedding();
    const Model::MemoryPlan plan = Model::plan_memory(cfg, 2, 128, true, 4, false);
    CHECK(plan.params_no_embedding == static_cast<u64>(real_ne),
          std::string(label) + ": non-embedding count matches (" +
              human_count(plan.params_no_embedding) + " vs " + human_count(static_cast<u64>(real_ne)) + ")");
    CHECK(plan.weights == static_cast<size_t>(real) * 4,
          std::string(label) + ": weights bytes == params*4");
    CHECK(plan.grads == plan.weights, std::string(label) + ": grads bytes == weights bytes");
    CHECK(plan.activations == model.estimate_activation_bytes(2, 128, true, 4),
          std::string(label) + ": activation estimate matches the member function");
    CHECK(plan.static_total + plan.activations == plan.total,
          std::string(label) + ": total == static + activations");
}

int main() {
    // ---- the shipped production recipes, straight from configs/ ------------
    // Loading a 1B model on CPU is exactly what F-23 avoids, so only the
    // arithmetic path is checked at production size; the equivalence is proven
    // on the small configs below.
    static const char* kConfigs[] = {
        "configs/en_pro.yaml", "configs/pro_v1.yaml", "configs/t4_1b.yaml",
        "configs/en_ollama.yaml", "configs/pro_auxfree.yaml", "configs/sft_en_pro.yaml",
        "configs/smoke.yaml",
    };
    for (const char* path : kConfigs) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cerr << "skip (absent): " << path << "\n";
            continue;
        }
        const Config c = Config::from_file(path);
        const ModelConfig cfg = ModelConfig::from_config(c, "model");
        cfg.validate();
        const u64 params = Model::count_parameters(cfg);
        const Model::MemoryPlan plan =
            Model::plan_memory(cfg, 1, 512, true, 4, /*fp16_cache=*/true);
        CHECK(params > 0, std::string(path) + ": parameter count is positive");
        CHECK(plan.fp16_cache > 0, std::string(path) + ": fp16 cache is accounted");
        CHECK(plan.total > plan.static_total,
              std::string(path) + ": activations contribute to the total");
        // The fp16 cache must cover every 2-D parameter AND the fused QKV cache.
        const size_t fused_only =
            static_cast<size_t>(cfg.num_layers) *
            static_cast<size_t>(cfg.q_dim() + 2 * cfg.kv_dim()) *
            static_cast<size_t>(cfg.hidden_size) * sizeof(u16);
        CHECK(Model::count_fp16_cache_bytes(cfg) > fused_only,
              std::string(path) + ": fp16 cache includes the fused QKV cache (F-14)");
        // A sane upper bound: the cache can never exceed the fp32 weights.
        CHECK(Model::count_fp16_cache_bytes(cfg) <= plan.weights,
              std::string(path) + ": fp16 cache does not exceed the fp32 weights");
    }

    // ---- equivalence on small models we CAN actually instantiate ----------
    {
        ModelConfig dense;
        dense.vocab_size = 64;
        dense.hidden_size = 32;
        dense.num_layers = 3;
        dense.num_heads = 4;
        dense.num_kv_heads = 2;
        dense.intermediate_size = 64;
        dense.max_seq_len = 64;
        check_config(dense, "dense");

        ModelConfig tied = dense;
        tied.tie_embeddings = true;
        check_config(tied, "dense+tied");

        ModelConfig moe = dense;
        moe.use_moe = true;
        moe.num_experts = 4;
        moe.moe_top_k = 2;
        moe.moe_expert_dim = 32;
        moe.moe_shared = true;
        check_config(moe, "moe+shared");

        ModelConfig moe_nosh = moe;
        moe_nosh.moe_shared = false;
        check_config(moe_nosh, "moe no-shared");

        ModelConfig qk = dense;
        qk.use_qk_norm = true;
        check_config(qk, "dense+qknorm");
    }

    // ---- F-14: the runtime accounting now matches the arithmetic -----------
    {
        ModelConfig cfg;
        cfg.vocab_size = 64;
        cfg.hidden_size = 32;
        cfg.num_layers = 4;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 64;
        cfg.max_seq_len = 64;
        Model model(cfg, Device::CPU);
        model.enable_fp16_weight_cache(true);
        const size_t runtime = model.fp16_weight_cache_bytes();
        const size_t arith = Model::count_fp16_cache_bytes(cfg);
        CHECK(runtime == arith,
              "F-14: runtime fp16_weight_cache_bytes() == count_fp16_cache_bytes() (" +
                  human_bytes(runtime) + " vs " + human_bytes(arith) + ")");
        CHECK(runtime > static_cast<size_t>(cfg.num_layers) *
                            static_cast<size_t>(cfg.q_dim() + 2 * cfg.kv_dim()) *
                            static_cast<size_t>(cfg.hidden_size) * sizeof(u16),
              "F-14: the fused QKV cache is included in the runtime figure");
    }

    // ---- F-16: the workspace plan is sane for the T4 recipe ---------------
    {
        ModelConfig big;
        big.vocab_size = 32000;
        big.hidden_size = 1024;
        big.num_layers = 26;
        big.num_heads = 16;
        big.num_kv_heads = 4;
        big.intermediate_size = 2816;
        big.max_seq_len = 512;
        big.use_moe = true;
        big.num_experts = 8;
        big.moe_top_k = 2;
        big.moe_expert_dim = 1280;
        big.moe_shared = true;
        big.moe_aux_scale = 0.0f;
        big.moe_aux_free = true;
        big.tie_embeddings = true;
        const Model::MemoryPlan plan = Model::plan_memory(big, 1, 512, true, 4, false);
        // ~1B params => ~3.8 GiB weights + grads, +4 B/param Lion state.
        CHECK(plan.params > 900000000ULL && plan.params < 1200000000ULL,
              "a 1B-class recipe prices at ~1B parameters without allocating it");
        // The whole point of the plan: price the recipe BEFORE allocating it and
        // see that a single 16 GB T4 can hold it with >=10% headroom.
        const size_t t4 = 15360ULL * 1024ULL * 1024ULL;
        CHECK(plan.total * 10 <= t4 * 9,
              "the 1B T4 recipe is predicted to fit in 16 GB with >=10% headroom: " +
                  human_bytes(plan.total));
    }

    // ---- F-16: workspace pre-sizing covers the allocation sites ------------
    {
        ModelConfig cfg;
        cfg.vocab_size = 32000;
        cfg.hidden_size = 768;
        cfg.num_layers = 26;
        cfg.num_heads = 12;
        cfg.num_kv_heads = 4;
        cfg.intermediate_size = 2048;
        cfg.max_seq_len = 1024;
        cfg.use_moe = true;
        cfg.num_experts = 8;
        cfg.moe_top_k = 2;
        cfg.moe_expert_dim = 768;
        cfg.moe_shared = true;
        const int B = 2, T = 1024;
        const Model::WorkspacePlan wsp = Model::workspace_plan(cfg, B, T);
        const u64 N = static_cast<u64>(B) * T;
        const u64 NK = N * 2;
        // Forward site in cuda/moe.cu: N*ne + 2*NK + 3*NK*E + NK*d floats.
        const u64 fwd_floats = N * 8 + 2 * NK + 3 * NK * 768 + NK * 768;
        CHECK(wsp.moe_bytes >= 4 * fwd_floats,
              "F-16: the MoE pool covers the forward call site");
        // Backward site: 6*N*E + NK + 4*NK*E + 2*NK*d + NK + N*ne floats.
        const u64 bwd_floats =
            6 * N * 768 + NK + 4 * NK * 768 + 2 * NK * 768 + NK + N * 8;
        CHECK(wsp.moe_bytes >= 4 * bwd_floats,
              "F-16: the MoE pool covers the backward call site");
        // GEMM pool: largest fp16 conversion pair (lm_head N*d + d*V).
        const u64 conv = 2 * (N * 768 + 768 * 32000);
        CHECK(wsp.gemm_bytes >= conv,
              "F-16: the GEMM pool covers the largest fp16 conversion");
        // Monotonicity: a bigger (B,T) never plans smaller pools.
        const Model::WorkspacePlan small = Model::workspace_plan(cfg, 1, 512);
        CHECK(wsp.gemm_bytes >= small.gemm_bytes && wsp.moe_bytes >= small.moe_bytes,
              "F-16: workspace plans grow monotonically with (B,T)");
        // Dense models need no MoE pool.
        ModelConfig dense = cfg;
        dense.use_moe = false;
        CHECK(Model::workspace_plan(dense, B, T).moe_bytes == 0,
              "F-16: dense models plan no MoE workspace");
    }


    if (failures == 0) {
        std::cout << "test_memory_plan: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_memory_plan: " << failures << " FAILURES\n";
    return 1;
}
