// F-11 / F-12 regression: the auxiliary-loss-free (DeepSeek-V3 Â§3.2) router
// bias is optimizer-step-coupled control state.
//
//   F-11: when the optimizer refuses an update (non-finite grad norm) the bias
//         must NOT move, and the counts of that step must not leak into the
//         next step.
//   F-12: the numerator and denominator of the per-layer load fraction must
//         describe the SAME population. The old code divided raw routing counts
//         by an externally supplied SUPERVISED-token count while the counts
//         covered ALL routed tokens, so any SFT loss mask made the fractions sum
//         to N/ntok instead of 1 and mis-scaled the bias. The denominator is
//         now derived from the counts themselves (row_sum == routed slots),
//         which makes the skew structurally impossible.
#include "model/model.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static ModelConfig test_config() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 32;
    cfg.num_layers = 2;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 64;
    cfg.max_seq_len = 64;
    cfg.use_moe = true;
    cfg.num_experts = 4;
    cfg.moe_top_k = 2;
    cfg.moe_expert_dim = 32;
    cfg.moe_shared = true;
    cfg.moe_aux_scale = 0.0f;   // validate() requires this with aux_free
    cfg.moe_aux_free = true;
    return cfg;
}

// Reference load counts taken straight from the routing the model chose.
static std::vector<double> routing_counts(const Activations& act, int B, int T,
                                          int ne, int K, int layer) {
    const size_t N = static_cast<size_t>(B) * static_cast<size_t>(T);
    std::vector<double> cnt(static_cast<size_t>(ne), 0.0);
    const i32* idx = act.saved_moe_idx[static_cast<size_t>(layer)].i32p();
    for (size_t t = 0; t < N; ++t)
        for (int k = 0; k < K; ++k) {
            const int e = idx[t * static_cast<size_t>(K) + static_cast<size_t>(k)];
            if (e >= 0 && e < ne) cnt[static_cast<size_t>(e)] += 1.0;
        }
    return cnt;
}

static void add_counts(std::vector<double>& dst, const std::vector<double>& src) {
    for (size_t e = 0; e < dst.size(); ++e) dst[e] += src[e];
}

static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

struct Micro {
    Activations act;
    std::vector<i32> ids, tgt;
    int B = 0, T = 0;
    i64 ntok = 0;
};

// `keep_every` < 0 supervises every position, 0 supervises none, otherwise
// only every (keep_every)-th position gets a target (what an SFT mask does).
static Micro run_micro(Model& m, int B, int T, int keep_every, u32 salt) {
    Micro mb;
    mb.B = B;
    mb.T = T;
    mb.act = m.make_activations(B, T, true, 1);
    const size_t n = static_cast<size_t>(B) * static_cast<size_t>(T);
    mb.ids.resize(n);
    mb.tgt.assign(n, -100);
    for (size_t i = 0; i < n; ++i)
        mb.ids[i] = static_cast<i32>((i * 7u + salt * 13u) % 61u) + 1;
    for (size_t i = 0; i + 1 < n; ++i) {
        const bool supervised = (keep_every < 0) ||
                                (keep_every > 0 && static_cast<int>(i) % keep_every == 0);
        if (supervised) mb.tgt[i] = mb.ids[i + 1];
    }
    m.forward_backward(mb.ids.data(), mb.tgt.data(), B, T, mb.act, &mb.ntok);
    return mb;
}

int main() {
    const ModelConfig cfg = test_config();
    const int ne = cfg.num_experts;
    const int K = cfg.moe_top_k;
    const int L = cfg.num_layers;
    const double bias_lr = 0.001;               // Model::moe_bias_lr_
    const double target = static_cast<double>(K) / static_cast<double>(ne);

    // ======================================================================
    // 1. F-12: PARTIALLY MASKED (SFT-like) microbatches of different sizes.
    //    The applied fraction must be the pooled routed-slot fraction, which
    //    always sums to exactly 1 regardless of how many positions were
    //    supervised. The old ntok-based denominator produced N/ntok here.
    // ======================================================================
    {
        Model m(cfg, Device::CPU);
        m.init_weights(11);
        m.enable_grad(true);
        m.ensure_moe_bias();

        Micro a = run_micro(m, 1, 8, 3, 1);    //  8 tokens, 1/3 supervised
        Micro b = run_micro(m, 2, 4, 2, 2);    //  8 tokens, 1/2 supervised
        const int tokens = (a.B * a.T) + (b.B * b.T);
        const i64 supervised = a.ntok + b.ntok;
        CHECK(supervised > 0, "masked microbatches still supervise some tokens");
        CHECK(supervised < tokens,
              "F-12: the test really is a partially masked SFT step (ntok < N)");

        std::vector<std::vector<double>> pooled(static_cast<size_t>(L),
                                                std::vector<double>(static_cast<size_t>(ne), 0.0));
        for (int l = 0; l < L; ++l) {
            add_counts(pooled[static_cast<size_t>(l)], routing_counts(a.act, a.B, a.T, ne, K, l));
            add_counts(pooled[static_cast<size_t>(l)], routing_counts(b.act, b.B, b.T, ne, K, l));
        }
        m.apply_moe_bias_step(m.moe_bias_acc_host().data(), true);

        const auto& bias = m.moe_bias_all();
        CHECK(bias.size() == static_cast<size_t>(L), "bias has one row per layer");
        for (int l = 0; l < L; ++l) {
            double row = 0.0;
            for (double c : pooled[static_cast<size_t>(l)]) row += c;
            CHECK(near(row, static_cast<double>(tokens) * K, 1e-9),
                  "F-12: pooled counts cover every routed token, supervised or not");
            double frac_sum = 0.0;
            for (int e = 0; e < ne; ++e) {
                const double frac = pooled[static_cast<size_t>(l)][static_cast<size_t>(e)] / row;
                frac_sum += frac;
                const double want = -bias_lr * (frac - target);
                CHECK(near(bias[static_cast<size_t>(l)][static_cast<size_t>(e)], want, 1e-7),
                      "F-12: bias follows the pooled routed-slot fraction (layer " +
                          std::to_string(l) + " expert " + std::to_string(e) + ")");
            }
            CHECK(near(frac_sum, 1.0, 1e-9),
                  "F-12: per-layer fractions sum to 1 even under a loss mask");
        }
        CHECK(m.moe_bias_acc_host().empty(),
              "F-11/F-12: the count accumulator is released after the step");
    }

    // ======================================================================
    // 2. F-11: a step the optimizer refused must leave the bias untouched and
    //    must not leak its counts into the next step.
    // ======================================================================
    {
        Model m(cfg, Device::CPU);
        m.init_weights(23);
        m.enable_grad(true);
        m.ensure_moe_bias();
        const std::vector<std::vector<float>> zero = m.moe_bias_all();

        // Step A: routed, then REJECTED (non-finite grad norm -> opt skipped).
        Micro rejected = run_micro(m, 1, 6, -1, 5);
        CHECK(rejected.ntok > 0, "rejected step was a real training step");
        CHECK(!m.moe_bias_acc_host().empty(), "rejected step still collected counts");
        m.apply_moe_bias_step(m.moe_bias_acc_host().data(), false);
        const auto& after_reject = m.moe_bias_all();
        bool unchanged = after_reject.size() == zero.size();
        for (size_t l = 0; unchanged && l < after_reject.size(); ++l)
            for (size_t e = 0; e < after_reject[l].size(); ++e)
                if (after_reject[l][e] != zero[l][e]) { unchanged = false; break; }
        CHECK(unchanged, "F-11: a rejected optimizer step does not move the router bias");
        CHECK(m.moe_bias_acc_host().empty(),
              "F-11: rejected-step counts are dropped, not carried forward");

        // Step B: accepted. Only THIS step's routing may influence the bias.
        Micro accepted = run_micro(m, 1, 5, -1, 9);
        const std::vector<double> cnt_b = routing_counts(accepted.act, accepted.B, accepted.T, ne, K, 0);
        m.apply_moe_bias_step(m.moe_bias_acc_host().data(), true);
        const auto& after_accept = m.moe_bias_all();
        double row = 0.0;
        for (double c : cnt_b) row += c;
        for (int e = 0; e < ne; ++e) {
            const double frac = cnt_b[static_cast<size_t>(e)] / row;
            CHECK(near(after_accept[0][static_cast<size_t>(e)], -bias_lr * (frac - target), 1e-7),
                  "F-11: the accepted step uses only its own routing, not the rejected one");
        }
    }

    // ======================================================================
    // 3. F-12: a FULLY masked microbatch is a true no-op. forward_backward
    //    returns before routing when ntok==0, so such a step contributes no
    //    counts and cannot skew the fractions of a later step.
    // ======================================================================
    {
        Model m(cfg, Device::CPU);
        m.init_weights(31);
        m.enable_grad(true);
        m.ensure_moe_bias();
        Micro masked = run_micro(m, 1, 7, 0, 4);   // no supervised position
        CHECK(masked.ntok == 0, "fully masked microbatch supervises nothing");
        CHECK(m.moe_bias_acc_host().empty(),
              "F-12: a fully masked microbatch contributes no routing counts");
        m.apply_moe_bias_step(nullptr, true);
        const std::vector<std::vector<float>> bias = m.moe_bias_all();
        for (size_t l = 0; l < bias.size(); ++l)
            for (size_t e = 0; e < bias[l].size(); ++e)
                CHECK(bias[l][e] == 0.0f, "F-12: no bias drift from a fully masked step");
    }

    // ======================================================================
    // 4. Steering clamp + no-op contract.
    // ======================================================================
    {
        Model m(cfg, Device::CPU);
        m.init_weights(43);
        m.enable_grad(true);
        m.ensure_moe_bias();
        Micro mb = run_micro(m, 1, 8, -1, 6);
        const std::vector<double> cnt = routing_counts(mb.act, mb.B, mb.T, ne, K, 0);
        double row = 0.0;
        for (double c : cnt) row += c;
        CHECK(row > 0.0, "the accepted step routed at least one slot");
        m.apply_moe_bias_step(m.moe_bias_acc_host().data(), true);
        for (size_t l = 0; l < m.moe_bias_all().size(); ++l)
            for (size_t e = 0; e < m.moe_bias_all()[l].size(); ++e)
                CHECK(m.moe_bias_all()[l][e] >= -0.5f && m.moe_bias_all()[l][e] <= 0.5f,
                      "the steering bias stays inside its clamp");
    }

    if (failures == 0) {
        std::cout << "test_moe_bias_step: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_moe_bias_step: " << failures << " FAILURES\n";
    return 1;
}
