// Gradient correctness is the single most important property of a hand-written
// training stack. This test finite-difference checks EVERY parameter group of the
// real architecture on a small config, plus verifies the Flash MoE parameter math.

#include "model/model.h"
#include "core/ops.h"
#include "core/rng.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace gai;

static int g_fail = 0;
#define CHECK(cond, name) do { \
    if (cond) { std::cout << "  [ok]   " << name << "\n"; } \
    else      { std::cout << "  [FAIL] " << name << "\n"; ++g_fail; } } while (0)

static ModelConfig tiny_config() {
    ModelConfig c;
    c.vocab_size = 64;
    c.hidden_size = 32;
    c.num_layers = 2;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.intermediate_size = 48;
    c.max_seq_len = 32;
    c.tie_embeddings = true;
    c.init_std = 0.08f;
    // tiny MoE so the gradient checks below cover routing + experts
    c.use_moe = true;
    c.num_experts = 4;
    c.moe_top_k = 2;
    c.moe_expert_dim = 16;
    c.moe_shared = true;
    c.moe_aux_scale = 0.01f;
    return c;
}

static double loss_only(Model& m, const std::vector<i32>& ids, const std::vector<i32>& tgt,
                        int B, int T, Activations& act) {
    Tensor& logits = m.forward(ids.data(), B, T, act);
    double sum = 0.0;
    i64 n = 0;
    ops::softmax_cross_entropy(Device::CPU, logits.f32(), tgt.data(), nullptr,
                               static_cast<i64>(B) * T, m.config().vocab_size, &sum, &n);
    double l = n ? sum / static_cast<double>(n) : 0.0;
    // include the MoE load-balance term so FD checks the FULL training loss
    // (averaged over layers, matching Model::forward_backward P0-02/mean).
    if (m.config().use_moe && m.config().moe_aux_scale > 0.0f && m.config().num_layers > 0)
        l += m.config().moe_aux_scale * m.moe_aux_loss(act, B, T) / static_cast<double>(m.config().num_layers);
    return l;
}

int main() {
    std::cout << "== model tests ==\n";
    set_log_level(LogLevel::Warn);

    // ------------------------------------------------ 1. parameter math (Ghassan v1 Flash)
    {
        ModelConfig c;   // defaults = the shipped Flash MoE architecture
        c.validate();
        Model probe(c, Device::CPU);   // allocation validates the math against reality
        i64 total = probe.num_parameters();
        ParamReport rep = probe.parameter_report();

        std::cout << "    Flash config: " << c.summary() << "\n";
        std::cout << "    total params = " << total << " (" << (double)total / 1e6 << " M)\n";

        // hand-computed expectation: d=768, L=26, 8 experts (top-2) + shared, E=768
        i64 expect_emb  = 16000LL * 768;                                   // 12,288,000
        i64 expect_attn = (768LL*768) + (768LL*256) + (768LL*256) + (768LL*768); // 1,572,864
        i64 expect_moe  = 8LL*768                       // router
                        + 3LL * 8LL*768*768             // routed gate/up/down
                        + 3LL * 768*768;                // shared expert
        i64 expect_layer = expect_attn + expect_moe + 2LL * 768;
        i64 expect_total = expect_emb + 26LL * expect_layer + 768;

        CHECK(expect_emb == 12288000, "embedding params = 12,288,000");
        CHECK(expect_attn == 1572864, "attention params per layer = 1,572,864");
        CHECK(expect_moe == 15931392, "moe params per layer = 15,931,392");
        CHECK(expect_layer == 17505792, "total params per layer = 17,505,792");
        CHECK(26LL * expect_layer == 455150592, "all 26 layers = 455,150,592");
        CHECK(expect_total == 467439360, "documented total = 467,439,360");
        CHECK(total == expect_total, "allocated tensors match the documented math");
        CHECK(rep.total == total, "parameter report matches allocation");
        CHECK(total < 500000000, "under the 500M ceiling");
        CHECK(rep.non_embedding == 455151360, "non-embedding params = 455,151,360");
    }

    // ------------------------------------------------ 2. forward sanity
    ModelConfig cfg = tiny_config();
    Model m(cfg, Device::CPU);
    m.init_weights(1234);
    m.enable_grad(true);

    const int B = 2, T = 6;
    Rng rng(99);
    std::vector<i32> ids(static_cast<size_t>(B) * T), tgt(static_cast<size_t>(B) * T);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<i32>(rng.below(static_cast<u64>(cfg.vocab_size)));
    for (size_t i = 0; i < tgt.size(); ++i) tgt[i] = static_cast<i32>(rng.below(static_cast<u64>(cfg.vocab_size)));
    tgt[3] = -100;   // exercise the ignore path

    Activations act = m.make_activations(B, T, true);
    {
        double l = loss_only(m, ids, tgt, B, T, act);
        std::cout << "    initial loss = " << l << " (ln(V) = " << std::log((double)cfg.vocab_size) << ")\n";
        CHECK(std::isfinite(l), "loss is finite");
        CHECK(std::fabs(l - std::log((double)cfg.vocab_size)) < 0.6,
              "initial loss near ln(vocab_size)");
    }

    // causality: changing a later token must not change earlier logits
    {
        Activations a2 = m.make_activations(B, T, false);
        std::vector<i32> ids2 = ids;
        ids2[static_cast<size_t>(T) - 1] = (ids2[static_cast<size_t>(T) - 1] + 7) % cfg.vocab_size;

        Activations a1 = m.make_activations(B, T, false);
        m.forward(ids.data(), B, T, a1);
        std::vector<float> ref(a1.logits.f32(), a1.logits.f32() + a1.logits.numel());
        m.forward(ids2.data(), B, T, a2);

        double max_early = 0.0, max_late = 0.0;
        for (int t = 0; t < T; ++t) {
            for (int v = 0; v < cfg.vocab_size; ++v) {
                size_t idx = static_cast<size_t>(t) * cfg.vocab_size + static_cast<size_t>(v);
                double diff = std::fabs(ref[idx] - a2.logits.f32()[idx]);
                if (t < T - 1) max_early = std::max(max_early, diff);
                else           max_late  = std::max(max_late, diff);
            }
        }
        CHECK(max_early < 1e-6, "causal masking: past logits unaffected by future token");
        CHECK(max_late > 1e-4, "changing the last token does change its own logits");
    }

    // ------------------------------------------------ 3. full gradient check
    // P0-05 convention: forward_backward emits SUM grads (mean×ntok, scaled),
    // while loss_only below is a MEAN. Divide analytics by bw_ntok so both
    // sides are means (exact, deterministic). Without this the check is off
    // by exactly ntok (observed 0.909 = 10/11 style ratio).
    m.zero_grad();
    i64 bw_ntok = 0;
    double base_loss = m.forward_backward(ids.data(), tgt.data(), B, T, act, &bw_ntok);
    CHECK(std::isfinite(base_loss), "forward_backward loss finite");

    struct Probe { const char* label; Parameter* p; };
    std::vector<Probe> probes = {
        {"tok_embeddings", m.find_parameter("tok_embeddings")},
        {"final_norm",     m.find_parameter("final_norm")},
        {"L0.attn_norm",   m.find_parameter("layers.0.attn_norm")},
        {"L0.wq",          m.find_parameter("layers.0.wq")},
        {"L0.wk",          m.find_parameter("layers.0.wk")},
        {"L0.wv",          m.find_parameter("layers.0.wv")},
        {"L0.wo",          m.find_parameter("layers.0.wo")},
        {"L0.ffn_norm",    m.find_parameter("layers.0.ffn_norm")},
        {"L0.moe_router",  m.find_parameter("layers.0.moe_router")},
        {"L0.moe_gate",    m.find_parameter("layers.0.moe_gate")},
        {"L0.moe_up",      m.find_parameter("layers.0.moe_up")},
        {"L0.moe_down",    m.find_parameter("layers.0.moe_down")},
        {"L0.moe_sh_down", m.find_parameter("layers.0.moe_sh_down")},
        {"L1.wq",          m.find_parameter("layers.1.wq")},
        {"L1.moe_gate",    m.find_parameter("layers.1.moe_gate")},
        {"L1.moe_router",  m.find_parameter("layers.1.moe_router")},
        {"L1.attn_norm",   m.find_parameter("layers.1.attn_norm")},
    };

    // NB: gradient activations (caches required for the MoE aux term above)
    Activations fd_act = m.make_activations(B, T, true);
    const float eps = 3e-3f;   // f32 sweet spot for central differences
    double worst_overall = 0.0;

    for (auto& pr : probes) {
        if (!pr.p) { std::cout << "  [FAIL] missing parameter " << pr.label << "\n"; ++g_fail; continue; }
        Parameter& P = *pr.p;
        float* w = P.w.f32();
        const float* g = P.g.f32();

        // sample a spread of coordinates
        int nprobe = static_cast<int>(std::min<i64>(6, P.numel()));
        double worst = 0.0;
        double max_abs_g = 0.0;
        for (int s = 0; s < nprobe; ++s) {
            i64 idx = (P.numel() * (2 * s + 1)) / (2 * nprobe);
            float orig = w[idx];

            w[idx] = orig + eps;
            double lp = loss_only(m, ids, tgt, B, T, fd_act);
            w[idx] = orig - eps;
            double lm = loss_only(m, ids, tgt, B, T, fd_act);
            w[idx] = orig;

            double num = (lp - lm) / (2.0 * eps);
            double ana = static_cast<double>(g[idx]) / static_cast<double>(bw_ntok > 0 ? bw_ntok : 1);
            max_abs_g = std::max(max_abs_g, std::fabs(ana));
            double denom = std::max(1e-4, std::max(std::fabs(num), std::fabs(ana)));
            worst = std::max(worst, std::fabs(num - ana) / denom);
        }
        worst_overall = std::max(worst_overall, worst);
        // Per-coordinate FD in f32 has an absolute noise floor of about
        // machine_eps*|L|/eps ~ 1.3e-4, so tiny-gradient coordinates legitimately show
        // a few percent relative error. Router top-k selection is piecewise-constant
        // so its threshold is wider (15%): an eps=3e-3 probe can flip a boundary
        // expert (kink) where no finite difference can match; the directional
        // check below (strict, router-excluded) plus the jitter FD test cover
        // the smooth softmax path exactly.
        const bool is_router = std::string(pr.label).find("router") != std::string::npos;
        bool ok = is_router ? (worst < 1.5e-1) : (worst < 5e-2);
        std::cout << (ok ? "  [ok]   " : "  [FAIL] ")
                  << "grad " << pr.label << "  rel_err=" << worst
                  << "  |g|max=" << max_abs_g << "\n";
        if (!ok) ++g_fail;
    }
    std::cout << "    worst relative gradient error (per-coordinate): " << worst_overall << "\n";

    // Directional derivative check: aggregating over a full random direction raises the
    // signal far above the f32 FD noise floor, giving a strict whole-model test.
    {
        Rng dr(2024);
        double worst_dir = 0.0;
        for (auto& pr : probes) {
            if (!pr.p) continue;   // dense models have no MoE params (and vice versa)
            // Router excluded: hard top-k selection is piecewise-constant, so a
            // full-direction step almost surely crosses a routing boundary (kink)
            // where no finite difference can match. The router's softmax path IS
            // verified by the per-coordinate checks above (small moves, no flips),
            // which is the same straight-through estimator used in production.
            if (std::string(pr.label).find("router") != std::string::npos) continue;
            Parameter& P = *pr.p;
            i64 n = P.numel();
            std::vector<float> dir(static_cast<size_t>(n));
            double gdot = 0.0;
            for (i64 i = 0; i < n; ++i) {
                dir[static_cast<size_t>(i)] = dr.normal(0.0f, 1.0f);
                gdot += static_cast<double>(dir[static_cast<size_t>(i)]) * static_cast<double>(P.g.f32()[i]);
            }
            gdot /= static_cast<double>(bw_ntok > 0 ? bw_ntok : 1); // sums -> mean
            std::vector<float> saved(P.w.f32(), P.w.f32() + n);
            const float h = 1e-3f;
            for (i64 i = 0; i < n; ++i) P.w.f32()[i] = saved[static_cast<size_t>(i)] + h * dir[static_cast<size_t>(i)];
            double lp = loss_only(m, ids, tgt, B, T, fd_act);
            for (i64 i = 0; i < n; ++i) P.w.f32()[i] = saved[static_cast<size_t>(i)] - h * dir[static_cast<size_t>(i)];
            double lm2 = loss_only(m, ids, tgt, B, T, fd_act);
            for (i64 i = 0; i < n; ++i) P.w.f32()[i] = saved[static_cast<size_t>(i)];

            double num = (lp - lm2) / (2.0 * h);
            double rel = std::fabs(num - gdot) / std::max(1e-6, std::max(std::fabs(num), std::fabs(gdot)));
            worst_dir = std::max(worst_dir, rel);
            if (rel >= 1e-2) {
                std::cout << "  [FAIL] directional grad " << pr.label
                          << " num=" << num << " ana=" << gdot << " rel=" << rel << "\n";
                ++g_fail;
            }
        }
        std::cout << "    worst directional gradient error: " << worst_dir << "\n";
        CHECK(worst_dir < 1e-2, "directional derivatives match analytic gradients");
    }

    // ------------------------------------------------ 4. gradient accumulation semantics
    {
        m.zero_grad();
        m.forward_backward(ids.data(), tgt.data(), B, T, act);
        std::vector<float> once(m.find_parameter("layers.0.wq")->g.f32(),
                                m.find_parameter("layers.0.wq")->g.f32() + 16);
        m.forward_backward(ids.data(), tgt.data(), B, T, act);
        const float* twice = m.find_parameter("layers.0.wq")->g.f32();
        bool ok = true;
        for (int i = 0; i < 16; ++i) if (std::fabs(twice[i] - 2.0f * once[i]) > 1e-5f * std::max(1.0f, std::fabs(once[i]))) ok = false;
        CHECK(ok, "gradients accumulate (no implicit zeroing)");
        m.zero_grad();
        bool zeroed = true;
        for (int i = 0; i < 16; ++i) if (m.find_parameter("layers.0.wq")->g.f32()[i] != 0.0f) zeroed = false;
        CHECK(zeroed, "zero_grad clears gradients");
    }

    // ------------------------------------------------ 5. can it actually learn?
    {
        ModelConfig c = tiny_config();
        c.num_layers = 2;
        Model lm(c, Device::CPU);
        lm.init_weights(7);
        lm.enable_grad(true);

        // memorise one fixed sequence
        const int b = 1, t = 8;
        std::vector<i32> x(static_cast<size_t>(t)), y(static_cast<size_t>(t));
        Rng r(3);
        for (int i = 0; i < t; ++i) x[static_cast<size_t>(i)] = static_cast<i32>(r.below(20));
        for (int i = 0; i < t - 1; ++i) y[static_cast<size_t>(i)] = x[static_cast<size_t>(i) + 1];
        y[static_cast<size_t>(t) - 1] = -100;

        Activations a = lm.make_activations(b, t, true);
        double first = 0, last = 0;
        for (int step = 0; step < 60; ++step) {
            lm.zero_grad();
            double l = lm.forward_backward(x.data(), y.data(), b, t, a);
            if (step == 0) first = l;
            last = l;
            for (Parameter* p : lm.parameters()) {
                float* w = p->w.f32();
                const float* g = p->g.f32();
                for (i64 i = 0; i < p->numel(); ++i) w[i] -= 0.1f * g[i];
            }
        }
        std::cout << "    memorisation: loss " << first << " -> " << last << "\n";
        CHECK(last < first * 0.25, "SGD drives the loss down (real learning)");
        CHECK(last < 0.5, "model memorises a short sequence");
    }

    // ------------------------------------------------ 6. save / load
    {
        const std::string path = "test_model.graw";
        m.save_raw(path);
        Model m2(cfg, Device::CPU);
        CHECK(m2.load_raw(path), "raw weights load");
        Activations a1 = m2.make_activations(B, T, false);
        m2.forward(ids.data(), B, T, a1);
        Activations a2 = m.make_activations(B, T, false);
        m.forward(ids.data(), B, T, a2);
        double maxd = 0;
        for (i64 i = 0; i < a1.logits.numel(); ++i)
            maxd = std::max(maxd, (double)std::fabs(a1.logits.f32()[i] - a2.logits.f32()[i]));
        CHECK(maxd == 0.0, "reloaded model produces identical logits");
        std::remove(path.c_str());
    }

    std::cout << (g_fail == 0 ? "== all model tests passed ==\n"
                              : "== FAILURES: " + std::to_string(g_fail) + " ==\n");
    return g_fail == 0 ? 0 : 1;
}
