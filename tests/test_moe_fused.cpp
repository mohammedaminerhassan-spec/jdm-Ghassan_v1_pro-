// F-03 regression: the fused grouped MoE forward must be bit-exact against
// the token-major reference. This validates the pack/save/scatter fusion
// LOGIC (indexing, grouped layout, weight application) that cuda/moe.cu
// implements in k_pack_all / k_save3_all / k_scatter_add_all — so a CUDA bug
// can only be a translation typo, never a design error.
#include "core/ops_cpu.h"
#include "core/rng.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static bool exact(const std::vector<float>& a, const std::vector<float>& b,
                  const char* what) {
    if (a.size() != b.size()) {
        std::cerr << "FAIL: " << what << " size " << a.size() << " vs " << b.size() << "\n";
        ++failures;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "FAIL: " << what << " differs at " << i
                      << " (" << a[i] << " vs " << b[i] << ")\n";
            ++failures;
            return false;
        }
    }
    return true;
}

// The final output accumulates expert contributions in grouped order instead
// of token order, so (like every other CPU/GPU pair in this repo) it agrees
// to ~1 ulp rather than bitwise. Everything that feeds backward (the saved
// G/U/A copies) is a pure copy and must be bitwise.
static bool near_vec(const std::vector<float>& a, const std::vector<float>& b,
                     const char* what) {
    if (a.size() != b.size()) {
        std::cerr << "FAIL: " << what << " size " << a.size() << " vs " << b.size() << "\n";
        ++failures;
        return false;
    }
    double max_rel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double denom = std::fabs((double)a[i]) + std::fabs((double)b[i]) + 1e-30;
        const double rel = std::fabs((double)a[i] - (double)b[i]) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    if (max_rel > 1e-6) {
        std::cerr << "FAIL: " << what << " max rel err " << max_rel << " (>1e-6)\n";
        ++failures;
        return false;
    }
    return true;
}

int main() {
    Rng rng;
    rng.seed_with(1234);
    const int N = 11;      // odd on purpose (tails, uneven expert loads)
    const int d = 12;      // divisible by 4 (float4 path) ...
    const int E = 10;      // ... while E is not (scalar tail path)
    const int ne = 5;
    const int K = 2;
    const i64 NK = static_cast<i64>(N) * K;

    auto rnd = [&](std::vector<float>& v, float s) {
        v.resize(v.size());
        for (float& x : v) x = (rng.uniform() * 2.0f - 1.0f) * s;
    };
    std::vector<float> x(static_cast<size_t>(N) * d), router(static_cast<size_t>(ne) * d),
        gates(static_cast<size_t>(ne) * E * d), ups(static_cast<size_t>(ne) * E * d),
        downs(static_cast<size_t>(ne) * d * E),
        sh_g(static_cast<size_t>(E) * d), sh_u(static_cast<size_t>(E) * d),
        sh_d(static_cast<size_t>(d) * E);
    rnd(x, 1.0f); rnd(router, 0.5f); rnd(gates, 0.3f); rnd(ups, 0.3f);
    rnd(downs, 0.3f); rnd(sh_g, 0.3f); rnd(sh_u, 0.3f); rnd(sh_d, 0.3f);

    const size_t nslot = static_cast<size_t>(NK);
    std::vector<float> out_ref(static_cast<size_t>(N) * d, 0.0f),
        out_fused(static_cast<size_t>(N) * d, 0.0f);
    std::vector<float> probs_ref(static_cast<size_t>(N) * ne), probs_fused(static_cast<size_t>(N) * ne);
    std::vector<i32> idx_ref(nslot), idx_fused(nslot);
    std::vector<float> w_ref(nslot), w_fused(nslot);
    std::vector<float> sg_ref(nslot * E), su_ref(nslot * E), sa_ref(nslot * E);
    std::vector<float> sg_fused(nslot * E), su_fused(nslot * E), sa_fused(nslot * E);
    std::vector<float> Xpack(nslot * d), Gpack(nslot * E), Upack(nslot * E),
        Apack(nslot * E), Ypack(nslot * d);
    std::vector<i32> grouped(nslot);
    std::vector<int> counts(ne), offsets(ne + 1);

    cpu::moe_forward(x.data(), router.data(), gates.data(), ups.data(), downs.data(),
                     sh_g.data(), sh_u.data(), sh_d.data(), out_ref.data(),
                     probs_ref.data(), idx_ref.data(), w_ref.data(),
                     sg_ref.data(), su_ref.data(), sa_ref.data(), N, d, E, ne, K);
    cpu::moe_forward_fused(x.data(), router.data(), gates.data(), ups.data(), downs.data(),
                           sh_g.data(), sh_u.data(), sh_d.data(), out_fused.data(),
                           probs_fused.data(), idx_fused.data(), w_fused.data(),
                           sg_fused.data(), su_fused.data(), sa_fused.data(),
                           Xpack.data(), Gpack.data(), Upack.data(), Apack.data(),
                           Ypack.data(), grouped.data(), counts.data(), offsets.data(),
                           N, d, E, ne, K);

    near_vec(out_ref, out_fused, "F-03: fused output == reference output (1-ulp)");
    exact(probs_ref, probs_fused, "F-03: router probs identical");
    exact(sg_ref, sg_fused, "F-03: saved gate activations identical");
    exact(su_ref, su_fused, "F-03: saved up activations identical");
    exact(sa_ref, sa_fused, "F-03: saved swiglu activations identical");

    // The grouped layout must be a permutation of the slots (each slot routed
    // exactly once; empty experts contribute nothing).
    {
        std::vector<int> seen(nslot, 0);
        int total = 0;
        for (int e = 0; e < ne; ++e) {
            CHECK(offsets[e + 1] - offsets[e] == counts[e], "F-03: offsets/counts agree");
            total += counts[e];
            for (int s = offsets[e]; s < offsets[e + 1]; ++s) {
                CHECK(grouped[s] >= 0 && grouped[s] < NK, "F-03: grouped slot in range");
                seen[grouped[s]]++;
            }
        }
        CHECK(total == NK, "F-03: every slot is grouped exactly once");
        for (size_t s = 0; s < nslot; ++s) CHECK(seen[s] == 1, "F-03: no slot lost or duplicated");
    }

    // idx/w caches agree (same routing decisions).
    {
        bool same = true;
        for (size_t s = 0; s < nslot; ++s)
            if (idx_ref[s] != idx_fused[s] || w_ref[s] != w_fused[s]) { same = false; break; }
        CHECK(same, "F-03: routing decisions identical");
    }

    // Dense (no shared expert) variant: routed contribution starts from zero.
    {
        std::fill(out_ref.begin(), out_ref.end(), 0.0f);
        std::fill(out_fused.begin(), out_fused.end(), 0.0f);
        cpu::moe_forward(x.data(), router.data(), gates.data(), ups.data(), downs.data(),
                         nullptr, nullptr, nullptr, out_ref.data(),
                         probs_ref.data(), idx_ref.data(), w_ref.data(),
                         sg_ref.data(), su_ref.data(), sa_ref.data(), N, d, E, ne, K);
        cpu::moe_forward_fused(x.data(), router.data(), gates.data(), ups.data(), downs.data(),
                               nullptr, nullptr, nullptr, out_fused.data(),
                               probs_fused.data(), idx_fused.data(), w_fused.data(),
                               sg_fused.data(), su_fused.data(), sa_fused.data(),
                               Xpack.data(), Gpack.data(), Upack.data(), Apack.data(),
                               Ypack.data(), grouped.data(), counts.data(), offsets.data(),
                               N, d, E, ne, K);
        near_vec(out_ref, out_fused, "F-03: fused == reference without a shared expert");
    }

    if (failures == 0) {
        std::cout << "test_moe_fused: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_moe_fused: " << failures << " FAILURES\n";
    return 1;
}
