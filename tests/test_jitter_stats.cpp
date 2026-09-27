// Jitter-hash statistical gate (audit P1): the MoE train-only router noise
// must be ~uniform on [-0.5, 0.5), deterministic per (token, expert, seed),
// and seed-sensitive. A biased hash would silently skew expert routing.
// Uses the real production hash via gai::cpu::jitter_u_for_test (fmix64,
// shared with cuda/moe.cu jitter_u), not a copy.
#include "core/ops_cpu.h"

#include <cmath>
#include <iostream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    cpu::set_moe_jitter_seed_cpu(12345u);
    const int NT = 65536;  // tokens (~590k samples: tighter gates than 36k)
    const int NE = 9;      // experts (1B recipe width)
    const int NBINS = 16;
    long bins[16] = {0};
    double sum = 0.0, sum2 = 0.0;
    double min_u = 1.0, max_u = -1.0;
    long n = 0;
    bool in_range = true;
    // Per-expert means: no expert may carry a systematic bias (that would
    // skew routing toward/away from it on every step).
    double expert_sum[9] = {0.0};
    for (int t = 0; t < NT; ++t) {
        for (int e = 0; e < NE; ++e) {
            float u = cpu::jitter_u_for_test(t, e);
            if (!(u >= -0.5f && u < 0.5f)) { in_range = false; continue; }
            sum += u; sum2 += u * u; ++n;
            expert_sum[e] += u;
            if (u < min_u) min_u = u;
            if (u > max_u) max_u = u;
            int b = static_cast<int>((u + 0.5f) * NBINS);
            if (b < 0) b = 0;
            if (b >= NBINS) b = NBINS - 1;
            ++bins[b];
        }
    }
    CHECK(in_range, "jitter values stay in [-0.5, 0.5)");
    CHECK(n == (long)NT * NE, "all samples collected");
    // Full 24-bit output range must be exercised (a degenerate hash that only
    // emits a narrow band would still pass mean/var tests).
    CHECK(min_u < -0.49 && max_u > 0.49, "output spans the full [-0.5, 0.5) range");
    const double mean = sum / n;
    const double var = sum2 / n - mean * mean;
    // U[-0.5,0.5): mean 0, var 1/12 ~= 0.08333. Tighter gates (n=590k).
    CHECK(std::fabs(mean) < 0.002, "mean near 0");
    CHECK(std::fabs(var - 1.0 / 12.0) < 0.001, "variance near 1/12");
    for (int e = 0; e < NE; ++e) {
        const double em = expert_sum[e] / NT;
        if (std::fabs(em) >= 0.005) {
            std::cerr << "FAIL: expert " << e << " mean " << em << " (biased)\n";
            ++failures;
        }
    }
    // Chi-square uniformity over 16 bins (df=15; 0.1% critical ~= 37.7).
    // Gate at 60: catches gross bias, tolerates hash lumpiness.
    {
        const double expected = (double)n / NBINS;
        double chi2 = 0.0;
        for (int b = 0; b < NBINS; ++b) {
            double d = bins[b] - expected;
            chi2 += d * d / expected;
        }
        CHECK(chi2 < 60.0, "chi-square uniformity (no gross bias)");
    }
    // Determinism: same (token, expert, seed) replays bit-exact.
    {
        float a = cpu::jitter_u_for_test(7, 3);
        float b = cpu::jitter_u_for_test(7, 3);
        CHECK(a == b, "deterministic replay per (token, expert, seed)");
    }
    // Seed sensitivity: a new seed must change the stream.
    {
        float before = cpu::jitter_u_for_test(7, 3);
        cpu::set_moe_jitter_seed_cpu(999u);
        float after = cpu::jitter_u_for_test(7, 3);
        CHECK(before != after, "seed changes the noise stream");
        cpu::set_moe_jitter_seed_cpu(12345u);
        CHECK(cpu::jitter_u_for_test(7, 3) == before, "seed restore replays");
    }
    // Neighbor diffusion: adjacent tokens/experts must not collide.
    {
        int same = 0;
        for (int t = 0; t < 512; ++t)
            if (cpu::jitter_u_for_test(t, 0) == cpu::jitter_u_for_test(t + 1, 0)) ++same;
        CHECK(same == 0, "no neighbor collisions across tokens");
    }

    if (failures == 0) { std::cout << "test_jitter_stats: ALL PASS\n"; return 0; }
    std::cerr << "test_jitter_stats: " << failures << " FAILURES\n";
    return 1;
}
