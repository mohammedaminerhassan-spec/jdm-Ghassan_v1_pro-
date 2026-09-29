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
    const int NT = 65536;
    const int NE = 9;
    const int NBINS = 16;
    long bins[16] = {0};
    double sum = 0.0, sum2 = 0.0;
    double min_u = 1.0, max_u = -1.0;
    long n = 0;
    bool in_range = true;

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

    CHECK(min_u < -0.49 && max_u > 0.49, "output spans the full [-0.5, 0.5) range");
    const double mean = sum / n;
    const double var = sum2 / n - mean * mean;

    CHECK(std::fabs(mean) < 0.002, "mean near 0");
    CHECK(std::fabs(var - 1.0 / 12.0) < 0.001, "variance near 1/12");
    for (int e = 0; e < NE; ++e) {
        const double em = expert_sum[e] / NT;
        if (std::fabs(em) >= 0.005) {
            std::cerr << "FAIL: expert " << e << " mean " << em << " (biased)\n";
            ++failures;
        }
    }

    {
        const double expected = (double)n / NBINS;
        double chi2 = 0.0;
        for (int b = 0; b < NBINS; ++b) {
            double d = bins[b] - expected;
            chi2 += d * d / expected;
        }
        CHECK(chi2 < 60.0, "chi-square uniformity (no gross bias)");
    }

    {
        float a = cpu::jitter_u_for_test(7, 3);
        float b = cpu::jitter_u_for_test(7, 3);
        CHECK(a == b, "deterministic replay per (token, expert, seed)");
    }

    {
        float before = cpu::jitter_u_for_test(7, 3);
        cpu::set_moe_jitter_seed_cpu(999u);
        float after = cpu::jitter_u_for_test(7, 3);
        CHECK(before != after, "seed changes the noise stream");
        cpu::set_moe_jitter_seed_cpu(12345u);
        CHECK(cpu::jitter_u_for_test(7, 3) == before, "seed restore replays");
    }

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
