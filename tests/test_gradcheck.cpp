// test_gradcheck - T4-only manual-backward safety net (CPU, no GPU needed).
// Finite-difference gradient check for the most training-critical ops:
//   linear, rmsnorm, swiglu, softmax_cross_entropy.
// Any failure returns 1 with a clear message (fail fast before a long run).
// Deterministic: fixed values, no RNG.

#include "core/ops.h"
#include "core/device.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using gai::i32;
using gai::i64;

constexpr float kEps = 1e-3f;
constexpr float kTol = 5e-2f;  // fp32 finite-diff tolerance (loose but catches sign/scale bugs)

int g_fail = 0;

void check_close(const char* name, float analytic, float numeric) {
    const float diff = std::fabs(analytic - numeric);
    const float denom = std::fabs(numeric) + 1.0f;
    if (diff / denom > kTol) {
        std::printf("[FAIL] %s analytic=%.6f numeric=%.6f diff=%.6f\n",
                    name, analytic, numeric, diff);
        g_fail = 1;
    }
}

// ---- linear: y = x @ W^T, loss = sum(y) -> dy=1, dx=sum_rows(W), dw=sum_rows(x)
void test_linear() {
    constexpr int M = 2, K = 3, N = 2;
    const std::vector<float> x = {0.5f, -0.25f, 1.0f, 0.75f, 0.1f, -0.5f};
    const std::vector<float> w = {0.2f, -0.4f, 0.6f, 1.0f, 0.3f, -0.7f};
    std::vector<float> y(static_cast<size_t>(M * N), 0.0f);
    gai::ops::linear_forward(gai::Device::CPU, x.data(), w.data(), y.data(), M, K, N);
    // numeric dw[0,0] = d sum(y)/d w[0,0] = x[0,0]+x[1,0] pattern: y[m,n]=sum_k x[m,k]w[n,k]
    // dL/dw[n,k] = sum_m dy[m,n]*x[m,k], dy=1 -> x[0,k]+x[1,k]
    std::vector<float> dy(static_cast<size_t>(M * N), 1.0f);
    std::vector<float> dx(static_cast<size_t>(M * K), 0.0f);
    std::vector<float> dw(static_cast<size_t>(N * K), 0.0f);
    gai::ops::linear_backward(gai::Device::CPU, x.data(), w.data(), dy.data(),
                              dx.data(), dw.data(), M, K, N);
    // analytic dw[0,0] = x[0,0]+x[1,0] = 0.5+0.75 = 1.25
    check_close("linear dw[0,0]", dw[0], 1.25f);
    // analytic dw[1,2] = x[0,2]+x[1,2] = 1.0+(-0.5) = 0.5
    check_close("linear dw[1,2]", dw[static_cast<size_t>(1 * K + 2)], 0.5f);
    // analytic dx[0,0] = sum_n w[n,0] = 0.2+1.0 = 1.2
    check_close("linear dx[0,0]", dx[0], 1.2f);
    // numeric check on one element via finite diff
    std::vector<float> xp = x;
    xp[0] += kEps;
    std::vector<float> yp(static_cast<size_t>(M * N), 0.0f);
    gai::ops::linear_forward(gai::Device::CPU, xp.data(), w.data(), yp.data(), M, K, N);
    float sum0 = 0.0f, sum1 = 0.0f;
    for (float v : y) sum0 += v;
    for (float v : yp) sum1 += v;
    check_close("linear dx[0,0]-numeric", dx[0], (sum1 - sum0) / kEps);
    std::printf("[ok] linear\n");
}

// ---- rmsnorm: check dweight via finite diff on tiny case
void test_rmsnorm() {
    constexpr i64 rows = 2;
    constexpr int dim = 4;
    const std::vector<float> x = {0.5f, -1.0f, 1.5f, 0.25f, 0.75f, 0.5f, -0.5f, 1.0f};
    const std::vector<float> wt = {1.0f, 0.5f, 2.0f, 1.5f};
    std::vector<float> out(static_cast<size_t>(rows * dim), 0.0f);
    std::vector<float> rrms(static_cast<size_t>(rows), 0.0f);
    gai::ops::rmsnorm_forward(gai::Device::CPU, x.data(), wt.data(), out.data(),
                              rrms.data(), rows, dim, 1e-5f);
    std::vector<float> dout(static_cast<size_t>(rows * dim), 1.0f);
    std::vector<float> dx(static_cast<size_t>(rows * dim), 0.0f);
    std::vector<float> dw(static_cast<size_t>(dim), 0.0f);
    gai::ops::rmsnorm_backward(gai::Device::CPU, x.data(), wt.data(), dout.data(),
                               rrms.data(), dx.data(), dw.data(), rows, dim);
    // numeric dw[0]: perturb wt[0]
    std::vector<float> wtp = wt;
    wtp[0] += kEps;
    std::vector<float> outp(static_cast<size_t>(rows * dim), 0.0f);
    std::vector<float> rtmp(static_cast<size_t>(rows), 0.0f);
    gai::ops::rmsnorm_forward(gai::Device::CPU, x.data(), wtp.data(), outp.data(),
                              rtmp.data(), rows, dim, 1e-5f);
    float s0 = 0.0f, s1 = 0.0f;
    for (float v : out) s0 += v;
    for (float v : outp) s1 += v;
    check_close("rmsnorm dw[0]", dw[0], (s1 - s0) / kEps);
    std::printf("[ok] rmsnorm\n");
}

// ---- swiglu: out = silu(g)*u, check dg[0] numeric
void test_swiglu() {
    constexpr i64 n = 4;
    const std::vector<float> g = {0.5f, -0.5f, 1.0f, 0.0f};
    const std::vector<float> u = {1.0f, 2.0f, -1.0f, 0.5f};
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    gai::ops::swiglu_forward(gai::Device::CPU, g.data(), u.data(), out.data(), n);
    std::vector<float> dout(static_cast<size_t>(n), 1.0f);
    std::vector<float> dg(static_cast<size_t>(n), 0.0f);
    std::vector<float> du(static_cast<size_t>(n), 0.0f);
    gai::ops::swiglu_backward(gai::Device::CPU, g.data(), u.data(), dout.data(),
                              dg.data(), du.data(), n);
    // numeric dg[0]
    std::vector<float> gp = g;
    gp[0] += kEps;
    std::vector<float> outp(static_cast<size_t>(n), 0.0f);
    gai::ops::swiglu_forward(gai::Device::CPU, gp.data(), u.data(), outp.data(), n);
    float s0 = 0.0f, s1 = 0.0f;
    for (float v : out) s0 += v;
    for (float v : outp) s1 += v;
    check_close("swiglu dg[0]", dg[0], (s1 - s0) / kEps);
    // du[0] = silu(g0): silu(0.5)=0.5*sigmoid(0.5)~=0.3112
    const float silu05 = 0.5f / (1.0f + std::exp(-0.5f));
    check_close("swiglu du[0]", du[0], silu05);
    std::printf("[ok] swiglu\n");
}

// ---- softmax CE: dlogits rows must sum to ~0, ignored targets give 0 grad
void test_softmax_ce() {
    constexpr i64 n = 2;
    constexpr int V = 4;
    const std::vector<float> logits = {1.0f, 2.0f, 3.0f, 0.5f, 0.1f, 0.2f, 0.3f, 0.4f};
    const std::vector<gai::i32> targets = {2, -100};
    std::vector<float> dlogits(static_cast<size_t>(n * V), 0.0f);
    double loss_sum = 0.0;
    gai::i64 count = 0;
    gai::ops::softmax_cross_entropy(gai::Device::CPU, logits.data(), targets.data(),
                                    dlogits.data(), n, V, &loss_sum, &count, 0.0f);
    if (count != 1) {
        std::printf("[FAIL] softmax_ce count=%lld want 1\n", static_cast<long long>(count));
        g_fail = 1;
    }
    float row0 = dlogits[0] + dlogits[1] + dlogits[2] + dlogits[3];
    if (std::fabs(row0) > 1e-4f) {
        std::printf("[FAIL] softmax_ce row0 sum=%.6f want ~0\n", row0);
        g_fail = 1;
    }
    const float row1 = dlogits[4] + dlogits[5] + dlogits[6] + dlogits[7];
    if (std::fabs(row1) > 1e-6f) {
        std::printf("[FAIL] softmax_ce ignored row grad=%.6f want 0\n", row1);
        g_fail = 1;
    }
    if (!(loss_sum > 0.0)) {
        std::printf("[FAIL] softmax_ce loss=%f want >0\n", loss_sum);
        g_fail = 1;
    }
    std::printf("[ok] softmax_ce\n");
}

}  // namespace

int main() {
    test_linear();
    test_rmsnorm();
    test_swiglu();
    test_softmax_ce();
    if (g_fail == 0) std::printf("gradcheck: ALL OK\n");
    else std::printf("gradcheck: FAILURES\n");
    return g_fail;
}
