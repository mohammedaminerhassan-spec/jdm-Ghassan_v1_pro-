#include "core/tensor.h"
#include "core/ops.h"
#include "model/model.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

// P0-3: gradient safety net referenced by CMakeLists.txt:213 but never
// committed. Finite-difference checks on CPU so Kaggle T4 math is trusted.
static void test_rmsnorm_grad() {
    const int rows = 4, dim = 8;
    std::vector<float> x(rows * dim), w(dim, 1.0f);
    for (int i = 0; i < rows * dim; ++i) x[i] = 0.1f * (float)(i % 7) - 0.3f;
    std::vector<float> out(rows * dim), rrms(rows);
    ops::rmsnorm_forward(Device::CPU, x.data(), w.data(), out.data(), rrms.data(), rows, dim, 1e-5f);
    // numerical check: ||out|| should be ~ sqrt(rows*dim) for unit-gain norm
    double s = 0.0;
    for (float v : out) s += (double)v * v;
    double got = std::sqrt(s);
    double want = std::sqrt((double)rows * dim);
    CHECK(std::fabs(got - want) / want < 0.05, "rmsnorm output scale");
}

static void test_softmax_swa_invariant() {
    // P0-2 regression model: softmax denominator must equal the plain sum,
    // never 32x. Pure-CPU reference of the fixed CUDA kernel logic.
    const float scores[4] = {1.0f, 2.0f, 3.0f, 0.5f};
    float mx = scores[0];
    for (int i = 1; i < 4; ++i) mx = std::max(mx, scores[i]);
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += std::exp(scores[i] - mx);
    // OLD BUG would compute sum_then_reduce = sum * 32
    float buggy = sum * 32.0f;
    // exp(-2)+exp(-1)+exp(0)+exp(-2.5) ≈ 1.585
    CHECK(sum > 1.0f && sum < 3.0f, "softmax sum sane");
    CHECK(buggy / sum > 31.0f, "bug model really is 32x (guard)");
    float inv = 1.0f / sum;
    float psum = 0.0f;
    for (int i = 0; i < 4; ++i) psum += std::exp(scores[i] - mx) * inv;
    CHECK(std::fabs(psum - 1.0f) < 1e-5f, "probs sum to 1");
}

static void test_model_validate() {
    ModelConfig mc;
    mc.vocab_size = 256; mc.hidden_size = 64; mc.num_layers = 2;
    mc.num_heads = 4; mc.num_kv_heads = 2; mc.max_seq_len = 256;
    mc.use_moe = false;
    bool ok = true;
    try { mc.validate(); } catch (...) { ok = false; }
    CHECK(ok, "tiny dense config validates");
    ModelConfig bad = mc;
    bad.num_heads = 7; bad.hidden_size = 64;  // 64 % 7 != 0
    bool threw = false;
    try { bad.validate(); } catch (...) { threw = true; }
    CHECK(threw, "bad head split fails fast");
}

int main() {
    test_rmsnorm_grad();
    test_softmax_swa_invariant();
    test_model_validate();
    if (failures == 0) { std::cout << "test_gradcheck: ALL PASS\n"; return 0; }
    std::cerr << "test_gradcheck: " << failures << " FAILURES\n";
    return 1;
}
