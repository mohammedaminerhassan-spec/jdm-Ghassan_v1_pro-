// F-10 regression: the device-side loss/count accumulate API must agree with
// repeated single-shot calls, and the host-side fully-masked-block skip must
// not change numerics. Runs the CPU backend; on CUDA the same dispatchers in
// core/ops.cpp route to the device accumulators in cuda/kernels.cu.
#include "core/ops.h"
#include "core/rng.h"
#include "model/model.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    Rng rng;
    rng.seed_with(77);
    const int V = 64;
    const i64 N = 40;

    std::vector<float> logits(static_cast<size_t>(N) * V);
    for (float& x : logits) x = (rng.uniform() * 2.0f - 1.0f) * 3.0f;
    std::vector<i32> targets(static_cast<size_t>(N));
    for (i64 i = 0; i < N; ++i)
        targets[static_cast<size_t>(i)] = (i % 3 == 0) ? -100 : static_cast<i32>(i % V);

    // ---- 1. accumulate API == sum of single-shot calls --------------------
    double ref_sum = 0.0;
    i64 ref_n = 0;
    std::vector<float> dlogits_ref(logits.size(), 0.0f);
    {
        const i64 Cc = 16;
        for (i64 r0 = 0; r0 < N; r0 += Cc) {
            const i64 Cr = std::min(Cc, N - r0);
            double csum = 0.0;
            i64 cn = 0;
            ops::softmax_cross_entropy(Device::CPU, logits.data() + r0 * V,
                                       targets.data() + r0,
                                       dlogits_ref.data() + r0 * V,
                                       Cr, V, &csum, &cn, 0.0f);
            ref_sum += csum;
            ref_n += cn;
        }
    }
    double acc_sum = 0.0;
    i64 acc_n = 0;
    std::vector<float> dlogits_acc(logits.size(), 0.0f);
    {
        const i64 Cc = 16;
        ops::sce_acc_begin(Device::CPU);
        for (i64 r0 = 0; r0 < N; r0 += Cc) {
            const i64 Cr = std::min(Cc, N - r0);
            ops::sce_accumulate(Device::CPU, logits.data() + r0 * V,
                                targets.data() + r0,
                                dlogits_acc.data() + r0 * V, Cr, V, 0.0f);
        }
        ops::sce_acc_end(Device::CPU, &acc_sum, &acc_n);
    }
    CHECK(ref_n == acc_n && ref_n > 0, "F-10: accumulate counts match single-shot");
    CHECK(std::fabs(ref_sum - acc_sum) / (std::fabs(ref_sum) + 1e-30) < 1e-9,
          "F-10: accumulate loss matches single-shot");
    {
        bool same = dlogits_ref.size() == dlogits_acc.size();
        for (size_t i = 0; same && i < dlogits_ref.size(); ++i)
            if (dlogits_ref[i] != dlogits_acc[i]) same = false;
        CHECK(same, "F-10: accumulate writes identical dlogits");
    }

    // ---- 2. z-loss scale flows through the accumulate path ----------------
    {
        ops::sce_acc_begin(Device::CPU);
        ops::sce_accumulate(Device::CPU, logits.data(), targets.data(),
                            dlogits_acc.data(), N, V, 1e-4f);
        double zsum = 0.0;
        i64 zn = 0;
        ops::sce_acc_end(Device::CPU, &zsum, &zn);
        double ref_z = 0.0;
        i64 ref_zn = 0;
        ops::softmax_cross_entropy(Device::CPU, logits.data(), targets.data(),
                                   dlogits_ref.data(), N, V, &ref_z, &ref_zn, 1e-4f);
        CHECK(zn == ref_zn && std::fabs(zsum - ref_z) / (std::fabs(ref_z) + 1e-30) < 1e-9,
              "F-10: z-loss scale is honoured by the accumulate path");
    }

    // ---- 3. host mirror skip: fully-masked blocks change nothing -----------
    {
        ModelConfig cfg;
        cfg.vocab_size = static_cast<int>(V);
        cfg.hidden_size = 16;
        cfg.num_layers = 1;
        cfg.num_heads = 2;
        cfg.num_kv_heads = 1;
        cfg.intermediate_size = 32;
        cfg.max_seq_len = 32;
        Model m(cfg, Device::CPU);
        m.init_weights(9);
        m.enable_grad(true);
        const int B = 1, T = 10;
        Activations act = m.make_activations(B, T, true, 4);
        std::vector<i32> ids(static_cast<size_t>(B) * T), tgt(static_cast<size_t>(B) * T, -100);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<i32>(i % (V - 1)) + 1;
        for (size_t i = 0; i < 4; ++i) tgt[i] = ids[i + 1];   // first block supervised...
        // ...the tail blocks are fully masked (PAD-like).
        i64 ntok_mirror = -1, ntok_plain = -2;
        const double l_mirror = m.forward_backward(ids.data(), tgt.data(), B, T, act,
                                                   &ntok_mirror, 1.0f, false, nullptr,
                                                   tgt.data());
        Activations act2 = m.make_activations(B, T, true, 4);
        const double l_plain = m.forward_backward(ids.data(), tgt.data(), B, T, act2,
                                                  &ntok_plain, 1.0f, false, nullptr, nullptr);
        CHECK(ntok_mirror == ntok_plain && ntok_mirror == 4,
              "F-10: host-mirror skip supervises exactly the same tokens");
        CHECK(std::fabs(l_mirror - l_plain) < 1e-9,
              "F-10: host-mirror skip does not change the loss");
    }

    if (failures == 0) {
        std::cout << "test_sce_accumulate: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_sce_accumulate: " << failures << " FAILURES\n";
    return 1;
}
