// GPU parity gate for the repair-prompt CUDA work (F-02, F-03, F-10).
//
// This test runs ONLY when built with CUDA (GAI_CUDA). On CPU-only builds it
// prints SKIP and passes, so `ctest` stays green everywhere. On Kaggle/T4 it
// compares the fused CUDA kernels against the CPU references:
//
//   F-03: cuda_ops::moe_forward (fused pack/swiglu/save/scatter) vs
//         cpu::moe_forward (token-major reference) — 1e-4 relative.
//   F-02: cuda_ops::moe_count_slots vs cpu::moe_count_slots — exact.
//   F-10: cuda_ops::sce_accumulate + sce_acc_end vs cpu single-shot — 1e-6.
//
// A failure here means the CUDA translation diverged from the validated logic
// and the run must NOT proceed to long training.
#include "core/ops.h"
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

#ifdef GAI_CUDA
#include "cuda/cuda_ops.h"
#include "core/device.h"

static bool near_vec(const std::vector<float>& a, const std::vector<float>& b,
                     double tol, const char* what) {
    if (a.size() != b.size()) {
        std::cerr << "FAIL: " << what << " size mismatch\n";
        ++failures;
        return false;
    }
    double max_rel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double denom = std::fabs((double)a[i]) + std::fabs((double)b[i]) + 1e-30;
        const double rel = std::fabs((double)a[i] - (double)b[i]) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    if (max_rel > tol) {
        std::cerr << "FAIL: " << what << " max rel err " << max_rel << "\n";
        ++failures;
        return false;
    }
    return true;
}

// Copy a host vector to a CUDA tensor and back.
static Tensor to_cuda(const std::vector<float>& h) {
    Tensor t = Tensor::empty({(i64)h.size()}, DType::F32, Device::CUDA);
    device_copy(t.data_ptr(), Device::CUDA, h.data(), Device::CPU,
                h.size() * sizeof(float));
    return t;
}
static Tensor to_cuda_i32(const std::vector<i32>& h) {
    Tensor t = Tensor::empty({(i64)h.size()}, DType::I32, Device::CUDA);
    device_copy(t.data_ptr(), Device::CUDA, h.data(), Device::CPU,
                h.size() * sizeof(i32));
    return t;
}
static std::vector<float> to_host(const Tensor& t) {
    std::vector<float> h(static_cast<size_t>(t.numel()));
    device_copy(h.data(), Device::CPU, t.data_ptr(), Device::CUDA,
                h.size() * sizeof(float));
    return h;
}

static void test_moe_forward_parity() {
    Rng rng;
    rng.seed_with(2024);
    const int N = 9, d = 16, E = 12, ne = 4, K = 2;
    const i64 NK = (i64)N * K;
    auto rnd = [&](size_t n, float s) {
        std::vector<float> v(n);
        for (float& x : v) x = (rng.uniform() * 2.0f - 1.0f) * s;
        return v;
    };
    const std::vector<float> x = rnd((size_t)N * d, 1.0f);
    const std::vector<float> router = rnd((size_t)ne * d, 0.5f);
    const std::vector<float> gates = rnd((size_t)ne * E * d, 0.3f);
    const std::vector<float> ups = rnd((size_t)ne * E * d, 0.3f);
    const std::vector<float> downs = rnd((size_t)ne * d * E, 0.3f);

    // CPU reference.
    std::vector<float> out_cpu((size_t)N * d, 0.0f);
    std::vector<float> probs((size_t)N * ne), w((size_t)NK);
    std::vector<i32> idx((size_t)NK);
    std::vector<float> sg((size_t)NK * E), su((size_t)NK * E), sa((size_t)NK * E);
    cpu::moe_forward(x.data(), router.data(), gates.data(), ups.data(), downs.data(),
                     nullptr, nullptr, nullptr, out_cpu.data(),
                     probs.data(), idx.data(), w.data(),
                     sg.data(), su.data(), sa.data(), N, d, E, ne, K);

    // CUDA fused path (caches double as the routing I/O).
    Tensor dx = to_cuda(x), dr = to_cuda(router), dg = to_cuda(gates),
           du = to_cuda(ups), dd = to_cuda(downs);
    Tensor dout = Tensor::zeros({(i64)N * d}, DType::F32, Device::CUDA);
    Tensor dprobs = Tensor::empty({(i64)N * ne}, DType::F32, Device::CUDA);
    Tensor didx = Tensor::empty({NK}, DType::I32, Device::CUDA);
    Tensor dw = Tensor::empty({NK}, DType::F32, Device::CUDA);
    Tensor dsg = Tensor::empty({NK * E}, DType::F32, Device::CUDA);
    Tensor dsu = Tensor::empty({NK * E}, DType::F32, Device::CUDA);
    Tensor dsa = Tensor::empty({NK * E}, DType::F32, Device::CUDA);
    cuda_ops::moe_forward(dx.f32(), dr.f32(), dg.f32(), du.f32(), dd.f32(),
                          nullptr, nullptr, nullptr, dout.f32(),
                          dprobs.f32(), didx.i32p(), dw.f32(),
                          dsg.f32(), dsu.f32(), dsa.f32(), N, d, E, ne, K);
    device_synchronize(Device::CUDA);
    near_vec(out_cpu, to_host(dout), 1e-4, "F-03: CUDA fused MoE forward vs CPU");
    near_vec(sg, to_host(dsg), 1e-5, "F-03: CUDA saved gate vs CPU");
    near_vec(su, to_host(dsu), 1e-5, "F-03: CUDA saved up vs CPU");
    near_vec(sa, to_host(dsa), 1e-5, "F-03: CUDA saved act vs CPU");
}

static void test_count_slots_parity() {
    const int ne = 4;
    const std::vector<i32> idx = {0, 1, 2, 3, 0, 0, 1, 3, 2, 1, 0, 3};
    const i64 NK = 12;
    std::vector<float> acc_cpu(ne, 0.0f);
    cpu::moe_count_slots(idx.data(), acc_cpu.data(), NK, ne);
    Tensor didx = to_cuda_i32(idx);
    Tensor dacc = Tensor::zeros({ne}, DType::F32, Device::CUDA);
    cuda_ops::moe_count_slots(didx.i32p(), dacc.f32(), NK, ne);
    device_synchronize(Device::CUDA);
    const std::vector<float> acc_gpu = to_host(dacc);
    bool same = acc_cpu.size() == acc_gpu.size();
    for (size_t i = 0; same && i < acc_cpu.size(); ++i)
        if (acc_cpu[i] != acc_gpu[i]) same = false;
    CHECK(same, "F-02: CUDA slot counts match CPU exactly");
}

static void test_sce_acc_parity() {
    Rng rng;
    rng.seed_with(99);
    const int V = 128;
    const i64 N = 32;
    std::vector<float> logits((size_t)N * V);
    for (float& x : logits) x = (rng.uniform() * 2.0f - 1.0f) * 2.0f;
    std::vector<i32> targets((size_t)N);
    for (i64 i = 0; i < N; ++i)
        targets[(size_t)i] = (i % 4 == 0) ? -100 : (i32)(i % V);
    std::vector<float> dlogits_cpu(logits.size(), 0.0f), dlogits_gpu(logits.size(), 0.0f);

    double ref_sum = 0.0;
    i64 ref_n = 0;
    cpu::softmax_cross_entropy(logits.data(), targets.data(), dlogits_cpu.data(),
                               N, V, &ref_sum, &ref_n, 1e-4f);

    Tensor dl = to_cuda(logits), dd = Tensor::empty({(i64)logits.size()}, DType::F32, Device::CUDA);
    Tensor dt = to_cuda_i32(targets);
    cuda_ops::sce_acc_begin();
    cuda_ops::sce_accumulate(dl.f32(), dt.i32p(), dd.f32(), N, V, 1e-4f);
    double gpu_sum = 0.0;
    i64 gpu_n = 0;
    cuda_ops::sce_acc_end(&gpu_sum, &gpu_n);
    CHECK(gpu_n == ref_n, "F-10: CUDA accumulate count matches CPU");
    CHECK(std::fabs(gpu_sum - ref_sum) / (std::fabs(ref_sum) + 1e-30) < 1e-6,
          "F-10: CUDA accumulate loss matches CPU");
    near_vec(dlogits_cpu, to_host(dd), 1e-5, "F-10: CUDA dlogits match CPU");
}

// Full MoE backward parity (P1-5): the most complex path (router GEMM grad,
// per-expert weight grads, dx scatter, shared-expert recompute) must match
// the serial CPU reference, or long training silently diverges.
static void test_moe_backward_parity() {
    Rng rng;
    rng.seed_with(2025);
    const int N = 9, d = 16, E = 12, ne = 4, K = 2;
    const i64 NK = (i64)N * K;
    auto rnd = [&](size_t n, float s) {
        std::vector<float> v(n);
        for (float& x : v) x = (rng.uniform() * 2.0f - 1.0f) * s;
        return v;
    };
    const std::vector<float> x = rnd((size_t)N * d, 1.0f);
    const std::vector<float> router = rnd((size_t)ne * d, 0.5f);
    const std::vector<float> gates = rnd((size_t)ne * E * d, 0.3f);
    const std::vector<float> ups = rnd((size_t)ne * E * d, 0.3f);
    const std::vector<float> downs = rnd((size_t)ne * d * E, 0.3f);
    const std::vector<float> dout = rnd((size_t)N * d, 0.5f);

    // Forward on CPU for the shared caches (probs/idx/w/s_*).
    std::vector<float> out_cpu((size_t)N * d, 0.0f);
    std::vector<float> probs((size_t)N * ne), w((size_t)NK);
    std::vector<i32> idx((size_t)NK);
    std::vector<float> sg((size_t)NK * E), su((size_t)NK * E), sa((size_t)NK * E);
    cpu::moe_forward(x.data(), router.data(), gates.data(), ups.data(), downs.data(),
                     nullptr, nullptr, nullptr, out_cpu.data(),
                     probs.data(), idx.data(), w.data(),
                     sg.data(), su.data(), sa.data(), N, d, E, ne, K);

    // CPU backward reference.
    std::vector<float> dx_cpu((size_t)N * d, 0.0f);
    std::vector<float> dr_cpu((size_t)ne * d, 0.0f);
    std::vector<float> dg_cpu((size_t)ne * E * d, 0.0f);
    std::vector<float> du_cpu((size_t)ne * E * d, 0.0f);
    std::vector<float> dd_cpu((size_t)ne * d * E, 0.0f);
    std::vector<float> s_dact_cpu((size_t)NK * E, 0.0f);
    cpu::moe_backward(x.data(), router.data(), gates.data(), ups.data(), downs.data(),
                      nullptr, nullptr, nullptr,
                      probs.data(), idx.data(), w.data(),
                      sg.data(), su.data(), sa.data(),
                      nullptr, 0.0f, dout.data(), dx_cpu.data(),
                      dr_cpu.data(), dg_cpu.data(), du_cpu.data(), dd_cpu.data(),
                      nullptr, nullptr, nullptr, s_dact_cpu.data(),
                      N, d, E, ne, K);

    // CUDA backward on the same inputs.
    Tensor dx_ = to_cuda(x), dr_ = to_cuda(router), dg_ = to_cuda(gates),
           du_ = to_cuda(ups), dd_ = to_cuda(downs);
    Tensor dprobs = to_cuda(probs), didx = to_cuda_i32(idx), dw = to_cuda(w);
    Tensor dsg = to_cuda(sg), dsu = to_cuda(su), dsa = to_cuda(sa);
    Tensor ddout = to_cuda(dout);
    Tensor gdx = Tensor::zeros({(i64)N * d}, DType::F32, Device::CUDA);
    Tensor gdr = Tensor::zeros({(i64)ne * d}, DType::F32, Device::CUDA);
    Tensor gdg = Tensor::zeros({(i64)ne * E * d}, DType::F32, Device::CUDA);
    Tensor gdu = Tensor::zeros({(i64)ne * E * d}, DType::F32, Device::CUDA);
    Tensor gdd = Tensor::zeros({(i64)ne * d * E}, DType::F32, Device::CUDA);
    Tensor gsd = Tensor::zeros({NK * E}, DType::F32, Device::CUDA);
    cuda_ops::moe_backward(dx_.f32(), dr_.f32(), dg_.f32(), du_.f32(), dd_.f32(),
                           nullptr, nullptr, nullptr,
                           dprobs.f32(), didx.i32p(), dw.f32(),
                           dsg.f32(), dsu.f32(), dsa.f32(),
                           nullptr, 0.0f, ddout.f32(), gdx.f32(),
                           gdr.f32(), gdg.f32(), gdu.f32(), gdd.f32(),
                           nullptr, nullptr, nullptr, gsd.f32(),
                           N, d, E, ne, K);
    device_synchronize(Device::CUDA);
    near_vec(dx_cpu, to_host(gdx), 1e-4, "moe bwd dx vs CPU");
    near_vec(dr_cpu, to_host(gdr), 1e-4, "moe bwd router grad vs CPU");
    near_vec(dg_cpu, to_host(gdg), 1e-4, "moe bwd gate grad vs CPU");
    near_vec(du_cpu, to_host(gdu), 1e-4, "moe bwd up grad vs CPU");
    near_vec(dd_cpu, to_host(gdd), 1e-4, "moe bwd down grad vs CPU");
}
#endif // GAI_CUDA

int main() {
#ifdef GAI_CUDA
    if (!cuda_available()) {
        std::cout << "test_moe_cuda_parity: SKIP (no CUDA device)\n";
        return 0;
    }
    test_moe_forward_parity();
    test_moe_backward_parity();
    test_count_slots_parity();
    test_sce_acc_parity();
#else
    std::cout << "test_moe_cuda_parity: SKIP (CPU-only build)\n";
#endif
    if (failures == 0) {
        std::cout << "test_moe_cuda_parity: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_moe_cuda_parity: " << failures << " FAILURES\n";
    return 1;
}
