// Attention CUDA parity gate (P1-4): the custom kernels in cuda/attention.cu
// (flash-style tiled forward, GQA, SWA, segment-masked packing) must match
// the CPU references in core/ops_cpu.cpp before any production run.
//
// Runs ONLY when built with CUDA on a machine with a device; otherwise prints
// SKIP and passes (same contract as test_moe_cuda_parity). Covers:
//   1. causal forward, GQA (H=6/KV=2)
//   2. full backward (dq/dk/dv) vs serial CPU reference
//   3. SWA window forward + backward
//   4. segment-id packing forward
//   5. T=1 edge case
// NOTE on tolerances: the CUDA backward accumulates dk/dv with atomics, so
// summation order differs from the serial CPU reference by design; 1e-4
// relative covers float reassociation without hiding real divergence.
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
        double denom = std::fabs((double)b[i]) + 1e-6;
        double rel = std::fabs((double)a[i] - (double)b[i]) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    if (max_rel > tol) {
        std::cerr << "FAIL: " << what << " max_rel=" << max_rel << " tol=" << tol << "\n";
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
    std::vector<float> h((size_t)t.numel());
    device_copy(h.data(), Device::CPU, t.data_ptr(), Device::CUDA,
                h.size() * sizeof(float));
    return h;
}

static std::vector<float> rnd_vec(Rng& rng, size_t n, float s) {
    std::vector<float> v(n);
    for (float& x : v) x = (rng.uniform() * 2.0f - 1.0f) * s;
    return v;
}

static void test_forward_gqa() {
    Rng rng;
    rng.seed_with(777);
    const int B = 2, T = 9, H = 6, KV = 2, hd = 8;
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::vector<float> q = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);
    const std::vector<float> k = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> v = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);

    std::vector<float> out_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> probs_cpu((size_t)B * H * T * T, 0.0f);
    cpu::attention_forward(q.data(), k.data(), v.data(), out_cpu.data(),
                           probs_cpu.data(), B, T, H, KV, hd, scale);

    Tensor dq = to_cuda(q), dk = to_cuda(k), dv = to_cuda(v);
    Tensor dout = Tensor::empty({(i64)B * T * H * hd}, DType::F32, Device::CUDA);
    Tensor dprobs = Tensor::empty({(i64)B * H * T * T}, DType::F32, Device::CUDA);
    cuda_ops::attention_forward(dq.f32(), dk.f32(), dv.f32(), dout.f32(),
                                dprobs.f32(), B, T, H, KV, hd, scale);
    device_synchronize(Device::CUDA);
    near_vec(out_cpu, to_host(dout), 1e-4, "attn fwd GQA causal vs CPU");
    near_vec(probs_cpu, to_host(dprobs), 1e-4, "attn fwd probs vs CPU");
}

static void test_backward_gqa() {
    Rng rng;
    rng.seed_with(778);
    const int B = 2, T = 9, H = 6, KV = 2, hd = 8;
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::vector<float> q = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);
    const std::vector<float> k = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> v = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> dout = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);

    // Forward on CPU first (probs feed both backward paths identically).
    std::vector<float> out_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> probs((size_t)B * H * T * T, 0.0f);
    cpu::attention_forward(q.data(), k.data(), v.data(), out_cpu.data(),
                           probs.data(), B, T, H, KV, hd, scale);

    std::vector<float> dq_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> dk_cpu((size_t)B * T * KV * hd, 0.0f);
    std::vector<float> dv_cpu((size_t)B * T * KV * hd, 0.0f);
    cpu::attention_backward(q.data(), k.data(), v.data(), probs.data(), dout.data(),
                            dq_cpu.data(), dk_cpu.data(), dv_cpu.data(),
                            B, T, H, KV, hd, scale);

    Tensor dq = to_cuda(q), dk = to_cuda(k), dv = to_cuda(v);
    Tensor dprobs = to_cuda(probs), ddout = to_cuda(dout);
    Tensor gdq = Tensor::zeros({(i64)B * T * H * hd}, DType::F32, Device::CUDA);
    Tensor gdk = Tensor::zeros({(i64)B * T * KV * hd}, DType::F32, Device::CUDA);
    Tensor gdv = Tensor::zeros({(i64)B * T * KV * hd}, DType::F32, Device::CUDA);
    cuda_ops::attention_backward(dq.f32(), dk.f32(), dv.f32(), dprobs.f32(),
                                 ddout.f32(), gdq.f32(), gdk.f32(), gdv.f32(),
                                 B, T, H, KV, hd, scale);
    device_synchronize(Device::CUDA);
    near_vec(dq_cpu, to_host(gdq), 1e-4, "attn bwd dq vs CPU");
    near_vec(dk_cpu, to_host(gdk), 1e-4, "attn bwd dk vs CPU (atomic order)");
    near_vec(dv_cpu, to_host(gdv), 1e-4, "attn bwd dv vs CPU (atomic order)");
}

static void test_swa_window() {
    Rng rng;
    rng.seed_with(779);
    const int B = 1, T = 12, H = 4, KV = 2, hd = 8, window = 4;
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::vector<float> q = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);
    const std::vector<float> k = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> v = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> dout = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);

    std::vector<float> out_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> probs((size_t)B * H * T * T, 0.0f);
    cpu::attention_forward_ex(q.data(), k.data(), v.data(), out_cpu.data(),
                              probs.data(), B, T, H, KV, hd, scale, window, nullptr);

    Tensor dq = to_cuda(q), dk = to_cuda(k), dv = to_cuda(v);
    Tensor dout_t = Tensor::empty({(i64)B * T * H * hd}, DType::F32, Device::CUDA);
    Tensor dprobs = Tensor::empty({(i64)B * H * T * T}, DType::F32, Device::CUDA);
    cuda_ops::attention_forward_ex(dq.f32(), dk.f32(), dv.f32(), dout_t.f32(),
                                   dprobs.f32(), B, T, H, KV, hd, scale, window, nullptr);
    device_synchronize(Device::CUDA);
    near_vec(out_cpu, to_host(dout_t), 1e-4, "attn SWA fwd vs CPU");

    std::vector<float> dq_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> dk_cpu((size_t)B * T * KV * hd, 0.0f);
    std::vector<float> dv_cpu((size_t)B * T * KV * hd, 0.0f);
    cpu::attention_backward_ex(q.data(), k.data(), v.data(), probs.data(), dout.data(),
                               dq_cpu.data(), dk_cpu.data(), dv_cpu.data(),
                               B, T, H, KV, hd, scale, window);

    Tensor dprobs2 = to_cuda(probs), ddout = to_cuda(dout);
    Tensor gdq = Tensor::zeros({(i64)B * T * H * hd}, DType::F32, Device::CUDA);
    Tensor gdk = Tensor::zeros({(i64)B * T * KV * hd}, DType::F32, Device::CUDA);
    Tensor gdv = Tensor::zeros({(i64)B * T * KV * hd}, DType::F32, Device::CUDA);
    cuda_ops::attention_backward_ex(dq.f32(), dk.f32(), dv.f32(), dprobs2.f32(),
                                    ddout.f32(), gdq.f32(), gdk.f32(), gdv.f32(),
                                    B, T, H, KV, hd, scale, window);
    device_synchronize(Device::CUDA);
    near_vec(dq_cpu, to_host(gdq), 1e-4, "attn SWA bwd dq vs CPU");
    near_vec(dk_cpu, to_host(gdk), 1e-4, "attn SWA bwd dk vs CPU");
    near_vec(dv_cpu, to_host(gdv), 1e-4, "attn SWA bwd dv vs CPU");
}

static void test_segment_packing_fwd() {
    Rng rng;
    rng.seed_with(780);
    const int B = 1, T = 8, H = 4, KV = 2, hd = 8;
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::vector<float> q = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);
    const std::vector<float> k = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> v = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    // Two packed docs: [0..4] and [5..7]; cross-segment attention is masked.
    const std::vector<i32> seg = {0, 0, 0, 0, 0, 1, 1, 1};

    std::vector<float> out_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> probs((size_t)B * H * T * T, 0.0f);
    cpu::attention_forward_ex(q.data(), k.data(), v.data(), out_cpu.data(),
                              probs.data(), B, T, H, KV, hd, scale, 0, seg.data());

    Tensor dq = to_cuda(q), dk = to_cuda(k), dv = to_cuda(v);
    Tensor dseg = to_cuda_i32(seg);
    Tensor dout = Tensor::empty({(i64)B * T * H * hd}, DType::F32, Device::CUDA);
    Tensor dprobs = Tensor::empty({(i64)B * H * T * T}, DType::F32, Device::CUDA);
    cuda_ops::attention_forward_ex(dq.f32(), dk.f32(), dv.f32(), dout.f32(),
                                   dprobs.f32(), B, T, H, KV, hd, scale, 0, dseg.i32p());
    device_synchronize(Device::CUDA);
    near_vec(out_cpu, to_host(dout), 1e-4, "attn packed-seg fwd vs CPU");
}

static void test_t1_edge() {
    Rng rng;
    rng.seed_with(781);
    const int B = 2, T = 1, H = 4, KV = 2, hd = 8;
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::vector<float> q = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);
    const std::vector<float> k = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> v = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);

    std::vector<float> out_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> probs((size_t)B * H * T * T, 0.0f);
    cpu::attention_forward(q.data(), k.data(), v.data(), out_cpu.data(),
                           probs.data(), B, T, H, KV, hd, scale);

    Tensor dq = to_cuda(q), dk = to_cuda(k), dv = to_cuda(v);
    Tensor dout = Tensor::empty({(i64)B * T * H * hd}, DType::F32, Device::CUDA);
    Tensor dprobs = Tensor::empty({(i64)B * H * T * T}, DType::F32, Device::CUDA);
    cuda_ops::attention_forward(dq.f32(), dk.f32(), dv.f32(), dout.f32(),
                                dprobs.f32(), B, T, H, KV, hd, scale);
    device_synchronize(Device::CUDA);
    near_vec(out_cpu, to_host(dout), 1e-4, "attn T=1 edge vs CPU");
}
#endif // GAI_CUDA

int main() {
#ifdef GAI_CUDA
    if (!cuda_available()) {
        std::cout << "test_attention_cuda_parity: SKIP (no CUDA device)\n";
        return 0;
    }
    test_forward_gqa();
    test_backward_gqa();
    test_swa_window();
    test_segment_packing_fwd();
    test_t1_edge();
#else
    std::cout << "test_attention_cuda_parity: SKIP (CPU-only build)\n";
#endif
    if (failures == 0) {
        std::cout << "test_attention_cuda_parity: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_attention_cuda_parity: " << failures << " FAILURES\n";
    return 1;
}
