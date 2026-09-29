#include "core/ops.h"
#include "core/ops_cpu.h"
#include "core/rng.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static std::vector<float> rnd_vec(Rng& rng, size_t n, float s) {
    std::vector<float> v(n);
    for (float& x : v) x = (rng.uniform() * 2.0f - 1.0f) * s;
    return v;
}

static bool near_vec(const std::vector<float>& a, const std::vector<float>& b,
                     double rtol, const char* what, double atol = 0.0) {
    if (a.size() != b.size()) {
        std::cerr << "FAIL: " << what << " size mismatch\n";
        ++failures;
        return false;
    }

    double scale = 0.0;
    for (float v : b) scale = std::max(scale, std::fabs((double)v));
    if (atol <= 0.0) atol = 1e-5 * (scale > 0.0 ? scale : 1.0);
    double max_rel = 0.0, max_abs = 0.0, worst = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::fabs((double)a[i] - (double)b[i]);
        const double tol = atol + rtol * std::fabs((double)b[i]);
        if (diff > max_abs) { max_abs = diff; worst = (double)b[i]; worst_i = i; }
        const double denom = std::fabs((double)b[i]) + 1e-30;
        max_rel = std::max(max_rel, diff / denom);
        if (diff > tol) {
            std::cerr << "FAIL: " << what << " elem " << i
                      << " got " << (double)a[i] << " want " << (double)b[i]
                      << " |diff| " << diff << " > tol " << tol
                      << " (atol " << atol << ", ref scale " << scale << ")\n";
            ++failures;
            return false;
        }
    }
    std::cout << "  ok  " << what
              << "  max_abs " << max_abs << " (worst ref " << worst << " @ " << worst_i
              << ")  max_rel " << max_rel
              << "  ref_scale " << scale << "  atol " << atol << "\n";
    return true;
}

static bool near_vec_report(const std::vector<float>& a, const std::vector<float>& b,
                            double rtol, const char* what, double atol, bool quiet) {
    const int before = failures;
    const bool r = near_vec(a, b, rtol, what, atol);
    if (quiet && r && failures > before) failures = before;
    if (quiet && !r) failures = before;
    return r;
}

static void test_tolerance_contract() {
    Rng rng;
    rng.seed_with(7781);
    const int B = 2, T = 9, H = 6, KV = 2, hd = 8;
    const int group = H / KV;
    const float scale = 1.0f / std::sqrt((float)hd);
    const std::vector<float> q = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);
    const std::vector<float> k = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> v = rnd_vec(rng, (size_t)B * T * KV * hd, 0.5f);
    const std::vector<float> dout = rnd_vec(rng, (size_t)B * T * H * hd, 0.5f);

    std::vector<float> out_cpu((size_t)B * T * H * hd, 0.0f);
    std::vector<float> probs((size_t)B * H * T * T, 0.0f);
    cpu::attention_forward(q.data(), k.data(), v.data(), out_cpu.data(),
                           probs.data(), B, T, H, KV, hd, scale);
    std::vector<float> dq_ref((size_t)B * T * H * hd, 0.0f);
    std::vector<float> dk((size_t)B * T * KV * hd, 0.0f);
    std::vector<float> dv((size_t)B * T * KV * hd, 0.0f);
    cpu::attention_backward(q.data(), k.data(), v.data(), probs.data(), dout.data(),
                            dq_ref.data(), dk.data(), dv.data(),
                            B, T, H, KV, hd, scale);

    const size_t qs = (size_t)T * H * hd, kvs = (size_t)T * KV * hd;
    auto dot = [&](const float* a, const float* c) {
        float s = 0.0f;
        for (int i = 0; i < hd; ++i) s += a[i] * c[i];
        return s;
    };

    auto pairwise_sum = [](std::vector<float>& a) {
        for (size_t w = 1; w < a.size(); w *= 2)
            for (size_t j = 0; j + w < a.size(); j += 2 * w) a[j] += a[j + w];
        return a[0];
    };
    auto serial_sum = [](const std::vector<float>& a) {
        float s = 0.0f;
        for (float x : a) s += x;
        return s;
    };
    auto build = [&](int mode) {
        std::vector<float> dq((size_t)B * T * H * hd, 0.0f);
        for (int b = 0; b < B; ++b)
            for (int kvhh = 0; kvhh < KV; ++kvhh)
                for (int hg = 0; hg < group; ++hg) {
                    const int h = kvhh * group + hg;
                    for (int t = 0; t < T; ++t) {
                        const int len = t + 1;
                        const float* pr = &probs[((size_t)b * H + h) * T * T + (size_t)t * T];
                        const float* go = &dout[(size_t)b * qs + ((size_t)t * H + h) * hd];
                        float* dqh = &dq[(size_t)b * qs + ((size_t)t * H + h) * hd];
                        std::vector<float> dsv(len), term(len);
                        for (int j = 0; j < len; ++j) {
                            dsv[j] = dot(go, &v[(size_t)b * kvs + ((size_t)j * KV + kvhh) * hd]);
                            term[j] = pr[j] * dsv[j];
                        }
                        const float dot_pg = (mode == 1) ? pairwise_sum(term)
                                                         : serial_sum(term);
                        for (int c = 0; c < hd; ++c) {
                            std::vector<float> row(len);
                            for (int j = 0; j < len; ++j) {
                                const float sc = (mode == 3) ? 1.0f : scale;
                                const float ds = pr[j] * ((mode == 2) ? dsv[j]
                                                                     : dsv[j] - dot_pg) * sc;
                                const int jj = (mode == 4) ? ((j + 1) % T) : j;
                                row[j] = ds * k[(size_t)b * kvs + ((size_t)jj * KV + kvhh) * hd + c];
                            }
                            dqh[c] += (mode == 1) ? pairwise_sum(row) : serial_sum(row);
                        }
                    }
                }
        return dq;
    };

    const std::vector<float> serial   = build(0);
    const std::vector<float> reassoc  = build(1);
    const std::vector<float> no_jac   = build(2);
    const std::vector<float> no_scale = build(3);
    const std::vector<float> shifted  = build(4);

    CHECK(near_vec(serial, dq_ref, 1e-4, "tolerance contract: serial dq == CPU dq"),
          "self-test oracle mismatch");

    CHECK(near_vec_report(reassoc, dq_ref, 1e-4, "reassociation is accepted", 0.0, true),
          "a legal reassociation of the same sums must pass the mixed criterion");

    CHECK(!near_vec_report(no_jac, dq_ref, 1e-4, "dropped jacobian", 0.0, true),
          "dropping the softmax-Jacobian dot_pg term must be caught");
    CHECK(!near_vec_report(no_scale, dq_ref, 1e-4, "dropped scale", 0.0, true),
          "dropping the 1/sqrt(hd) scale must be caught");
    CHECK(!near_vec_report(shifted, dq_ref, 1e-4, "shifted k index", 0.0, true),
          "a one-position k index shift must be caught");
}

#ifdef GAI_CUDA
#include "cuda/cuda_ops.h"
#include "core/device.h"

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
#endif

int main() {

    test_tolerance_contract();
#ifdef GAI_CUDA
    if (!cuda_available()) {
        std::cout << "test_attention_cuda_parity: tolerance contract PASSED, "
                     "device tests SKIP (no CUDA device)\n";
    } else {
        test_forward_gqa();
        test_backward_gqa();
        test_swa_window();
        test_segment_packing_fwd();
        test_t1_edge();
    }
#else
    std::cout << "test_attention_cuda_parity: tolerance contract PASSED, "
                 "device tests SKIP (CPU-only build)\n";
#endif
    if (failures == 0) {
        std::cout << "test_attention_cuda_parity: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_attention_cuda_parity: " << failures << " FAILURES\n";
    return 1;
}
