// test_cuda.cpp — CPU vs CUDA numerical parity test.
// Skips gracefully when CUDA is not available (CPU-only build/machine).
#include "model/model.h"
#include "core/device.h"
#include "core/ops.h"
#include "core/common.h"
#include "core/rng.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

int main() {
    if (!cuda_available()) {
        std::cout << "[SKIP] CUDA not available — parity tests skipped.\n";
        std::cout << "== CUDA parity tests: SKIPPED (no GPU) ==\n";
        return 0;
    }

    std::cout << "== CUDA parity tests ==\n";

    // Exact-math gate: fp16 tensor GEMMs are disabled here so CPU and CUDA must
    // agree to 1e-2/5%. (The fp16 path is validated separately by pilot loss.)
    ops::set_gemm_fp16(false);

    // Small config: ~1M params, fast to run
    ModelConfig cfg;
    cfg.vocab_size       = 256;
    cfg.hidden_size      = 64;
    cfg.num_layers       = 2;
    cfg.num_heads        = 4;
    cfg.num_kv_heads     = 1;
    cfg.intermediate_size = 128;
    cfg.max_seq_len      = 64;
    cfg.use_moe          = true;
    cfg.num_experts      = 4;
    cfg.moe_top_k        = 2;
    cfg.moe_expert_dim   = 32;

    constexpr u64 SEED = 42;
    constexpr int B = 2;
    constexpr int T = 32;

    // Build two identical models: one on CPU, one on GPU
    Model cpu_model(cfg);
    cpu_model.init_weights(SEED);
    cpu_model.enable_grad(true);
    cpu_model.zero_grad();

    Model cuda_model(cfg);
    cuda_model.init_weights(SEED);
    cuda_model.enable_grad(true);
    cuda_model.to(best_device());
    cuda_model.zero_grad();

    // Build a deterministic token batch
    std::vector<i32> host_ids(B * T);
    std::vector<i32> host_targets(B * T);
    u64 rng = 1337;
    for (int i = 0; i < B * T; ++i) {
        rng = splitmix64(rng);
        host_ids[i]     = static_cast<i32>(rng % static_cast<u64>(cfg.vocab_size));
        rng = splitmix64(rng);
        host_targets[i] = static_cast<i32>(rng % static_cast<u64>(cfg.vocab_size));
    }

    Activations cpu_act  = cpu_model.make_activations(B, T, true);
    Activations cuda_act = cuda_model.make_activations(B, T, true);

    // Copy batch to device
    Tensor dev_ids     = Tensor::empty({static_cast<i64>(B * T)}, DType::I32, cuda_model.device());
    Tensor dev_targets = Tensor::empty({static_cast<i64>(B * T)}, DType::I32, cuda_model.device());
    device_copy(dev_ids.data_ptr(),     cuda_model.device(), host_ids.data(),     Device::CPU, B * T * sizeof(i32));
    device_copy(dev_targets.data_ptr(), cuda_model.device(), host_targets.data(), Device::CPU, B * T * sizeof(i32));

    // Run forward+backward on both
    i64 cpu_ntok = 0, cuda_ntok = 0;
    double cpu_loss  = cpu_model.forward_backward(
        host_ids.data(), host_targets.data(), B, T, cpu_act, &cpu_ntok);
    double cuda_loss = cuda_model.forward_backward(
        dev_ids.i32p(), dev_targets.i32p(), B, T, cuda_act, &cuda_ntok);

    std::cout << "  CPU  loss = " << cpu_loss  << "  (ntok=" << cpu_ntok  << ")\n";
    std::cout << "  CUDA loss = " << cuda_loss << "  (ntok=" << cuda_ntok << ")\n";

    double loss_diff = std::abs(cpu_loss - cuda_loss);
    std::cout << "  |loss_diff| = " << loss_diff << "\n";

    constexpr double LOSS_TOL = 1e-2;
    if (loss_diff >= LOSS_TOL) {
        std::cerr << "FAIL: loss mismatch exceeds tolerance " << LOSS_TOL << "\n";
        return 1;
    }
    std::cout << "  [ok]   loss parity\n";

    // Compare gradients on a subset of parameters
    auto& cpu_params  = cpu_model.parameters();
    auto& cuda_params = cuda_model.parameters();
    if (cpu_params.size() != cuda_params.size()) {
        std::cerr << "FAIL: parameter count mismatch\n";
        return 1;
    }

    constexpr double GRAD_TOL = 0.05; // 5% relative error tolerance
    int n_checked = 0;
    int n_moe_checked = 0;
    for (size_t i = 0; i < cpu_params.size(); ++i) {
        auto* cp = cpu_params[i];
        auto* kp = cuda_params[i];
        if (!cp->g.data_ptr() || !kp->g.data_ptr()) continue;

        i64 N = cp->g.numel();
        // Copy CUDA grad back to host for comparison
        std::vector<float> cpu_buf(static_cast<size_t>(N));
        std::vector<float> gpu_buf(static_cast<size_t>(N));
        device_copy(cpu_buf.data(), Device::CPU, cp->g.data_ptr(), Device::CPU, N * sizeof(float));
        device_copy(gpu_buf.data(), Device::CPU, kp->g.data_ptr(), kp->g.device(), N * sizeof(float));

        double max_abs = 0, max_rel = 0;
        for (i64 j = 0; j < N; ++j) {
            double ae = std::abs(cpu_buf[static_cast<size_t>(j)] - gpu_buf[static_cast<size_t>(j)]);
            double ref = std::abs(cpu_buf[static_cast<size_t>(j)]) + 1e-8;
            if (ae > max_abs) max_abs = ae;
            if (ae / ref > max_rel) max_rel = ae / ref;
        }
        bool is_moe = cp->name.find("moe") != std::string::npos ||
                      cp->name.find("router") != std::string::npos;
        if (is_moe) ++n_moe_checked;
        std::cout << "  [ok]   grad " << cp->name
                  << "  max_abs=" << max_abs << "  max_rel=" << max_rel << "\n";
        if (max_rel >= GRAD_TOL) {
            std::cerr << "FAIL: grad " << cp->name << " max_rel=" << max_rel
                      << " exceeds " << GRAD_TOL << "\n";
            return 1;
        }
        ++n_checked;
    }
    std::cout << "  checked " << n_checked << " param grads (" << n_moe_checked << " MoE/router)\n";
    if (n_moe_checked == 0) {
        std::cerr << "FAIL: no MoE/router grads were checked (parity gate is blind to MoE)\n";
        return 1;
    }

    // ---- fp16 GEMM smoke test (large enough to hit the tensor-core path,
    // threshold is 1M MACs). Compares fp32 vs fp16 on the trans combos used
    // by linear_forward (N,T) and linear_backward (N,N / T,N).
    {
        std::cout << "  [fp16] GEMM trans-combo check...\n";
        ops::set_gemm_fp16(false);
        const int M = 256, N = 256, K = 64; // 4M MACs > 1M threshold
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hB2((size_t)N * K);
        std::vector<float> hC0((size_t)M * N), hC1((size_t)M * N);
        u64 rng = 0x12345;
        for (auto& v : hA) { rng = splitmix64(rng); v = (float)((int)(rng % 2000) - 1000) / 1000.0f; }
        for (auto& v : hB) { rng = splitmix64(rng); v = (float)((int)(rng % 2000) - 1000) / 1000.0f; }
        for (auto& v : hB2) { rng = splitmix64(rng); v = (float)((int)(rng % 2000) - 1000) / 1000.0f; }
        Tensor dA({(i64)M * K}, DType::F32, best_device());
        Tensor dB({(i64)K * N}, DType::F32, best_device());
        Tensor dB2({(i64)N * K}, DType::F32, best_device());
        Tensor dC({(i64)M * N}, DType::F32, best_device());
        device_copy(dA.data_ptr(), dA.device(), hA.data(), Device::CPU, hA.size() * sizeof(float));
        device_copy(dB.data_ptr(), dB.device(), hB.data(), Device::CPU, hB.size() * sizeof(float));
        device_copy(dB2.data_ptr(), dB2.device(), hB2.data(), Device::CPU, hB2.size() * sizeof(float));
        struct Combo { bool ta, tb; const float* B; int ldb; const std::vector<float>* hB; const char* n; };
        // Note: for tb=true the B layout is [N,K]; otherwise [K,N].
        Combo combos[2] = {
            {false, false, (const float*)dB.data_ptr(), N, &hB, "NN (dx path)"},
            {false, true,  (const float*)dB2.data_ptr(), K, &hB2, "NT (fwd path)"},
        };
        for (auto& c : combos) {
            ops::set_gemm_fp16(false);
            ops::gemm(best_device(), c.ta, c.tb, M, N, K, 1.0f,
                      (const float*)dA.data_ptr(), K, c.B, c.ldb, 0.0f,
                      (float*)dC.data_ptr(), N);
            device_copy(hC0.data(), Device::CPU, dC.data_ptr(), dC.device(), hC0.size() * sizeof(float));
            ops::set_gemm_fp16(true);
            ops::gemm(best_device(), c.ta, c.tb, M, N, K, 1.0f,
                      (const float*)dA.data_ptr(), K, c.B, c.ldb, 0.0f,
                      (float*)dC.data_ptr(), N);
            device_copy(hC1.data(), Device::CPU, dC.data_ptr(), dC.device(), hC1.size() * sizeof(float));
            double mr = 0;
            for (size_t j = 0; j < hC0.size(); ++j) {
                double ae = std::abs(hC0[j] - hC1[j]);
                double ref = std::abs(hC0[j]) + 1e-3;
                mr = std::max(mr, ae / ref);
            }
            std::cout << "    fp16 " << c.n << " max_rel=" << mr << "\n";
            if (!(mr < 0.08)) {
                std::cerr << "FAIL: fp16 GEMM " << c.n << " diverged (max_rel=" << mr << ")\n";
                return 1;
            }
        }
        ops::set_gemm_fp16(false);
    }

    std::cout << "== CUDA parity tests passed ==\n";
    return 0;
}
