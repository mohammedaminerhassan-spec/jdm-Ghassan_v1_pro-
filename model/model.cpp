#include "model/model.h"
#include "core/ops.h"
#include "core/device.h"
#include "core/common.h"

#ifdef GAI_OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <limits>

namespace gai {

double Model::moe_layer_aux(Device dev, const float* probs, const i32* idx,
                             float* auxfrac_dev, int layer, i64 N, int K, int ne,
                             bool want_stats, double* d_raw_accum) {

    if (dev == Device::CUDA) {
        std::vector<float> h_frac, h_psum;
        float* hf = nullptr;
        float* hp = nullptr;
        if (want_stats || !d_raw_accum) {

            h_frac.assign(static_cast<size_t>(ne), 0.0f);
            h_psum.assign(static_cast<size_t>(ne), 0.0f);
            hf = h_frac.data();
            hp = h_psum.data();
        }
        if (ops::moe_aux_gpu(dev, probs, idx, auxfrac_dev,
                             hf, hp, N, K, ne, d_raw_accum)) {
            if (!hf) return 0.0;
            std::vector<double> cnt(ne, 0.0);
            const double denom = static_cast<double>(N) * K;
            for (int e = 0; e < ne; ++e) cnt[e] = (double)h_frac[(size_t)e] * denom;
            double raw = 0.0;
            for (int e = 0; e < ne; ++e) {
                double f = denom > 0 ? cnt[e] / denom : 0.0;
                double m = N > 0 ? (double)h_psum[(size_t)e] / (double)N : 0.0;
                raw += m * f;
            }
            raw *= ne;
            if (moe_tok_acc_.size() != layers_.size())
                moe_tok_acc_.assign(layers_.size(), std::vector<double>(ne, 0.0));
            if (layer >= 0 && static_cast<size_t>(layer) < moe_tok_acc_.size()) {
                for (int e = 0; e < ne && e < static_cast<int>(moe_tok_acc_[static_cast<size_t>(layer)].size()); ++e)
                    moe_tok_acc_[static_cast<size_t>(layer)][static_cast<size_t>(e)] += cnt[e];
                ++moe_aux_batches_;
            }
            return raw;
        }

    }
    std::vector<i32> h_idx;
    std::vector<float> h_probs;
    const i32* idx_h = idx;
    const float* probs_h = probs;
    if (dev != Device::CPU) {
        h_idx.resize(static_cast<size_t>(N) * K);
        device_copy(h_idx.data(), Device::CPU, idx, dev,
                    sizeof(i32) * h_idx.size());
        h_probs.resize(static_cast<size_t>(N) * ne);
        device_copy(h_probs.data(), Device::CPU, probs, dev,
                    sizeof(float) * h_probs.size());
        idx_h = h_idx.data();
        probs_h = h_probs.data();
    }

    std::vector<double> cnt(ne, 0.0), psum(ne, 0.0);
    for (i64 t = 0; t < N; ++t)
        for (int k = 0; k < K; ++k) {
            int e = idx_h[static_cast<size_t>(t) * K + k];
            if (e >= 0 && e < ne) cnt[e] += 1.0;
        }
    for (i64 t = 0; t < N; ++t)
        for (int e = 0; e < ne; ++e)
            psum[e] += probs_h[static_cast<size_t>(t) * ne + e];

    std::vector<float> frac(ne);
    double raw = 0.0;
    const double denom = static_cast<double>(N) * K;
    for (int e = 0; e < ne; ++e) {
        double f = denom > 0 ? cnt[e] / denom : 0.0;
        double m = N > 0 ? psum[e] / static_cast<double>(N) : 0.0;
        frac[e] = static_cast<float>(f);
        raw += m * f;
    }
    raw *= ne;
    device_copy(auxfrac_dev, dev, frac.data(), Device::CPU,
                sizeof(float) * frac.size());

    if (moe_tok_acc_.size() != layers_.size())
        moe_tok_acc_.assign(layers_.size(), std::vector<double>(ne, 0.0));
    if (layer >= 0 && static_cast<size_t>(layer) < moe_tok_acc_.size()) {
        for (int e = 0; e < ne && e < static_cast<int>(moe_tok_acc_[static_cast<size_t>(layer)].size()); ++e)
            moe_tok_acc_[static_cast<size_t>(layer)][static_cast<size_t>(e)] += cnt[e];
        ++moe_aux_batches_;
    }
    return raw;
}

std::string Model::moe_balance_report() const {
    if (!cfg_.use_moe || moe_tok_acc_.empty() || moe_aux_batches_ == 0) return "";
    std::ostringstream ss;
    ss << "expert load (share of routed slots):";
    for (size_t l = 0; l < moe_tok_acc_.size(); ++l) {
        double tot = 0;
        for (double c : moe_tok_acc_[l]) tot += c;
        if (tot <= 0) continue;
        double mn = 1e9, mx = 0;
        for (double c : moe_tok_acc_[l]) {
            double s = c / tot;
            mn = std::min(mn, s);
            mx = std::max(mx, s);
        }

        ss << strfmt(" L%zu[%.2f..%.2f]%s", l, mn, mx,
                     (mn < 0.25 / cfg_.num_experts || mx > 4.0 / cfg_.num_experts)
                     ? "!" : "");
    }
    return ss.str();
}

void Model::moe_balance_reset() {
    moe_tok_acc_.clear();
    moe_aux_batches_ = 0;
}

void Model::ensure_moe_bias() {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) return;
    const int L = cfg_.num_layers, ne = cfg_.num_experts;
    if (static_cast<int>(moe_bias_.size()) != L) {
        moe_bias_.assign(static_cast<size_t>(L), std::vector<float>(static_cast<size_t>(ne), 0.0f));
        moe_bias_dev_.clear();
        moe_bias_dev_.reserve(static_cast<size_t>(L));
        for (int l = 0; l < L; ++l) {
            Tensor t({(i64)ne}, DType::F32, device_);
            t.zero_();
            moe_bias_dev_.push_back(std::move(t));
        }
    }
}

const float* Model::moe_bias_ptr(int layer) const {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) return nullptr;
    if (layer < 0 || static_cast<size_t>(layer) >= moe_bias_.size()) return nullptr;
    if (device_ == Device::CPU) return moe_bias_[static_cast<size_t>(layer)].data();
    if (static_cast<size_t>(layer) >= moe_bias_dev_.size()) return nullptr;
    return moe_bias_dev_[static_cast<size_t>(layer)].f32();
}

float* Model::moe_bias_ptr_mut(int layer) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) return nullptr;
    ensure_moe_bias();
    if (layer < 0 || static_cast<size_t>(layer) >= moe_bias_.size()) return nullptr;
    return moe_bias_[static_cast<size_t>(layer)].data();
}

void Model::update_moe_bias(int layer, const float* frac_host, int ne) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free || !frac_host || ne <= 0) return;
    ensure_moe_bias();
    if (layer < 0 || static_cast<size_t>(layer) >= moe_bias_.size()) return;

    const float target = static_cast<float>(cfg_.moe_top_k) / static_cast<float>(ne);
    auto& b = moe_bias_[static_cast<size_t>(layer)];
    for (int e = 0; e < ne && e < static_cast<int>(b.size()); ++e) {
        float err = frac_host[e] - target;
        b[static_cast<size_t>(e)] -= moe_bias_lr_ * err;

        if (b[static_cast<size_t>(e)] > 0.5f) b[static_cast<size_t>(e)] = 0.5f;
        if (b[static_cast<size_t>(e)] < -0.5f) b[static_cast<size_t>(e)] = -0.5f;
    }
    if (device_ == Device::CUDA && static_cast<size_t>(layer) < moe_bias_dev_.size()) {
        device_copy(moe_bias_dev_[static_cast<size_t>(layer)].data_ptr(), device_,
                    b.data(), Device::CPU, sizeof(float) * b.size());
    }
}

void Model::accumulate_moe_bias_fracs(Activations& act, int B, int T) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) return;
    const int L  = cfg_.num_layers;
    const int ne = cfg_.num_experts;
    const int K  = cfg_.moe_top_k;
    const i64 N  = static_cast<i64>(B) * T;
    ensure_moe_bias();
    const size_t total = static_cast<size_t>(L) * static_cast<size_t>(ne);
    if (static_cast<int>(act.saved_moe_probs.size()) != L ||
        static_cast<int>(act.saved_moe_idx.size())   != L) {
        return;
    }
    if (device_ == Device::CUDA) {

        if (!moe_bias_acc_dev_.defined() ||
            static_cast<size_t>(moe_bias_acc_dev_.numel()) != total) {
            moe_bias_acc_dev_ = Tensor::zeros({static_cast<i64>(total)}, DType::F32, device_);
        }
        float* base = moe_bias_acc_dev_.f32();
        for (int l = 0; l < L; ++l) {
            const size_t sl = static_cast<size_t>(l);
            if (!act.saved_moe_idx[sl].defined()) continue;
            ops::moe_count_slots(device_, act.saved_moe_idx[sl].i32p(),
                                 base + sl * static_cast<size_t>(ne),
                                 N * K, ne);
        }
        return;
    }

    if (moe_bias_acc_.size() != total) moe_bias_acc_.assign(total, 0.0f);
    for (int l = 0; l < L; ++l) {
        const size_t sl = static_cast<size_t>(l);
        if (!act.saved_moe_idx[sl].defined()) continue;

        std::vector<i32> h_idx(static_cast<size_t>(N) * static_cast<size_t>(K));
        device_copy(h_idx.data(), Device::CPU, act.saved_moe_idx[sl].i32p(), device_,
                    sizeof(i32) * h_idx.size());
        float* acc = moe_bias_acc_.data() + sl * static_cast<size_t>(ne);
        for (i64 t = 0; t < N; ++t) {
            for (int k = 0; k < K; ++k) {
                const i32 e = h_idx[static_cast<size_t>(t) * static_cast<size_t>(K) + k];
                if (e >= 0 && e < ne) acc[e] += 1.0f;
            }
        }
    }
}

void Model::apply_moe_bias_step(const float* global_count_sum, bool apply) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free) {
        moe_bias_acc_.clear();
        moe_bias_acc_dev_ = Tensor();
        return;
    }
    const int L  = cfg_.num_layers;
    const int ne = cfg_.num_experts;
    if (!apply || !global_count_sum || (moe_bias_acc_.empty() && !moe_bias_acc_dev_.defined())) {

        moe_bias_acc_.clear();
        moe_bias_acc_dev_ = Tensor();
        return;
    }
    ensure_moe_bias();
    std::vector<float> frac(static_cast<size_t>(ne));
    for (int l = 0; l < L; ++l) {
        const size_t base = static_cast<size_t>(l) * static_cast<size_t>(ne);

        double row_sum = 0.0;
        for (int e = 0; e < ne; ++e) row_sum += static_cast<double>(global_count_sum[base + e]);
        const double inv = (row_sum > 0.0) ? 1.0 / row_sum : 0.0;
        if (inv <= 0.0) continue;
        for (int e = 0; e < ne; ++e)
            frac[static_cast<size_t>(e)] =
                static_cast<float>(static_cast<double>(global_count_sum[base + e]) * inv);
        update_moe_bias(l, frac.data(), ne);
    }
    moe_bias_acc_.clear();
    moe_bias_acc_dev_ = Tensor();
}

void Model::set_moe_bias(int layer, const float* values, size_t count) {
    if (!cfg_.use_moe || !cfg_.moe_aux_free || !values) return;
    ensure_moe_bias();
    if (layer < 0 || static_cast<size_t>(layer) >= moe_bias_.size() ||
        count != moe_bias_[static_cast<size_t>(layer)].size()) {
        GAI_FAIL("checkpoint moe bias shape mismatch");
    }
    auto& bias = moe_bias_[static_cast<size_t>(layer)];
    std::copy(values, values + count, bias.begin());
    if (device_ == Device::CUDA && static_cast<size_t>(layer) < moe_bias_dev_.size()) {
        device_copy(moe_bias_dev_[static_cast<size_t>(layer)].data_ptr(), device_,
                    bias.data(), Device::CPU, sizeof(float) * bias.size());
    }
}

void ModelConfig::validate() const {
    GAI_CHECK(vocab_size > 0, "vocab_size must be > 0");
    GAI_CHECK(hidden_size > 0, "hidden_size must be > 0");
    GAI_CHECK(num_layers > 0, "num_layers must be > 0");
    GAI_CHECK(num_heads > 0, "num_heads must be > 0");
    GAI_CHECK(num_kv_heads > 0, "num_kv_heads must be > 0");
    GAI_CHECK(hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads");
    GAI_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    GAI_CHECK(head_dim() % 2 == 0, "head_dim must be even (RoPE pairs)");
    GAI_CHECK(intermediate_size > 0, "intermediate_size must be > 0");
    GAI_CHECK(max_seq_len > 0, "max_seq_len must be > 0");

    GAI_CHECK(max_seq_len <= 65536, "max_seq_len must be <= 65536 (larger is unvalidated)");
    GAI_CHECK(std::isfinite(rope_theta) && rope_theta > 0.0f, "rope_theta must be finite and > 0");

    GAI_CHECK(std::isfinite(rms_eps) && rms_eps >= 1e-12f && rms_eps <= 1e-2f,
              "rms_eps must be finite and in [1e-12,1e-2]");
    GAI_CHECK(std::isfinite(init_std) && init_std > 0.0f && init_std <= 0.1f,
              "init_std must be finite and in (0,0.1]");
    GAI_CHECK(std::isfinite(rope_scale) && rope_scale >= 1.0f, "rope_scale must be finite and >= 1.0 (1.0 = off)");

    GAI_CHECK(rope_scale <= 8.0f, "rope_scale must be <= 8.0 (larger is unvalidated extrapolation)");
    GAI_CHECK(std::isfinite(z_loss_scale) && z_loss_scale >= 0.0f && z_loss_scale <= 0.01f,
              "z_loss_scale must be finite and in [0,0.01]");
    GAI_CHECK(std::isfinite(moe_jitter) && moe_jitter >= 0.0f && moe_jitter <= 0.5f,
              "moe_jitter must be finite and in [0,0.5]");
    GAI_CHECK(std::isfinite(rope_yarn_mscale) && rope_yarn_mscale >= 0.0f && rope_yarn_mscale <= 2.0f,
              "rope_yarn_mscale must be finite and in [0,2]");

    GAI_CHECK(std::isfinite(rope_yarn_low) && rope_yarn_low >= 1.0f && rope_yarn_low <= 128.0f,
              "rope_yarn_low must be finite and in [1,128]");
    GAI_CHECK(std::isfinite(rope_yarn_high) && rope_yarn_high >= 1.0f && rope_yarn_high <= 128.0f,
              "rope_yarn_high must be finite and in [1,128]");
    GAI_CHECK(rope_yarn_high >= rope_yarn_low, "rope_yarn_high must be >= rope_yarn_low");
    GAI_CHECK(sliding_window >= 0 && sliding_window <= 16384, "sliding_window must be in [0,16384]");
    GAI_CHECK(rope_type == 0 || rope_type == 1, "rope_type must be 0 (interleaved) or 1 (neox)");

    if (moe_aux_free && moe_aux_scale != 0.0f) {

        GAI_CHECK(false, "moe_aux_free=true requires moe_aux_scale: 0.0 (aux-loss-free has no aux grad)");
    }
    if (use_moe) {
        GAI_CHECK(num_experts > 0 && num_experts <= 64, "num_experts must be in [1,64]");
        GAI_CHECK(moe_top_k > 0 && moe_top_k <= num_experts, "moe_top_k must be in [1,num_experts]");
        GAI_CHECK(moe_top_k <= 8, "moe_top_k must be <= 8 (router kernel limit)");

        GAI_CHECK(moe_allow_dense || moe_top_k * 2 <= num_experts,
                  "moe_top_k must be <= num_experts/2 (sparse MoE; dense routing OOMs T4 and removes the MoE advantage; "
                  "set model.moe_allow_dense=true only for short research runs)");
        GAI_CHECK(moe_expert_dim > 0, "moe_expert_dim must be > 0");
        GAI_CHECK(std::isfinite(moe_aux_scale) && moe_aux_scale >= 0.0f && moe_aux_scale <= 1.0f,
                  "moe_aux_scale must be finite and in [0,1] (larger drowns CE loss)");
    }

    {
        const int KV_TILE = 64;
        const int group = num_heads / num_kv_heads;
        size_t shmem_fwd = sizeof(float) * (static_cast<size_t>(head_dim()) + static_cast<size_t>(KV_TILE) * head_dim() * 2 + KV_TILE + static_cast<size_t>(head_dim()));
        size_t shmem_bwd = sizeof(float) * (static_cast<size_t>(2) * KV_TILE * head_dim() + group);
        size_t shmem_max = std::max(shmem_fwd, shmem_bwd);
        GAI_CHECK(shmem_max <= 48 * 1024,
                  strfmt("head_dim=%d too large for attention kernel shared memory (need %s, T4 limit 48KB). "
                         "Reduce num_heads or hidden_size, or use a GPU with more shared memory.",
                         head_dim(), human_bytes(shmem_max).c_str()));
    }
}

ModelConfig ModelConfig::from_config(const Config& c, const std::string& p) {
    ModelConfig m;
    auto key = [&](const char* k) { return p.empty() ? std::string(k) : p + "." + k; };

    auto get_int_checked = [&](const char* k, int def) {
        i64 v = c.get_int(key(k), static_cast<i64>(def));
        GAI_CHECK(v > 0 && v <= static_cast<i64>(std::numeric_limits<int>::max()),
                  std::string("model.") + k + " out of int range");
        return static_cast<int>(v);
    };

    auto get_int_checked0 = [&](const char* k, int def) {
        i64 v = c.get_int(key(k), static_cast<i64>(def));
        GAI_CHECK(v >= 0 && v <= static_cast<i64>(std::numeric_limits<int>::max()),
                  std::string("model.") + k + " out of int range");
        return static_cast<int>(v);
    };
    m.vocab_size        = get_int_checked("vocab_size", m.vocab_size);
    m.hidden_size       = get_int_checked("hidden_size", m.hidden_size);
    m.num_layers        = get_int_checked("num_layers", m.num_layers);
    m.num_heads         = get_int_checked("num_heads", m.num_heads);
    m.num_kv_heads      = get_int_checked("num_kv_heads", m.num_kv_heads);
    m.intermediate_size = get_int_checked("intermediate_size", m.intermediate_size);
    m.max_seq_len       = get_int_checked("max_seq_len", m.max_seq_len);
    m.rope_theta        = c.get_f32(key("rope_theta"), m.rope_theta);
    m.rms_eps           = c.get_f32(key("rms_eps"), m.rms_eps);
    m.tie_embeddings    = c.get_bool(key("tie_embeddings"), m.tie_embeddings);
    m.init_std          = c.get_f32(key("init_std"), m.init_std);
    m.use_moe           = c.get_bool(key("use_moe"), m.use_moe);
    m.num_experts       = get_int_checked("num_experts", m.num_experts);
    m.moe_top_k         = get_int_checked("moe_top_k", m.moe_top_k);
    m.moe_expert_dim    = get_int_checked("moe_expert_dim", m.moe_expert_dim);
    m.moe_shared        = c.get_bool(key("moe_shared"), m.moe_shared);
    m.moe_aux_scale     = c.get_f32(key("moe_aux_scale"), m.moe_aux_scale);
    m.moe_jitter        = c.get_f32(key("moe_jitter"), m.moe_jitter);
    m.moe_allow_dense   = c.get_bool(key("moe_allow_dense"), m.moe_allow_dense);
    m.moe_aux_free      = c.get_bool(key("moe_aux_free"), m.moe_aux_free);
    m.rope_yarn_low     = c.get_f32(key("rope_yarn_low"), m.rope_yarn_low);
    m.rope_yarn_high    = c.get_f32(key("rope_yarn_high"), m.rope_yarn_high);
    m.sliding_window    = get_int_checked0("sliding_window", m.sliding_window);
    m.rope_type         = get_int_checked0("rope_type", m.rope_type);
    m.use_qk_norm       = c.get_bool(key("use_qk_norm"), m.use_qk_norm);
    m.z_loss_scale      = c.get_f32(key("z_loss_scale"), m.z_loss_scale);
    m.rope_scale        = c.get_f32(key("rope_scale"), m.rope_scale);
    m.rope_yarn_mscale  = c.get_f32(key("rope_yarn_mscale"), m.rope_yarn_mscale);
    m.validate();
    return m;
}

static float rope_theta_eff_local(const ModelConfig& m) {
    if (m.rope_scale <= 1.0f) return m.rope_theta;
    float hd = static_cast<float>(m.head_dim());
    float e = (hd > 2.0f) ? hd / (hd - 2.0f) : 1.0f;
    return m.rope_theta * std::pow(m.rope_scale, e);
}

static float rope_mscale_local(const ModelConfig& m) {
    if (m.rope_scale <= 1.0f) return 1.0f;
    if (m.rope_yarn_mscale > 0.0f) return m.rope_yarn_mscale;

    float base = 0.1f * std::log(m.rope_scale) + 1.0f;
    if (m.rope_yarn_low <= 1.0f && m.rope_yarn_high >= 32.0f) return base;
    float span = m.rope_yarn_high - m.rope_yarn_low;
    float ramp = (span > 0.0f) ? (32.0f - m.rope_yarn_low) / span : 1.0f;
    if (ramp < 0.0f) ramp = 0.0f;
    if (ramp > 1.0f) ramp = 1.0f;
    return 1.0f + (base - 1.0f) * (0.5f + 0.5f * ramp);
}

std::string ModelConfig::summary() const {
    std::ostringstream ss;
    ss << "vocab=" << vocab_size << " d=" << hidden_size << " L=" << num_layers
       << " H=" << num_heads << " KV=" << num_kv_heads << " hd=" << head_dim()
       << " ctx=" << max_seq_len << " rope=" << rope_theta;
    if (rope_scale > 1.0f) ss << strfmt("(x%.1f NTK->%.0f)", (double)rope_scale,
                                        (double)rope_theta_eff_local(*this)).c_str();
    ss << (tie_embeddings ? " tied" : " untied");
    if (use_qk_norm) ss << " QK-Norm";
    if (z_loss_scale > 0) ss << strfmt(" Z(%.0e)", (double)z_loss_scale).c_str();
    if (rope_scale > 1.0f) ss << strfmt(" YaRN-m%.2f", (double)rope_mscale_local(*this)).c_str();
    if (moe_jitter > 0) ss << strfmt(" jit=%.2f", (double)moe_jitter).c_str();
    if (use_moe)
        ss << " MoE(ne=" << num_experts << ",k=" << moe_top_k
           << ",E=" << moe_expert_dim << (moe_shared ? ",shared" : "")
           << (moe_aux_free ? ",aux-free" : "")
           << (moe_allow_dense ? ",dense-ok" : "") << ")";
    else
        ss << " ffn=" << intermediate_size << " (dense)";
    if (sliding_window > 0) ss << " SWA(" << sliding_window << ")";
    if (rope_type == 1) ss << " RoPE-neox";
    return ss.str();
}

std::string ModelConfig::arch_identity() const {

    std::ostringstream ss;
    ss << "v1|voc=" << vocab_size << "|d=" << hidden_size << "|L=" << num_layers
       << "|H=" << num_heads << "|KV=" << num_kv_heads << "|ffn=" << intermediate_size
       << "|ctx=" << max_seq_len << "|theta=" << strfmt("%.6f", (double)rope_theta).c_str()
       << "|eps=" << strfmt("%.9f", (double)rms_eps).c_str()
       << "|tie=" << (tie_embeddings ? 1 : 0)
       << "|moe=" << (use_moe ? 1 : 0) << "|ne=" << num_experts << "|k=" << moe_top_k
       << "|E=" << moe_expert_dim << "|sh=" << (moe_shared ? 1 : 0)
       << "|aux=" << strfmt("%.6f", (double)moe_aux_scale).c_str()
       << "|auxfree=" << (moe_aux_free ? 1 : 0)
       << "|denseok=" << (moe_allow_dense ? 1 : 0)
       << "|jit=" << strfmt("%.6f", (double)moe_jitter).c_str()
       << "|qkn=" << (use_qk_norm ? 1 : 0)
       << "|z=" << strfmt("%.7f", (double)z_loss_scale).c_str()
       << "|rs=" << strfmt("%.4f", (double)rope_scale).c_str()
       << "|ym=" << strfmt("%.4f", (double)rope_yarn_mscale).c_str()
       << "|yl=" << strfmt("%.4f", (double)rope_yarn_low).c_str()
       << "|yh=" << strfmt("%.4f", (double)rope_yarn_high).c_str()
       << "|swa=" << sliding_window << "|rt=" << rope_type;
    return ss.str();
}

bool ModelConfig::same_architecture_as(const ModelConfig& o, std::string* reason) const {
    auto fail = [&](const char* f) {
        if (reason) *reason = std::string("arch mismatch: ") + f;
        return false;
    };
    if (vocab_size != o.vocab_size) return fail("vocab_size");
    if (hidden_size != o.hidden_size) return fail("hidden_size");
    if (num_layers != o.num_layers) return fail("num_layers");
    if (num_heads != o.num_heads) return fail("num_heads");
    if (num_kv_heads != o.num_kv_heads) return fail("num_kv_heads");
    if (intermediate_size != o.intermediate_size) return fail("intermediate_size");
    if (max_seq_len != o.max_seq_len) return fail("max_seq_len");
    if (rope_theta != o.rope_theta) return fail("rope_theta");
    if (rms_eps != o.rms_eps) return fail("rms_eps");
    if (tie_embeddings != o.tie_embeddings) return fail("tie_embeddings");
    if (use_moe != o.use_moe) return fail("use_moe");
    if (num_experts != o.num_experts) return fail("num_experts");
    if (moe_top_k != o.moe_top_k) return fail("moe_top_k");
    if (moe_expert_dim != o.moe_expert_dim) return fail("moe_expert_dim");
    if (moe_shared != o.moe_shared) return fail("moe_shared");
    if (moe_aux_scale != o.moe_aux_scale) return fail("moe_aux_scale");
    if (moe_aux_free != o.moe_aux_free) return fail("moe_aux_free");
    if (moe_allow_dense != o.moe_allow_dense) return fail("moe_allow_dense");
    if (moe_jitter != o.moe_jitter) return fail("moe_jitter");
    if (use_qk_norm != o.use_qk_norm) return fail("use_qk_norm");
    if (z_loss_scale != o.z_loss_scale) return fail("z_loss_scale");
    if (rope_scale != o.rope_scale) return fail("rope_scale");
    if (rope_yarn_mscale != o.rope_yarn_mscale) return fail("rope_yarn_mscale");
    if (rope_yarn_low != o.rope_yarn_low) return fail("rope_yarn_low");
    if (rope_yarn_high != o.rope_yarn_high) return fail("rope_yarn_high");
    if (sliding_window != o.sliding_window) return fail("sliding_window");
    if (rope_type != o.rope_type) return fail("rope_type");

    return true;
}

float rope_theta_eff(const ModelConfig& m) { return rope_theta_eff_local(m); }
float rope_mscale(const ModelConfig& m) { return rope_mscale_local(m); }

void Model::rebuild_rope_cache() {
    const int half = cfg_.head_dim() / 2;
    std::vector<float> host(static_cast<size_t>(half));
    const float theta = rope_theta_eff_local(cfg_);
    for (int i = 0; i < half; ++i) {
        float base = 1.0f / std::pow(theta,
            (2.0f * static_cast<float>(i)) / static_cast<float>(cfg_.head_dim()));
        if (cfg_.rope_scale > 1.0f) {
            const float wavelength = 2.0f * 3.14159265358979f / base;
            if (wavelength > cfg_.rope_yarn_high) {
                base /= cfg_.rope_scale;
            } else if (wavelength > cfg_.rope_yarn_low) {
                const float span = cfg_.rope_yarn_high - cfg_.rope_yarn_low;
                const float t = span > 0.0f ? (wavelength - cfg_.rope_yarn_low) / span : 1.0f;
                base *= 1.0f - t + t / cfg_.rope_scale;
            }
        }
        host[static_cast<size_t>(i)] = base;
    }
    rope_inv_freq_ = Tensor({half}, DType::F32, device_);
    device_copy(rope_inv_freq_.data_ptr(), device_, host.data(), Device::CPU,
                host.size() * sizeof(float));
}

const float* Model::rope_inv_freq_ptr() const {
    return rope_inv_freq_.defined() ? rope_inv_freq_.f32() : nullptr;
}

void Model::alloc_param(Parameter& p, const std::string& name, std::vector<i64> shape, bool decay) {
    p.name  = name;
    p.shape = shape;
    p.decay = decay;
    p.w = Tensor::zeros(shape, DType::F32, device_);
    p.w.set_name(name);
    params_.push_back(&p);
}

Model::~Model() {
    enable_fp16_weight_cache(false);
}

Model::Model(ModelConfig cfg, Device dev) : cfg_(std::move(cfg)), device_(dev) {
    cfg_.validate();

    if (cfg_.rope_scale > 1.0f) {
        log_warn(strfmt("[rope] rope_scale=%.1f is approximate NTK/YaRN extrapolation "
                        "(theta->%.0f, mscale=%.2f): validate ppl at %d ctx before long runs",
                        (double)cfg_.rope_scale, (double)rope_theta_eff_local(cfg_),
                        (double)rope_mscale_local(cfg_), cfg_.max_seq_len));
    }
    if (cfg_.use_moe && cfg_.moe_allow_dense) {
        log_warn("[moe ] moe_allow_dense=true: dense routing enabled for research "
                 "(expect T4 OOM at 1B; keep runs short)");
    }
    ops::set_moe_jitter(cfg_.moe_jitter);

    const i64 d   = cfg_.hidden_size;
    const i64 V   = cfg_.vocab_size;
    const i64 qd  = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    const i64 F   = cfg_.intermediate_size;
    const i64 E   = cfg_.moe_expert_dim;
    const i64 ne  = cfg_.num_experts;

    alloc_param(tok_emb_, "tok_embeddings", {V, d}, false);

    layers_.resize(static_cast<size_t>(cfg_.num_layers));
    for (int i = 0; i < cfg_.num_layers; ++i) {
        LayerParams& L = layers_[static_cast<size_t>(i)];
        std::string pre = "layers." + std::to_string(i) + ".";
        alloc_param(L.attn_norm, pre + "attn_norm", {d},      false);
        alloc_param(L.wq,        pre + "wq",        {qd, d},  true);
        alloc_param(L.wk,        pre + "wk",        {kvd, d}, true);
        alloc_param(L.wv,        pre + "wv",        {kvd, d}, true);
        alloc_param(L.wo,        pre + "wo",        {d, qd},  true);
        alloc_param(L.ffn_norm,  pre + "ffn_norm",  {d},      false);
        if (cfg_.use_qk_norm) {
            const i64 hd = cfg_.head_dim();
            alloc_param(L.qk_qnorm, pre + "qk_qnorm", {hd}, false);
            alloc_param(L.qk_knorm, pre + "qk_knorm", {hd}, false);
        }
        if (cfg_.use_moe) {
            alloc_param(L.router,   pre + "moe_router", {ne, d},    true);
            alloc_param(L.moe_gate, pre + "moe_gate",   {ne * E, d}, true);
            alloc_param(L.moe_up,   pre + "moe_up",     {ne * E, d}, true);
            alloc_param(L.moe_down, pre + "moe_down",   {ne * d, E}, true);
            if (cfg_.moe_shared) {
                alloc_param(L.sh_gate, pre + "moe_sh_gate", {E, d}, true);
                alloc_param(L.sh_up,   pre + "moe_sh_up",   {E, d}, true);
                alloc_param(L.sh_down, pre + "moe_sh_down", {d, E}, true);
            }
        } else {
            alloc_param(L.w_gate,    pre + "w_gate",    {F, d},   true);
            alloc_param(L.w_up,      pre + "w_up",      {F, d},   true);
            alloc_param(L.w_down,    pre + "w_down",    {d, F},   true);
        }
    }

    alloc_param(final_norm_, "final_norm", {d}, false);
    if (!cfg_.tie_embeddings) alloc_param(lm_head_, "lm_head", {V, d}, true);
    if (cfg_.use_moe && cfg_.moe_aux_free) {
        ensure_moe_bias();
        log_info("[moe ] aux-loss-free bias enabled (DeepSeek-V3 §3.2): EMA steering, no aux grad");
    }
    rebuild_rope_cache();
}

const u16* Model::fused_qkv_ptr(int i) const {
    if (!fp16_weight_cache_ || i < 0 || static_cast<size_t>(i) >= layers_.size()) return nullptr;
    const Tensor& fused = layers_[static_cast<size_t>(i)].wqkv_fp16;
    return fused.defined() ? fused.ptr<u16>() : nullptr;
}

Parameter* Model::find_parameter(const std::string& name) {
    for (Parameter* p : params_) if (p->name == name) return p;
    return nullptr;
}

i64 Model::num_parameters() const {
    i64 n = 0;
    for (const Parameter* p : params_) n += p->numel();
    return n;
}

i64 Model::num_parameters_non_embedding() const {
    i64 n = 0;
    for (const Parameter* p : params_) {
        if (p->name == "tok_embeddings" || p->name == "lm_head") continue;
        n += p->numel();
    }
    return n;
}

ParamReport Model::parameter_report() const {
    ParamReport r;
    const i64 d   = cfg_.hidden_size;
    const i64 qd  = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    const i64 F   = cfg_.intermediate_size;
    const i64 E   = cfg_.moe_expert_dim;
    const i64 ne  = cfg_.num_experts;
    const int L   = cfg_.num_layers;

    auto add = [&](const std::string& n, i64 per, int units) {
        r.groups.push_back(ParamGroupCount{n, per * units, per, units});
    };

    add(cfg_.tie_embeddings ? "tok_embeddings (tied with lm_head)" : "tok_embeddings",
        cfg_.vocab_size * d, 1);
    add("attn.wq   [q_dim, d]",  qd * d,  L);
    add("attn.wk   [kv_dim, d]", kvd * d, L);
    add("attn.wv   [kv_dim, d]", kvd * d, L);
    add("attn.wo   [d, q_dim]",  d * qd,  L);
    if (cfg_.use_moe) {
        add("moe.router [ne, d]",      ne * d,     L);
        add("moe.gate   [ne*E, d]",    ne * E * d, L);
        add("moe.up     [ne*E, d]",    ne * E * d, L);
        add("moe.down   [ne*d, E]",    ne * d * E, L);
        if (cfg_.moe_shared) {
            add("moe.shared.gate [E, d]", E * d,   L);
            add("moe.shared.up   [E, d]", E * d,   L);
            add("moe.shared.down [d, E]", d * E,   L);
        }
    } else {
        add("ffn.gate  [ffn, d]",    F * d,   L);
        add("ffn.up    [ffn, d]",    F * d,   L);
        add("ffn.down  [d, ffn]",    d * F,   L);
    }
    add("attn_norm [d]",         d,       L);
    add("ffn_norm  [d]",         d,       L);
    if (cfg_.use_qk_norm) {
        const i64 hd = cfg_.head_dim();
        add("qk_qnorm  [hd]", hd, L);
        add("qk_knorm  [hd]", hd, L);
    }
    add("final_norm [d]",        d,       1);
    if (!cfg_.tie_embeddings) add("lm_head [V, d]", cfg_.vocab_size * d, 1);

    for (const auto& g : r.groups) r.total += g.count;
    r.embedding = cfg_.vocab_size * d * (cfg_.tie_embeddings ? 1 : 2);
    r.non_embedding = r.total - r.embedding;
    return r;
}

void Model::to(Device dev) { move_to_device(dev, true); }

void Model::move_to_device(Device dev, bool announce) {
    if (device_ == dev) return;
    const bool restore_fp16_cache = fp16_weight_cache_;
    if (restore_fp16_cache) enable_fp16_weight_cache(false);

    if (announce)
        log_warn("[model] Model::to() moved weights; REBUILD Trainer activations and "
                 "Generator cache/scratch on the new device before next step "
                 "(cross-device use-after-move crashes T4).");
    for (Parameter* p : params_) {
        if (p->w.defined()) p->w = p->w.to(dev);
        if (p->g.defined()) p->g = p->g.to(dev);
    }
    device_ = dev;
    if (rope_inv_freq_.defined()) rope_inv_freq_ = rope_inv_freq_.to(dev);
    if (cfg_.use_moe && cfg_.moe_aux_free) {
        const size_t L = static_cast<size_t>(cfg_.num_layers);
        if (moe_bias_.size() != L)
            moe_bias_.assign(L, std::vector<float>(static_cast<size_t>(cfg_.num_experts), 0.0f));
        moe_bias_dev_.clear();
        if (dev == Device::CUDA) {
            moe_bias_dev_.reserve(L);
            for (size_t l = 0; l < L; ++l) {
                Tensor t({cfg_.num_experts}, DType::F32, dev);
                device_copy(t.data_ptr(), dev, moe_bias_[l].data(), Device::CPU,
                            sizeof(float) * moe_bias_[l].size());
                moe_bias_dev_.push_back(std::move(t));
            }
        }
    }
    if (restore_fp16_cache) enable_fp16_weight_cache(true);
}

void Model::set_rope_runtime(float scale, float yarn_mscale) {
    cfg_.rope_scale = scale;
    cfg_.rope_yarn_mscale = yarn_mscale;
    cfg_.validate();
    rebuild_rope_cache();
    log_info(strfmt("[rope] runtime scale=%.3f mscale=%.3f theta=%.1f",
                    (double)scale, (double)rope_mscale_local(cfg_),
                    (double)rope_theta_eff_local(cfg_)));
}

std::string ParamReport::to_string(const ModelConfig& cfg) const {
    std::ostringstream ss;
    ss << "\n";
    ss << "+--------------------------------------+-------------+-------+---------------+\n";
    ss << "| group                                |    per unit | units |         total |\n";
    ss << "+--------------------------------------+-------------+-------+---------------+\n";
    for (const auto& g : groups) {
        ss << "| " << g.name;
        for (size_t i = g.name.size(); i < 36; ++i) ss << ' ';
        ss << " | " << std::string(11 - std::min<size_t>(11, std::to_string(g.per_unit).size()), ' ')
           << g.per_unit
           << " | " << std::string(5 - std::min<size_t>(5, std::to_string(g.units).size()), ' ')
           << g.units
           << " | " << std::string(13 - std::min<size_t>(13, std::to_string(g.count).size()), ' ')
           << g.count << " |\n";
    }
    ss << "+--------------------------------------+-------------+-------+---------------+\n";
    ss << strfmt("  total parameters      : %lld  (%.2f M)\n",
                 static_cast<long long>(total), double(total) / 1e6);
    ss << strfmt("  embedding parameters  : %lld  (%.1f%%)\n",
                 static_cast<long long>(embedding),
                 total ? 100.0 * double(embedding) / double(total) : 0.0);
    ss << strfmt("  non-embedding params  : %lld  (%.2f M)\n",
                 static_cast<long long>(non_embedding), double(non_embedding) / 1e6);
    ss << strfmt("  weight memory  fp32   : %s\n", human_bytes(static_cast<u64>(total) * 4).c_str());
    ss << strfmt("  weight memory  fp16   : %s\n", human_bytes(static_cast<u64>(total) * 2).c_str());
    ss << strfmt("  weight memory  int8   : %s\n", human_bytes(static_cast<u64>(total)).c_str());
    ss << strfmt("  weight memory  int4   : %s\n", human_bytes(static_cast<u64>(total) / 2).c_str());
    {

        u64 kv = static_cast<u64>(cfg.num_layers) * 2ull *
                 static_cast<u64>(cfg.kv_dim()) * static_cast<u64>(cfg.max_seq_len) * 2ull;
        ss << strfmt("  kv cache @%d ctx fp16 : %s\n", cfg.max_seq_len, human_bytes(kv).c_str());
    }
    return ss.str();
}

void Model::print_parameter_report() const {
    log_info("---------------- Ghassan 1 preview model ----------------");
    log_info("  " + cfg_.summary());
    log_raw(LogLevel::Info, parameter_report().to_string(cfg_));
}

void Model::init_weights(u64 seed) {
    Rng rng(seed);
    if (cfg_.use_moe && cfg_.moe_aux_free) {
        for (auto& bias : moe_bias_) std::fill(bias.begin(), bias.end(), 0.0f);
    }
    const float std_base = cfg_.init_std;

    const float std_res = std_base / std::sqrt(2.0f * static_cast<float>(cfg_.num_layers));

    Device orig_dev = device_;
    if (device_ != Device::CPU) {
        move_to_device(Device::CPU, false);
    }

    auto fill_normal = [&](Parameter& p, float sd) {
        float* w = p.w.f32();
        for (i64 i = 0; i < p.w.numel(); ++i) w[i] = rng.truncated_normal(sd);
    };
    auto fill_ones = [&](Parameter& p) {
        float* w = p.w.f32();
        for (i64 i = 0; i < p.w.numel(); ++i) w[i] = 1.0f;
    };

    fill_normal(tok_emb_, std_base);
    if (!cfg_.tie_embeddings) fill_normal(lm_head_, std_base);
    fill_ones(final_norm_);

    for (auto& L : layers_) {
        fill_ones(L.attn_norm);
        fill_ones(L.ffn_norm);
        if (L.qk_qnorm.w.defined()) fill_ones(L.qk_qnorm);
        if (L.qk_knorm.w.defined()) fill_ones(L.qk_knorm);
        fill_normal(L.wq, std_base);
        fill_normal(L.wk, std_base);
        fill_normal(L.wv, std_base);
        fill_normal(L.wo, std_res);
        if (L.has_moe()) {

            fill_normal(L.router, std::min(std_base, 0.01f));
            fill_normal(L.moe_gate, std_base);
            fill_normal(L.moe_up,   std_base);
            fill_normal(L.moe_down, std_res);
            if (L.sh_gate.w.defined()) {
                fill_normal(L.sh_gate, std_base);
                fill_normal(L.sh_up,   std_base);
                fill_normal(L.sh_down, std_res);
            }
        } else {
            fill_normal(L.w_gate, std_base);
            fill_normal(L.w_up,   std_base);
            fill_normal(L.w_down, std_res);
        }
    }

    if (orig_dev != Device::CPU) {
        move_to_device(orig_dev, false);
    }
    if (fp16_weight_cache_) enable_fp16_weight_cache(true);
}

void Model::enable_fp16_weight_cache(bool on) {
    if (!on) {
        for (Parameter* p : params_) {
            if (p->fp16_cache.defined()) {
                ops::unregister_fp16_weight(p->w.f32());
                p->fp16_cache = Tensor();
            }
        }
        for (LayerParams& layer : layers_) layer.wqkv_fp16 = Tensor();
        fp16_weight_cache_ = false;
        fp16_weights_dirty_ = false;
        return;
    }
    if (!fp16_weight_cache_) {
        for (Parameter* p : params_) {
            if (p->shape.size() != 2) continue;
            p->fp16_cache = Tensor::empty(p->shape, DType::F16, device_);
        }
        fp16_weight_cache_ = true;
    }
    for (Parameter* p : params_) {
        if (!p->fp16_cache.defined()) continue;
        ops::convert_f32_to_f16(device_, p->w.f32(), p->fp16_cache.ptr<u16>(), p->numel());
        ops::register_fp16_weight(p->w.f32(), p->fp16_cache.ptr<u16>(), p->numel());
    }
    const i64 d = cfg_.hidden_size;
    const i64 qd = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    for (LayerParams& layer : layers_) {
        layer.wqkv_fp16 = Tensor::empty({qd + 2 * kvd, d}, DType::F16, device_);
        u16* fused = layer.wqkv_fp16.ptr<u16>();
        ops::convert_f32_to_f16(device_, layer.wq.w.f32(), fused, qd * d);
        ops::convert_f32_to_f16(device_, layer.wk.w.f32(), fused + qd * d, kvd * d);
        ops::convert_f32_to_f16(device_, layer.wv.w.f32(), fused + (qd + kvd) * d, kvd * d);
    }
    fp16_weights_dirty_ = false;
}

void Model::mark_weights_dirty() {
    if (fp16_weight_cache_) fp16_weights_dirty_ = true;
}

size_t Model::fp16_weight_cache_bytes() const {
    size_t bytes = 0;
    for (const Parameter* p : params_) bytes += p->fp16_cache.nbytes();

    for (const LayerParams& layer : layers_) bytes += layer.wqkv_fp16.nbytes();
    return bytes;
}

void Model::enable_grad(bool on) {
    grad_enabled_ = on;
    if (on) {

        for (const Parameter* p : params_) {
            GAI_CHECK(!p->w.is_external(),
                      "enable_grad on mmap-backed weights is forbidden (read-only file pages): " +
                      p->name + " — reload without --mmap for training");
        }
    }
    for (Parameter* p : params_) {
        if (on) {
            if (!p->g.defined()) {
                p->g = Tensor::zeros(p->shape, DType::F32, device_);
                p->g.set_name(p->name + ".grad");
            }
        } else {
            p->g = Tensor();
        }
    }
}

void Model::zero_grad() {

    for (Parameter* p : params_)
        if (p->g.defined()) p->g.zero_();
}

static Tensor mk(std::vector<i64> shape, Device dev, size_t& acc) {
    Tensor t = Tensor::zeros(std::move(shape), DType::F32, dev);
    acc += t.nbytes();
    return t;
}

Activations Model::make_activations(int B, int T, bool with_grad, int ce_chunks) const {
    Activations a;
    a.B = B;
    a.T = T;
    a.with_grad = with_grad;

    const i64 N   = static_cast<i64>(B) * T;

    if (ce_chunks < 1) ce_chunks = 1;
    const i64 Cc = (with_grad && ce_chunks > 1) ? (N + ce_chunks - 1) / ce_chunks : N;
    a.ce_chunks = (with_grad && ce_chunks > 1) ? ce_chunks : 1;
    a.ce_rows = Cc;
    const i64 d   = cfg_.hidden_size;
    const i64 qd  = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    const i64 F   = cfg_.intermediate_size;
    const i64 E   = cfg_.moe_expert_dim;
    const i64 ne  = cfg_.num_experts;
    const i64 K   = cfg_.moe_top_k;
    const i64 V   = cfg_.vocab_size;
    const int L   = cfg_.num_layers;
    const bool moe = cfg_.use_moe;
    size_t& acc = a.bytes;

    auto mk_dt = [&](std::vector<i64> shape, DType dt) {
        Tensor t = Tensor::zeros(std::move(shape), dt, device_);
        acc += t.nbytes();
        return t;
    };

    a.x       = mk({N, d},  device_, acc);
    a.xb      = mk({N, d},  device_, acc);
    a.xb2     = mk({N, d},  device_, acc);
    if (fp16_weight_cache_) a.qkv = mk({N, qd + 2 * kvd}, device_, acc);
    a.q = mk({N, qd}, device_, acc);
    a.k = mk({N, kvd}, device_, acc);
    a.v = mk({N, kvd}, device_, acc);
    a.att_out = mk({N, qd}, device_, acc);
    a.proj    = mk({N, d},  device_, acc);
    if (moe) {

        a.moe_gate  = mk({N, K, E}, device_, acc);
        a.moe_up    = mk({N, K, E}, device_, acc);
        a.moe_act   = mk({N, K, E}, device_, acc);
        a.moe_probs = mk({N, ne},   device_, acc);
        a.moe_idx   = mk_dt({N, K}, DType::I32);
        a.moe_w     = mk({N, K},     device_, acc);
    } else {
        a.gate    = mk({N, F},  device_, acc);
        a.up      = mk({N, F},  device_, acc);
        a.act     = mk({N, F},  device_, acc);
    }
    a.ffn_out = mk({N, d},  device_, acc);
    a.logits  = mk({Cc, V}, device_, acc);

    if (with_grad) {
        a.saved_x.resize(static_cast<size_t>(L));
        a.saved_xb.resize(static_cast<size_t>(L));
        a.saved_rrms1.resize(static_cast<size_t>(L));
        a.saved_q.resize(static_cast<size_t>(L));
        a.saved_k.resize(static_cast<size_t>(L));
        a.saved_v.resize(static_cast<size_t>(L));
        if (cfg_.use_qk_norm) {
            a.saved_qk_rms_q.resize(static_cast<size_t>(L));
            a.saved_qk_rms_k.resize(static_cast<size_t>(L));
            a.saved_qk_raw_q.resize(static_cast<size_t>(L));
            a.saved_qk_raw_k.resize(static_cast<size_t>(L));
        }

        {
            const size_t need_tmp =
                static_cast<size_t>(B) * static_cast<size_t>(cfg_.num_heads) *
                static_cast<size_t>(T) * static_cast<size_t>(T) * sizeof(float);
            if (need_tmp > (800ULL << 20)) {
                GAI_FAIL(strfmt("attention tmp needs %s (B=%d H=%d T=%d). "
                                "Lower batch_size/seq_len or enable ce_chunks "
                                "(T=4096 needs 1.6GB transient).",
                                human_bytes(need_tmp).c_str(), B, cfg_.num_heads, T));
            } else if (need_tmp > (400ULL << 20)) {
                log_warn(strfmt("[model] attention tmp large (%s); T4 frag risk",
                                human_bytes(need_tmp).c_str()));
            }
        }
        a.attn_probs_tmp = mk({static_cast<i64>(B) * cfg_.num_heads * T * T}, device_, acc);
        a.saved_attout.resize(static_cast<size_t>(L));
        a.saved_xmid.resize(static_cast<size_t>(L));
        a.saved_xb2.resize(static_cast<size_t>(L));
        a.saved_rrms2.resize(static_cast<size_t>(L));
        a.saved_gate.resize(static_cast<size_t>(L));
        a.saved_up.resize(static_cast<size_t>(L));
        a.saved_act.resize(static_cast<size_t>(L));
        if (moe) {
            a.saved_moe_probs.resize(static_cast<size_t>(L));
            a.saved_moe_idx.resize(static_cast<size_t>(L));
            a.saved_moe_w.resize(static_cast<size_t>(L));
            a.moe_dact    = mk({N, K, E}, device_, acc);
            a.moe_auxfrac = mk({ne},      device_, acc);
        }

        for (int i = 0; i < L; ++i) {
            size_t s = static_cast<size_t>(i);
            a.saved_x[s]      = mk({N, d},   device_, acc);
            a.saved_xb[s]     = mk({N, d},   device_, acc);
            a.saved_rrms1[s]  = mk({N},      device_, acc);
            a.saved_q[s]      = mk({N, qd},  device_, acc);
            a.saved_k[s]      = mk({N, kvd}, device_, acc);
            a.saved_v[s]      = mk({N, kvd}, device_, acc);
            if (cfg_.use_qk_norm) {
                const i64 Hh = cfg_.num_heads;
                const i64 KVh = cfg_.num_kv_heads;
                a.saved_qk_rms_q[s] = mk({N * Hh}, device_, acc);
                a.saved_qk_rms_k[s] = mk({N * KVh}, device_, acc);
                a.saved_qk_raw_q[s] = mk({N, qd},  device_, acc);
                a.saved_qk_raw_k[s] = mk({N, kvd}, device_, acc);
            }
            a.saved_attout[s] = mk({N, qd},  device_, acc);
            a.saved_xmid[s]   = mk({N, d},   device_, acc);
            a.saved_xb2[s]    = mk({N, d},   device_, acc);
            a.saved_rrms2[s]  = mk({N},      device_, acc);
            if (moe) {
                a.saved_gate[s]      = mk({N, K, E}, device_, acc);
                a.saved_up[s]        = mk({N, K, E}, device_, acc);
                a.saved_act[s]       = mk({N, K, E}, device_, acc);
                a.saved_moe_probs[s] = mk({N, ne},   device_, acc);
                a.saved_moe_idx[s]   = mk_dt({N, K}, DType::I32);
                a.saved_moe_w[s]     = mk({N, K},     device_, acc);
            } else {
                a.saved_gate[s]   = mk({N, F},   device_, acc);
                a.saved_up[s]     = mk({N, F},   device_, acc);
                a.saved_act[s]    = mk({N, F},   device_, acc);
            }
        }
        a.saved_xfinal    = mk({N, d}, device_, acc);
        a.saved_rrms_final= mk({N},    device_, acc);
        a.saved_hnorm     = mk({N, d}, device_, acc);

        a.dx      = mk({N, d},   device_, acc);
        a.dxb     = mk({N, d},   device_, acc);
        a.dq      = mk({N, qd},  device_, acc);
        a.dk      = mk({N, kvd}, device_, acc);
        a.dv      = mk({N, kvd}, device_, acc);
        a.dattout = mk({N, qd},  device_, acc);
        a.dproj   = mk({N, d},   device_, acc);
        if (!moe) {
            a.dgate   = mk({N, F},   device_, acc);
            a.dup     = mk({N, F},   device_, acc);
            a.dact    = mk({N, F},   device_, acc);
            a.dffn    = mk({N, d},   device_, acc);
        }
        a.dlogits = mk({Cc, V},  device_, acc);
        a.dtmp    = mk({N, d},   device_, acc);
    }
    return a;
}

size_t Model::estimate_activation_bytes(int B, int T, bool with_grad, int ce_chunks) const {
    return estimate_activation_bytes_for(cfg_, fp16_weight_cache_, B, T, with_grad, ce_chunks);
}

size_t Model::estimate_activation_bytes_for(const ModelConfig& cfg, bool fp16_on,
                                            int B, int T, bool with_grad, int ce_chunks) {

    auto sat_add = [](u64 a, u64 b) -> u64 {
        const u64 cap = (1ull << 40);
        if (a > cap || b > cap) return cap;
        u64 s = a + b;
        return (s < a || s > cap) ? cap : s;
    };
    auto sat_mul = [&](u64 a, u64 b) -> u64 {
        if (a == 0 || b == 0) return 0;
        const u64 cap = (1ull << 40);
        if (a > cap / b) return cap;
        u64 p = a * b;
        return p > cap ? cap : p;
    };
    const u64 N   = static_cast<u64>(B) * static_cast<u64>(T);
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 qd  = static_cast<u64>(cfg.q_dim());
    const u64 kvd = static_cast<u64>(cfg.kv_dim());
    const u64 F   = cfg.use_moe ? static_cast<u64>(cfg.moe_top_k) * static_cast<u64>(cfg.moe_expert_dim)
                                 : static_cast<u64>(cfg.intermediate_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 L   = static_cast<u64>(cfg.num_layers);

    u64 e = sat_mul(N, (4 * d + qd * 2 + kvd * 2 + F * 3 + V));
    if (fp16_on) e = sat_add(e, sat_mul(N, (qd + 2 * kvd)));
    if (cfg.use_moe) e = sat_add(e, sat_mul(N, (static_cast<u64>(cfg.num_experts) + 2 * static_cast<u64>(cfg.moe_top_k))));

    e = sat_add(e, N);
    if (with_grad) {

        u64 per_layer = sat_mul(N, (4 * d + qd * 2 + kvd * 2 + F * 3 + 2));
        if (cfg.use_moe) per_layer = sat_add(per_layer, sat_mul(N, (static_cast<u64>(cfg.num_experts) + 3 * static_cast<u64>(cfg.moe_top_k))));
        if (cfg.use_qk_norm) per_layer = sat_add(per_layer, sat_mul(N, (static_cast<u64>(cfg.num_heads) + static_cast<u64>(cfg.num_kv_heads) + qd + kvd)));
        e = sat_add(e, sat_mul(L, per_layer));
        e = sat_add(e, sat_mul(sat_mul(static_cast<u64>(B), static_cast<u64>(cfg.num_heads)), sat_mul(static_cast<u64>(T), static_cast<u64>(T))));
        if (!cfg.use_moe) {
            e = sat_add(e, sat_mul(N, (5 * d + qd * 2 + kvd * 2 + F * 3 + V + 2)));
        } else {
            e = sat_add(e, sat_mul(N, (5 * d + qd * 2 + kvd * 2 + V + 2)));
            e = sat_add(e, sat_add(sat_mul(N, sat_mul(static_cast<u64>(cfg.moe_top_k), static_cast<u64>(cfg.moe_expert_dim))), static_cast<u64>(cfg.num_experts)));
        }

        e = sat_add(e, sat_mul(N, (4 * d + qd * 2 + kvd * 2 + V + d * 2)));
    }

    if (with_grad && ce_chunks > 1 && N > 0) {
        const u64 cc = static_cast<u64>(ce_chunks);
        const u64 crow = N / cc + (N % cc ? 1ULL : 0ULL);
        const u64 sub = sat_mul(sat_mul(2ULL, V), N - crow);
        e = (e >= sub) ? e - sub : 0;
    }
    if (e > ((1ull << 40) / 4)) return (1ull << 40);
    return static_cast<size_t>(e) * sizeof(float);
}

u64 Model::count_parameters(const ModelConfig& cfg) {
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 qd  = static_cast<u64>(cfg.q_dim());
    const u64 kvd = static_cast<u64>(cfg.kv_dim());
    const u64 F   = static_cast<u64>(cfg.intermediate_size);
    const u64 E   = static_cast<u64>(cfg.moe_expert_dim);
    const u64 ne  = static_cast<u64>(cfg.num_experts);
    const u64 L   = static_cast<u64>(cfg.num_layers);
    const u64 hd  = static_cast<u64>(cfg.head_dim());

    u64 n = V * d;
    for (u64 i = 0; i < L; ++i) {
        n += d;
        n += qd * d + kvd * d + kvd * d + d * qd;
        n += d;
        if (cfg.use_qk_norm) n += 2 * hd;
        if (cfg.use_moe) {
            n += ne * d;
            n += ne * E * d + ne * E * d;
            n += ne * d * E;
            if (cfg.moe_shared) n += E * d + E * d + d * E;
        } else {
            n += F * d + F * d + d * F;
        }
    }
    n += d;
    if (!cfg.tie_embeddings) n += V * d;
    return n;
}

size_t Model::count_fp16_cache_bytes(const ModelConfig& cfg) {
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 qd  = static_cast<u64>(cfg.q_dim());
    const u64 kvd = static_cast<u64>(cfg.kv_dim());
    const u64 F   = static_cast<u64>(cfg.intermediate_size);
    const u64 E   = static_cast<u64>(cfg.moe_expert_dim);
    const u64 ne  = static_cast<u64>(cfg.num_experts);
    const u64 L   = static_cast<u64>(cfg.num_layers);

    u64 n2 = 0;
    n2 += V * d;
    for (u64 i = 0; i < L; ++i) {
        n2 += qd * d + kvd * d + kvd * d + d * qd;
        if (cfg.use_moe) {
            n2 += ne * d;
            n2 += ne * E * d + ne * E * d;
            n2 += ne * d * E;
            if (cfg.moe_shared) n2 += E * d + E * d + d * E;
        } else {
            n2 += F * d + F * d + d * F;
        }
    }
    if (!cfg.tie_embeddings) n2 += V * d;

    const u64 fused = L * (qd + 2 * kvd) * d;
    return static_cast<size_t>((n2 + fused) * sizeof(u16));
}

Model::MemoryPlan Model::plan_memory(const ModelConfig& cfg, int B, int T,
                                      bool with_grad, int ce_chunks,
                                      bool fp16_weight_cache) {
    MemoryPlan p;
    p.params = count_parameters(cfg);
    p.params_no_embedding = p.params - static_cast<u64>(cfg.vocab_size) * static_cast<u64>(cfg.hidden_size);
    if (!cfg.tie_embeddings)
        p.params_no_embedding -= static_cast<u64>(cfg.vocab_size) * static_cast<u64>(cfg.hidden_size);
    p.weights = static_cast<size_t>(p.params) * sizeof(float);
    p.grads = with_grad ? static_cast<size_t>(p.params) * sizeof(float) : 0;
    p.fp16_cache = fp16_weight_cache ? count_fp16_cache_bytes(cfg) : 0;
    p.activations = estimate_activation_bytes_for(cfg, fp16_weight_cache, B, T,
                                                 with_grad, ce_chunks);
    p.static_total = p.weights + p.grads + p.fp16_cache;
    p.total = p.static_total + p.activations;
    return p;
}

Model::WorkspacePlan Model::workspace_plan(const ModelConfig& cfg, int B, int T) {

    const u64 cap = (1ull << 40);
    auto sat = [&](u64 v) -> u64 { return v > cap ? cap : v; };
    const u64 N   = static_cast<u64>(B) * static_cast<u64>(T);
    const u64 NK  = N * static_cast<u64>(cfg.moe_top_k > 0 ? cfg.moe_top_k : 1);
    const u64 d   = static_cast<u64>(cfg.hidden_size);
    const u64 V   = static_cast<u64>(cfg.vocab_size);
    const u64 E   = static_cast<u64>(cfg.moe_expert_dim);
    const u64 ne  = static_cast<u64>(cfg.num_experts);

    WorkspacePlan p;

    const u64 gemm_conv = sat(2 * sat(N * d + d * V));
    const u64 sce = sat(N + 260);
    p.gemm_bytes = static_cast<size_t>(sat(2 * sat(gemm_conv + sce)));
    if (cfg.use_moe) {

        const u64 fwd = sat(N * ne + 2 * NK + 3 * NK * E + NK * d);

        const u64 bwd = sat(6 * N * E + NK + 4 * NK * E + 2 * NK * d + NK + N * ne);
        const u64 peak_floats = fwd > bwd ? fwd : bwd;
        p.moe_bytes = static_cast<size_t>(sat(4 * peak_floats + (16ull << 20)));
    }
    return p;
}

static void check_act_device(Device dev, const char* name, const Tensor& t) {
    if (t.defined() && t.device() != dev) {
        GAI_FAIL(std::string("stale activations: ") + name +
                 " is on " + device_name(t.device()) + ", model is on " +
                 device_name(dev) + " (rebuild activations after Model::to)");
    }
}

void Model::forward_body(const i32* ids, int B, int T, Activations& act,
                         const i32* segment_ids) {

    GAI_CHECK(ids != nullptr, "forward: null ids");
    GAI_CHECK(B > 0 && T > 0, "forward: B and T must be > 0");
    const int d   = cfg_.hidden_size;
    const int qd  = cfg_.q_dim();
    const int kvd = cfg_.kv_dim();
    const int F   = cfg_.intermediate_size;
    const int V   = cfg_.vocab_size;
    const int H   = cfg_.num_heads;
    const int KV  = cfg_.num_kv_heads;
    const int hd  = cfg_.head_dim();
    const i64 N   = static_cast<i64>(B) * T;
    const float scale = (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale_local(cfg_);
    const bool  train = act.with_grad;
    Device dev = device_;
    if (fp16_weight_cache_ && fp16_weights_dirty_) {
        for (Parameter* p : params_) {
            if (!p->fp16_cache.defined()) continue;
            ops::convert_f32_to_f16(dev, p->w.f32(), p->fp16_cache.ptr<u16>(), p->numel());
        }
        const i64 d = cfg_.hidden_size;
        const i64 qd = cfg_.q_dim();
        const i64 kvd = cfg_.kv_dim();
        for (LayerParams& layer : layers_) {
            u16* fused = layer.wqkv_fp16.ptr<u16>();
            ops::convert_f32_to_f16(dev, layer.wq.w.f32(), fused, qd * d);
            ops::convert_f32_to_f16(dev, layer.wk.w.f32(), fused + qd * d, kvd * d);
            ops::convert_f32_to_f16(dev, layer.wv.w.f32(), fused + (qd + kvd) * d, kvd * d);
        }
        fp16_weights_dirty_ = false;
    }

    if (dev == Device::CPU) {
        for (i64 i = 0; i < N; ++i) {
            i32 id = ids[i];
            GAI_CHECK(id >= 0 && id < V,
                      strfmt("forward: token id out of range ids[%lld]=%d (vocab=%d; tokenizer/dataloader mismatch?)",
                             (long long)i, (int)id, V));
        }
    }

    GAI_CHECK((act.B == B || (act.with_grad && act.B > B)) && act.T == T,
              "activation buffer shape mismatch");
    GAI_CHECK(T <= cfg_.max_seq_len, "sequence longer than max_seq_len");

    GAI_CHECK(N <= static_cast<i64>(std::numeric_limits<int>::max()),
              "forward: B*T exceeds int range (reduce batch/seq_len)");

    if (train) {
        GAI_CHECK(static_cast<int>(act.saved_x.size()) == cfg_.num_layers,
                  "forward: stale Activations (layer count mismatch, rebuild)");
    }
    if (train && cfg_.use_moe) {
        GAI_CHECK(static_cast<int>(act.saved_moe_probs.size()) == cfg_.num_layers,
                  "forward: MoE training needs with_grad Activations");
    }
    check_act_device(dev, "act.x", act.x);
    check_act_device(dev, "act.xb", act.xb);
    check_act_device(dev, "act.logits", act.logits);
    check_act_device(dev, "act.pos", act.pos);
    if (train) {
        check_act_device(dev, "act.saved_hnorm", act.saved_hnorm);
        check_act_device(dev, "act.dx", act.dx);
    }

    if (!act.pos.defined() || act.pos.numel() < N) {

        if (act.pos.defined()) act.bytes -= act.pos.nbytes();
        act.pos = Tensor::empty({N}, DType::I32, device_);
        act.bytes += act.pos.nbytes();
        act.pos_cached_B = -1;
        act.pos_cached_T = -1;
        act.pos_cached_offset = -1;
    }
    if (act.pos_cached_B != B || act.pos_cached_T != T || act.pos_cached_offset != act.pos_offset) {
        thread_local std::vector<i32> pos_staging;
        if (pos_staging.size() < static_cast<size_t>(N))
            pos_staging.resize(static_cast<size_t>(N));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                pos_staging[static_cast<size_t>(b) * T + t] = act.pos_offset + t;
        device_copy(act.pos.data_ptr(), device_, pos_staging.data(), Device::CPU,
                    static_cast<size_t>(N) * sizeof(i32));
        act.pos_cached_B = B;
        act.pos_cached_T = T;
        act.pos_cached_offset = act.pos_offset;
    }

    ops::embedding_forward(dev, ids, tok_emb_.w.f32(), act.x.f32(), N, d, V);

    const float* rope_freq = rope_inv_freq_ptr();
    GAI_CHECK(rope_freq != nullptr, "forward: RoPE frequency cache is missing");

    for (int l = 0; l < cfg_.num_layers; ++l) {
        LayerParams& L = layers_[static_cast<size_t>(l)];
        size_t sl = static_cast<size_t>(l);

        if (train) ops::copy(dev, act.saved_x[sl].f32(), act.x.f32(), N * d);

        float* rrms1 = train ? act.saved_rrms1[sl].f32() : nullptr;
        ops::rmsnorm_forward(dev, act.x.f32(), L.attn_norm.w.f32(), act.xb.f32(),
                             rrms1, N, d, cfg_.rms_eps);
        if (train) ops::copy(dev, act.saved_xb[sl].f32(), act.xb.f32(), N * d);

        float* qp = nullptr;
        float* kp = nullptr;
        float* vp = nullptr;
        if (act.qkv.defined()) {
            ops::linear_forward_fp16(dev, act.xb.f32(), L.wqkv_fp16.ptr<u16>(),
                                     act.qkv.f32(), static_cast<int>(N), d, qd + 2 * kvd);
            ops::split_qkv(dev, act.qkv.f32(), act.q.f32(), act.k.f32(), act.v.f32(),
                            N, qd, kvd);
            qp = act.q.f32();
            kp = act.k.f32();
            vp = act.v.f32();
        } else {
            qp = act.q.f32();
            kp = act.k.f32();
            vp = act.v.f32();
            ops::linear_forward(dev, act.xb.f32(), L.wq.w.f32(), qp, static_cast<int>(N), d, qd);
            ops::linear_forward(dev, act.xb.f32(), L.wk.w.f32(), kp, static_cast<int>(N), d, kvd);
            ops::linear_forward(dev, act.xb.f32(), L.wv.w.f32(), vp, static_cast<int>(N), d, kvd);
        }

        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined() && L.qk_knorm.w.defined()) {
            if (train) {
                ops::copy(dev, act.saved_qk_raw_q[sl].f32(), qp, N * qd);
                ops::copy(dev, act.saved_qk_raw_k[sl].f32(), kp, N * kvd);
            }
            float* rrms_q = train ? act.saved_qk_rms_q[sl].f32() : nullptr;
            float* rrms_k = train ? act.saved_qk_rms_k[sl].f32() : nullptr;
            ops::rmsnorm_forward(dev, qp, L.qk_qnorm.w.f32(), qp,
                                 rrms_q, N * H, hd, cfg_.rms_eps);
            ops::rmsnorm_forward(dev, kp, L.qk_knorm.w.f32(), kp,
                                 rrms_k, N * KV, hd, cfg_.rms_eps);
        }

        ops::rope_forward_cached(dev, qp, kp, act.pos.i32p(), rope_freq, N, H, KV, hd,
                                 cfg_.rope_type);

        if (train) {
            ops::copy(dev, act.saved_q[sl].f32(), qp, N * qd);
            ops::copy(dev, act.saved_k[sl].f32(), kp, N * kvd);
            ops::copy(dev, act.saved_v[sl].f32(), vp, N * kvd);
        }

        ops::attention_forward_ex(dev, qp, kp, vp,
                                  act.att_out.f32(), nullptr,
                                  B, T, H, KV, hd, scale, cfg_.sliding_window,
                                  segment_ids);
        if (train) ops::copy(dev, act.saved_attout[sl].f32(), act.att_out.f32(), N * qd);

        ops::linear_forward(dev, act.att_out.f32(), L.wo.w.f32(), act.proj.f32(), static_cast<int>(N), qd, d);
        ops::add_inplace(dev, act.x.f32(), act.proj.f32(), N * d);

        if (train) ops::copy(dev, act.saved_xmid[sl].f32(), act.x.f32(), N * d);

        float* rrms2 = train ? act.saved_rrms2[sl].f32() : nullptr;
        ops::rmsnorm_forward(dev, act.x.f32(), L.ffn_norm.w.f32(), act.xb2.f32(),
                             rrms2, N, d, cfg_.rms_eps);
        if (train) ops::copy(dev, act.saved_xb2[sl].f32(), act.xb2.f32(), N * d);

        if (L.has_moe()) {
            const int E  = cfg_.moe_expert_dim;
            const int ne = cfg_.num_experts;
            const int K  = cfg_.moe_top_k;
            float* probs = train ? act.saved_moe_probs[sl].f32() : nullptr;
            i32*   ridx  = train ? act.saved_moe_idx[sl].i32p()  : nullptr;
            float* rw    = train ? act.saved_moe_w[sl].f32()     : nullptr;
            const float* bias = (cfg_.moe_aux_free && !moe_bias_.empty())
                ? moe_bias_ptr(l) : nullptr;
            ops::moe_forward_bias(dev, act.xb2.f32(), L.router.w.f32(), bias,
                                  L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                  L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                  L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                  L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                  act.ffn_out.f32(), probs, ridx, rw,
                                  act.moe_gate.f32(), act.moe_up.f32(), act.moe_act.f32(),
                                  N, d, E, ne, K);
            if (train) {
                ops::copy(dev, act.saved_gate[sl].f32(), act.moe_gate.f32(), N * K * E);
                ops::copy(dev, act.saved_up[sl].f32(),   act.moe_up.f32(),   N * K * E);
                ops::copy(dev, act.saved_act[sl].f32(),  act.moe_act.f32(),  N * K * E);
            }
        } else {
            ops::linear_forward(dev, act.xb2.f32(), L.w_gate.w.f32(), act.gate.f32(), static_cast<int>(N), d, F);
            ops::linear_forward(dev, act.xb2.f32(), L.w_up.w.f32(),   act.up.f32(),   static_cast<int>(N), d, F);
            ops::swiglu_forward(dev, act.gate.f32(), act.up.f32(), act.act.f32(), N * F);

            if (train) {
                ops::copy(dev, act.saved_gate[sl].f32(), act.gate.f32(), N * F);
                ops::copy(dev, act.saved_up[sl].f32(),   act.up.f32(),   N * F);
                ops::copy(dev, act.saved_act[sl].f32(),  act.act.f32(),  N * F);
            }

            ops::linear_forward(dev, act.act.f32(), L.w_down.w.f32(), act.ffn_out.f32(), static_cast<int>(N), F, d);
        }
        ops::add_inplace(dev, act.x.f32(), act.ffn_out.f32(), N * d);
    }

    if (train) ops::copy(dev, act.saved_xfinal.f32(), act.x.f32(), N * d);
    float* rrmsf = train ? act.saved_rrms_final.f32() : nullptr;
    ops::rmsnorm_forward(dev, act.x.f32(), final_norm_.w.f32(), act.xb.f32(), rrmsf, N, d, cfg_.rms_eps);
    if (train) ops::copy(dev, act.saved_hnorm.f32(), act.xb.f32(), N * d);
}

Tensor& Model::forward(const i32* ids, int B, int T, Activations& act,
                       const i32* segment_ids) {

    forward_body(ids, B, T, act, segment_ids);
    const int d = cfg_.hidden_size;
    const int V = cfg_.vocab_size;
    const i64 N = static_cast<i64>(B) * T;
    {
        const i64 need = N * static_cast<i64>(V);
        GAI_CHECK(act.logits.numel() >= need,
                  "forward: logits buffer too small (chunked training activations "
                  "need forward_backward, not forward)");
    }
    ops::linear_forward(device_, act.xb.f32(), lm_head().w.f32(), act.logits.f32(),
                        static_cast<int>(N), d, V);
    return act.logits;
}

double Model::forward_backward(const i32* ids, const i32* targets, int B, int T,
                               Activations& act, i64* out_ntok, float dout_scale,
                               bool want_aux_stats, const i32* segment_ids,
                               const i32* host_targets) {
    GAI_CHECK(act.with_grad, "forward_backward requires gradient activations");
    GAI_CHECK(grad_enabled_, "call enable_grad(true) before forward_backward");

    const int d   = cfg_.hidden_size;
    const int qd  = cfg_.q_dim();
    const int kvd = cfg_.kv_dim();
    const int F   = cfg_.intermediate_size;
    const int V   = cfg_.vocab_size;
    const int H   = cfg_.num_heads;
    const int KV  = cfg_.num_kv_heads;
    const int hd  = cfg_.head_dim();
    const i64 N   = static_cast<i64>(B) * T;
    const float scale = (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale_local(cfg_);
    Device dev = device_;

    forward_body(ids, B, T, act, segment_ids);

    float eff_scale = dout_scale;

    const i64 Cc = act.ce_rows > 0 ? act.ce_rows : N;
    const float* hnorm_full = act.saved_hnorm.f32();
    float* logits_c = act.logits.f32();
    float* dlogits_c = act.dlogits.f32();
    ops::zero(dev, act.dxb.f32(), N * d);

    ops::sce_acc_begin(dev);
    for (i64 r0 = 0; r0 < N; r0 += Cc) {
        const i64 Cr = std::min(Cc, N - r0);
        const int Ci = static_cast<int>(Cr);
        if (host_targets) {
            bool any_supervised = false;
            for (i64 i = 0; i < Cr; ++i) {
                if (host_targets[r0 + i] >= 0) { any_supervised = true; break; }
            }
            if (!any_supervised) continue;
        }
        ops::linear_forward(dev, hnorm_full + r0 * d, lm_head().w.f32(),
                            logits_c, Ci, d, V);
        ops::sce_accumulate(dev, logits_c, targets + r0, dlogits_c,
                            Cr, V, cfg_.z_loss_scale);

        if (eff_scale != 1.0f) ops::scale_inplace(dev, dlogits_c, eff_scale, Cr * V);
        ops::linear_backward(dev, hnorm_full + r0 * d, lm_head().w.f32(), dlogits_c,
                             act.dxb.f32() + r0 * d, lm_head().g.f32(), Ci, d, V);
    }
    double loss_sum = 0.0;
    i64 ntok = 0;
    ops::sce_acc_end(dev, &loss_sum, &ntok);
    if (out_ntok) *out_ntok = ntok;
    if (ntok == 0) return 0.0;

    double aux_total = 0.0;
    const bool use_dev_aux = (dev == Device::CUDA) && cfg_.use_moe &&
                             cfg_.moe_aux_scale > 0.0f;
    double* aux_accum = use_dev_aux ? ops::moe_aux_begin(dev) : nullptr;

    const float* rope_freq_bwd = rope_inv_freq_ptr();
    GAI_CHECK(rope_freq_bwd != nullptr, "backward: RoPE frequency cache is missing");

    ops::zero(dev, act.dx.f32(), N * d);
    ops::rmsnorm_backward(dev, act.saved_xfinal.f32(), final_norm_.w.f32(), act.dxb.f32(),
                          act.saved_rrms_final.f32(), act.dx.f32(), final_norm_.g.f32(), N, d);

    for (int l = cfg_.num_layers - 1; l >= 0; --l) {
        LayerParams& L = layers_[static_cast<size_t>(l)];
        size_t sl = static_cast<size_t>(l);

        if (L.has_moe()) {
            const int E  = cfg_.moe_expert_dim;
            const int ne = cfg_.num_experts;
            const int K  = cfg_.moe_top_k;

            const float* aux_frac = nullptr;
            float aux_scale_grad = 0.0f;
            if (cfg_.moe_aux_scale > 0.0f) {
                aux_total += moe_layer_aux(dev, act.saved_moe_probs[sl].f32(),
                                           act.saved_moe_idx[sl].i32p(),
                                           act.moe_auxfrac.f32(),
                                           l, N, K, ne,
                                           want_aux_stats, aux_accum);
                aux_frac = act.moe_auxfrac.f32();
                aux_scale_grad = cfg_.moe_aux_scale * eff_scale * static_cast<float>(ntok) /
                                 static_cast<float>(cfg_.num_layers > 0 ? cfg_.num_layers : 1);
            }
            ops::zero(dev, act.dxb.f32(), N * d);
            ops::moe_backward(dev, act.saved_xb2[sl].f32(), L.router.w.f32(),
                              L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                              L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                              L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                              L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                              act.saved_moe_probs[sl].f32(),
                              act.saved_moe_idx[sl].i32p(),
                              act.saved_moe_w[sl].f32(),
                              act.saved_gate[sl].f32(), act.saved_up[sl].f32(),
                              act.saved_act[sl].f32(),
                              aux_frac, aux_scale_grad,
                              act.dx.f32(), act.dxb.f32(),
                              L.router.g.f32(), L.moe_gate.g.f32(),
                              L.moe_up.g.f32(), L.moe_down.g.f32(),
                              L.sh_gate.g.defined() ? L.sh_gate.g.f32() : nullptr,
                              L.sh_up.g.defined()   ? L.sh_up.g.f32()   : nullptr,
                              L.sh_down.g.defined() ? L.sh_down.g.f32() : nullptr,
                              act.moe_dact.f32(),
                              N, d, E, ne, K);
        } else {
            ops::zero(dev, act.dact.f32(), N * F);
            ops::linear_backward(dev, act.saved_act[sl].f32(), L.w_down.w.f32(), act.dx.f32(),
                                 act.dact.f32(), L.w_down.g.f32(), static_cast<int>(N), F, d);

            ops::zero(dev, act.dgate.f32(), N * F);
            ops::zero(dev, act.dup.f32(),   N * F);
            ops::swiglu_backward(dev, act.saved_gate[sl].f32(), act.saved_up[sl].f32(),
                                 act.dact.f32(), act.dgate.f32(), act.dup.f32(), N * F);

            ops::zero(dev, act.dxb.f32(), N * d);
            ops::linear_backward(dev, act.saved_xb2[sl].f32(), L.w_gate.w.f32(), act.dgate.f32(),
                                 act.dxb.f32(), L.w_gate.g.f32(), static_cast<int>(N), d, F);
            ops::linear_backward(dev, act.saved_xb2[sl].f32(), L.w_up.w.f32(), act.dup.f32(),
                                 act.dxb.f32(), L.w_up.g.f32(), static_cast<int>(N), d, F);
        }

        ops::rmsnorm_backward(dev, act.saved_xmid[sl].f32(), L.ffn_norm.w.f32(), act.dxb.f32(),
                              act.saved_rrms2[sl].f32(), act.dx.f32(), L.ffn_norm.g.f32(), N, d);

        ops::zero(dev, act.dattout.f32(), N * qd);
        ops::linear_backward(dev, act.saved_attout[sl].f32(), L.wo.w.f32(), act.dx.f32(),
                             act.dattout.f32(), L.wo.g.f32(), static_cast<int>(N), qd, d);

        ops::zero(dev, act.dq.f32(), N * qd);
        ops::zero(dev, act.dk.f32(), N * kvd);
        ops::zero(dev, act.dv.f32(), N * kvd);

        ops::attention_forward_ex(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                                  act.saved_v[sl].f32(), act.att_out.f32(),
                                  act.attn_probs_tmp.f32(),
                                  B, T, H, KV, hd, scale, cfg_.sliding_window,
                                  segment_ids);
        ops::attention_backward_ex(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                                   act.saved_v[sl].f32(), act.attn_probs_tmp.f32(),
                                   act.dattout.f32(),
                                   act.dq.f32(), act.dk.f32(), act.dv.f32(),
                                   B, T, H, KV, hd, scale, cfg_.sliding_window);

        ops::rope_backward_cached(dev, act.dq.f32(), act.dk.f32(), act.pos.i32p(), rope_freq_bwd,
                                  N, H, KV, hd, cfg_.rope_type);

        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined()) {

            ops::copy(dev, act.att_out.f32(), act.dq.f32(), N * qd);
            ops::zero(dev, act.dq.f32(), N * qd);
            ops::rmsnorm_backward(dev, act.saved_qk_raw_q[sl].f32(), L.qk_qnorm.w.f32(), act.att_out.f32(),
                                  act.saved_qk_rms_q[sl].f32(), act.dq.f32(), L.qk_qnorm.g.f32(),
                                  N * H, hd);

            ops::copy(dev, act.dattout.f32(), act.dk.f32(), N * kvd);
            ops::zero(dev, act.dk.f32(), N * kvd);
            ops::rmsnorm_backward(dev, act.saved_qk_raw_k[sl].f32(), L.qk_knorm.w.f32(), act.dattout.f32(),
                                  act.saved_qk_rms_k[sl].f32(), act.dk.f32(), L.qk_knorm.g.f32(),
                                  N * KV, hd);
        }

        ops::zero(dev, act.dxb.f32(), N * d);
        ops::linear_backward(dev, act.saved_xb[sl].f32(), L.wq.w.f32(), act.dq.f32(),
                             act.dxb.f32(), L.wq.g.f32(), static_cast<int>(N), d, qd);
        ops::linear_backward(dev, act.saved_xb[sl].f32(), L.wk.w.f32(), act.dk.f32(),
                             act.dxb.f32(), L.wk.g.f32(), static_cast<int>(N), d, kvd);
        ops::linear_backward(dev, act.saved_xb[sl].f32(), L.wv.w.f32(), act.dv.f32(),
                             act.dxb.f32(), L.wv.g.f32(), static_cast<int>(N), d, kvd);

        ops::rmsnorm_backward(dev, act.saved_x[sl].f32(), L.attn_norm.w.f32(), act.dxb.f32(),
                              act.saved_rrms1[sl].f32(), act.dx.f32(), L.attn_norm.g.f32(), N, d);
    }

    ops::embedding_backward(dev, ids, act.dx.f32(), tok_emb_.g.f32(), N, d, V);

    double loss = loss_sum / static_cast<double>(ntok);

    if (use_dev_aux) aux_total = ops::moe_aux_end(dev);
    if (cfg_.moe_aux_free) moe_aux_loss(act, B, T);

    if (cfg_.use_moe && cfg_.moe_aux_scale > 0.0f && cfg_.num_layers > 0)
        loss += cfg_.moe_aux_scale * aux_total / static_cast<double>(cfg_.num_layers);
    return loss;
}

double Model::moe_aux_loss(Activations& act, int B, int T) {
    if (!cfg_.use_moe) return 0.0;

    if (cfg_.moe_aux_free) {

        accumulate_moe_bias_fracs(act, B, T);
        return 0.0;
    }

    GAI_CHECK(act.with_grad, "moe_aux_loss needs training Activations (with_grad)");
    GAI_CHECK(static_cast<int>(act.saved_moe_probs.size()) == cfg_.num_layers,
              "moe_aux_loss: stale Activations");
    GAI_CHECK(act.moe_auxfrac.defined(), "moe_aux_loss: missing auxfrac buffer");
    const i64 N = static_cast<i64>(B) * T;
    double total = 0.0;
    for (int l = 0; l < cfg_.num_layers; ++l) {
        size_t sl = static_cast<size_t>(l);
        total += moe_layer_aux(device_, act.saved_moe_probs[sl].f32(),
                               act.saved_moe_idx[sl].i32p(),
                               act.moe_auxfrac.f32(), l,
                               N, cfg_.moe_top_k, cfg_.num_experts);
    }
    return total;
}

static constexpr u32 RAW_MAGIC = 0x57415247u;

void Model::save_raw(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot write model: " + path);
    u32 magic = RAW_MAGIC;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    u32 n = static_cast<u32>(params_.size());
    f.write(reinterpret_cast<const char*>(&n), 4);
    for (const Parameter* p : params_) {
        u32 len = static_cast<u32>(p->name.size());
        f.write(reinterpret_cast<const char*>(&len), 4);
        f.write(p->name.data(), len);
        u64 ne = static_cast<u64>(p->numel());
        f.write(reinterpret_cast<const char*>(&ne), 8);

        u32 nd = static_cast<u32>(p->shape.size());
        f.write(reinterpret_cast<const char*>(&nd), 4);
        for (i64 d : p->shape) {
            i64 dd = d;
            f.write(reinterpret_cast<const char*>(&dd), 8);
        }
        Tensor cpu = p->w.to(Device::CPU);
        f.write(reinterpret_cast<const char*>(cpu.data_ptr()),
                static_cast<std::streamsize>(cpu.nbytes()));
    }
    GAI_CHECK(f.good(), "model write failed");
}

bool Model::load_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    u32 magic = 0, n = 0;
    if (!f.read(reinterpret_cast<char*>(&magic), 4)) return false;
    if (magic != RAW_MAGIC) return false;
    if (!f.read(reinterpret_cast<char*>(&n), 4)) return false;

    if (n != static_cast<u32>(params_.size())) return false;
    for (u32 i = 0; i < n; ++i) {
        u32 len = 0;
        if (!f.read(reinterpret_cast<char*>(&len), 4) || len == 0 || len > 512) return false;
        std::string name(len, '\0');
        if (!f.read(name.data(), static_cast<std::streamsize>(len))) return false;
        u64 ne = 0;
        if (!f.read(reinterpret_cast<char*>(&ne), 8)) return false;
        Parameter* p = find_parameter(name);
        if (!p || static_cast<u64>(p->numel()) != ne) return false;
        if (p->shape.empty()) return false;

        u32 nd = 0;
        if (!f.read(reinterpret_cast<char*>(&nd), 4)) return false;
        if (nd != static_cast<u32>(p->shape.size())) return false;
        for (size_t k = 0; k < p->shape.size(); ++k) {
            i64 dd = 0;
            if (!f.read(reinterpret_cast<char*>(&dd), 8)) return false;
            if (dd != p->shape[k]) return false;
        }
        Tensor cpu(p->shape, DType::F32, Device::CPU);
        if (cpu.nbytes() == 0) return false;
        if (!f.read(reinterpret_cast<char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()))) return false;
        p->w.copy_from(cpu);
    }
    if (fp16_weight_cache_) enable_fp16_weight_cache(true);
    return true;
}

}
