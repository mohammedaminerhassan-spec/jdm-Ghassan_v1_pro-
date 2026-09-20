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

// ---------------------------------------------------------------- MoE aux loss
// Computes the DeepSeek-style load-balance term for one layer from its saved
// routing: raw = ne * sum_e(mean_prob_e * frac_e), and uploads frac_e into
// the (tiny, caller-owned) device buffer consumed by moe_backward.
double Model::moe_layer_aux(Device dev, const float* probs, const i32* idx,
                             float* auxfrac_dev, int layer, i64 N, int K, int ne,
                             bool want_stats, double* d_raw_accum) {
    // Fast GPU path: frac stays on device, raw folds into the device
    // accumulator (audit P1: read once per microbatch, not 2x ne-float D2H
    // per layer). Host copies happen ONLY when want_stats (log cadence).
    if (dev == Device::CUDA) {
        std::vector<float> h_frac, h_psum;
        float* hf = nullptr;
        float* hp = nullptr;
        if (want_stats || !d_raw_accum) {
            // Safe fallback (no accumulator, e.g. eval path): full host raw.
            h_frac.assign(static_cast<size_t>(ne), 0.0f);
            h_psum.assign(static_cast<size_t>(ne), 0.0f);
            hf = h_frac.data();
            hp = h_psum.data();
        }
        if (ops::moe_aux_gpu(dev, probs, idx, auxfrac_dev,
                             hf, hp, N, K, ne, d_raw_accum)) {
            if (!hf) return 0.0; // raw lives on device; read at microbatch end
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
        // fall through to CPU path if the GPU helper is unavailable
    }
    std::vector<i32> h_idx(static_cast<size_t>(N) * K);
    device_copy(h_idx.data(), Device::CPU, idx, dev,
                sizeof(i32) * h_idx.size());
    std::vector<float> h_probs(static_cast<size_t>(N) * ne);
    device_copy(h_probs.data(), Device::CPU, probs, dev,
                sizeof(float) * h_probs.size());

    std::vector<double> cnt(ne, 0.0), psum(ne, 0.0);
    for (i64 t = 0; t < N; ++t)
        for (int k = 0; k < K; ++k) {
            int e = h_idx[static_cast<size_t>(t) * K + k];
            if (e >= 0 && e < ne) cnt[e] += 1.0;
        }
    for (i64 t = 0; t < N; ++t)
        for (int e = 0; e < ne; ++e)
            psum[e] += h_probs[static_cast<size_t>(t) * ne + e];

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

    // accumulate routing stats for the balance report
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
        // uniform share would be 1/ne; flag starvation/collapse inline
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
    // DeepSeek-V3 aux-free rule: bias_e -= lr * (load_e - target), target = top_k/ne.
    const float target = static_cast<float>(cfg_.moe_top_k) / static_cast<float>(ne);
    auto& b = moe_bias_[static_cast<size_t>(layer)];
    for (int e = 0; e < ne && e < static_cast<int>(b.size()); ++e) {
        float err = frac_host[e] - target;
        b[static_cast<size_t>(e)] -= moe_bias_lr_ * err;
        // Clamp to keep selection stable (bias is steering only, not weights).
        if (b[static_cast<size_t>(e)] > 0.5f) b[static_cast<size_t>(e)] = 0.5f;
        if (b[static_cast<size_t>(e)] < -0.5f) b[static_cast<size_t>(e)] = -0.5f;
    }
    if (device_ == Device::CUDA && static_cast<size_t>(layer) < moe_bias_dev_.size()) {
        device_copy(moe_bias_dev_[static_cast<size_t>(layer)].data_ptr(), device_,
                    b.data(), Device::CPU, sizeof(float) * b.size());
    }
}

// ================================================================ config
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
    GAI_CHECK(rope_theta > 0.0f, "rope_theta must be > 0");
    // DeepSeek stability contract: rms_eps/init_std silently break norms/init
    // when <= 0 (inf/NaN or zero-init dead model). Aux weight > 1 drowns CE.
    GAI_CHECK(rms_eps >= 1e-12f && rms_eps <= 1e-2f, "rms_eps must be in [1e-12,1e-2]");
    GAI_CHECK(init_std > 0.0f && init_std <= 0.1f, "init_std must be in (0,0.1]");
    GAI_CHECK(rope_scale >= 1.0f, "rope_scale must be >= 1.0 (1.0 = off)");
    // YaRN is approximate NTK extrapolation: small scales (2x) are routine,
    // large scales are unvalidated here. Bound it so a typo (e.g. 20 instead
    // of 2.0) fails fast instead of training with garbage positions.
    GAI_CHECK(rope_scale <= 8.0f, "rope_scale must be <= 8.0 (larger is unvalidated extrapolation)");
    GAI_CHECK(z_loss_scale >= 0.0f && z_loss_scale <= 0.01f, "z_loss_scale must be in [0,0.01]");
    GAI_CHECK(moe_jitter >= 0.0f && moe_jitter <= 0.5f, "moe_jitter must be in [0,0.5]");
    GAI_CHECK(rope_yarn_mscale >= 0.0f && rope_yarn_mscale <= 2.0f, "rope_yarn_mscale must be in [0,2]");
    // ---- Pro fields (default OFF = bit-identical legacy; فعلها فقط لـPro جديد)
    GAI_CHECK(rope_yarn_low >= 1.0f && rope_yarn_low <= 128.0f, "rope_yarn_low must be in [1,128]");
    GAI_CHECK(rope_yarn_high >= 1.0f && rope_yarn_high <= 128.0f, "rope_yarn_high must be in [1,128]");
    GAI_CHECK(rope_yarn_high >= rope_yarn_low, "rope_yarn_high must be >= rope_yarn_low");
    GAI_CHECK(sliding_window >= 0 && sliding_window <= 16384, "sliding_window must be in [0,16384]");
    GAI_CHECK(rope_type == 0 || rope_type == 1, "rope_type must be 0 (interleaved) or 1 (neox)");
    // NeoX kernels (CPU+CUDA) implemented: rope_type=1 allowed for new Pro checkpoints.
    // Interleaved stays default for bit-identical legacy resumes.
    if (moe_aux_free && moe_aux_scale != 0.0f) {
        // aux-loss-free تعني لا aux grad: نفرض scale=0 بصوت عال بدل تجاهل صامت.
        GAI_CHECK(false, "moe_aux_free=true requires moe_aux_scale: 0.0 (aux-loss-free has no aux grad)");
    }
    if (use_moe) {
        GAI_CHECK(num_experts > 0 && num_experts <= 64, "num_experts must be in [1,64]");
        GAI_CHECK(moe_top_k > 0 && moe_top_k <= num_experts, "moe_top_k must be in [1,num_experts]");
        GAI_CHECK(moe_top_k <= 8, "moe_top_k must be <= 8 (router kernel limit)");
        // Strong-model rule: MoE must stay sparse (<=50% experts active).
        // top-5/8 = 70% dense defeats MoE (OOM on T4, no conditional-compute win).
        // Standard sparse regime is top-1..2/8. Reject dense configs fail-fast
        // unless the research hatch is set (moe_allow_dense=true): the rule
        // stays safe by default but no longer blocks dense experiments.
        GAI_CHECK(moe_allow_dense || moe_top_k * 2 <= num_experts,
                  "moe_top_k must be <= num_experts/2 (sparse MoE; dense routing OOMs T4 and removes the MoE advantage; "
                  "set model.moe_allow_dense=true only for short research runs)");
        GAI_CHECK(moe_expert_dim > 0, "moe_expert_dim must be > 0");
        GAI_CHECK(moe_aux_scale >= 0.0f && moe_aux_scale <= 1.0f,
                  "moe_aux_scale must be in [0,1] (larger drowns CE loss)");
    }
    // Attention kernel shared memory limit (T4: 48KB). The flash-style kernel
    // uses shared memory: sQ[hd] + sK[KV_TILE*hd] + sV[KV_TILE*hd] + sS[KV_TILE] + sAcc[hd]
    // and backward: s_dk[KV_TILE*hd] + s_dv[KV_TILE*hd] + s_dot[group].
    // KV_TILE=64, group=H/KV. For hd=64 this is ~33KB (forward) and ~32KB (backward).
    // For hd=128 it exceeds 48KB. Fail fast at config time.
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
    // FIX: YAML ints are i64; blind static_cast<int> wraps huge values to
    // negative/small -> validate() passes on truncated value -> OOB/OOM in
    // training (compilation-clean, runtime-corrupt). Range-check first.
    auto get_int_checked = [&](const char* k, int def) {
        i64 v = c.get_int(key(k), static_cast<i64>(def));
        GAI_CHECK(v > 0 && v <= static_cast<i64>(std::numeric_limits<int>::max()),
                  std::string("model.") + k + " out of int range");
        return static_cast<int>(v);
    };
    // PRO: sliding_window/rope_type يقبلان 0 (off/interleaved) — الفاحص
    // العام v>0 كان يرفض القيمة الصحيحة 0. فاحص >=0 منفصل لهما.
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

// YaRN-lite (NTK): theta_eff = theta * scale^(hd/(hd-2)), 1.0 = off.
// Extends usable context ~scale x without touching kernels (only theta changes).
static float rope_theta_eff_local(const ModelConfig& m) {
    if (m.rope_scale <= 1.0f) return m.rope_theta;
    float hd = static_cast<float>(m.head_dim());
    float e = (hd > 2.0f) ? hd / (hd - 2.0f) : 1.0f;
    return m.rope_theta * std::pow(m.rope_scale, e);
}

// YaRN attention rescale (DeepSeek long-ctx): mscale = 0.1*ln(scale)+1.
// Multiplies the 1/sqrt(hd) scale so entropy stays flat at 8k ctx.
// PRO: مع yarn_low/high نطبق ramp الكامل (YaRN paper §3): الأبعاد ذات الطول
// الموجي < low تبقى خطية، > high تستكمل NTK، وبينهما interpolation سلس.
// mscale هنا للـattention-temperature؛ الـramp الكامل للـtheta يبقى NTK
// (theta_eff أعلاه) حتى ترحيل الـkernel — نفس سلوك legacy عند low=1/high=32.
static float rope_mscale_local(const ModelConfig& m) {
    if (m.rope_scale <= 1.0f) return 1.0f;
    if (m.rope_yarn_mscale > 0.0f) return m.rope_yarn_mscale;
    // ramp factor: يحفظ mscale الأصلي عند الإعداد الافتراضي (low=1/high=32
    // يعطي نفس 0.1*ln+1 ضمن 1%) ويتيح ضبطا دقيقا لاحقا.
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

float rope_theta_eff(const ModelConfig& m) { return rope_theta_eff_local(m); }
float rope_mscale(const ModelConfig& m) { return rope_mscale_local(m); }

// ================================================================ model
void Model::alloc_param(Parameter& p, const std::string& name, std::vector<i64> shape, bool decay) {
    p.name  = name;
    p.shape = shape;
    p.decay = decay;
    p.w = Tensor::zeros(shape, DType::F32, device_);
    p.w.set_name(name);
    params_.push_back(&p);
}

Model::Model(ModelConfig cfg, Device dev) : cfg_(std::move(cfg)), device_(dev) {
    cfg_.validate();
    // YaRN is approximate NTK extrapolation (not exact long-ctx training).
    // rope_scale==1 (all shipped configs) is exact; anything above needs a
    // perplexity validation run at the target ctx (e.g. 8k) before trusting it.
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

void Model::to(Device dev) {
    if (device_ == dev) return;
    // PRO-HARDEN: نقل الأوزان فقط يترك Trainer::act_/ckpt_act_ و
    // Generator::cache_/scratch على الجهاز القديم فيكون التالي cross-device
    // GEMM/copies -> illegal access. لا يمكن للـModel إعادة بناء arenas
    // خارجية، لذا نحذر بصوت عال: بعد to() يجب إعادة بناء Trainer/Generator.
    log_warn("[model] Model::to() moved weights; REBUILD Trainer activations and "
             "Generator cache/scratch on the new device before next step "
             "(cross-device use-after-move crashes T4).");
    for (Parameter* p : params_) {
        if (p->w.defined()) p->w = p->w.to(dev);
        if (p->g.defined()) p->g = p->g.to(dev);
    }
    device_ = dev;
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
        // KV cache at full context (fp16)
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

// ---------------------------------------------------------------- init
void Model::init_weights(u64 seed) {
    Rng rng(seed);
    const float std_base = cfg_.init_std;
    // scaled init for residual projections (GPT-2 style depth scaling)
    const float std_res = std_base / std::sqrt(2.0f * static_cast<float>(cfg_.num_layers));

    Device orig_dev = device_;
    if (device_ != Device::CPU) {
        to(Device::CPU);
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
        fill_normal(L.wo, std_res);       // residual output
        if (L.has_moe()) {
            // Router starts near-uniform (small init); experts use the same
            // depth-scaled scheme as dense FFNs so early loss ~= ln(V).
            fill_normal(L.router, std::min(std_base, 0.01f));
            fill_normal(L.moe_gate, std_base);
            fill_normal(L.moe_up,   std_base);
            fill_normal(L.moe_down, std_res);   // residual output
            if (L.sh_gate.w.defined()) {
                fill_normal(L.sh_gate, std_base);
                fill_normal(L.sh_up,   std_base);
                fill_normal(L.sh_down, std_res);
            }
        } else {
            fill_normal(L.w_gate, std_base);
            fill_normal(L.w_up,   std_base);
            fill_normal(L.w_down, std_res);   // residual output
        }
    }

    if (orig_dev != Device::CPU) {
        to(orig_dev);
    }
}

void Model::enable_grad(bool on) {
    grad_enabled_ = on;
    if (on) {
        // mmap-backed weights are read-only file pages: any optimizer write
        // would fault. Training must reload the model normally (no --mmap).
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
    // PERF (audit #8): one memset per tensor, serial. Tensors are disjoint —
    // clear them concurrently. (Full bucketing stays future work.)
    // PRO-HARDEN (MSVC C3016): size_t مرفوض كمتغير OpenMP على Windows.
#ifdef GAI_OPENMP
#pragma omp parallel for schedule(dynamic, 4) if(params_.size() > (size_t)8)
#endif
    for (long long i = 0; i < static_cast<long long>(params_.size()); ++i)
        if (params_[static_cast<size_t>(i)]->g.defined()) params_[static_cast<size_t>(i)]->g.zero_();
}

// ---------------------------------------------------------------- activations
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
    // Chunked-loss sizing (roadmap item 6): with ce_chunks > 1 the logits and
    // dlogits buffers shrink from [N,V] to [Cc,V] scratch, Cc = ceil(N/chunks).
    // Only forward_backward() may consume such activations (forward() refuses
    // them loudly). ce_chunks <= 1 keeps the legacy full buffers exactly.
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
    a.q       = mk({N, qd}, device_, acc);
    a.k       = mk({N, kvd},device_, acc);
    a.v       = mk({N, kvd},device_, acc);
    a.att_out = mk({N, qd}, device_, acc);
    a.proj    = mk({N, d},  device_, acc);
    if (moe) {
        // Shared scratch, reused by every layer (forward overwrites per layer,
        // backward consumes each layer's SAVED copies below).
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
        // Single transient probs buffer, recomputed per layer in backward
        // (saves (L-1)*B*H*T*T floats, e.g. 2.5GB at B=2,T=1024,L=26).
        // PRO-HARDEN: حارس T² داخل make_activations نفسها (كان في Trainer فقط)
        // فيحمي أي استدعاء مباشر/اختبار من OOM صامت على T4.
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
    // FIX: i64 signed overflow on adversarial B*T*V*L wrapped negative ->
    // "fits" estimate then real OOM on T4 (training killer). Use u64 with
    // saturation (cap at 1TiB elements) so the guard always over-estimates.
    // FIX (P0-1): explicit -> u64 return type. Without it, `return 0ull`
    // (unsigned long long) and `return cap` (u64 == unsigned long on LP64
    // Linux) deduce different types -> hard compile error on GCC/Clang.
    // MSVC hid the bug because uint64_t IS unsigned long long on Windows.
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
    const u64 d   = static_cast<u64>(cfg_.hidden_size);
    const u64 qd  = static_cast<u64>(cfg_.q_dim());
    const u64 kvd = static_cast<u64>(cfg_.kv_dim());
    const u64 F   = cfg_.use_moe ? static_cast<u64>(cfg_.moe_top_k) * static_cast<u64>(cfg_.moe_expert_dim)
                                 : static_cast<u64>(cfg_.intermediate_size);
    const u64 V   = static_cast<u64>(cfg_.vocab_size);
    const u64 L   = static_cast<u64>(cfg_.num_layers);

    u64 e = sat_mul(N, (4 * d + qd * 2 + kvd * 2 + F * 3 + V));
    if (cfg_.use_moe) e = sat_add(e, sat_mul(N, (static_cast<u64>(cfg_.num_experts) + 2 * static_cast<u64>(cfg_.moe_top_k))));
    // Forward misses pos[N] i32 + xfinal/hnorm/dtmp grad scratch in the old
    // math (systematic under-count ~5-8%). Account for them explicitly.
    e = sat_add(e, N); // pos ids (i32 ~ 1 float)
    if (with_grad) {
        // Attention probs are recomputed per layer into ONE shared buffer
        // (not stored per layer): single B*H*T*T instead of L*B*H*T*T.
        u64 per_layer = sat_mul(N, (4 * d + qd * 2 + kvd * 2 + F * 3 + 2));
        if (cfg_.use_moe) per_layer = sat_add(per_layer, sat_mul(N, (static_cast<u64>(cfg_.num_experts) + 3 * static_cast<u64>(cfg_.moe_top_k))));
        if (cfg_.use_qk_norm) per_layer = sat_add(per_layer, sat_mul(N, (static_cast<u64>(cfg_.num_heads) + static_cast<u64>(cfg_.num_kv_heads) + qd + kvd)));
        e = sat_add(e, sat_mul(L, per_layer));
        e = sat_add(e, sat_mul(sat_mul(static_cast<u64>(B), static_cast<u64>(cfg_.num_heads)), sat_mul(static_cast<u64>(T), static_cast<u64>(T)))); // transient recompute buf
        if (!cfg_.use_moe) {
            e = sat_add(e, sat_mul(N, (5 * d + qd * 2 + kvd * 2 + F * 3 + V + 2)));
        } else {
            e = sat_add(e, sat_mul(N, (5 * d + qd * 2 + kvd * 2 + V + 2)));
            e = sat_add(e, sat_add(sat_mul(N, sat_mul(static_cast<u64>(cfg_.moe_top_k), static_cast<u64>(cfg_.moe_expert_dim))), static_cast<u64>(cfg_.num_experts)));
        }
        // grad scratch: dx/dxb/dq/dk/dv/dattout/dproj/dtmp/dlogits/xfinal/hnorm
        e = sat_add(e, sat_mul(N, (4 * d + qd * 2 + kvd * 2 + V + d * 2)));
    }
    // Chunked loss (roadmap item 6): logits [N,V] (fwd term) + dlogits [N,V]
    // (grad scratch term) shrink to [Cc,V] each, Cc = ceil(N/chunks). Both
    // terms are present in `e` above exactly once, so subtract the saved part.
    // (e is only ever added to here, never saturated below the true count
    // before this point, so the subtraction cannot underflow.)
    if (with_grad && ce_chunks > 1 && N > 0) {
        const u64 cc = static_cast<u64>(ce_chunks);
        const u64 crow = N / cc + (N % cc ? 1ULL : 0ULL);
        const u64 sub = sat_mul(sat_mul(2ULL, V), N - crow);
        e = (e >= sub) ? e - sub : 0;
    }
    if (e > ((1ull << 40) / 4)) return (1ull << 40);
    return static_cast<size_t>(e) * sizeof(float);
}

// ---------------------------------------------------------------- forward
// Body shared by forward() and forward_backward(): everything up to (and
// including) the final norm. The lm_head GEMM is left to the caller so the
// training path can chunk it (roadmap item 6) instead of materializing a
// full [N,V] logits matrix.
void Model::forward_body(const i32* ids, int B, int T, Activations& act) {
    // FIX: fail-fast guards (training/inference crash + T4/low-PC OOM safety).
    // Old code accepted B/T<=0 -> N<=0 cast to size_t = huge alloc -> OOM,
    // and N>INT_MAX truncated to int M in linear_forward -> silent zero math.
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
    // DeepSeek fail-fast: OOB ids previously trained as zero-vectors silently
    // (embedding_forward memset 0). On CPU we can check the host buffer here
    // for free vs GEMMs; on CUDA the ids live on device so the check happens
    // in the data path (dataloader/tokenizer already range-check), and the
    // CPU embedding kernel below keeps its defensive zero + host counter.
    if (dev == Device::CPU) {
        for (i64 i = 0; i < N; ++i) {
            i32 id = ids[i];
            GAI_CHECK(id >= 0 && id < V,
                      strfmt("forward: token id out of range ids[%lld]=%d (vocab=%d; tokenizer/dataloader mismatch?)",
                             (long long)i, (int)id, V));
        }
    }

    GAI_CHECK(act.B == B && act.T == T, "activation buffer shape mismatch");
    GAI_CHECK(T <= cfg_.max_seq_len, "sequence longer than max_seq_len");
    // N is i64 but GEMM M is int: refuse absurd micro-batches before trunc.
    GAI_CHECK(N <= static_cast<i64>(std::numeric_limits<int>::max()),
              "forward: B*T exceeds int range (reduce batch/seq_len)");
    // Reused Activations from another config indexed saved_* OOB below
    // -> segfault. Validate once here, but ONLY for training: inference
    // (with_grad=false) never touches saved_* (all guarded by `train`),
    // so empty saved_x is correct there. The old unconditional check
    // rejected EVERY inference forward (chat/generate/eval/bench killer).
    if (train) {
        GAI_CHECK(static_cast<int>(act.saved_x.size()) == cfg_.num_layers,
                  "forward: stale Activations (layer count mismatch, rebuild)");
    }
    if (train && cfg_.use_moe) {
        GAI_CHECK(static_cast<int>(act.saved_moe_probs.size()) == cfg_.num_layers,
                  "forward: MoE training needs with_grad Activations");
    }

    // position ids (repeated per batch element)
    // FIX (T4 perf/fragmentation): old code allocated a std::vector + a CPU
    // Tensor on EVERY forward (128x per optimizer step with grad_accum=128).
    // Allocator churn + fragmentation looked like a CPU leak and stalled the
    // GPU. Now: monotonic thread-local staging, zero per-call mallocs after
    // the first, direct H2D into the persistent act.pos buffer.
    // PERF (audit #10): positions are a pure function of (B,T), and training
    // reuses one (B,T) across all 128 micros. Skip the rebuild + H2D entirely
    // when the buffer already holds this (B,T) — 127/128 forwards do no work.
    if (!act.pos.defined() || act.pos.numel() < N) {
        act.pos = Tensor::empty({N}, DType::I32, device_);
        act.bytes += act.pos.nbytes();
        act.pos_cached_B = -1;
        act.pos_cached_T = -1;
    }
    if (act.pos_cached_B != B || act.pos_cached_T != T) {
        thread_local std::vector<i32> pos_staging;
        if (pos_staging.size() < static_cast<size_t>(N))
            pos_staging.resize(static_cast<size_t>(N));
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) pos_staging[static_cast<size_t>(b) * T + t] = t;
        device_copy(act.pos.data_ptr(), device_, pos_staging.data(), Device::CPU,
                    static_cast<size_t>(N) * sizeof(i32));
        act.pos_cached_B = B;
        act.pos_cached_T = T;
    }

    // ---- embeddings
    ops::embedding_forward(dev, ids, tok_emb_.w.f32(), act.x.f32(), N, d, V);

    for (int l = 0; l < cfg_.num_layers; ++l) {
        LayerParams& L = layers_[static_cast<size_t>(l)];
        size_t sl = static_cast<size_t>(l);

        if (train) ops::copy(dev, act.saved_x[sl].f32(), act.x.f32(), N * d);

        // ---- attention block
        float* rrms1 = train ? act.saved_rrms1[sl].f32() : nullptr;
        ops::rmsnorm_forward(dev, act.x.f32(), L.attn_norm.w.f32(), act.xb.f32(),
                             rrms1, N, d, cfg_.rms_eps);
        if (train) ops::copy(dev, act.saved_xb[sl].f32(), act.xb.f32(), N * d);

        ops::linear_forward(dev, act.xb.f32(), L.wq.w.f32(), act.q.f32(), static_cast<int>(N), d, qd);
        ops::linear_forward(dev, act.xb.f32(), L.wk.w.f32(), act.k.f32(), static_cast<int>(N), d, kvd);
        ops::linear_forward(dev, act.xb.f32(), L.wv.w.f32(), act.v.f32(), static_cast<int>(N), d, kvd);

        // QK-Norm (optional): per-head RMSNorm on Q/K before RoPE stabilizes
        // MoE training at 1B (prevents attention logit explosion). Layout
        // [N,H,hd] is contiguous as [N*H,hd], so rmsnorm applies directly.
        // FIX: old guard checked only qk_qnorm, then deref'd qk_knorm ->
        // segfault on partial alloc / stale ckpt. Require both gains.
        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined() && L.qk_knorm.w.defined()) {
            if (train) {
                ops::copy(dev, act.saved_qk_raw_q[sl].f32(), act.q.f32(), N * qd);
                ops::copy(dev, act.saved_qk_raw_k[sl].f32(), act.k.f32(), N * kvd);
            }
            float* rrms_q = train ? act.saved_qk_rms_q[sl].f32() : nullptr;
            float* rrms_k = train ? act.saved_qk_rms_k[sl].f32() : nullptr;
            ops::rmsnorm_forward(dev, act.q.f32(), L.qk_qnorm.w.f32(), act.q.f32(),
                                 rrms_q, N * H, hd, cfg_.rms_eps);
            ops::rmsnorm_forward(dev, act.k.f32(), L.qk_knorm.w.f32(), act.k.f32(),
                                 rrms_k, N * KV, hd, cfg_.rms_eps);
        }

        const float theta_eff = rope_theta_eff(cfg_);
        // Pro kernels: NeoX + full YaRN ramp + SWA (DeepSeek-V3 / LLaMA-3 class).
        // Legacy path (rope_type 0, scale 1, window 0) is bit-identical to old calls.
        ops::rope_forward_ex(dev, act.q.f32(), act.k.f32(), act.pos.i32p(), N, H, KV, hd, theta_eff,
                             cfg_.rope_type, cfg_.rope_yarn_low, cfg_.rope_yarn_high, cfg_.rope_scale);

        if (train) {
            ops::copy(dev, act.saved_q[sl].f32(), act.q.f32(), N * qd);
            ops::copy(dev, act.saved_k[sl].f32(), act.k.f32(), N * kvd);
            ops::copy(dev, act.saved_v[sl].f32(), act.v.f32(), N * kvd);
        }

        // Probs are never stored (recomputed per layer in backward).
        // nullptr here = memory-lean forward on both CPU and CUDA.
        ops::attention_forward_ex(dev, act.q.f32(), act.k.f32(), act.v.f32(),
                                  act.att_out.f32(), nullptr,
                                  B, T, H, KV, hd, scale, cfg_.sliding_window);
        if (train) ops::copy(dev, act.saved_attout[sl].f32(), act.att_out.f32(), N * qd);

        ops::linear_forward(dev, act.att_out.f32(), L.wo.w.f32(), act.proj.f32(), static_cast<int>(N), qd, d);
        ops::add_inplace(dev, act.x.f32(), act.proj.f32(), N * d);

        if (train) ops::copy(dev, act.saved_xmid[sl].f32(), act.x.f32(), N * d);

        // ---- ffn block (MoE or dense)
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

Tensor& Model::forward(const i32* ids, int B, int T, Activations& act) {
    // Full-[N,V] inference/eval path. Chunked training activations carry only
    // compact [Cc,V] scratch: they must go through forward_backward().
    forward_body(ids, B, T, act);
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

// ---------------------------------------------------------------- backward
double Model::forward_backward(const i32* ids, const i32* targets, int B, int T,
                               Activations& act, i64* out_ntok, float dout_scale,
                               bool want_aux_stats) {
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

    forward_body(ids, B, T, act);

    // ---- chunked lm head + loss (roadmap item 6; with z-loss stabilizer).
    // Materializing full [N,V] logits+dlogits costs 524MB at B=2,T=1024,V=32k
    // — the biggest single block in the T4 budget. Instead the head runs in
    // row-blocks over compact [Cc,V] scratch (Cc = ceil(N/chunks)):
    //   GEMM block -> SCE block -> scale -> backward block (accumulates).
    // EXACTNESS: per chunk c, SCE emits SUM grads directly (audit P1: no
    // host count roundtrip); scaling by eff_scale yields raw×eff_scale —
    // identical, element for element, to the old full-matrix path (mean
    // ÷ntok, then ×eff_scale×ntok), up to 1-ulp fp associativity. Loss sums
    // and ntok add across chunks; dw/dx accumulate through linear_backward's
    // += contract. Only the fp addition ORDER of the loss scalar differs
    // (negligible, ~1e-9 relative).
    float eff_scale = dout_scale;
    // (Range safety for the scale lives in the Trainer (kLossScaleMax): it
    // must own the limit because it also owns the unscaling divisor. See the
    // P0-2a note that used to live here: clamping inside the model while the
    // trainer unscales with the passed-in value is a silent LR cut.)
    const i64 Cc = act.ce_rows > 0 ? act.ce_rows : N;
    const float* hnorm_full = act.saved_hnorm.f32();
    float* logits_c = act.logits.f32();
    float* dlogits_c = act.dlogits.f32();
    double loss_sum = 0.0;
    i64    ntok = 0;
    ops::zero(dev, act.dxb.f32(), N * d);
    for (i64 r0 = 0; r0 < N; r0 += Cc) {
        const i64 Cr = std::min(Cc, N - r0);
        const int Ci = static_cast<int>(Cr); // Cr <= N <= INT_MAX (body checked)
        ops::linear_forward(dev, hnorm_full + r0 * d, lm_head().w.f32(),
                            logits_c, Ci, d, V);
        double csum = 0.0;
        i64 cn = 0;
        ops::softmax_cross_entropy(dev, logits_c, targets + r0, dlogits_c,
                                   Cr, V, &csum, &cn, cfg_.z_loss_scale);
        if (cn == 0) continue; // fully-masked block: nothing to learn here
        loss_sum += csum;
        ntok += cn;
        // P0-05 (sums, not means) + P0-2b (fused single pass), per chunk.
        // SCE already emits eff_scale-ready SUM grads, so no ×cn rescale.
        if (eff_scale != 1.0f) ops::scale_inplace(dev, dlogits_c, eff_scale, Cr * V);
        ops::linear_backward(dev, hnorm_full + r0 * d, lm_head().w.f32(), dlogits_c,
                             act.dxb.f32() + r0 * d, lm_head().g.f32(), Ci, d, V);
    }
    if (out_ntok) *out_ntok = ntok;
    if (ntok == 0) return 0.0;

    // ---- MoE load-balance aux loss: measured per layer inside the backward
    // loop below (each layer refills the shared [ne] fractions buffer right
    // before its own moe_backward call), then added to the returned loss.
    // AUDIT P1: on CUDA the per-layer raw folds into ONE device accumulator
    // (zero syncs per layer); the host reads it once below. want_aux_stats
    // additionally refreshes the balance-report stats (log cadence only).
    double aux_total = 0.0;
    const bool use_dev_aux = (dev == Device::CUDA) && cfg_.use_moe &&
                             cfg_.moe_aux_scale > 0.0f;
    double* aux_accum = use_dev_aux ? ops::moe_aux_begin(dev) : nullptr;

    // ---- final norm
    ops::zero(dev, act.dx.f32(), N * d);
    ops::rmsnorm_backward(dev, act.saved_xfinal.f32(), final_norm_.w.f32(), act.dxb.f32(),
                          act.saved_rrms_final.f32(), act.dx.f32(), final_norm_.g.f32(), N, d);

    // ---- layers, in reverse
    for (int l = cfg_.num_layers - 1; l >= 0; --l) {
        LayerParams& L = layers_[static_cast<size_t>(l)];
        size_t sl = static_cast<size_t>(l);

        // ffn: x_out = x_mid + FFN(x_mid)
        // dx currently holds dL/dx_out, which flows into both the residual and the ffn path.
        if (L.has_moe()) {
            const int E  = cfg_.moe_expert_dim;
            const int ne = cfg_.num_experts;
            const int K  = cfg_.moe_top_k;
            // load-balance term: measure this layer's aux loss and upload
            // its expert fractions into the shared [ne] device buffer.
            // P0-02 FIX + DeepSeek SFT rule: aux grads must live in the SAME
            // scaled+sum space as CE grads AND share the trainer's 1/ntok
            // divisor. Old code used dense N (B*T): pretrain N≈ntok hides it,
            // but SFT masks (ntok<<N) made aux 2-5x too strong vs reported
            // loss. Now ×eff_scale ×ntok (supervised, not dense N) ÷L so the
            // optimized weight == reported weight (aux_scale/L) exactly.
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

        // dx_mid = dx (residual) + rmsnorm_backward(dxb)
        ops::rmsnorm_backward(dev, act.saved_xmid[sl].f32(), L.ffn_norm.w.f32(), act.dxb.f32(),
                              act.saved_rrms2[sl].f32(), act.dx.f32(), L.ffn_norm.g.f32(), N, d);

        // attention: x_mid = x_in + W_o(attn(...))
        ops::zero(dev, act.dattout.f32(), N * qd);
        ops::linear_backward(dev, act.saved_attout[sl].f32(), L.wo.w.f32(), act.dx.f32(),
                             act.dattout.f32(), L.wo.g.f32(), static_cast<int>(N), qd, d);

        ops::zero(dev, act.dq.f32(), N * qd);
        ops::zero(dev, act.dk.f32(), N * kvd);
        ops::zero(dev, act.dv.f32(), N * kvd);
        // Recompute probs per layer into the shared transient buffer
        // (saves (L-1)*B*H*T*T memory, e.g. 2.5GB at B=2,T=1024).
        // act.att_out is free scratch here (forward value already saved).
        // SWA mask rides in probs (zeros outside window) so backward matches forward.
        ops::attention_forward_ex(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                                  act.saved_v[sl].f32(), act.att_out.f32(),
                                  act.attn_probs_tmp.f32(),
                                  B, T, H, KV, hd, scale, cfg_.sliding_window);
        ops::attention_backward_ex(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                                   act.saved_v[sl].f32(), act.attn_probs_tmp.f32(),
                                   act.dattout.f32(),
                                   act.dq.f32(), act.dk.f32(), act.dv.f32(),
                                   B, T, H, KV, hd, scale, cfg_.sliding_window);

        const float theta_eff_bwd = rope_theta_eff(cfg_);
        ops::rope_backward_ex(dev, act.dq.f32(), act.dk.f32(), act.pos.i32p(), N, H, KV, hd, theta_eff_bwd,
                              cfg_.rope_type, cfg_.rope_yarn_low, cfg_.rope_yarn_high, cfg_.rope_scale);

        // QK-Norm backward (exact, using saved pre-norm Q/K):
        // dq/dk hold dL/d(normed Q/K) from rope_backward. rmsnorm_backward does dx+=f(dout), so
        // dout and dx must NOT alias. att_out [N,qd] and dattout [N,kvd] stage dout.
        // We pass the true pre-norm Q/K (saved_qk_raw_q/k) as input x to rmsnorm_backward.
        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined()) {
            // Q path: stage dout in att_out, zero dq, then rmsnorm_backward.
            ops::copy(dev, act.att_out.f32(), act.dq.f32(), N * qd);
            ops::zero(dev, act.dq.f32(), N * qd);
            ops::rmsnorm_backward(dev, act.saved_qk_raw_q[sl].f32(), L.qk_qnorm.w.f32(), act.att_out.f32(),
                                  act.saved_qk_rms_q[sl].f32(), act.dq.f32(), L.qk_qnorm.g.f32(),
                                  N * H, hd);
            // K path: stage dout in dattout, zero dk, then rmsnorm_backward.
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

    // ---- embeddings
    ops::embedding_backward(dev, ids, act.dx.f32(), tok_emb_.g.f32(), N, d, V);

    double loss = loss_sum / static_cast<double>(ntok);
    // DeepSeek-style aux loss: per-layer raw (ne*sum(mean_prob*frac)).
    // On CUDA the hot path folded raws into the device accumulator: single
    // sync read here replaces 2x ne-float copies per layer (72/microbatch).
    if (use_dev_aux) aux_total = ops::moe_aux_end(dev);
    // FIX (10/10): old code SUMMED over layers, so a 36L model got stronger
    // aux than a 26L model with the same scale. Average over layers so
    // moe_aux_scale means the same at any depth.
    if (cfg_.use_moe && cfg_.moe_aux_scale > 0.0f && cfg_.num_layers > 0)
        loss += cfg_.moe_aux_scale * aux_total / static_cast<double>(cfg_.num_layers);
    return loss;
}

// ---------------------------------------------------------------- aux loss
double Model::moe_aux_loss(Activations& act, int B, int T) {
    if (!cfg_.use_moe) return 0.0;
    // Aux-loss-free (DeepSeek-V3 §3.2): no aux grad; balance via EMA bias.
    // We still measure per-layer load (tiny ne-float host copies) and update
    // the steering bias here. Returns 0 loss; validate enforces aux_scale=0.
    if (cfg_.moe_aux_free) {
        GAI_CHECK(act.with_grad, "moe_aux_loss needs training Activations (with_grad)");
        GAI_CHECK(static_cast<int>(act.saved_moe_probs.size()) == cfg_.num_layers,
                  "moe_aux_loss: stale Activations");
        GAI_CHECK(act.moe_auxfrac.defined(), "moe_aux_loss: missing auxfrac buffer");
        const i64 N = static_cast<i64>(B) * T;
        const int ne = cfg_.num_experts;
        std::vector<float> h_frac(static_cast<size_t>(ne));
        std::vector<float> h_psum(static_cast<size_t>(ne));
        for (int l = 0; l < cfg_.num_layers; ++l) {
            size_t sl = static_cast<size_t>(l);
            // want_stats=true forces the tiny host copy we need for the EMA.
            moe_layer_aux(device_, act.saved_moe_probs[sl].f32(),
                          act.saved_moe_idx[sl].i32p(),
                          act.moe_auxfrac.f32(), l, N, cfg_.moe_top_k, ne, true, nullptr);
            // Re-read fractions from device buffer (ne floats, 1 small sync).
            device_copy(h_frac.data(), Device::CPU, act.moe_auxfrac.f32(), device_,
                        sizeof(float) * h_frac.size());
            update_moe_bias(l, h_frac.data(), ne);
        }
        return 0.0;
    }
    // FIX: calling after inference forward (with_grad=false) indexed empty
    // saved_moe_* -> UB/segfault. Require training activations explicitly.
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

// ---------------------------------------------------------------- raw io
static constexpr u32 RAW_MAGIC = 0x57415247u;   // "GRAW"

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
    // FIX: old code accepted dense->MoE / tied->untied files (extra model
    // params stayed zero, returned true -> silent corrupt training) and
    // ignored truncated reads. Enforce exact param-count + shape match.
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
        Tensor cpu(p->shape, DType::F32, Device::CPU);
        if (cpu.nbytes() == 0) return false;
        if (!f.read(reinterpret_cast<char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()))) return false;
        p->w.copy_from(cpu);
    }
    return true;
}

} // namespace gai
