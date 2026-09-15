#include "model/model.h"
#include "core/ops.h"
#include "core/device.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>

namespace gai {

// ---------------------------------------------------------------- MoE aux loss
// Computes the DeepSeek-style load-balance term for one layer from its saved
// routing: raw = ne * sum_e(mean_prob_e * frac_e), and uploads frac_e into
// the (tiny, caller-owned) device buffer consumed by moe_backward.
double Model::moe_layer_aux(Device dev, const float* probs, const i32* idx,
                             float* auxfrac_dev, int layer, i64 N, int K, int ne) {
    // Fast GPU path: frac stays on device, only ne floats come back.
    // Avoids 2x large D2H (N*K + N*ne) + 1x H2D per layer per micro-batch.
    if (dev == Device::CUDA) {
        std::vector<float> h_frac(static_cast<size_t>(ne));
        std::vector<float> h_psum(static_cast<size_t>(ne));
        if (ops::moe_aux_gpu(dev, probs, idx, auxfrac_dev,
                             h_frac.data(), h_psum.data(), N, K, ne)) {
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
    GAI_CHECK(rope_scale >= 1.0f, "rope_scale must be >= 1.0 (1.0 = off)");
    GAI_CHECK(z_loss_scale >= 0.0f && z_loss_scale <= 0.01f, "z_loss_scale must be in [0,0.01]");
    GAI_CHECK(moe_jitter >= 0.0f && moe_jitter <= 0.5f, "moe_jitter must be in [0,0.5]");
    GAI_CHECK(rope_yarn_mscale >= 0.0f && rope_yarn_mscale <= 2.0f, "rope_yarn_mscale must be in [0,2]");
    if (use_moe) {
        GAI_CHECK(num_experts > 0 && num_experts <= 64, "num_experts must be in [1,64]");
        GAI_CHECK(moe_top_k > 0 && moe_top_k <= num_experts, "moe_top_k must be in [1,num_experts]");
        GAI_CHECK(moe_top_k <= 8, "moe_top_k must be <= 8 (router kernel limit)");
        GAI_CHECK(moe_expert_dim > 0, "moe_expert_dim must be > 0");
        GAI_CHECK(moe_aux_scale >= 0.0f, "moe_aux_scale must be >= 0");
    }
}

ModelConfig ModelConfig::from_config(const Config& c, const std::string& p) {
    ModelConfig m;
    auto key = [&](const char* k) { return p.empty() ? std::string(k) : p + "." + k; };
    m.vocab_size        = static_cast<int>(c.get_int(key("vocab_size"), m.vocab_size));
    m.hidden_size       = static_cast<int>(c.get_int(key("hidden_size"), m.hidden_size));
    m.num_layers        = static_cast<int>(c.get_int(key("num_layers"), m.num_layers));
    m.num_heads         = static_cast<int>(c.get_int(key("num_heads"), m.num_heads));
    m.num_kv_heads      = static_cast<int>(c.get_int(key("num_kv_heads"), m.num_kv_heads));
    m.intermediate_size = static_cast<int>(c.get_int(key("intermediate_size"), m.intermediate_size));
    m.max_seq_len       = static_cast<int>(c.get_int(key("max_seq_len"), m.max_seq_len));
    m.rope_theta        = c.get_f32(key("rope_theta"), m.rope_theta);
    m.rms_eps           = c.get_f32(key("rms_eps"), m.rms_eps);
    m.tie_embeddings    = c.get_bool(key("tie_embeddings"), m.tie_embeddings);
    m.init_std          = c.get_f32(key("init_std"), m.init_std);
    m.use_moe           = c.get_bool(key("use_moe"), m.use_moe);
    m.num_experts       = static_cast<int>(c.get_int(key("num_experts"), m.num_experts));
    m.moe_top_k         = static_cast<int>(c.get_int(key("moe_top_k"), m.moe_top_k));
    m.moe_expert_dim    = static_cast<int>(c.get_int(key("moe_expert_dim"), m.moe_expert_dim));
    m.moe_shared        = c.get_bool(key("moe_shared"), m.moe_shared);
    m.moe_aux_scale     = c.get_f32(key("moe_aux_scale"), m.moe_aux_scale);
    m.moe_jitter        = c.get_f32(key("moe_jitter"), m.moe_jitter);
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
static float rope_mscale_local(const ModelConfig& m) {
    if (m.rope_scale <= 1.0f) return 1.0f;
    if (m.rope_yarn_mscale > 0.0f) return m.rope_yarn_mscale;
    return 0.1f * std::log(m.rope_scale) + 1.0f;
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
           << ",E=" << moe_expert_dim << (moe_shared ? ",shared" : "") << ")";
    else
        ss << " ffn=" << intermediate_size << " (dense)";
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
    for (Parameter* p : params_) if (p->g.defined()) p->g.zero_();
}

// ---------------------------------------------------------------- activations
static Tensor mk(std::vector<i64> shape, Device dev, size_t& acc) {
    Tensor t = Tensor::zeros(std::move(shape), DType::F32, dev);
    acc += t.nbytes();
    return t;
}

Activations Model::make_activations(int B, int T, bool with_grad) const {
    Activations a;
    a.B = B;
    a.T = T;
    a.with_grad = with_grad;

    const i64 N   = static_cast<i64>(B) * T;
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
    a.logits  = mk({N, V},  device_, acc);

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
        }
        // Single transient probs buffer, recomputed per layer in backward
        // (saves (L-1)*B*H*T*T floats, e.g. 2.5GB at B=2,T=1024,L=26).
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
        a.dgate   = mk({N, F},   device_, acc);
        a.dup     = mk({N, F},   device_, acc);
        a.dact    = mk({N, F},   device_, acc);
        a.dffn    = mk({N, d},   device_, acc);
        a.dlogits = mk({N, V},   device_, acc);
        a.dtmp    = mk({N, d},   device_, acc);
    }
    return a;
}

size_t Model::estimate_activation_bytes(int B, int T, bool with_grad) const {
    const i64 N   = static_cast<i64>(B) * T;
    const i64 d   = cfg_.hidden_size;
    const i64 qd  = cfg_.q_dim();
    const i64 kvd = cfg_.kv_dim();
    const i64 F   = cfg_.use_moe ? cfg_.moe_top_k * cfg_.moe_expert_dim
                                 : cfg_.intermediate_size;
    const i64 V   = cfg_.vocab_size;
    const i64 L   = cfg_.num_layers;

    i64 e = N * (4 * d + qd * 2 + kvd * 2 + F * 3 + V);
    if (cfg_.use_moe) e += N * (cfg_.num_experts + 2 * cfg_.moe_top_k);
    if (with_grad) {
        // Attention probs are recomputed per layer into ONE shared buffer
        // (not stored per layer): single B*H*T*T instead of L*B*H*T*T.
        i64 per_layer = N * (4 * d + qd * 2 + kvd * 2 + F * 3 + 2);
        if (cfg_.use_moe) per_layer += N * (cfg_.num_experts + 3 * cfg_.moe_top_k);
        if (cfg_.use_qk_norm) per_layer += N * (cfg_.num_heads + cfg_.num_kv_heads);
        e += L * per_layer;
        e += static_cast<i64>(B) * cfg_.num_heads * T * T; // transient recompute buf
        e += N * (5 * d + qd * 2 + kvd * 2 + F * 3 + V + 2);
        if (cfg_.use_moe) e += N * cfg_.moe_top_k * cfg_.moe_expert_dim + cfg_.num_experts;
    }
    return static_cast<size_t>(e) * sizeof(float);
}

// ---------------------------------------------------------------- forward
Tensor& Model::forward(const i32* ids, int B, int T, Activations& act) {
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

    GAI_CHECK(act.B == B && act.T == T, "activation buffer shape mismatch");
    GAI_CHECK(T <= cfg_.max_seq_len, "sequence longer than max_seq_len");

    // position ids (repeated per batch element)
    if (!act.pos.defined() || act.pos.numel() < N) {
        act.pos = Tensor::empty({N}, DType::I32, device_);
        act.bytes += act.pos.nbytes();
    }
    std::vector<i32> pos_all(static_cast<size_t>(N));
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) pos_all[static_cast<size_t>(b) * T + t] = t;
    Tensor cpu_pos({N}, DType::I32, Device::CPU);
    std::memcpy(cpu_pos.data_ptr(), pos_all.data(), N * sizeof(i32));
    act.pos.copy_from(cpu_pos);

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
        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined()) {
            float* rrms_q = train ? act.saved_qk_rms_q[sl].f32() : nullptr;
            float* rrms_k = train ? act.saved_qk_rms_k[sl].f32() : nullptr;
            ops::rmsnorm_forward(dev, act.q.f32(), L.qk_qnorm.w.f32(), act.q.f32(),
                                 rrms_q, N * H, hd, cfg_.rms_eps);
            ops::rmsnorm_forward(dev, act.k.f32(), L.qk_knorm.w.f32(), act.k.f32(),
                                 rrms_k, N * KV, hd, cfg_.rms_eps);
        }

        const float theta_eff = rope_theta_eff(cfg_);
        ops::rope_forward(dev, act.q.f32(), act.k.f32(), act.pos.i32p(), N, H, KV, hd, theta_eff);

        if (train) {
            ops::copy(dev, act.saved_q[sl].f32(), act.q.f32(), N * qd);
            ops::copy(dev, act.saved_k[sl].f32(), act.k.f32(), N * kvd);
            ops::copy(dev, act.saved_v[sl].f32(), act.v.f32(), N * kvd);
        }

        // Probs are never stored (recomputed per layer in backward).
        // nullptr here = memory-lean forward on both CPU and CUDA.
        ops::attention_forward(dev, act.q.f32(), act.k.f32(), act.v.f32(),
                               act.att_out.f32(), nullptr,
                               B, T, H, KV, hd, scale);
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
            ops::moe_forward(dev, act.xb2.f32(), L.router.w.f32(),
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

    ops::linear_forward(dev, act.xb.f32(), lm_head().w.f32(), act.logits.f32(), static_cast<int>(N), d, V);
    return act.logits;
}

// ---------------------------------------------------------------- backward
double Model::forward_backward(const i32* ids, const i32* targets, int B, int T,
                               Activations& act, i64* out_ntok, float dout_scale) {
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

    forward(ids, B, T, act);

    // ---- loss (with z-loss stabilizer when configured)
    double loss_sum = 0.0;
    i64    ntok = 0;
    ops::softmax_cross_entropy(dev, act.logits.f32(), targets, act.dlogits.f32(),
                               N, V, &loss_sum, &ntok, cfg_.z_loss_scale);
    if (out_ntok) *out_ntok = ntok;
    if (ntok == 0) return 0.0;
    // FP16 loss scaling: amplify the seed gradient so small values survive
    // fp16 GEMMs; the trainer divides them back out in the optimizer step.
    // Guard: scaled seed max ~ dout_scale/ntok must stay in fp16 range.
    // Cap at 8192 (far from 65504) so tiny batches can't overflow to inf.
    float eff_scale = dout_scale;
    if (eff_scale != 1.0f && ops::gemm_fp16_enabled() && ntok > 0) {
        float cap = 8192.0f * (float)ntok;
        if (eff_scale > cap) eff_scale = cap;
    }
    if (eff_scale != 1.0f) ops::scale_inplace(dev, act.dlogits.f32(), eff_scale, N * V);
    // P0-05 FIX (token-weighted grad accumulation): SCE above produces MEAN
    // grads (divided by ntok of THIS micro). Old trainer averaged micro-means
    // ((gA+gB)/accum), so a 100-tok micro counted as much as a 1000-tok micro
    // — wrong SFT objective when masks vary. Now convert to SUM grads
    // (×ntok, still ×eff_scale) so micros accumulate as sums; the trainer
    // divides once by the step-total ntok_step. Uniform batches are bit-
    // identical to before (mean*ntok/ntok); masked SFT becomes correctly
    // weighted. Loss value returned below stays a MEAN for logging.
    if (ntok > 0) ops::scale_inplace(dev, act.dlogits.f32(), static_cast<float>(ntok), N * V);

    // ---- MoE load-balance aux loss: measured per layer inside the backward
    // loop below (each layer refills the shared [ne] fractions buffer right
    // before its own moe_backward call), then added to the returned loss.
    double aux_total = 0.0;

    // ---- lm head
    ops::zero(dev, act.dxb.f32(), N * d);
    ops::linear_backward(dev, act.saved_hnorm.f32(), lm_head().w.f32(), act.dlogits.f32(),
                         act.dxb.f32(), lm_head().g.f32(), static_cast<int>(N), d, V);

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
            // P0-02 FIX: aux grads must live in the SAME scaled+sum space as
            // the main CE grads (above: mean×eff_scale×ntok = sum×eff_scale).
            // Old code passed raw moe_aux_scale (mean, unscaled), so after the
            // trainer's 1/(ntok*dscale) unscale the router regularization was
            // ~1/(ntok*dscale) too weak (invisible with dscale=65536).
            // Now: ×eff_scale (same loss-scale) ×N (mean→sum over N) ÷L (loss
            // is averaged over layers) so both paths accumulate as scaled sums
            // and share one divisor, exactly matching the returned loss
            // (CE mean + aux_scale*aux_total/L).
            const float* aux_frac = nullptr;
            float aux_scale_grad = 0.0f;
            if (cfg_.moe_aux_scale > 0.0f) {
                aux_total += moe_layer_aux(dev, act.saved_moe_probs[sl].f32(),
                                           act.saved_moe_idx[sl].i32p(),
                                           act.moe_auxfrac.f32(),
                                           l, N, K, ne);
                aux_frac = act.moe_auxfrac.f32();
                aux_scale_grad = cfg_.moe_aux_scale * eff_scale * static_cast<float>(N) /
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
        ops::attention_forward(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                               act.saved_v[sl].f32(), act.att_out.f32(),
                               act.attn_probs_tmp.f32(),
                               B, T, H, KV, hd, scale);
        ops::attention_backward(dev, act.saved_q[sl].f32(), act.saved_k[sl].f32(),
                                act.saved_v[sl].f32(), act.attn_probs_tmp.f32(),
                                act.dattout.f32(),
                                act.dq.f32(), act.dk.f32(), act.dv.f32(),
                                B, T, H, KV, hd, scale);

        const float theta_eff_bwd = rope_theta_eff(cfg_);
        ops::rope_backward(dev, act.dq.f32(), act.dk.f32(), act.pos.i32p(), N, H, KV, hd, theta_eff_bwd);

        // QK-Norm backward (exact, no extra GEMM):
        // dq/dk hold dL/d(normed Q/K). rmsnorm_backward does dx+=f(dout), so
        // dout and dx must NOT alias. att_out [N,qd] and dattout [N,qd] are
        // both free here (attn recompute + backward already consumed them),
        // so use them as dout staging. act.q/act.k (forward scratch, dead in
        // backward) stage the unrotated Q/K inputs.
        if (cfg_.use_qk_norm && L.qk_qnorm.w.defined()) {
            // Forward order is rmsnorm -> RoPE, so exact backward needs
            // pre-rope x1 = R^T * x2 and pre-rope dout1 = R^T * dout2.
            // dout1 is already ready: rope_backward above unrotated dq/dk.
            // Unrotate the saved post-rope Q/K into act.q/act.k staging.
            // NOTE: the old code used post-rope saved_q directly. The dot term
            // is rotation-invariant but per-channel gain grads
            // (dweight[i] = sum dout[i]*x[i]*rrms) are NOT — RoPE mixes pairs,
            // so that was biased. Unrotating x makes both dx and dweight exact.
            ops::copy(dev, act.q.f32(), act.saved_q[sl].f32(), N * qd);
            ops::rope_backward(dev, act.q.f32(), nullptr, act.pos.i32p(),
                               N, H, 0, hd, theta_eff_bwd);
            // Q path: stage dout in att_out, zero dq, then backward.
            ops::copy(dev, act.att_out.f32(), act.dq.f32(), N * qd);
            ops::zero(dev, act.dq.f32(), N * qd);
            ops::rmsnorm_backward(dev, act.q.f32(), L.qk_qnorm.w.f32(), act.att_out.f32(),
                                  act.saved_qk_rms_q[sl].f32(), act.dq.f32(), L.qk_qnorm.g.f32(),
                                  N * H, hd);
            // K path: unrotate saved_k into act.k, stage dout in dattout.
            ops::copy(dev, act.k.f32(), act.saved_k[sl].f32(), N * kvd);
            ops::rope_backward(dev, nullptr, act.k.f32(), act.pos.i32p(),
                               N, 0, KV, hd, theta_eff_bwd);
            ops::copy(dev, act.dattout.f32(), act.dk.f32(), N * kvd);
            ops::zero(dev, act.dk.f32(), N * kvd);
            ops::rmsnorm_backward(dev, act.k.f32(), L.qk_knorm.w.f32(), act.dattout.f32(),
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
    // DeepSeek-style aux loss: moe_layer_aux returns per-layer raw
    // (ne*sum(mean_prob*frac)). FIX (10/10): old code SUMMED over layers, so a
    // 36L model got 1.38x stronger aux than a 26L model with the same scale.
    // Average over layers so moe_aux_scale means the same at any depth.
    if (cfg_.use_moe && cfg_.moe_aux_scale > 0.0f && cfg_.num_layers > 0)
        loss += cfg_.moe_aux_scale * aux_total / static_cast<double>(cfg_.num_layers);
    return loss;
}

// ---------------------------------------------------------------- aux loss
double Model::moe_aux_loss(Activations& act, int B, int T) {
    if (!cfg_.use_moe) return 0.0;
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
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != RAW_MAGIC) return false;
    f.read(reinterpret_cast<char*>(&n), 4);
    for (u32 i = 0; i < n; ++i) {
        u32 len = 0;
        if (!f.read(reinterpret_cast<char*>(&len), 4) || len > 512) return false;
        std::string name(len, '\0');
        f.read(name.data(), len);
        u64 ne = 0;
        f.read(reinterpret_cast<char*>(&ne), 8);
        Parameter* p = find_parameter(name);
        if (!p || static_cast<u64>(p->numel()) != ne) return false;
        Tensor cpu(p->shape, DType::F32, Device::CPU);
        if (!f.read(reinterpret_cast<char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()))) return false;
        p->w.copy_from(cpu);
    }
    return true;
}

} // namespace gai
