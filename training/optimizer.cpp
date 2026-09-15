#include "training/optimizer.h"
#include "core/ops.h"

#include <cmath>
#include <ostream>
#include <istream>
#include <utility>
#include <vector>

namespace gai {

AdamW::AdamW(Model& model, AdamWConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "AdamW requires enable_grad(true)");
    m_.reserve(model_.parameters().size());
    v_.reserve(model_.parameters().size());
    for (Parameter* p : model_.parameters()) {
        m_.push_back(Tensor::zeros(p->shape, DType::F32, model_.device()));
        v_.push_back(Tensor::zeros(p->shape, DType::F32, model_.device()));
    }
}

double AdamW::step(float lr, float grad_scale) {
    ++t_;
    Device dev = model_.device();
    auto& params = model_.parameters();

    // ---- global grad norm (fused: 1 sync, was ~200). DeepSeek-style.
    std::vector<std::pair<const float*, i64>> parts;
    parts.reserve(params.size());
    for (Parameter* p : params) {
        if (!p->g.defined()) continue;
        if (p->frozen) continue;
        if (p->numel() == 0) continue;
        parts.emplace_back(p->g.f32(), p->numel());
    }
    double sq = ops::global_sq_norm_multi(dev, parts);
    double gnorm = std::sqrt(sq) * static_cast<double>(grad_scale);
    last_grad_norm_ = gnorm;

    float clip_scale = 1.0f;
    if (cfg_.grad_clip > 0.0f && gnorm > static_cast<double>(cfg_.grad_clip)) {
        clip_scale = static_cast<float>(static_cast<double>(cfg_.grad_clip) / (gnorm + 1e-6));
    }
    const float effective_scale = grad_scale * clip_scale;

    // Skip the update on a non-finite gradient (fp16 overflow, bad batch).
    if (!std::isfinite(gnorm)) {
        log_warn(strfmt("adamw: non-finite grad norm at step %lld, update skipped",
                        static_cast<long long>(t_)));
        --t_;
        return gnorm;
    }

    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(t_));

    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params[i];
        if (!p->g.defined()) continue;
        if (p->frozen) continue;   // frozen params receive no update at all
        float wd = p->decay ? cfg_.weight_decay : 0.0f;
        ops::adamw_step(dev, p->w.f32(), p->g.f32(), m_[i].f32(), v_[i].f32(),
                        p->numel(), lr, cfg_.beta1, cfg_.beta2, cfg_.eps, wd,
                        bc1, bc2, effective_scale);
    }
    return gnorm;
}

size_t AdamW::state_bytes() const {
    size_t n = 0;
    for (const auto& t : m_) n += t.nbytes();
    for (const auto& t : v_) n += t.nbytes();
    return n;
}

void AdamW::save_state(std::ostream& os) const {
    u64 count = static_cast<u64>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), 8);
    os.write(reinterpret_cast<const char*>(&t_), 8);
    os.write(reinterpret_cast<const char*>(&cfg_), sizeof(AdamWConfig));
    for (size_t i = 0; i < m_.size(); ++i) {
        Tensor mc = m_[i].to(Device::CPU);
        Tensor vc = v_[i].to(Device::CPU);
        u64 ne = static_cast<u64>(mc.numel());
        os.write(reinterpret_cast<const char*>(&ne), 8);
        os.write(reinterpret_cast<const char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()));
        os.write(reinterpret_cast<const char*>(vc.data_ptr()), static_cast<std::streamsize>(vc.nbytes()));
    }
}

bool AdamW::load_state(std::istream& is) {
    u64 count = 0;
    if (!is.read(reinterpret_cast<char*>(&count), 8)) return false;
    if (count != m_.size()) return false;
    if (!is.read(reinterpret_cast<char*>(&t_), 8)) return false;
    AdamWConfig saved{};
    if (!is.read(reinterpret_cast<char*>(&saved), sizeof(AdamWConfig))) return false;
    // The recipe on disk wins for beta/eps (they define the moment semantics);
    // lr and clipping come from the live config so a run can be re-tuned on resume.
    cfg_.beta1 = saved.beta1;
    cfg_.beta2 = saved.beta2;
    cfg_.eps   = saved.eps;

    for (size_t i = 0; i < m_.size(); ++i) {
        u64 ne = 0;
        if (!is.read(reinterpret_cast<char*>(&ne), 8)) return false;
        if (ne != static_cast<u64>(m_[i].numel())) return false;
        Tensor mc(m_[i].shape(), DType::F32, Device::CPU);
        Tensor vc(v_[i].shape(), DType::F32, Device::CPU);
        if (!is.read(reinterpret_cast<char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()))) return false;
        if (!is.read(reinterpret_cast<char*>(vc.data_ptr()), static_cast<std::streamsize>(vc.nbytes()))) return false;
        m_[i].copy_from(mc);
        v_[i].copy_from(vc);
    }
    return true;
}

// ---------------------------------------------------------------- Lion
Lion::Lion(Model& model, LionConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "Lion requires enable_grad(true)");
    m_.reserve(model_.parameters().size());
    for (Parameter* p : model_.parameters()) {
        m_.push_back(Tensor::zeros(p->shape, DType::F32, model_.device()));
    }
}

double Lion::step(float lr, float grad_scale) {
    ++t_;
    Device dev = model_.device();
    auto& params = model_.parameters();

    // Fused norm (same as AdamW above): 1 D2H instead of ~200.
    std::vector<std::pair<const float*, i64>> parts;
    parts.reserve(params.size());
    for (Parameter* p : params) {
        if (!p->g.defined()) continue;
        if (p->frozen) continue;
        if (p->numel() == 0) continue;
        parts.emplace_back(p->g.f32(), p->numel());
    }
    double sq = ops::global_sq_norm_multi(dev, parts);
    double gnorm = std::sqrt(sq) * static_cast<double>(grad_scale);
    last_grad_norm_ = gnorm;

    float clip_scale = 1.0f;
    if (cfg_.grad_clip > 0.0f && gnorm > static_cast<double>(cfg_.grad_clip)) {
        clip_scale = static_cast<float>(static_cast<double>(cfg_.grad_clip) / (gnorm + 1e-6));
    }
    const float effective_scale = grad_scale * clip_scale;

    if (!std::isfinite(gnorm)) {
        log_warn(strfmt("lion: non-finite grad norm at step %lld, update skipped",
                        static_cast<long long>(t_)));
        --t_;
        return gnorm;
    }

    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params[i];
        if (!p->g.defined()) continue;
        if (p->frozen) continue;
        float wd = p->decay ? cfg_.weight_decay : 0.0f;
        ops::lion_step(dev, p->w.f32(), p->g.f32(), m_[i].f32(),
                       p->numel(), lr, cfg_.beta1, cfg_.beta2, wd, effective_scale);
    }
    return gnorm;
}

size_t Lion::state_bytes() const {
    size_t n = 0;
    for (const auto& t : m_) n += t.nbytes();
    return n;
}

void Lion::save_state(std::ostream& os) const {
    u64 count = static_cast<u64>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), 8);
    os.write(reinterpret_cast<const char*>(&t_), 8);
    os.write(reinterpret_cast<const char*>(&cfg_), sizeof(LionConfig));
    for (size_t i = 0; i < m_.size(); ++i) {
        Tensor mc = m_[i].to(Device::CPU);
        u64 ne = static_cast<u64>(mc.numel());
        os.write(reinterpret_cast<const char*>(&ne), 8);
        os.write(reinterpret_cast<const char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()));
    }
}

bool Lion::load_state(std::istream& is) {
    u64 count = 0;
    if (!is.read(reinterpret_cast<char*>(&count), 8)) return false;
    if (count != m_.size()) return false;
    if (!is.read(reinterpret_cast<char*>(&t_), 8)) return false;
    LionConfig saved{};
    if (!is.read(reinterpret_cast<char*>(&saved), sizeof(LionConfig))) return false;
    cfg_.beta1 = saved.beta1;
    cfg_.beta2 = saved.beta2;
    for (size_t i = 0; i < m_.size(); ++i) {
        u64 ne = 0;
        if (!is.read(reinterpret_cast<char*>(&ne), 8)) return false;
        if (ne != static_cast<u64>(m_[i].numel())) return false;
        Tensor mc(m_[i].shape(), DType::F32, Device::CPU);
        if (!is.read(reinterpret_cast<char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()))) return false;
        m_[i].copy_from(mc);
    }
    return true;
}

} // namespace gai
