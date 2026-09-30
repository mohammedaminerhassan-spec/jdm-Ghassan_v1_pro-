#include "training/optimizer.h"
#include "core/ops.h"

#ifdef GAI_OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <ostream>
#include <istream>
#include <utility>
#include <vector>
#include <algorithm>

namespace gai {

namespace {

struct OptStepTimer {
    Timer t;
    ~OptStepTimer() { ops::perf_note_opt_step(t.elapsed_us()); }
};
}

static Tensor snapshot_tensor(const Tensor& src) {
    if (!src.defined()) return {};
    return src.device() == Device::CPU ? src.clone() : src.to(Device::CPU);
}

static void validate_moment_table(const char* kind,
                                  const std::vector<Parameter*>& params,
                                  const std::vector<Tensor>& m,
                                  const std::vector<Tensor>& v) {
    GAI_CHECK(m.size() == params.size(),
              std::string(kind) + ": moment table has " + std::to_string(m.size()) +
              " entries for " + std::to_string(params.size()) + " parameters");
    if (!v.empty()) {
        GAI_CHECK(v.size() == params.size(),
                  std::string(kind) + ": second moment table has " + std::to_string(v.size()) +
                  " entries for " + std::to_string(params.size()) + " parameters");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params[i];
        if (!p->g.defined() || p->frozen) continue;
        const std::string where = std::string(kind) + " param '" + p->name + "' (index " +
                                  std::to_string(i) + "): ";
        GAI_CHECK(m[i].defined(), where + "optimizer moment is missing (index misalignment)");
        GAI_CHECK(m[i].numel() == p->numel(),
                  where + "moment numel " + std::to_string(m[i].numel()) +
                  " != param numel " + std::to_string(p->numel()));
        GAI_CHECK(m[i].device() == p->w.device(),
                  where + "moment on " + device_name(m[i].device()) +
                  " but weights on " + device_name(p->w.device()));
        if (!v.empty()) {
            GAI_CHECK(v[i].defined(), where + "second moment is missing (index misalignment)");
            GAI_CHECK(v[i].numel() == p->numel(),
                      where + "second moment numel " + std::to_string(v[i].numel()) +
                      " != param numel " + std::to_string(p->numel()));
            GAI_CHECK(v[i].device() == p->w.device(),
                      where + "second moment on " + device_name(v[i].device()) +
                      " but weights on " + device_name(p->w.device()));
        }
    }
}

AdamW::AdamW(Model& model, AdamWConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "AdamW requires enable_grad(true)");
    m_.reserve(model_.parameters().size());
    v_.reserve(model_.parameters().size());

    for (Parameter* p : model_.parameters()) {
        if (p->frozen) {
            m_.emplace_back();
            v_.emplace_back();
            continue;
        }
        m_.push_back(Tensor::zeros(p->shape, DType::F32, model_.device()));
        v_.push_back(Tensor::zeros(p->shape, DType::F32, model_.device()));
    }
}

double AdamW::step(float lr, float grad_scale) {
    OptStepTimer opt_step_timer;
    ++t_;
    Device dev = model_.device();
    auto& params = model_.parameters();
    validate_moment_table("adamw", params, m_, v_);

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
        log_warn(strfmt("adamw: non-finite grad norm at step %lld, update skipped",
                        static_cast<long long>(t_)));
        --t_;
        return gnorm;
    }

    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(t_));

    for (long long i = 0; i < static_cast<long long>(params.size()); ++i) {
        Parameter* p = params[static_cast<size_t>(i)];
        if (!p->g.defined()) continue;
        if (p->frozen) continue;
        float wd = p->decay ? cfg_.weight_decay : 0.0f;

        if (log_level() <= LogLevel::Debug)
            log_debug(strfmt("[opt] adamw '%s' numel=%lld w=%p g=%p m=%p v=%p",
                             p->name.c_str(), static_cast<long long>(p->numel()),
                             static_cast<const void*>(p->w.f32()),
                             static_cast<const void*>(p->g.f32()),
                             static_cast<const void*>(m_[static_cast<size_t>(i)].f32()),
                             static_cast<const void*>(v_[static_cast<size_t>(i)].f32())));
        ops::adamw_step(dev, p->w.f32(), p->g.f32(), m_[static_cast<size_t>(i)].f32(), v_[static_cast<size_t>(i)].f32(),
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

OptimizerStateSnapshot AdamW::snapshot_state() const {
    OptimizerStateSnapshot s;
    s.kind = OptimizerSnapshotKind::AdamW;
    s.step = t_;
    s.adamw = cfg_;
    s.first.reserve(m_.size());
    s.second.reserve(v_.size());
    for (const Tensor& t : m_) s.first.push_back(snapshot_tensor(t));
    for (const Tensor& t : v_) s.second.push_back(snapshot_tensor(t));
    return s;
}

static void wr_f32(std::ostream& os, float v) {
    os.write(reinterpret_cast<const char*>(&v), 4);
}
static bool rd_f32(std::istream& is, float& v) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), 4));
}

void AdamW::save_state(std::ostream& os) const {
    u64 count = static_cast<u64>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), 8);
    os.write(reinterpret_cast<const char*>(&t_), 8);
    const u8 fmt = 1;
    os.write(reinterpret_cast<const char*>(&fmt), 1);
    wr_f32(os, cfg_.lr);
    wr_f32(os, cfg_.beta1);
    wr_f32(os, cfg_.beta2);
    wr_f32(os, cfg_.eps);
    wr_f32(os, cfg_.weight_decay);
    wr_f32(os, cfg_.grad_clip);
    auto& params = model_.parameters();
    for (size_t i = 0; i < m_.size(); ++i) {
        const u8 has = (m_[i].defined() && !params[i]->frozen) ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has), 1);
        if (!has) continue;
        Tensor mc = m_[i].to(Device::CPU);
        Tensor vc = v_[i].to(Device::CPU);
        u64 ne = static_cast<u64>(mc.numel());
        os.write(reinterpret_cast<const char*>(&ne), 8);
        os.write(reinterpret_cast<const char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()));
        os.write(reinterpret_cast<const char*>(vc.data_ptr()), static_cast<std::streamsize>(vc.nbytes()));
    }
}

namespace {
// [FIX P0-02] float recipe comparison: configs are parsed deterministically
// from YAML so identical recipes give bit-identical floats; a small epsilon
// guards against float<->double round-trips in serialization paths.
inline bool opt_f32_same(float a, float b) {
    const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
    return d <= 1e-9 * (1.0 + std::fabs(static_cast<double>(a)));
}
}

bool AdamW::load_state(std::istream& is, int state_version, bool exact) {
    u64 count = 0;
    if (!is.read(reinterpret_cast<char*>(&count), 8)) return false;
    if (count != m_.size()) return false;
    if (!is.read(reinterpret_cast<char*>(&t_), 8)) return false;
    auto& params = model_.parameters();
    if (state_version >= OPT_STATE_CURRENT) {
        u8 fmt = 0;
        if (!is.read(reinterpret_cast<char*>(&fmt), 1) || fmt != 1) return false;
        float lr = 0, b1 = 0, b2 = 0, eps = 0, wd = 0, clip = 0;
        if (!rd_f32(is, lr) || !rd_f32(is, b1) || !rd_f32(is, b2) ||
            !rd_f32(is, eps) || !rd_f32(is, wd) || !rd_f32(is, clip))
            return false;

        if (exact) {
            // [FIX P0-02] exact resume: every behavior-affecting field must
            // match; otherwise the restored moments would be updated with a
            // different rule (different wd/clip/beta/eps changes the
            // trajectory even from identical weights+moments).
            if (!opt_f32_same(b1, cfg_.beta1) || !opt_f32_same(b2, cfg_.beta2) ||
                !opt_f32_same(eps, cfg_.eps) || !opt_f32_same(wd, cfg_.weight_decay) ||
                !opt_f32_same(clip, cfg_.grad_clip)) {
                log_error(strfmt("exact resume: adamw hyperparams differ from checkpoint "
                                 "(ckpt beta1=%.6f beta2=%.6f eps=%.3e wd=%.4f clip=%.3f vs "
                                 "cur beta1=%.6f beta2=%.6f eps=%.3e wd=%.4f clip=%.3f)",
                                 b1, b2, (double)eps, (double)wd, (double)clip,
                                 cfg_.beta1, cfg_.beta2, (double)cfg_.eps,
                                 (double)cfg_.weight_decay, (double)cfg_.grad_clip));
                return false;
            }
        }
        cfg_.beta1 = b1;
        cfg_.beta2 = b2;
        cfg_.eps   = eps;
        for (size_t i = 0; i < m_.size(); ++i) {
            u8 has = 0;
            if (!is.read(reinterpret_cast<char*>(&has), 1)) return false;
            const i64 want = params[i]->numel();
            if (has) {
                u64 ne = 0;
                if (!is.read(reinterpret_cast<char*>(&ne), 8)) return false;
                if (ne != static_cast<u64>(want)) return false;
                Tensor mc(params[i]->shape, DType::F32, Device::CPU);
                Tensor vc(params[i]->shape, DType::F32, Device::CPU);
                if (!is.read(reinterpret_cast<char*>(mc.data_ptr()),
                             static_cast<std::streamsize>(mc.nbytes())))
                    return false;
                if (!is.read(reinterpret_cast<char*>(vc.data_ptr()),
                             static_cast<std::streamsize>(vc.nbytes())))
                    return false;

                if (!m_[i].defined()) continue;
                m_[i].copy_from(mc);
                v_[i].copy_from(vc);
            }

        }
        return true;
    }

    AdamWConfig saved{};
    if (!is.read(reinterpret_cast<char*>(&saved), sizeof(AdamWConfig))) return false;
    if (exact) {
        // [FIX P0-02] legacy path: same exactness contract as current format.
        if (!opt_f32_same(saved.beta1, cfg_.beta1) ||
            !opt_f32_same(saved.beta2, cfg_.beta2) ||
            !opt_f32_same(saved.eps, cfg_.eps) ||
            !opt_f32_same(saved.weight_decay, cfg_.weight_decay) ||
            !opt_f32_same(saved.grad_clip, cfg_.grad_clip)) {
            log_error("exact resume: adamw hyperparams differ from legacy checkpoint");
            return false;
        }
    }
    cfg_.beta1 = saved.beta1;
    cfg_.beta2 = saved.beta2;
    cfg_.eps   = saved.eps;

    for (size_t i = 0; i < m_.size(); ++i) {
        u64 ne = 0;
        if (!is.read(reinterpret_cast<char*>(&ne), 8)) return false;
        if (ne != static_cast<u64>(params[i]->numel())) return false;
        Tensor mc(params[i]->shape, DType::F32, Device::CPU);
        Tensor vc(params[i]->shape, DType::F32, Device::CPU);
        if (!is.read(reinterpret_cast<char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()))) return false;
        if (!is.read(reinterpret_cast<char*>(vc.data_ptr()), static_cast<std::streamsize>(vc.nbytes()))) return false;
        if (!m_[i].defined()) continue;
        m_[i].copy_from(mc);
        v_[i].copy_from(vc);
    }
    return true;
}

namespace {
constexpr float kNSa = 1.5f, kNSb = -0.5f;
}

void Muon::orthogonalize(const float* G, float* O, int rows, int cols) {
    Device dev = model_.device();
    GAI_CHECK(rows > 0 && cols > 0, "orthogonalize: empty shape");
    GAI_CHECK(G != nullptr && O != nullptr, "orthogonalize: null pointer");
    GAI_CHECK(scratch_.defined(),
              "orthogonalize: scratch workspace is not allocated");
    const i64 rc = static_cast<i64>(rows) * cols;
    const i64 cc = static_cast<i64>(cols) * cols;

    GAI_CHECK(rc <= omax_rc_ && cc <= amax_cc_,
              "orthogonalize: shape exceeds the construction-time workspace "
              "(rebuild Muon with a larger min_ns_dim coverage)");
    Timer ns_t;
    int iters_done = 0;

    auto note = [&]() { ops::perf_note_muon_ns(iters_done, ns_t.elapsed_us()); };

    float* base = scratch_.f32();
    float* T = base + omax_rc_;
    float* A = base + omax_rc_ * 2;

    ops::copy(dev, O, G, rc);

    const double frob = std::sqrt(ops::global_sq_norm(dev, O, rc));
    if (!std::isfinite(frob) || frob < 1e-12) { note(); return; }
    ops::scale_inplace(dev, O, static_cast<float>(1.0 / frob), rc);

    const i64 saved_thr = ops::gemm_fp16_mnk_threshold();
    ops::set_gemm_fp16_mnk_threshold((i64)1 << 60);
    for (int it = 0; it < cfg_.ns_steps; ++it) {

        ops::gemm(dev, true, false, cols, cols, rows, 1.0f, O, cols, O, cols, 0.0f, A, cols);
        ops::gemm(dev, false, false, rows, cols, cols, 1.0f, O, cols, A, cols, 0.0f, T, cols);

        ops::scale_inplace(dev, O, kNSa, rc);
        ops::scale_inplace(dev, T, kNSb, rc);
        ops::add_inplace(dev, O, T, rc);
        ++iters_done;
    }
    ops::set_gemm_fp16_mnk_threshold(saved_thr);
    note();
}

Muon::Muon(Model& model, MuonConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "Muon requires enable_grad(true)");
    if (cfg_.ns_steps < 1) cfg_.ns_steps = 1;
    if (cfg_.ns_steps > 10) cfg_.ns_steps = 10;

    {
        const double f1 = static_cast<double>(kNSa) + static_cast<double>(kNSb);
        GAI_CHECK(std::fabs(f1 - 1.0) < 1e-6, "Muon NS coefficients must fix 1.0");
    }
    m_.reserve(model_.parameters().size());
    v_.reserve(model_.parameters().size());
    i64 max_rc = 0, max_cc = 0;
    i64 ns_matrices = 0;
    for (Parameter* p : model_.parameters()) {

        if (p->frozen) {
            m_.emplace_back();
            v_.emplace_back();
            continue;
        }

        const std::vector<i64> shape = p->shape;
        m_.push_back(Tensor::zeros(shape, DType::F32, model_.device()));
        const bool is_ns = uses_ns(p);
        const bool is_vec = !is_ns && shape.size() == 1;
        if (is_vec) v_.push_back(Tensor::zeros(shape, DType::F32, model_.device()));
        else v_.emplace_back();
        if (is_ns) {
            ++ns_matrices;
            const i64 r = p->shape[0], c = p->shape[1];
            max_rc = std::max(max_rc, r * c);
            max_cc = std::max(max_cc, c * c);
        }
    }

    omax_rc_ = max_rc;
    amax_cc_ = max_cc;
    const i64 need = max_rc * 2 + max_cc;
    if (need > 0) scratch_ = Tensor::zeros({need}, DType::F32, model_.device());
    if (ns_matrices > 0)
        log_info(strfmt("[opt ] muon: Newton-Schulz on %lld matrices x %d iters (min_ns_dim=%d)",
                        static_cast<long long>(ns_matrices), cfg_.ns_steps, cfg_.min_ns_dim));
    else
        log_info("[opt ] muon: no matrix eligible for Newton-Schulz (min_ns_dim gate); "
                 "all params take the cheap branches");
}

double Muon::step(float lr, float grad_scale) {
    OptStepTimer opt_step_timer;
    ++t_;
    Device dev = model_.device();
    auto& params = model_.parameters();

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
        log_warn(strfmt("muon: non-finite grad norm at step %lld, update skipped",
                        static_cast<long long>(t_)));
        --t_;
        return gnorm;
    }

    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(t_));
    const float vec_lr = lr * cfg_.vec_lr_ratio;

    float* Ostage = scratch_.defined() ? scratch_.f32() : nullptr;

    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params[i];
        if (!p->g.defined()) continue;
        if (p->frozen) continue;

        const bool is_mat = uses_ns(p);
        float wd = p->decay ? cfg_.weight_decay : 0.0f;
        if (is_mat) {

            GAI_CHECK(Ostage != nullptr, "muon: missing scratch for matrix update");
            ops::copy(dev, Ostage, p->g.f32(), p->numel());
            ops::scale_inplace(dev, Ostage, effective_scale * (1.0f - cfg_.beta1), p->numel());
            ops::scale_inplace(dev, m_[i].f32(), cfg_.beta1, p->numel());
            ops::add_inplace(dev, m_[i].f32(), Ostage, p->numel());
            const int rows = static_cast<int>(p->shape[0]);
            const int cols = static_cast<int>(p->shape[1]);
            orthogonalize(m_[i].f32(), Ostage, rows, cols);

            ops::scale_inplace(dev, Ostage, -lr, p->numel());
            ops::add_inplace(dev, p->w.f32(), Ostage, p->numel());
            if (wd != 0.0f) ops::scale_inplace(dev, p->w.f32(), 1.0f - lr * wd, p->numel());
        } else if (p->shape.size() == 1) {

            ops::adamw_step(dev, p->w.f32(), p->g.f32(), m_[i].f32(), v_[i].f32(),
                            p->numel(), vec_lr, cfg_.beta1, cfg_.beta2, cfg_.eps, wd,
                            bc1, bc2, effective_scale);
        } else {

            ops::lion_step(dev, p->w.f32(), p->g.f32(), m_[i].f32(),
                           p->numel(), vec_lr, cfg_.beta1, 0.99f, wd, effective_scale);
        }
    }
    return gnorm;
}

size_t Muon::state_bytes() const {

    size_t n = 0;
    for (const auto& t : m_) n += t.nbytes();
    for (const auto& t : v_) n += t.nbytes();
    n += scratch_.nbytes();
    return n;
}

OptimizerStateSnapshot Muon::snapshot_state() const {
    OptimizerStateSnapshot s;
    s.kind = OptimizerSnapshotKind::Muon;
    s.step = t_;
    s.muon = cfg_;
    s.first.reserve(m_.size());
    s.second.reserve(v_.size());
    for (const Tensor& t : m_) s.first.push_back(snapshot_tensor(t));
    for (const Tensor& t : v_) s.second.push_back(snapshot_tensor(t));
    return s;
}

void Muon::save_state(std::ostream& os) const {
    u64 count = static_cast<u64>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), 8);
    os.write(reinterpret_cast<const char*>(&t_), 8);
    // [FIX P0-02] fmt=2 appends min_ns_dim (fmt=1 files remain loadable;
    // exact resume on fmt=1 skips the min_ns_dim check with a warning).
    const u8 fmt = 2;
    os.write(reinterpret_cast<const char*>(&fmt), 1);
    wr_f32(os, cfg_.lr);
    wr_f32(os, cfg_.vec_lr_ratio);
    wr_f32(os, cfg_.beta1);
    wr_f32(os, cfg_.beta2);
    wr_f32(os, cfg_.eps);
    wr_f32(os, cfg_.weight_decay);
    wr_f32(os, cfg_.grad_clip);
    {
        const i32 ns = static_cast<i32>(cfg_.ns_steps);
        os.write(reinterpret_cast<const char*>(&ns), 4);
        const i32 min_ns = static_cast<i32>(cfg_.min_ns_dim);
        os.write(reinterpret_cast<const char*>(&min_ns), 4);
    }
    auto& params = model_.parameters();
    for (size_t i = 0; i < m_.size(); ++i) {
        u8 mask = 0;
        if (m_[i].defined() && !params[i]->frozen) mask |= 1u;
        if (v_[i].defined() && !params[i]->frozen) mask |= 2u;
        os.write(reinterpret_cast<const char*>(&mask), 1);
        if (mask & 1u) {
            Tensor mc = m_[i].to(Device::CPU);
            u64 ne = static_cast<u64>(mc.numel());
            os.write(reinterpret_cast<const char*>(&ne), 8);
            os.write(reinterpret_cast<const char*>(mc.data_ptr()),
                     static_cast<std::streamsize>(mc.nbytes()));
        }
        if (mask & 2u) {
            Tensor vc = v_[i].to(Device::CPU);
            u64 ne = static_cast<u64>(vc.numel());
            os.write(reinterpret_cast<const char*>(&ne), 8);
            os.write(reinterpret_cast<const char*>(vc.data_ptr()),
                     static_cast<std::streamsize>(vc.nbytes()));
        }
    }
}

static bool rd_muon_moments(std::istream& is, const std::vector<i64>& shape, Tensor& dst) {
    u64 ne = 0;
    if (!is.read(reinterpret_cast<char*>(&ne), 8)) return false;
    if (ne != static_cast<u64>(numel_of(shape))) return false;
    Tensor tmp(shape, DType::F32, Device::CPU);
    if (!is.read(reinterpret_cast<char*>(tmp.data_ptr()),
                 static_cast<std::streamsize>(tmp.nbytes())))
        return false;
    if (!dst.defined()) return true;
    dst.copy_from(tmp);
    return true;
}

bool Muon::load_state(std::istream& is, int state_version, bool exact) {
    if (state_version < OPT_STATE_CURRENT) return false;
    u64 count = 0;
    if (!is.read(reinterpret_cast<char*>(&count), 8)) return false;
    if (count != m_.size()) return false;
    if (!is.read(reinterpret_cast<char*>(&t_), 8)) return false;
    u8 fmt = 0;
    if (!is.read(reinterpret_cast<char*>(&fmt), 1) || (fmt != 1 && fmt != 2)) return false;
    float lr = 0, vr = 0, b1 = 0, b2 = 0, eps = 0, wd = 0, clip = 0;
    i32 ns = 0;
    if (!rd_f32(is, lr) || !rd_f32(is, vr) || !rd_f32(is, b1) || !rd_f32(is, b2) ||
        !rd_f32(is, eps) || !rd_f32(is, wd) || !rd_f32(is, clip))
        return false;
    if (!is.read(reinterpret_cast<char*>(&ns), 4)) return false;
    i32 saved_min_ns = 0;
    if (fmt == 2) {
        if (!is.read(reinterpret_cast<char*>(&saved_min_ns), 4)) return false;
    }

    if (exact) {
        // [FIX P0-02] exact resume: Muon has the widest behavior surface
        // (vec ratio, NS steps, NS gate). Any drift invalidates exactness.
        if (!opt_f32_same(b1, cfg_.beta1) || !opt_f32_same(b2, cfg_.beta2) ||
            !opt_f32_same(eps, cfg_.eps) || !opt_f32_same(wd, cfg_.weight_decay) ||
            !opt_f32_same(clip, cfg_.grad_clip) || !opt_f32_same(vr, cfg_.vec_lr_ratio) ||
            ns != static_cast<i32>(cfg_.ns_steps)) {
            log_error(strfmt("exact resume: muon hyperparams differ from checkpoint "
                             "(ckpt vec_ratio=%.4f ns=%d wd=%.4f clip=%.3f vs "
                             "cur vec_ratio=%.4f ns=%d wd=%.4f clip=%.3f)",
                             (double)vr, (int)ns, (double)wd, (double)clip,
                             (double)cfg_.vec_lr_ratio, (int)cfg_.ns_steps,
                             (double)cfg_.weight_decay, (double)cfg_.grad_clip));
            return false;
        }
        if (fmt == 2 && saved_min_ns != static_cast<i32>(cfg_.min_ns_dim)) {
            log_error(strfmt("exact resume: muon min_ns_dim differs (ckpt %d vs cur %d); "
                             "the NS/vec branch assignment would change",
                             (int)saved_min_ns, (int)cfg_.min_ns_dim));
            return false;
        }
        if (fmt == 1 && cfg_.min_ns_dim != 0) {
            log_warn("[ckpt] exact resume from pre-min_ns_dim checkpoint: cannot verify "
                     "muon min_ns_dim; continuing (moments shapes validated per-tensor)");
        }
    }
    cfg_.beta1 = b1;
    cfg_.beta2 = b2;
    cfg_.eps = eps;
    auto& params = model_.parameters();
    for (size_t i = 0; i < m_.size(); ++i) {
        u8 mask = 0;
        if (!is.read(reinterpret_cast<char*>(&mask), 1)) return false;
        if (mask & 1u) {
            if (!rd_muon_moments(is, params[i]->shape, m_[i])) return false;
        }
        if (mask & 2u) {
            if (!rd_muon_moments(is, params[i]->shape, v_[i])) return false;
        }
    }
    return true;
}

Lion::Lion(Model& model, LionConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "Lion requires enable_grad(true)");
    m_.reserve(model_.parameters().size());
    for (Parameter* p : model_.parameters()) {

        if (p->frozen) {
            m_.emplace_back();
            continue;
        }
        m_.push_back(Tensor::zeros(p->shape, DType::F32, model_.device()));
    }
}

double Lion::step(float lr, float grad_scale) {
    OptStepTimer opt_step_timer;
    ++t_;
    Device dev = model_.device();
    auto& params = model_.parameters();

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

    validate_moment_table("lion", params, m_, {});
    for (long long i = 0; i < static_cast<long long>(params.size()); ++i) {
        Parameter* p = params[static_cast<size_t>(i)];
        if (!p->g.defined()) continue;
        if (p->frozen) continue;
        float wd = p->decay ? cfg_.weight_decay : 0.0f;
        ops::lion_step(dev, p->w.f32(), p->g.f32(), m_[static_cast<size_t>(i)].f32(),
                       p->numel(), lr, cfg_.beta1, cfg_.beta2, wd, effective_scale);
    }
    return gnorm;
}

size_t Lion::state_bytes() const {
    size_t n = 0;
    for (const auto& t : m_) n += t.nbytes();
    return n;
}

OptimizerStateSnapshot Lion::snapshot_state() const {
    OptimizerStateSnapshot s;
    s.kind = OptimizerSnapshotKind::Lion;
    s.step = t_;
    s.lion = cfg_;
    s.first.reserve(m_.size());
    for (const Tensor& t : m_) s.first.push_back(snapshot_tensor(t));
    return s;
}

void Lion::save_state(std::ostream& os) const {
    u64 count = static_cast<u64>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), 8);
    os.write(reinterpret_cast<const char*>(&t_), 8);
    const u8 fmt = 1;
    os.write(reinterpret_cast<const char*>(&fmt), 1);
    wr_f32(os, cfg_.lr);
    wr_f32(os, cfg_.beta1);
    wr_f32(os, cfg_.beta2);
    wr_f32(os, cfg_.weight_decay);
    wr_f32(os, cfg_.grad_clip);
    auto& params = model_.parameters();
    for (size_t i = 0; i < m_.size(); ++i) {
        const u8 has = (m_[i].defined() && !params[i]->frozen) ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has), 1);
        if (!has) continue;
        Tensor mc = m_[i].to(Device::CPU);
        u64 ne = static_cast<u64>(mc.numel());
        os.write(reinterpret_cast<const char*>(&ne), 8);
        os.write(reinterpret_cast<const char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()));
    }
}

bool Lion::load_state(std::istream& is, int state_version, bool exact) {
    u64 count = 0;
    if (!is.read(reinterpret_cast<char*>(&count), 8)) return false;
    if (count != m_.size()) return false;
    if (!is.read(reinterpret_cast<char*>(&t_), 8)) return false;
    auto& params = model_.parameters();
    if (state_version >= OPT_STATE_CURRENT) {
        u8 fmt = 0;
        if (!is.read(reinterpret_cast<char*>(&fmt), 1) || fmt != 1) return false;
        float lr = 0, b1 = 0, b2 = 0, wd = 0, clip = 0;
        if (!rd_f32(is, lr) || !rd_f32(is, b1) || !rd_f32(is, b2) ||
            !rd_f32(is, wd) || !rd_f32(is, clip))
            return false;
        if (exact) {
            // [FIX P0-02] exact resume: Lion update depends on wd/clip/betas.
            if (!opt_f32_same(b1, cfg_.beta1) || !opt_f32_same(b2, cfg_.beta2) ||
                !opt_f32_same(wd, cfg_.weight_decay) ||
                !opt_f32_same(clip, cfg_.grad_clip)) {
                log_error(strfmt("exact resume: lion hyperparams differ from checkpoint "
                                 "(ckpt beta1=%.6f beta2=%.6f wd=%.4f clip=%.3f vs "
                                 "cur beta1=%.6f beta2=%.6f wd=%.4f clip=%.3f)",
                                 b1, b2, (double)wd, (double)clip,
                                 cfg_.beta1, cfg_.beta2,
                                 (double)cfg_.weight_decay, (double)cfg_.grad_clip));
                return false;
            }
        }
        cfg_.beta1 = b1;
        cfg_.beta2 = b2;
        for (size_t i = 0; i < m_.size(); ++i) {
            u8 has = 0;
            if (!is.read(reinterpret_cast<char*>(&has), 1)) return false;
            const i64 want = params[i]->numel();
            if (has) {
                u64 ne = 0;
                if (!is.read(reinterpret_cast<char*>(&ne), 8)) return false;
                if (ne != static_cast<u64>(want)) return false;
                Tensor mc(params[i]->shape, DType::F32, Device::CPU);
                if (!is.read(reinterpret_cast<char*>(mc.data_ptr()),
                             static_cast<std::streamsize>(mc.nbytes())))
                    return false;
                if (!m_[i].defined()) continue;
                m_[i].copy_from(mc);
            }
        }
        return true;
    }

    LionConfig saved{};
    if (!is.read(reinterpret_cast<char*>(&saved), sizeof(LionConfig))) return false;
    if (exact) {
        // [FIX P0-02] legacy path: same exactness contract.
        if (!opt_f32_same(saved.beta1, cfg_.beta1) ||
            !opt_f32_same(saved.beta2, cfg_.beta2) ||
            !opt_f32_same(saved.weight_decay, cfg_.weight_decay) ||
            !opt_f32_same(saved.grad_clip, cfg_.grad_clip)) {
            log_error("exact resume: lion hyperparams differ from legacy checkpoint");
            return false;
        }
    }
    cfg_.beta1 = saved.beta1;
    cfg_.beta2 = saved.beta2;
    for (size_t i = 0; i < m_.size(); ++i) {
        u64 ne = 0;
        if (!is.read(reinterpret_cast<char*>(&ne), 8)) return false;
        if (ne != static_cast<u64>(params[i]->numel())) return false;
        Tensor mc(params[i]->shape, DType::F32, Device::CPU);
        if (!is.read(reinterpret_cast<char*>(mc.data_ptr()), static_cast<std::streamsize>(mc.nbytes()))) return false;
        if (!m_[i].defined()) continue;
        m_[i].copy_from(mc);
    }
    return true;
}

}
