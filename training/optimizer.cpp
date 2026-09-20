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

AdamW::AdamW(Model& model, AdamWConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "AdamW requires enable_grad(true)");
    m_.reserve(model_.parameters().size());
    v_.reserve(model_.parameters().size());
    // P2-3: frozen parameters never receive updates, so they get no moments
    // (undefined placeholder keeps indices aligned with model_.parameters()).
    // With freeze_embeddings this saves the full embedding m+v (196MB at
    // V=32000,d=768). Requires freeze flags set BEFORE construction.
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

    // PERF (audit #7): one kernel per parameter serializes ~200 launches.
    // Full bucketing needs a layout migration; the safe win available now is
    // threading the per-parameter updates — iterations are independent
    // (disjoint w/g/m/v tensors), so a parallel-for is exact. Muon keeps its
    // serial loop (shared NS scratch Ostage).
    // PRO-HARDEN (MSVC C3016): OpenMP loop var must be signed on Windows.
#ifdef GAI_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(params.size() > (size_t)8)
#endif
    for (long long i = 0; i < static_cast<long long>(params.size()); ++i) {
        Parameter* p = params[static_cast<size_t>(i)];
        if (!p->g.defined()) continue;
        if (p->frozen) continue;   // frozen params receive no update at all
        float wd = p->decay ? cfg_.weight_decay : 0.0f;
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

// P2-4: v1 layout is [count][t][u8 fmt=1][6 field-wise f32]
// [per param: u8 has, then ne + m + v iff has]. Field-wise (never a raw
// struct) is padding- and compiler-independent; the presence flag lets frozen
// parameters store nothing while staying stream-aligned.
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

bool AdamW::load_state(std::istream& is, int state_version) {
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
        // The recipe on disk wins for beta/eps (they define the moment
        // semantics); lr and clipping come from the live config so a run can
        // be re-tuned on resume (same policy as the legacy layout).
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
                // File holds moments we no longer want (now frozen): consume
                // and discard. Our own moments stay undefined (zero bytes).
                if (!m_[i].defined()) continue;
                m_[i].copy_from(mc);
                v_[i].copy_from(vc);
            }
            // else: file holds nothing; live moments stay as constructed.
        }
        return true;
    }
    // Legacy v0 layout: raw config struct + moments for every parameter.
    AdamWConfig saved{};
    if (!is.read(reinterpret_cast<char*>(&saved), sizeof(AdamWConfig))) return false;
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
        if (!m_[i].defined()) continue; // now frozen: consume and discard
        m_[i].copy_from(mc);
        v_[i].copy_from(vc);
    }
    return true;
}

// ---------------------------------------------------------------- Muon
// Newton-Schulz orthogonalization of G (rows x cols) into O (same shape).
// Uses the RIGHT-multiplied cubic X <- 1.5*X - 0.5*(X(X^TX)): for G = U S V^T
// this maps every singular value s -> ~1, giving the same orthogonal factor
// as the left form while work buffers stay c-by-c (small side for [out,in]
// weights). Fixes s=1 (1.5-0.5=1, enforced in the ctor) and converges for
// all s in (0, sqrt(3)); the frob-normalized input always has s<=1.
// Textbook (Higham): no magic constants, 2 GEMMs per iteration.
// All math goes through ops:: (CPU + CUDA backends). T/A live in persistent
// scratch_ regions (grown once at construction): a step performs zero allocs.
namespace {
constexpr float kNSa = 1.5f, kNSb = -0.5f;
} // namespace

void Muon::orthogonalize(const float* G, float* O, int rows, int cols) {
    Device dev = model_.device();
    const i64 rc = static_cast<i64>(rows) * cols;
    // Layout [O|T|A]: O is caller-owned output (step() passes the scratch O
    // region, tests pass their own vector); T/A are scratch-internal. O never
    // aliases T/A by construction.
    float* base = scratch_.f32();
    float* T = base + omax_rc_;
    float* A = base + omax_rc_ * 2;

    ops::copy(dev, O, G, rc);
    // Normalize: NS converges from X0 = G / ||G||_F (scale-invariant update).
    const double frob = std::sqrt(ops::global_sq_norm(dev, O, rc));
    if (!std::isfinite(frob) || frob == 0.0) return; // degenerate: keep copy
    ops::scale_inplace(dev, O, static_cast<float>(1.0 / frob), rc);

    for (int it = 0; it < cfg_.ns_steps; ++it) {
        // A = X^T X [c,c], then T = X A [r,c].
        ops::gemm(dev, true, false, cols, cols, rows, 1.0f, O, cols, O, cols, 0.0f, A, cols);
        ops::gemm(dev, false, false, rows, cols, cols, 1.0f, O, cols, A, cols, 0.0f, T, cols);
        // X = aX + bT via scale/add (no axpy primitive in ops::).
        ops::scale_inplace(dev, O, kNSa, rc);
        ops::scale_inplace(dev, T, kNSb, rc);
        ops::add_inplace(dev, O, T, rc);
    }
}

Muon::Muon(Model& model, MuonConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "Muon requires enable_grad(true)");
    if (cfg_.ns_steps < 1) cfg_.ns_steps = 1;
    if (cfg_.ns_steps > 10) cfg_.ns_steps = 10;
    // NS math check (fail fast, not silently): cubic must fix 1.0.
    {
        const double f1 = static_cast<double>(kNSa) + static_cast<double>(kNSb);
        GAI_CHECK(std::fabs(f1 - 1.0) < 1e-6, "Muon NS coefficients must fix 1.0");
    }
    m_.reserve(model_.parameters().size());
    v_.reserve(model_.parameters().size());
    i64 max_rc = 0, max_cc = 0;
    for (Parameter* p : model_.parameters()) {
        // P2-3: frozen params carry no moments (placeholders keep indices).
        if (p->frozen) {
            m_.emplace_back();
            v_.emplace_back();
            continue;
        }
        // NOTE: shape is first bound to a named local (not passed as
        // p->shape directly into the by-value factory parameter). This is
        // deliberate hardening: some GCC 14 configurations diagnose the
        // direct member-to-value-parameter vector copy under -Werror, which
        // would block the official Linux/Kaggle build for no runtime reason.
        const std::vector<i64> shape = p->shape;
        m_.push_back(Tensor::zeros(shape, DType::F32, model_.device()));
        const bool is_mat = shape.size() == 2 && p->decay;
        const bool is_vec = !is_mat && shape.size() == 1;
        if (is_vec) v_.push_back(Tensor::zeros(shape, DType::F32, model_.device()));
        else v_.emplace_back();
        if (is_mat) {
            const i64 r = p->shape[0], c = p->shape[1];
            max_rc = std::max(max_rc, r * c);
            max_cc = std::max(max_cc, c * c);
        }
    }
    // Scratch [O|T|A]: O/T need max rc, A needs max cc. step() stages the
    // momentum combine and the NS output in O; orthogonalize() uses only
    // T/A internally, so output never aliases temps. No trainable matrices
    // leave scratch undefined and orthogonalize() is then unreachable.
    omax_rc_ = max_rc;
    amax_cc_ = max_cc;
    const i64 need = max_rc * 2 + max_cc;
    if (need > 0) scratch_ = Tensor::zeros({need}, DType::F32, model_.device());
}

double Muon::step(float lr, float grad_scale) {
    ++t_;
    Device dev = model_.device();
    auto& params = model_.parameters();

    // Fused global norm (same 1-sync pattern as AdamW/Lion).
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

    // Bias corrections for the AdamW-lite vector path (mirrors AdamW::step).
    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(t_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(t_));
    const float vec_lr = lr * cfg_.vec_lr_ratio;

    // O staging for matrix updates reuses the scratch T region (free outside
    // orthogonalize, which uses all regions only within its own call).
    float* Ostage = scratch_.defined() ? scratch_.f32() : nullptr;

    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params[i];
        if (!p->g.defined()) continue;
        if (p->frozen) continue;
        const bool is_mat = p->shape.size() == 2 && p->decay;
        float wd = p->decay ? cfg_.weight_decay : 0.0f;
        if (is_mat) {
            // Momentum combine into O staging: O = (1-b1)*g, m = b1*m, m += O.
            // ARCHITECTURE (DeepSeek/Moonshot rule): Newton-Schulz normalizes
            // away ANY pre-orthogonalization scale, so grad_scale (1/ntok/dscale)
            // is intentionally NOT applied here — matrix updates are
            // batch-size invariant by construction. BUT clip_scale MUST survive:
            // scaling before NS would be erased, so it scales the orthogonal
            // update AFTER NS (lr_eff = lr * clip_scale).
            GAI_CHECK(Ostage != nullptr, "muon: missing scratch for matrix update");
            ops::copy(dev, Ostage, p->g.f32(), p->numel());
            ops::scale_inplace(dev, Ostage, 1.0f - cfg_.beta1, p->numel());
            ops::scale_inplace(dev, m_[i].f32(), cfg_.beta1, p->numel());
            ops::add_inplace(dev, m_[i].f32(), Ostage, p->numel());
            const int rows = static_cast<int>(p->shape[0]);
            const int cols = static_cast<int>(p->shape[1]);
            orthogonalize(m_[i].f32(), Ostage, rows, cols);
            // Decoupled update: w -= (lr*clip_scale)*O, then decoupled decay.
            const float lr_eff = lr * clip_scale;
            ops::scale_inplace(dev, Ostage, -lr_eff, p->numel());
            ops::add_inplace(dev, p->w.f32(), Ostage, p->numel());
            if (wd != 0.0f) ops::scale_inplace(dev, p->w.f32(), 1.0f - lr * wd, p->numel());
        } else if (p->shape.size() == 1) {
            // 1D norms: AdamW-lite with own m/v at vec_lr.
            ops::adamw_step(dev, p->w.f32(), p->g.f32(), m_[i].f32(), v_[i].f32(),
                            p->numel(), vec_lr, cfg_.beta1, cfg_.beta2, cfg_.eps, wd,
                            bc1, bc2, effective_scale);
        } else {
            // Embeddings and other 2D non-matrices: Lion-style sign momentum
            // (m only) at vec_lr — the tested rule for non-matrix params
            // (beta2 = 0.99 mirrors Lion exactly).
            ops::lion_step(dev, p->w.f32(), p->g.f32(), m_[i].f32(),
                           p->numel(), vec_lr, cfg_.beta1, 0.99f, wd, effective_scale);
        }
    }
    return gnorm;
}

size_t Muon::state_bytes() const {
    // Moments + persistent NS scratch (the scratch is live optimizer memory
    // the T4 guard must account for, though it is never checkpointed).
    size_t n = 0;
    for (const auto& t : m_) n += t.nbytes();
    for (const auto& t : v_) n += t.nbytes();
    n += scratch_.nbytes();
    return n;
}

void Muon::save_state(std::ostream& os) const {
    u64 count = static_cast<u64>(m_.size());
    os.write(reinterpret_cast<const char*>(&count), 8);
    os.write(reinterpret_cast<const char*>(&t_), 8);
    const u8 fmt = 1;
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
    }
    auto& params = model_.parameters();
    for (size_t i = 0; i < m_.size(); ++i) {
        u8 mask = 0;
        if (m_[i].defined() && !params[i]->frozen) mask |= 1u; // m
        if (v_[i].defined() && !params[i]->frozen) mask |= 2u; // v
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
    if (!dst.defined()) return true; // now frozen: consume and discard
    dst.copy_from(tmp);
    return true;
}

bool Muon::load_state(std::istream& is, int state_version) {
    if (state_version < OPT_STATE_CURRENT) return false; // Muon is new: no legacy
    u64 count = 0;
    if (!is.read(reinterpret_cast<char*>(&count), 8)) return false;
    if (count != m_.size()) return false;
    if (!is.read(reinterpret_cast<char*>(&t_), 8)) return false;
    u8 fmt = 0;
    if (!is.read(reinterpret_cast<char*>(&fmt), 1) || fmt != 1) return false;
    float lr = 0, vr = 0, b1 = 0, b2 = 0, eps = 0, wd = 0, clip = 0;
    i32 ns = 0;
    if (!rd_f32(is, lr) || !rd_f32(is, vr) || !rd_f32(is, b1) || !rd_f32(is, b2) ||
        !rd_f32(is, eps) || !rd_f32(is, wd) || !rd_f32(is, clip))
        return false;
    if (!is.read(reinterpret_cast<char*>(&ns), 4)) return false;
    // Moment semantics come from disk; step-size policy stays live (same rule
    // as AdamW/Lion: lr/wd/clip/ns are re-tunable on resume — recipe on disk
    // must NOT silently override --weight-decay on resume).
    cfg_.beta1 = b1;
    cfg_.beta2 = b2;
    cfg_.eps = eps;
    (void)wd;  // intentionally ignored: keep live weight_decay like AdamW/Lion
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

// ---------------------------------------------------------------- Lion
Lion::Lion(Model& model, LionConfig cfg) : model_(model), cfg_(cfg) {
    GAI_CHECK(model.grad_enabled(), "Lion requires enable_grad(true)");
    m_.reserve(model_.parameters().size());
    for (Parameter* p : model_.parameters()) {
        // P2-3: see AdamW ctor (frozen params carry no moments).
        if (p->frozen) {
            m_.emplace_back();
            continue;
        }
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

    // PERF: same threading as AdamW above (independent per-parameter updates).
    // PRO-HARDEN (MSVC C3016): OpenMP loop var must be signed on Windows.
#ifdef GAI_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(params.size() > (size_t)8)
#endif
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

bool Lion::load_state(std::istream& is, int state_version) {
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
                if (!m_[i].defined()) continue; // now frozen: consume, discard
                m_[i].copy_from(mc);
            }
        }
        return true;
    }
    // Legacy v0 layout (see AdamW above).
    LionConfig saved{};
    if (!is.read(reinterpret_cast<char*>(&saved), sizeof(LionConfig))) return false;
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

} // namespace gai
