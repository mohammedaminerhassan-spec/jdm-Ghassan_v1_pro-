#pragma once

#include "model/model.h"

namespace gai {

constexpr int OPT_STATE_LEGACY = 0;
constexpr int OPT_STATE_CURRENT = 1;

struct OptimizerStateSnapshot;

struct AdamWConfig {
    float lr           = 3e-4f;
    float beta1        = 0.9f;
    float beta2        = 0.95f;
    float eps          = 1e-8f;
    float weight_decay = 0.1f;
    float grad_clip    = 1.0f;
};

class AdamW {
public:
    AdamW(Model& model, AdamWConfig cfg);

    double step(float lr, float grad_scale = 1.0f);

    double last_grad_norm() const { return last_grad_norm_; }
    i64    step_count() const { return t_; }
    void   set_step_count(i64 t) { t_ = t; }

    void save_state(std::ostream& os) const;
    // [FIX P0-02] exact=true validates ALL behavior-affecting hyperparams
    // saved in the checkpoint against the current cfg_ and returns false on
    // any mismatch (exact resume must never silently continue with a
    // different update rule). exact=false preserves the legacy migrate
    // behavior (restore moments + beta/eps only, keep current wd/clip).
    bool load_state(std::istream& is, int state_version, bool exact = false);
    size_t state_bytes() const;
    OptimizerStateSnapshot snapshot_state() const;

    const AdamWConfig& config() const { return cfg_; }
    void set_config(const AdamWConfig& c) { cfg_ = c; }

private:
    Model&      model_;
    AdamWConfig cfg_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
    i64    t_ = 0;
    double last_grad_norm_ = 0.0;
};

struct LionConfig {
    float lr           = 3e-5f;
    float beta1        = 0.9f;
    float beta2        = 0.99f;
    float weight_decay = 0.1f;
    float grad_clip    = 1.0f;
};

class Lion {
public:
    Lion(Model& model, LionConfig cfg);

    double step(float lr, float grad_scale = 1.0f);

    double last_grad_norm() const { return last_grad_norm_; }
    i64    step_count() const { return t_; }
    void   set_step_count(i64 t) { t_ = t; }

    void save_state(std::ostream& os) const;
    // [FIX P0-02] see AdamW::load_state.
    bool load_state(std::istream& is, int state_version, bool exact = false);
    size_t state_bytes() const;
    OptimizerStateSnapshot snapshot_state() const;

    const LionConfig& config() const { return cfg_; }
    void set_config(const LionConfig& c) { cfg_ = c; }

private:
    Model&     model_;
    LionConfig cfg_;
    std::vector<Tensor> m_;
    i64    t_ = 0;
    double last_grad_norm_ = 0.0;
};

struct MuonConfig {
    float lr           = 0.02f;

    float vec_lr_ratio = 0.1f;

    float beta1        = 0.9f;
    float beta2        = 0.95f;
    float eps          = 1e-8f;
    float weight_decay = 0.1f;
    float grad_clip    = 1.0f;
    int   ns_steps     = 5;

    int   min_ns_dim   = 0;
};

enum class OptimizerSnapshotKind : u8 { None = 0, AdamW = 1, Lion = 2, Muon = 3 };

struct OptimizerStateSnapshot {
    OptimizerSnapshotKind kind = OptimizerSnapshotKind::None;
    i64 step = 0;
    AdamWConfig adamw{};
    LionConfig lion{};
    MuonConfig muon{};
    std::vector<Tensor> first;
    std::vector<Tensor> second;

    size_t bytes() const {
        size_t b = 0;
        for (const auto& t : first) if (t.defined()) b += t.nbytes();
        for (const auto& t : second) if (t.defined()) b += t.nbytes();
        return b;
    }
};

class Muon {
public:
    Muon(Model& model, MuonConfig cfg);

    double step(float lr, float grad_scale = 1.0f);

    double last_grad_norm() const { return last_grad_norm_; }
    i64    step_count() const { return t_; }
    void   set_step_count(i64 t) { t_ = t; }

    void save_state(std::ostream& os) const;
    // [FIX P0-02] see AdamW::load_state. Also validates vec_lr_ratio,
    // ns_steps and min_ns_dim (the latter stored as fmt=2 trailer).
    bool load_state(std::istream& is, int state_version, bool exact = false);
    size_t state_bytes() const;
    OptimizerStateSnapshot snapshot_state() const;

    const MuonConfig& config() const { return cfg_; }

    void set_config(const MuonConfig& c) {
        GAI_CHECK(c.min_ns_dim == cfg_.min_ns_dim,
                  "Muon::set_config cannot change min_ns_dim after construction "
                  "(rebuild the optimizer instead)");
        cfg_ = c;
    }

    void orthogonalize(const float* G, float* O, int rows, int cols);

private:

    bool uses_ns(const Parameter* p) const {
        if (!p || p->frozen) return false;
        if (p->shape.size() != 2 || !p->decay) return false;
        if (cfg_.min_ns_dim <= 0) return true;
        const i64 r = p->shape[0], c = p->shape[1];
        return (r < c ? r : c) >= static_cast<i64>(cfg_.min_ns_dim);
    }

    Model&     model_;
    MuonConfig cfg_;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
    Tensor scratch_;
    i64 omax_rc_ = 0;
    i64 amax_cc_ = 0;
    i64    t_ = 0;
    double last_grad_norm_ = 0.0;
};

}
