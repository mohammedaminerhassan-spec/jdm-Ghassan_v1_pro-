#pragma once

#include "model/model.h"

namespace gai {

// Optimizer state-blob versions (P2-4): v0 is the legacy layout (raw config
// struct + moments for every parameter). v1 is field-wise config + a per
// parameter presence flag, so frozen parameters store nothing and old/new
// binaries never misparse each other (the checkpoint container version
// selects the layout; see Checkpoint::load).
constexpr int OPT_STATE_LEGACY = 0;
constexpr int OPT_STATE_CURRENT = 1;

struct OptimizerStateSnapshot;

struct AdamWConfig {
    float lr           = 3e-4f;
    float beta1        = 0.9f;
    float beta2        = 0.95f;
    float eps          = 1e-8f;
    float weight_decay = 0.1f;
    float grad_clip    = 1.0f;   // 0 disables
};

// AdamW with decoupled weight decay, applied only to matmul weights
// (norms and embeddings are excluded, which is standard and matters at this scale).
class AdamW {
public:
    AdamW(Model& model, AdamWConfig cfg);

    // grad_scale multiplies gradients before the update: used for gradient
    // accumulation (1/accum_steps) and fp16 loss-scaling recovery.
    // Returns the pre-clip global gradient norm.
    double step(float lr, float grad_scale = 1.0f);

    double last_grad_norm() const { return last_grad_norm_; }
    i64    step_count() const { return t_; }
    void   set_step_count(i64 t) { t_ = t; }

    // state io (v1 layout; load_state also reads legacy v0 blobs)
    void save_state(std::ostream& os) const;
    bool load_state(std::istream& is, int state_version);
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
    float lr           = 3e-5f;   // ~10x smaller than AdamW peak (sign updates are large)
    float beta1        = 0.9f;
    float beta2        = 0.99f;
    float weight_decay = 0.1f;
    float grad_clip    = 1.0f;   // 0 disables
};

// Lion: single momentum state (m only, no v) -> ~50% optimizer memory vs AdamW.
// Ideal for T4 16GB: 467M params need ~1.9GB moments instead of ~3.7GB.
class Lion {
public:
    Lion(Model& model, LionConfig cfg);

    double step(float lr, float grad_scale = 1.0f);

    double last_grad_norm() const { return last_grad_norm_; }
    i64    step_count() const { return t_; }
    void   set_step_count(i64 t) { t_ = t; }

    void save_state(std::ostream& os) const;
    bool load_state(std::istream& is, int state_version);
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
    float lr           = 0.02f;   // Muon takes lr DIRECTLY (no 0.1x Lion rule):
                                  // orthogonal updates have ~Adam-scale steps
                                  // at this magnitude. Set 0.01-0.03; the stock
                                  // 3e-4 still learns, just slowly.
    float vec_lr_ratio = 0.1f;    // non-matrix params use lr*vec_lr_ratio
                                  // (same 0.1x convention as Lion here)
    float beta1        = 0.9f;    // momentum (matrices + vectors)
    float beta2        = 0.95f;   // AdamW-lite second moment (1D norms only)
    float eps          = 1e-8f;   // AdamW-lite (1D norms only)
    float weight_decay = 0.1f;
    float grad_clip    = 1.0f;   // 0 disables
    int   ns_steps     = 5;      // Newton-Schulz iterations (1..10)
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
};

// Muon (orthogonalized momentum, cf. Moonshot Kimi K2): 2D matmul weights
// (decay==true) update along Newton-Schulz-orthogonalized momentum — the
// largest single-optimizer speedup reported for small models (1.5-2x vs
// AdamW to the same loss). Everything else reuses proven rules: embeddings
// (2D, decay==false) take Lion-style sign momentum (m only); 1D norms take
// AdamW-lite. Memory ~= Lion (m everywhere + v on tiny norms only).
// Matrix updates are scale-normalized by construction, so loss-scaling
// recovery is automatic there. Cost: 2 small GEMMs per NS iteration per
// matrix (cuBLAS on CUDA); lower ns_steps if steps get heavy.
class Muon {
public:
    Muon(Model& model, MuonConfig cfg);

    double step(float lr, float grad_scale = 1.0f);

    double last_grad_norm() const { return last_grad_norm_; }
    i64    step_count() const { return t_; }
    void   set_step_count(i64 t) { t_ = t; }

    void save_state(std::ostream& os) const;
    bool load_state(std::istream& is, int state_version);
    size_t state_bytes() const;
    OptimizerStateSnapshot snapshot_state() const;

    const MuonConfig& config() const { return cfg_; }
    void set_config(const MuonConfig& c) { cfg_ = c; }

    // Newton-Schulz orthogonalization of G (rows x cols, fp32) into O.
    // Public for unit testing; uses the persistent scratch workspace.
    void orthogonalize(const float* G, float* O, int rows, int cols);

private:
    Model&     model_;
    MuonConfig cfg_;
    std::vector<Tensor> m_;   // momentum for every non-frozen param
    std::vector<Tensor> v_;   // AdamW-lite v for 1D norms only (undefined else)
    Tensor scratch_;          // persistent NS workspace [O|T|A] (device)
    i64 omax_rc_ = 0;         // max rows*cols over trainable matrices
    i64 amax_cc_ = 0;         // max cols*cols over trainable matrices
    i64    t_ = 0;
    double last_grad_norm_ = 0.0;
};

} // namespace gai
