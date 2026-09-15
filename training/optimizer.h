#pragma once

#include "model/model.h"

namespace gai {

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

    // state io
    void save_state(std::ostream& os) const;
    bool load_state(std::istream& is);
    size_t state_bytes() const;

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
    bool load_state(std::istream& is);
    size_t state_bytes() const;

    const LionConfig& config() const { return cfg_; }
    void set_config(const LionConfig& c) { cfg_ = c; }

private:
    Model&     model_;
    LionConfig cfg_;
    std::vector<Tensor> m_;
    i64    t_ = 0;
    double last_grad_norm_ = 0.0;
};

} // namespace gai
