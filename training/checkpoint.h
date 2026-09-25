#pragma once

#include "model/model.h"
#include "training/optimizer.h"
#include "training/dataloader.h"

namespace gai {

struct TrainState {
    i64    step        = 0;
    i64    tokens_seen = 0;
    double best_val    = 1e30;
    double last_loss   = 0.0;
    u64    seed        = 42;
    DataLoader::State loader{};
    // v3 additions: without these a resume silently changes training dynamics
    double loss_scale  = 65536.0;  // dynamic loss-scaler value
    int    clean_steps = 0;        // scaler clean-step counter
    int    tok_vocab   = 0;        // tokenizer vocab at save time (0 = unknown)
    // v8 additions: scheduler snapshot so resume with a different
    // max_steps/epochs/warmup/lr cannot silently reshape PAST lr_at(N).
    i64    sched_total = 0;
    i64    sched_warmup = 0;
    float  sched_peak = 0.0f;
    float  sched_min_ratio = 0.1f;
    float  sched_decay_frac = 0.2f;
    int    sched_kind = 0;  // 0=cosine, 1=wsd
    int    ddp_world = 1;   // world_size at save (DDP resume must match)
};

// Full training checkpoint: weights + optimizer moments + schedule position +
// dataloader position + RNG. Resume is bit-exact for the data order.
struct CheckpointSnapshot {
    ModelConfig config;
    TrainState state;
    std::vector<Tensor> weights;
    std::vector<std::string> parameter_names;
    std::vector<std::vector<float>> moe_bias;
    OptimizerStateSnapshot optimizer;

    size_t bytes() const {
        size_t b = 0;
        for (const auto& t : weights) if (t.defined()) b += t.nbytes();
        b += optimizer.bytes();
        return b;
    }
};

class Checkpoint {
public:
    static void save(const std::string& path,
                     const Model& model,
                     const AdamW& opt,
                     const TrainState& state);
    static void save(const std::string& path,
                     const Model& model,
                     const Lion& opt,
                     const TrainState& state);
    static void save(const std::string& path,
                     const Model& model,
                     const Muon& opt,
                     const TrainState& state);
    static CheckpointSnapshot capture(const Model& model,
                                      const AdamW& opt,
                                      const TrainState& state);
    static CheckpointSnapshot capture(const Model& model,
                                      const Lion& opt,
                                      const TrainState& state);
    static CheckpointSnapshot capture(const Model& model,
                                      const Muon& opt,
                                      const TrainState& state);
    static void save(const CheckpointSnapshot& snapshot, const std::string& path);

    // out_moments_restored (optional): true iff optimizer moments were
    // actually restored. False on kind mismatch / corrupt blob / legacy /
    // weights-only (opt==null). Callers MUST reset bias-correction t_=0 when
    // false, else resumed steps are ~10x too small (m≈(1-b)*g, bc≈1).
    //
    // F-08 `resume_mode`: strict=true implements `resume_mode: exact`. It turns
    // every "warn and continue" recovery path into a hard failure (parameter
    // missing from the file, optimizer kind mismatch, unreadable moments,
    // moments present but no optimizer supplied), so an exact resume can never
    // silently run with fresh-init parameters or fresh optimizer moments.
    // strict=false is the historical `migrate` behavior.
    static bool load(const std::string& path,
                     Model& model,
                     AdamW* opt,          // may be null (inference / eval)
                     TrainState& state,
                     bool* out_moments_restored = nullptr,
                     bool strict = false);
    static bool load(const std::string& path,
                     Model& model,
                     Lion* opt,
                     TrainState& state,
                     bool* out_moments_restored = nullptr,
                     bool strict = false);
    static bool load(const std::string& path,
                     Model& model,
                     Muon* opt,
                     TrainState& state,
                     bool* out_moments_restored = nullptr,
                     bool strict = false);

    // reads only the header + config, for `ghassan-ai info`
    static bool peek(const std::string& path, ModelConfig& cfg, TrainState& state);

    static std::string latest_in(const std::string& dir);
};

} // namespace gai
