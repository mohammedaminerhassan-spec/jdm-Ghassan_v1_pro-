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
};

// Full training checkpoint: weights + optimizer moments + schedule position +
// dataloader position + RNG. Resume is bit-exact for the data order.
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

    static bool load(const std::string& path,
                     Model& model,
                     AdamW* opt,          // may be null (inference / eval)
                     TrainState& state);
    static bool load(const std::string& path,
                     Model& model,
                     Lion* opt,
                     TrainState& state);

    // reads only the header + config, for `ghassan-ai info`
    static bool peek(const std::string& path, ModelConfig& cfg, TrainState& state);

    static std::string latest_in(const std::string& dir);
};

} // namespace gai
