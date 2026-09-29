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

    double loss_scale  = 65536.0;
    int    clean_steps = 0;
    int    tok_vocab   = 0;

    u64    tok_fingerprint = 0;

    DataLoader::State val_loader{};

    i64    sched_total = 0;
    i64    sched_warmup = 0;
    float  sched_peak = 0.0f;
    float  sched_min_ratio = 0.1f;
    float  sched_decay_frac = 0.2f;
    int    sched_kind = 0;
    int    ddp_world = 1;
};

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

size_t unique_snapshot_bytes(const std::vector<std::shared_ptr<CheckpointSnapshot>>& refs);

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

    static bool load(const std::string& path,
                     Model& model,
                     AdamW* opt,
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

    static bool peek(const std::string& path, ModelConfig& cfg, TrainState& state);

    static std::string latest_in(const std::string& dir);
};

}
