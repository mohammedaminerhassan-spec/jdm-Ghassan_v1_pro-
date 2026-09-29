#pragma once

#include "core/common.h"
#include "core/rng.h"
#include <vector>

namespace gai {

struct SamplingConfig {
    float temperature       = 0.8f;
    int   top_k             = 40;
    float top_p             = 0.92f;
    float min_p             = 0.05f;
    float repetition_penalty= 1.12f;
    int   repetition_window = 128;
    float frequency_penalty = 0.0f;
    float presence_penalty  = 0.0f;
    int   no_repeat_ngram   = 0;
    bool  greedy            = false;
    bool  gpu_fast_sample   = true;
    u64   seed              = 0;

    static SamplingConfig from_config(const class Config& c, const std::string& prefix = "sampling");

    void validate();
};

class Sampler {
public:
    explicit Sampler(SamplingConfig cfg);

    void set_config(const SamplingConfig& c) { cfg_ = c; }
    const SamplingConfig& config() const { return cfg_; }
    void reseed(u64 seed) { rng_.seed_with(seed); }

    i32 sample(float* logits, int vocab, const std::vector<i32>& history);

    i32 sample_candidates(const float* vals, const i32* ids, int K);

    static double log_prob(const float* logits, int vocab, i32 token);

private:
    void apply_penalties(float* logits, int vocab, const std::vector<i32>& history) const;

    i32 draw_from_scratch();

    SamplingConfig cfg_;
    Rng rng_;
    std::vector<std::pair<float, i32>> scratch_;

    std::vector<double> probs_buf_;
};

}
