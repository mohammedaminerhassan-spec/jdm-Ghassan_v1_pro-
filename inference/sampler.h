#pragma once

#include "core/common.h"
#include "core/rng.h"
#include <vector>

namespace gai {

struct SamplingConfig {
    float temperature       = 0.8f;
    int   top_k             = 40;      // 0 disables
    float top_p             = 0.92f;   // 1.0 disables
    float min_p             = 0.05f;   // 0 disables; robust alternative to top_p
    float repetition_penalty= 1.12f;   // 1.0 disables
    int   repetition_window = 128;
    float frequency_penalty = 0.0f;
    float presence_penalty  = 0.0f;
    int   no_repeat_ngram   = 0;       // 0=off; 2..8 bans tokens completing a seen n-gram (Darija anti-loop)
    bool  greedy            = false;
    bool  gpu_fast_sample   = true;    // GPU penalties+top-k, ~1KB D2H/token
    u64   seed              = 0;       // 0 = random

    static SamplingConfig from_config(const class Config& c, const std::string& prefix = "sampling");
    // PRO-HARDEN: يصلح القيم الشاذة القادمة من YAML/CLI قبل أن تسمم التوليد:
    // temp=NaN/Inf، top_p=0 (كان يعطل nucleus خطأ)، top_k>V، penalties سالبة،
    // seed=0 تعني عشوائي. يستدعى من Sampler ctor + generate().
    void validate();
};

// Applies penalties + filtering + sampling to a logits row. Stateless apart from
// the RNG, so a Sampler can be reused across sessions.
class Sampler {
public:
    explicit Sampler(SamplingConfig cfg);

    void set_config(const SamplingConfig& c) { cfg_ = c; }
    const SamplingConfig& config() const { return cfg_; }
    void reseed(u64 seed) { rng_.seed_with(seed); }

    // `history` is the full generated context, used for repetition penalties.
    // `logits` is modified in place.
    i32 sample(float* logits, int vocab, const std::vector<i32>& history);

    // Fast path: candidates are pre-selected device-side (top-K, penalties
    // already applied there). `vals`/`ids` hold K raw (unscaled) logits.
    // The softmax/min-p/top-p/draw tail is SHARED with sample(), so given
    // identical candidate sets both paths draw bit-identical tokens.
    i32 sample_candidates(const float* vals, const i32* ids, int K);

    // Utility used by evaluation: log-probability of a specific token.
    static double log_prob(const float* logits, int vocab, i32 token);

private:
    void apply_penalties(float* logits, int vocab, const std::vector<i32>& history) const;
    // Shared draw tail: scratch_ holds (logit, id) candidates, sorted desc.
    i32 draw_from_scratch();

    SamplingConfig cfg_;
    Rng rng_;
    std::vector<std::pair<float, i32>> scratch_;
    // PRO-HARDEN: إعادة استعمال buffer الاحتمالات بدل malloc لكل توكن
    // (كان vector<double> probs(size) ~256KB عند V=32k لكل توكن).
    std::vector<double> probs_buf_;
};

} // namespace gai
