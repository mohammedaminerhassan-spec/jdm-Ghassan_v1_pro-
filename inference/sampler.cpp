#include "inference/sampler.h"
#include "core/config.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace gai {

SamplingConfig SamplingConfig::from_config(const Config& c, const std::string& p) {
    SamplingConfig s;
    auto key = [&](const char* k) { return p.empty() ? std::string(k) : p + "." + k; };
    s.temperature        = c.get_f32(key("temperature"), s.temperature);
    s.top_k              = static_cast<int>(c.get_int(key("top_k"), s.top_k));
    s.top_p              = c.get_f32(key("top_p"), s.top_p);
    s.min_p              = c.get_f32(key("min_p"), s.min_p);
    s.repetition_penalty = c.get_f32(key("repetition_penalty"), s.repetition_penalty);
    s.repetition_window  = static_cast<int>(c.get_int(key("repetition_window"), s.repetition_window));
    s.frequency_penalty  = c.get_f32(key("frequency_penalty"), s.frequency_penalty);
    s.presence_penalty   = c.get_f32(key("presence_penalty"), s.presence_penalty);
    s.greedy             = c.get_bool(key("greedy"), s.greedy);
    s.seed               = static_cast<u64>(c.get_int(key("seed"), static_cast<i64>(s.seed)));
    return s;
}

Sampler::Sampler(SamplingConfig cfg) : cfg_(cfg) {
    rng_.seed_with(cfg.seed != 0 ? cfg.seed
                                 : static_cast<u64>(std::chrono::steady_clock::now()
                                       .time_since_epoch().count()));
}

double Sampler::log_prob(const float* logits, int vocab, i32 token) {
    if (token < 0 || token >= vocab) return -1e30;
    float mx = logits[0];
    for (int i = 1; i < vocab; ++i) mx = std::max(mx, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < vocab; ++i) sum += std::exp(static_cast<double>(logits[i] - mx));
    return static_cast<double>(logits[token] - mx) - std::log(sum);
}

void Sampler::apply_penalties(float* logits, int vocab, const std::vector<i32>& hist) const {
    if (hist.empty()) return;
    const bool rep  = cfg_.repetition_penalty != 1.0f;
    const bool freq = cfg_.frequency_penalty != 0.0f;
    const bool pres = cfg_.presence_penalty != 0.0f;
    if (!rep && !freq && !pres) return;

    size_t window = cfg_.repetition_window > 0
                  ? std::min(hist.size(), static_cast<size_t>(cfg_.repetition_window))
                  : hist.size();

    std::unordered_map<i32, int> counts;
    counts.reserve(window * 2);
    for (size_t i = hist.size() - window; i < hist.size(); ++i) counts[hist[i]]++;

    for (const auto& [tok, n] : counts) {
        if (tok < 0 || tok >= vocab) continue;
        float& l = logits[tok];
        if (rep) {
            // CTRL-style: divide positive logits, multiply negative ones
            l = (l > 0.0f) ? l / cfg_.repetition_penalty : l * cfg_.repetition_penalty;
        }
        if (freq) l -= cfg_.frequency_penalty * static_cast<float>(n);
        if (pres) l -= cfg_.presence_penalty;
    }
}

i32 Sampler::sample(float* logits, int vocab, const std::vector<i32>& history) {
    GAI_CHECK(vocab > 0, "empty vocabulary");
    apply_penalties(logits, vocab, history);

    if (cfg_.greedy || cfg_.temperature <= 0.0f) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

    const float inv_t = 1.0f / cfg_.temperature;

    // top-k selection (partial sort keeps this cheap at V=32000)
    int k = (cfg_.top_k > 0 && cfg_.top_k < vocab) ? cfg_.top_k : vocab;
    scratch_.clear();
    scratch_.reserve(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) scratch_.emplace_back(logits[i], i);

    if (k < vocab) {
        std::partial_sort(scratch_.begin(), scratch_.begin() + k, scratch_.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        scratch_.resize(static_cast<size_t>(k));
    } else {
        std::sort(scratch_.begin(), scratch_.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // softmax over the candidate set (temperature applied here)
    float mx = scratch_[0].first;
    double sum = 0.0;
    std::vector<double> probs(scratch_.size());
    for (size_t i = 0; i < scratch_.size(); ++i) {
        double p = std::exp(static_cast<double>((scratch_[i].first - mx) * inv_t));
        probs[i] = p;
        sum += p;
    }
    for (auto& p : probs) p /= sum;

    // min-p: drop everything below min_p * p_max (applied before top-p; it is far
    // more stable than top-p alone on a peaked small-model distribution)
    size_t limit = probs.size();
    if (cfg_.min_p > 0.0f) {
        double thresh = cfg_.min_p * probs[0];
        size_t n = 0;
        while (n < probs.size() && probs[n] >= thresh) ++n;
        limit = std::max<size_t>(1, n);
    }

    // top-p nucleus
    if (cfg_.top_p > 0.0f && cfg_.top_p < 1.0f) {
        double cum = 0.0;
        size_t n = 0;
        while (n < limit) {
            cum += probs[n];
            ++n;
            if (cum >= static_cast<double>(cfg_.top_p)) break;
        }
        limit = std::max<size_t>(1, n);
    }

    double renorm = 0.0;
    for (size_t i = 0; i < limit; ++i) renorm += probs[i];
    double r = static_cast<double>(rng_.uniform()) * renorm;
    double acc = 0.0;
    for (size_t i = 0; i < limit; ++i) {
        acc += probs[i];
        if (r <= acc) return scratch_[i].second;
    }
    return scratch_[limit - 1].second;
}

} // namespace gai
