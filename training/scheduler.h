#pragma once

#include "core/common.h"
#include <cmath>
#include <string>

namespace gai {

// Linear warmup -> cosine decay to min_lr_ratio * peak (default).
// WSD (warmup-stable-decay): warmup -> constant peak -> linear/cooldown decay.
// WSD is preferred for long T4 runs: you can extend total steps without
// retuning the cosine shape; just keep training in the stable phase.
class LrScheduler {
public:
    LrScheduler() = default;
    LrScheduler(float peak_lr, i64 warmup_steps, i64 total_steps, float min_ratio = 0.1f,
                const std::string& type = "cosine", float decay_frac = 0.2f)
        : peak_(peak_lr), warmup_(warmup_steps), total_(total_steps),
          min_ratio_(min_ratio), type_(type), decay_frac_(decay_frac) {}

    float lr_at(i64 step) const {
        if (total_ <= 0) return peak_;
        if (warmup_ > 0 && step < warmup_) {
            // start at 1/warmup of peak rather than exactly 0 so step 0 makes progress
            return peak_ * (static_cast<float>(step + 1) / static_cast<float>(warmup_));
        }
        if (type_ == "wsd") {
            i64 stable_end = total_ - static_cast<i64>(decay_frac_ * static_cast<float>(total_ - warmup_));
            if (stable_end < warmup_) stable_end = warmup_;
            if (step < stable_end) return peak_;   // stable phase
            i64 decay_steps = total_ - stable_end;
            if (decay_steps <= 0) return peak_;
            float p = static_cast<float>(step - stable_end) / static_cast<float>(decay_steps);
            if (p > 1.0f) p = 1.0f;
            // linear decay to min_ratio (stable + predictable on T4 restarts)
            return peak_ * (1.0f - (1.0f - min_ratio_) * p);
        }
        i64 decay_steps = total_ - warmup_;
        if (decay_steps <= 0) return peak_;
        i64 t = step - warmup_;
        if (t > decay_steps) t = decay_steps;
        float progress = static_cast<float>(t) / static_cast<float>(decay_steps);
        float cos_out = 0.5f * (1.0f + std::cos(3.14159265358979f * progress));
        return peak_ * (min_ratio_ + (1.0f - min_ratio_) * cos_out);
    }

    float peak() const { return peak_; }
    i64   warmup() const { return warmup_; }
    i64   total() const { return total_; }
    float min_ratio() const { return min_ratio_; }
    const std::string& type() const { return type_; }

private:
    float peak_      = 3e-4f;
    i64   warmup_    = 2000;
    i64   total_     = 20000;
    float min_ratio_ = 0.1f;
    std::string type_ = "cosine";  // cosine | wsd
    float decay_frac_ = 0.2f;      // wsd: last 20% decays
};

} // namespace gai
