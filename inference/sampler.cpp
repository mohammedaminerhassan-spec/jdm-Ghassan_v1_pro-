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
    s.no_repeat_ngram    = static_cast<int>(c.get_int(key("no_repeat_ngram"), s.no_repeat_ngram));
    s.greedy             = c.get_bool(key("greedy"), s.greedy);
    s.gpu_fast_sample    = c.get_bool(key("gpu_fast_sample"), s.gpu_fast_sample);
    s.seed               = static_cast<u64>(c.get_int(key("seed"), static_cast<i64>(s.seed)));
    return s;
}

Sampler::Sampler(SamplingConfig cfg) : cfg_(cfg) {
    cfg_.validate();
    rng_.seed_with(cfg.seed != 0 ? cfg.seed
                                 : static_cast<u64>(std::chrono::steady_clock::now()
                                       .time_since_epoch().count()));
}

void SamplingConfig::validate() {
    // temperature: NaN/Inf/negative -> greedy آمن؛ 0 يعني greedy أصلا.
    if (!std::isfinite(temperature) || temperature < 0.0f) temperature = 0.0f;
    if (temperature > 5.0f) temperature = 5.0f;  // يمنع توزيع مسطح مدمر
    // top_p: خارج (0,1] يعني معطل؛ 0 كان يعطل خطأ فيجب أن يعني توكن واحد؟ لا:
    // القاعدة الصحيحة top_p<=0 أو >=1 = معطل (full set). نطبع القيم الشاذة.
    if (!std::isfinite(top_p) || top_p < 0.0f) top_p = 1.0f;
    if (top_p > 1.0f) top_p = 1.0f;
    if (!std::isfinite(min_p) || min_p < 0.0f) min_p = 0.0f;
    if (min_p >= 1.0f) min_p = 0.0f;
    if (top_k < 0) top_k = 0;
    if (top_k == 1) greedy = true;  // top_k=1 هو greedy ضمنيا
    // penalties: قيم سالبة أو ضخمة تقلب الإشارة/تسمم logits.
    if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0.0f)
        repetition_penalty = 1.0f;
    if (repetition_penalty > 5.0f) repetition_penalty = 5.0f;
    if (!std::isfinite(frequency_penalty)) frequency_penalty = 0.0f;
    if (!std::isfinite(presence_penalty)) presence_penalty = 0.0f;
    if (frequency_penalty < -5.0f) frequency_penalty = -5.0f;
    if (frequency_penalty > 5.0f) frequency_penalty = 5.0f;
    if (presence_penalty < -5.0f) presence_penalty = -5.0f;
    if (presence_penalty > 5.0f) presence_penalty = 5.0f;
    if (repetition_window < 0) repetition_window = 0;
    if (repetition_window > 8192) repetition_window = 8192;
    // no_repeat_ngram: 0=off; clamp to [0,8] (larger n rarely helps, costs scan).
    if (no_repeat_ngram < 0) no_repeat_ngram = 0;
    if (no_repeat_ngram == 1) no_repeat_ngram = 2;
    if (no_repeat_ngram > 8) no_repeat_ngram = 8;
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

    // PRO-HARDEN: النسخة القديمة كانت تبني unordered_map + reserve لكل توكن
    // (hash لكل توكن يبطئ decode). ننسخ النافذة إلى buffer خيطي، نفرز، ثم
    // نطبق العقوبة على كل run — نفس الرياضيات CTRL-style بلا أي hash.
    thread_local std::vector<i32> win_buf;
    win_buf.clear();
    win_buf.reserve(window);
    for (size_t i = hist.size() - window; i < hist.size(); ++i) {
        i32 t = hist[i];
        if (t >= 0 && t < vocab) win_buf.push_back(t);
    }
    if (win_buf.empty()) return;
    std::sort(win_buf.begin(), win_buf.end());
    for (size_t i = 0; i < win_buf.size();) {
        size_t j = i + 1;
        while (j < win_buf.size() && win_buf[j] == win_buf[i]) ++j;
        const int n = static_cast<int>(j - i);
        float& l = logits[win_buf[i]];
        if (rep) {
            // CTRL-style: divide positive logits, multiply negative ones
            l = (l > 0.0f) ? l / cfg_.repetition_penalty : l * cfg_.repetition_penalty;
        }
        if (freq) l -= cfg_.frequency_penalty * static_cast<float>(n);
        if (pres) l -= cfg_.presence_penalty;
        i = j;
    }
}

static void ban_no_repeat_ngram(float* logits, int vocab,
                                  const std::vector<i32>& hist, int n) {
    if (n <= 0 || vocab <= 0 || hist.size() < static_cast<size_t>(n - 1)) return;
    // Prefix = last n-1 tokens; ban any token that previously followed it.
    const size_t m = hist.size();
    const size_t pre = static_cast<size_t>(n - 1);
    for (size_t i = 0; i + static_cast<size_t>(n) <= m; ++i) {
        bool match = true;
        for (size_t k = 0; k < pre; ++k) {
            if (hist[i + k] != hist[m - pre + k]) { match = false; break; }
        }
        if (!match) continue;
        i32 nxt = hist[i + pre];
        if (nxt >= 0 && nxt < vocab) logits[nxt] = -1e30f;
    }
}

i32 Sampler::sample(float* logits, int vocab, const std::vector<i32>& history) {
    GAI_CHECK(vocab > 0, "empty vocabulary");
    apply_penalties(logits, vocab, history);
    if (cfg_.no_repeat_ngram > 0) ban_no_repeat_ngram(logits, vocab, history, cfg_.no_repeat_ngram);

    if (cfg_.greedy || cfg_.temperature <= 0.0f) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }

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

    return draw_from_scratch();
}

i32 Sampler::sample_candidates(const float* vals, const i32* ids, int K) {
    GAI_CHECK(K > 0 && vals && ids, "sample_candidates: empty set");
    // Penalties were already applied device-side; rebuild the sorted set.
    scratch_.clear();
    scratch_.reserve(static_cast<size_t>(K));
    for (int i = 0; i < K; ++i) scratch_.emplace_back(vals[i], ids[i]);
    std::sort(scratch_.begin(), scratch_.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    return draw_from_scratch();
}

i32 Sampler::draw_from_scratch() {
    GAI_CHECK(!scratch_.empty(), "sampler: empty candidate set");
    // top_k أكبر من المفردات كان يقلص بصمت؛ now clamp صريح قبل softmax.
    const float inv_t = 1.0f / cfg_.temperature;
    // softmax over the candidate set (temperature applied here)
    float mx = scratch_[0].first;
    double sum = 0.0;
    // PRO-HARDEN: إعادة استعمال probs_buf_ العضو بدل vector<double>(size)
    // جديد لكل توكن (256KB malloc عند V=32k × آلاف التوكنات = تهش RAM).
    probs_buf_.resize(scratch_.size());
    double* probs = probs_buf_.data();
    for (size_t i = 0; i < scratch_.size(); ++i) {
        double p = std::exp(static_cast<double>((scratch_[i].first - mx) * inv_t));
        probs[i] = p;
        sum += p;
    }
    // guard: كل logits -inf (قناع عدواني) -> sum=0/NaN. نرجع argmax بدل NaN.
    if (!std::isfinite(sum) || sum <= 0.0) return scratch_[0].second;
    for (size_t i = 0; i < scratch_.size(); ++i) probs[i] /= sum;

    // min-p: drop everything below min_p * p_max (applied before top-p; it is far
    // more stable than top-p alone on a peaked small-model distribution)
    const size_t nprob = scratch_.size();
    size_t limit = nprob;
    if (cfg_.min_p > 0.0f) {
        double thresh = cfg_.min_p * probs[0];
        size_t n = 0;
        while (n < nprob && probs[n] >= thresh) ++n;
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
