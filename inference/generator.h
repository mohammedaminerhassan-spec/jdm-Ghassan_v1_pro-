#pragma once

#include "model/model.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/chat_template.h"
#include "inference/kv_cache.h"
#include "inference/sampler.h"
#include <functional>

namespace gai {

struct GenerationConfig {
    int  max_new_tokens = 256;
    int  max_context    = 0;
    std::vector<i32>         stop_tokens{special::END, special::EOS};
    std::vector<std::string> stop_strings;
    bool echo_prompt = false;
    SamplingConfig sampling{};
};

struct GenerationStats {
    int    prompt_tokens = 0;
    int    new_tokens    = 0;
    double prefill_sec   = 0;
    double decode_sec    = 0;
    double prefill_tps() const { return prefill_sec > 0 ? prompt_tokens / prefill_sec : 0; }
    double decode_tps()  const { return decode_sec  > 0 ? new_tokens / decode_sec : 0; }
    std::string summary() const;
};

class Generator {
public:
    Generator(Model& model, const Tokenizer& tok, int max_context = 0);

    using StreamFn = std::function<bool(const std::string& piece, i32 token)>;

    void reset();

    void prefill(const std::vector<i32>& tokens, int front_drop = 0);

    std::vector<i32> generate(const GenerationConfig& cfg, StreamFn on_token = nullptr);

    std::string complete(const std::string& prompt, const GenerationConfig& cfg,
                         StreamFn on_token = nullptr);

    std::string chat(const std::vector<Message>& msgs, const GenerationConfig& cfg,
                     StreamFn on_token = nullptr);

    double score_tokens(const std::vector<i32>& tokens, const std::vector<u8>* mask = nullptr,
                        i64* out_ntok = nullptr,
                        const std::vector<i32>* segment_ids = nullptr);

    const GenerationStats& stats() const { return stats_; }
    KVCache& cache() { return cache_; }
    int context_used() const { return cache_.length(); }

private:

    float* decode_step(i32 token, int position);

    float* decode_step_logits(i32 token, int position);
    void   forward_prefill(const std::vector<i32>& tokens, int start_pos);

    static constexpr int FAST_TOPK_MAX = 128;
    static constexpr int FAST_HIST_MAX = 2048;

    static constexpr size_t HIST_MAX = 8192;
    void push_history(i32 tok) {
        if (history_.size() >= HIST_MAX)
            history_.erase(history_.begin(),
                           history_.begin() + (history_.size() - HIST_MAX + 1));
        history_.push_back(tok);
    }

    Model&           model_;
    const Tokenizer& tok_;
    KVCache          cache_;
    int              max_context_;
    Device           expected_device_ = Device::CPU;

    int64_t          absolute_pos_ = 0;

    Tensor x_, xb_, q_, k_, v_, qkv_, attn_, proj_, gate_, up_, act_, logits_dev_, scores_;
    Tensor tok_dev_, pos_dev_;
    Tensor hlast_;
    Tensor topk_vals_dev_, topk_ids_dev_;
    std::vector<float> topk_vals_host_;
    std::vector<i32>   topk_ids_host_;

    Tensor moe_gate_, moe_up_, moe_act_;
    std::vector<float> logits_host_;
    std::vector<i32>   history_;
    GenerationStats    stats_;
};

}
