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
    int  max_context    = 0;         // 0 = model max_seq_len
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

// Autoregressive generator: prefill in one batched forward, then single-token
// decode against the KV cache.
class Generator {
public:
    Generator(Model& model, const Tokenizer& tok, int max_context = 0);

    using StreamFn = std::function<bool(const std::string& piece, i32 token)>;   // return false to stop

    void reset();

    // Feeds tokens into the cache without sampling (prompt processing).
    void prefill(const std::vector<i32>& tokens);

    // Continues from the current cache state.
    std::vector<i32> generate(const GenerationConfig& cfg, StreamFn on_token = nullptr);

    // Convenience: full prompt -> text
    std::string complete(const std::string& prompt, const GenerationConfig& cfg,
                         StreamFn on_token = nullptr);

    std::string chat(const std::vector<Message>& msgs, const GenerationConfig& cfg,
                     StreamFn on_token = nullptr);

    // Teacher-forced scoring, used by the evaluation harness.
    double score_tokens(const std::vector<i32>& tokens, const std::vector<u8>* mask = nullptr,
                        i64* out_ntok = nullptr);

    const GenerationStats& stats() const { return stats_; }
    KVCache& cache() { return cache_; }
    int context_used() const { return cache_.length(); }

private:
    // one decode step; returns pointer to HOST logits row for sampling
    float* decode_step(i32 token, int position);
    // same, but stops at the DEVICE logits (no D2H); fast-sampling path
    // applies GPU penalties + top-k/argmax on top of this.
    float* decode_step_logits(i32 token, int position);
    void   forward_prefill(const std::vector<i32>& tokens, int start_pos);

    // fast-sampling scratch (audit P1: ~1KB D2H/token instead of 128KB)
    static constexpr int FAST_TOPK_MAX = 128;
    static constexpr int FAST_HIST_MAX = 2048;
    // PRO-HARDEN: history_ كان ينمو بلا حدود في المحادثات الطويلة (leak منطقي
    // يرفع RAM ويبطئ penalties). السقف 4096 يحفظ نافذة repetition كاملة.
    static constexpr size_t HIST_MAX = 4096;
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

    // P0-03 FIX: Absolute token position counter. After KV evict_front,
    // cache_.length() shrinks, but the true RoPE coordinate of the next
    // token must keep increasing monotonically. Using cache_.length() as
    // position after eviction corrupts the relative positional distances
    // between retained cached keys and the new query.
    int64_t          absolute_pos_ = 0;

    // decode scratch (single token) on the MODEL device (CPU or CUDA).
    // Using Tensors (not std::vector) fixes the latent CUDA bug where host
    // pointers were passed to device kernels. Host staging is only for the
    // token id, position, and final logits row consumed by the sampler.
    Tensor x_, xb_, q_, k_, v_, attn_, proj_, gate_, up_, act_, logits_dev_, scores_;
    Tensor tok_dev_, pos_dev_;
    Tensor hlast_;   // [d] last-row norm scratch for prefill head (audit P1: no [P,V])
    Tensor topk_vals_dev_, topk_ids_dev_;   // [FAST_TOPK_MAX] device candidates
    std::vector<float> topk_vals_host_;     // [FAST_TOPK_MAX] copied per token
    std::vector<i32>   topk_ids_host_;
    // MoE decode scratch: [K, E] slot buffers
    Tensor moe_gate_, moe_up_, moe_act_;
    std::vector<float> logits_host_;   // [V] sampled on CPU
    std::vector<i32>   history_;
    GenerationStats    stats_;
};

} // namespace gai
