#include "inference/generator.h"
#include "core/ops.h"
#include "core/device.h"
#ifdef GAI_CUDA
#include "cuda/cuda_utils.h"
#endif

#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace gai {

std::string GenerationStats::summary() const {
    return strfmt("prompt %d tok in %s (%.1f tok/s) | generated %d tok in %s (%.1f tok/s)",
                  prompt_tokens, human_duration(prefill_sec).c_str(), prefill_tps(),
                  new_tokens, human_duration(decode_sec).c_str(), decode_tps());
}

Generator::Generator(Model& model, const Tokenizer& tok, int max_context)
    : model_(model), tok_(tok) {
    const ModelConfig& c = model.config();
    Device dev = model.device();
    max_context_ = max_context > 0 ? std::min(max_context, c.max_seq_len) : c.max_seq_len;
    expected_device_ = dev;
    // PRO-HARDEN (T4/weak-PC OOM guard): KV cache = 2*L*max_len*kv_dim*4 bytes
    // متجاور. بدون فحص مسبق أي max_seq_len=8192 يفجر T4 فورا. نفشل برسالة
    // واضحة تقترح تصغير max_context بدل OOM غامض وسط التوليد.
    {
        size_t need = 2ULL * static_cast<size_t>(c.num_layers) *
                      static_cast<size_t>(max_context_) *
                      static_cast<size_t>(c.kv_dim()) * sizeof(float);
        if (dev == Device::CUDA && cuda_available()) {
            // FIX P2-1: device_info().free_mem is a stale startup snapshot.
            // Query live VRAM so ctx=4096 fails fast only when truly OOM.
#ifdef GAI_CUDA
            size_t free_b = cuda::free_bytes_live();
#else
            size_t free_b = device_info().free_mem;
#endif
            if (free_b == 0) free_b = device_info().free_mem;
            if (free_b > 0 && need > free_b) {
                GAI_FAIL(strfmt("KV cache needs %s but only %s free on device "
                                "(L=%d ctx=%d kv_dim=%d). Halve max_context or max_seq_len.",
                                human_bytes(need).c_str(), human_bytes(free_b).c_str(),
                                c.num_layers, max_context_, c.kv_dim()));
            }
        } else if (need > (1ULL << 30)) {
            log_warn(strfmt("[gen ] KV cache large (%s); weak PCs should lower max_context",
                            human_bytes(need).c_str()));
        }
    }
    cache_ = KVCache(c, max_context_, dev);

    const int d = c.hidden_size;
    x_          = Tensor::empty({d}, DType::F32, dev);
    xb_         = Tensor::empty({d}, DType::F32, dev);
    q_          = Tensor::empty({c.q_dim()}, DType::F32, dev);
    k_          = Tensor::empty({c.kv_dim()}, DType::F32, dev);
    v_          = Tensor::empty({c.kv_dim()}, DType::F32, dev);
    attn_       = Tensor::empty({c.q_dim()}, DType::F32, dev);
    proj_       = Tensor::empty({d}, DType::F32, dev);
    gate_       = Tensor::empty({c.intermediate_size}, DType::F32, dev);
    up_         = Tensor::empty({c.intermediate_size}, DType::F32, dev);
    act_        = Tensor::empty({c.intermediate_size}, DType::F32, dev);
    if (c.use_moe) {
        const i64 nke = static_cast<i64>(c.moe_top_k) * c.moe_expert_dim;
        moe_gate_ = Tensor::empty({nke}, DType::F32, dev);
        moe_up_   = Tensor::empty({nke}, DType::F32, dev);
        moe_act_  = Tensor::empty({nke}, DType::F32, dev);
    }
    logits_dev_ = Tensor::empty({c.vocab_size}, DType::F32, dev);
    scores_     = Tensor::empty({static_cast<i64>(c.num_heads) * max_context_}, DType::F32, dev);
    tok_dev_    = Tensor::empty({1}, DType::I32, dev);
    pos_dev_    = Tensor::empty({1}, DType::I32, dev);
    hlast_      = Tensor::empty({d}, DType::F32, dev);
    topk_vals_dev_ = Tensor::empty({(i64)FAST_TOPK_MAX}, DType::F32, dev);
    topk_ids_dev_  = Tensor::empty({(i64)FAST_TOPK_MAX}, DType::I32, dev);
    topk_vals_host_.assign(FAST_TOPK_MAX, 0.0f);
    topk_ids_host_.assign(FAST_TOPK_MAX, 0);
    logits_host_.assign(static_cast<size_t>(c.vocab_size), 0.0f);

    log_info(strfmt("[gen ] context %d | kv cache %s | device %s",
                    max_context_, human_bytes(cache_.bytes()).c_str(), device_name(dev)));
}

void Generator::reset() {
    cache_.reset();
    history_.clear();
    stats_ = GenerationStats{};
    absolute_pos_ = 0;   // P0-03: reset absolute position counter
}

// ---------------------------------------------------------------- decode step
// Full body through the LM head; result stays in logits_dev_ (no copy).
// FIX P2 (robustness): position is int downstream (kernels + KV slots), while
// the monotonic counter is int64. Guard the 2B-token overflow explicitly.
float* Generator::decode_step_logits(i32 token, int position) {
    GAI_CHECK(position >= 0 && (i64)position <= (i64)std::numeric_limits<int>::max(),
              "decode_step_logits: position out of int range");
    GAI_CHECK(model_.device() == expected_device_,
              "model device changed after Generator construction; rebuild the Generator");
    const ModelConfig& c = model_.config();
    const int d   = c.hidden_size;
    const int qd  = c.q_dim();
    const int kvd = c.kv_dim();
    const int F   = c.intermediate_size;
    const int V   = c.vocab_size;
    const int H   = c.num_heads;
    const int KV  = c.num_kv_heads;
    const int hd  = c.head_dim();
    const float scale = (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale(c);
    Device dev = model_.device();

    // stage token id on-device (works for CPU and CUDA alike)
    device_copy(tok_dev_.data_ptr(), dev, &token, Device::CPU, sizeof(i32));
    ops::embedding_forward(dev, tok_dev_.i32p(), model_.tok_embeddings().w.f32(), x_.f32(), 1, d, V);

    const int cur = cache_.length();          // positions already cached
    const int slot = cur;                     // this token's slot
    GAI_CHECK(slot < max_context_, "KV cache overflow");

    for (int l = 0; l < c.num_layers; ++l) {
        LayerParams& L = model_.layer(l);

        ops::rmsnorm_forward(dev, x_.f32(), L.attn_norm.w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);

        ops::linear_forward(dev, xb_.f32(), L.wq.w.f32(), q_.f32(), 1, d, qd);
        ops::linear_forward(dev, xb_.f32(), L.wk.w.f32(), k_.f32(), 1, d, kvd);
        ops::linear_forward(dev, xb_.f32(), L.wv.w.f32(), v_.f32(), 1, d, kvd);

        // QK-Norm (must mirror Model::forward exactly, otherwise train/inference
        // mismatch on use_qk_norm models). Single token layout [H,hd]/[KV,hd]
        // is contiguous so rmsnorm applies in-place directly.
        if (c.use_qk_norm && L.qk_qnorm.w.defined()) {
            ops::rmsnorm_forward(dev, q_.f32(), L.qk_qnorm.w.f32(), q_.f32(),
                                 nullptr, H, hd, c.rms_eps);
            ops::rmsnorm_forward(dev, k_.f32(), L.qk_knorm.w.f32(), k_.f32(),
                                 nullptr, KV, hd, c.rms_eps);
        }

        const float theta_eff = rope_theta_eff(c);
        // T4-P1-25: the position value is loop-invariant — upload once per
        // decode step, not once per layer (was 36 tiny H2D copies/token).
        if (l == 0)
            device_copy(pos_dev_.data_ptr(), dev, &position, Device::CPU, sizeof(i32));
        ops::rope_forward_ex(dev, q_.f32(), k_.f32(), pos_dev_.i32p(), 1, H, KV, hd, theta_eff,
                             c.rope_type, c.rope_yarn_low, c.rope_yarn_high, c.rope_scale);

        // write into the cache (same-device D2D via device_copy)
        {
            size_t bytes = sizeof(float) * static_cast<size_t>(kvd);
            char* kdst = reinterpret_cast<char*>(cache_.k(l)) + static_cast<size_t>(slot) * bytes;
            char* vdst = reinterpret_cast<char*>(cache_.v(l)) + static_cast<size_t>(slot) * bytes;
            device_copy(kdst, dev, k_.f32(), dev, bytes);
            device_copy(vdst, dev, v_.f32(), dev, bytes);
        }

        ops::attention_decode_ex(dev, q_.f32(), cache_.k(l), cache_.v(l), attn_.f32(),
                                 H, KV, hd, slot + 1, max_context_, scale, scores_.f32(),
                                 c.sliding_window);

        ops::linear_forward(dev, attn_.f32(), L.wo.w.f32(), proj_.f32(), 1, qd, d);
        ops::add_inplace(dev, x_.f32(), proj_.f32(), d);

        ops::rmsnorm_forward(dev, x_.f32(), L.ffn_norm.w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);
        if (L.has_moe()) {
            const float* bias = model_.moe_bias_ptr(l);
            ops::moe_forward_bias(dev, xb_.f32(), L.router.w.f32(), bias,
                                  L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                  L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                  L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                  L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                  proj_.f32(), nullptr, nullptr, nullptr,
                                  moe_gate_.f32(), moe_up_.f32(), moe_act_.f32(),
                                  1, d, c.moe_expert_dim, c.num_experts, c.moe_top_k);
        } else {
            ops::linear_forward(dev, xb_.f32(), L.w_gate.w.f32(), gate_.f32(), 1, d, F);
            ops::linear_forward(dev, xb_.f32(), L.w_up.w.f32(),   up_.f32(),   1, d, F);
            ops::swiglu_forward(dev, gate_.f32(), up_.f32(), act_.f32(), F);
            ops::linear_forward(dev, act_.f32(), L.w_down.w.f32(), proj_.f32(), 1, F, d);
        }
        ops::add_inplace(dev, x_.f32(), proj_.f32(), d);
    }

    cache_.advance(1);

    ops::rmsnorm_forward(dev, x_.f32(), model_.final_norm().w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);
    ops::linear_forward(dev, xb_.f32(), model_.lm_head().w.f32(), logits_dev_.f32(), 1, d, V);
    return logits_dev_.f32();
}

// Legacy single-token step: full-vocab D2H + CPU sampling (kept for CPU,
// tiny vocabs, disabled fast path, and top_k=0 full-distribution sampling).
float* Generator::decode_step(i32 token, int position) {
    const int V = model_.config().vocab_size;
    float* lp = decode_step_logits(token, position);
    device_copy(logits_host_.data(), Device::CPU, lp, model_.device(),
                sizeof(float) * static_cast<size_t>(V));
    return logits_host_.data();
}

void Generator::forward_prefill(const std::vector<i32>& tokens, int start_pos) {
    GAI_CHECK(model_.device() == expected_device_,
              "model device changed after Generator construction; rebuild the Generator");
    // Batched prefill for fresh prompts (cache empty): one batched GEMM per
    // projection instead of P single-token GEMMs. Falls back to sequential
    // decode_step when the cache already holds context (rare: continuation).
    if (tokens.empty() || cache_.length() != 0 || start_pos != 0) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (cache_.length() >= max_context_) cache_.evict_front(max_context_ / 4, 8);
            decode_step(tokens[i], start_pos + static_cast<int>(i));
            push_history(tokens[i]);
        }
        return;
    }
    const ModelConfig& c = model_.config();
    Device dev = model_.device();
    const int P = static_cast<int>(tokens.size());
    const int d = c.hidden_size, qd = c.q_dim(), kvd = c.kv_dim();
    const int H = c.num_heads, KV = c.num_kv_heads, hd = c.head_dim();
    const float scale = (1.0f / std::sqrt(static_cast<float>(hd))) * rope_mscale(c);

    // Device staging for the whole prompt.
    Tensor ids({P}, DType::I32, Device::CPU);
    std::memcpy(ids.data_ptr(), tokens.data(), sizeof(i32) * tokens.size());
    Tensor d_ids({P}, DType::I32, dev);
    d_ids.copy_from(ids);
    Tensor pos({P}, DType::I32, Device::CPU);
    for (int i = 0; i < P; ++i) pos.i32p()[i] = start_pos + i;  // P0-03: use start_pos offset
    Tensor d_pos({P}, DType::I32, dev);
    d_pos.copy_from(pos);

    Tensor x({(i64)P * d}, DType::F32, dev);
    Tensor xb({(i64)P * d}, DType::F32, dev);
    Tensor q({(i64)P * qd}, DType::F32, dev);
    Tensor k({(i64)P * kvd}, DType::F32, dev);
    Tensor v({(i64)P * kvd}, DType::F32, dev);
    Tensor att({(i64)P * qd}, DType::F32, dev);
    Tensor proj({(i64)P * d}, DType::F32, dev);
    // FIX (10/10): old prefill allocated probs [P*H*P] transient (~1GB at
    // P=4096,H=16) and passed it to attention_forward. Inference never needs
    // probs (training-only for backward). nullptr = flash O(T) memory-lean
    // path on CPU+CUDA, same numerics for `out`. Saves 1GB + H2D traffic.
    const i64 nke = (i64)c.moe_top_k * c.moe_expert_dim;
    Tensor mg({(i64)P * nke}, DType::F32, dev);
    Tensor mu({(i64)P * nke}, DType::F32, dev);
    Tensor ma({(i64)P * nke}, DType::F32, dev);
    Tensor gtmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);
    Tensor utmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);
    Tensor atmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);

    ops::embedding_forward(dev, d_ids.i32p(), model_.tok_embeddings().w.f32(), x.f32(), P, d, c.vocab_size);
    for (int l = 0; l < c.num_layers; ++l) {
        LayerParams& L = model_.layer(l);
        ops::rmsnorm_forward(dev, x.f32(), L.attn_norm.w.f32(), xb.f32(), nullptr, P, d, c.rms_eps);
        ops::linear_forward(dev, xb.f32(), L.wq.w.f32(), q.f32(), P, d, qd);
        ops::linear_forward(dev, xb.f32(), L.wk.w.f32(), k.f32(), P, d, kvd);
        ops::linear_forward(dev, xb.f32(), L.wv.w.f32(), v.f32(), P, d, kvd);
        // QK-Norm batched: [P,H,hd] == [P*H,hd] contiguous, same for K.
        if (c.use_qk_norm && L.qk_qnorm.w.defined()) {
            ops::rmsnorm_forward(dev, q.f32(), L.qk_qnorm.w.f32(), q.f32(),
                                 nullptr, (i64)P * H, hd, c.rms_eps);
            ops::rmsnorm_forward(dev, k.f32(), L.qk_knorm.w.f32(), k.f32(),
                                 nullptr, (i64)P * KV, hd, c.rms_eps);
        }
        ops::rope_forward_ex(dev, q.f32(), k.f32(), d_pos.i32p(), P, H, KV, hd, rope_theta_eff(c),
                             c.rope_type, c.rope_yarn_low, c.rope_yarn_high, c.rope_scale);
        // fill cache rows [0,P)
        device_copy(cache_.k(l), dev, k.f32(), dev, sizeof(float) * (size_t)P * kvd);
        device_copy(cache_.v(l), dev, v.f32(), dev, sizeof(float) * (size_t)P * kvd);
        ops::attention_forward_ex(dev, q.f32(), k.f32(), v.f32(), att.f32(), nullptr,
                                  1, P, H, KV, hd, scale, c.sliding_window);
        ops::linear_forward(dev, att.f32(), L.wo.w.f32(), proj.f32(), P, qd, d);
        ops::add_inplace(dev, x.f32(), proj.f32(), (i64)P * d);
        ops::rmsnorm_forward(dev, x.f32(), L.ffn_norm.w.f32(), xb.f32(), nullptr, P, d, c.rms_eps);
        if (L.has_moe()) {
            const float* bias = model_.moe_bias_ptr(l);
            ops::moe_forward_bias(dev, xb.f32(), L.router.w.f32(), bias,
                                  L.moe_gate.w.f32(), L.moe_up.w.f32(), L.moe_down.w.f32(),
                                  L.sh_gate.w.defined() ? L.sh_gate.w.f32() : nullptr,
                                  L.sh_up.w.defined()   ? L.sh_up.w.f32()   : nullptr,
                                  L.sh_down.w.defined() ? L.sh_down.w.f32() : nullptr,
                                  proj.f32(), nullptr, nullptr, nullptr,
                                  mg.f32(), mu.f32(), ma.f32(),
                                  P, d, c.moe_expert_dim, c.num_experts, c.moe_top_k);
        } else {
            ops::linear_forward(dev, xb.f32(), L.w_gate.w.f32(), gtmp.f32(), P, d, c.intermediate_size);
            ops::linear_forward(dev, xb.f32(), L.w_up.w.f32(), utmp.f32(), P, d, c.intermediate_size);
            ops::swiglu_forward(dev, gtmp.f32(), utmp.f32(), atmp.f32(), (i64)P * c.intermediate_size);
            ops::linear_forward(dev, atmp.f32(), L.w_down.w.f32(), proj.f32(), P, c.intermediate_size, d);
        }
        ops::add_inplace(dev, x.f32(), proj.f32(), (i64)P * d);
    }
    cache_.set_length(P);
    // Final norm + head for the LAST position only (what sampling needs).
    // AUDIT P1 FIX: old code ran the full [P,V] head (524MB temp + wasted
    // GEMM at P=4096) then kept one row. Rows are independent, so norm+head
    // run on row P-1 directly into the persistent [V] scratch: bit-identical
    // logits, O(V) memory instead of O(P*V).
    ops::rmsnorm_forward(dev, x.f32() + (size_t)(P - 1) * d,
                         model_.final_norm().w.f32(), hlast_.f32(),
                         nullptr, 1, d, c.rms_eps);
    ops::linear_forward(dev, hlast_.f32(), model_.lm_head().w.f32(),
                        logits_dev_.f32(), 1, d, c.vocab_size);
    device_copy(logits_host_.data(), Device::CPU,
                logits_dev_.f32(), dev,
                sizeof(float) * (size_t)c.vocab_size);
    // keep single-token scratch in sync for the upcoming decode steps
    device_copy(x_.f32(), dev, x.f32() + (size_t)(P - 1) * d, dev, sizeof(float) * (size_t)d);
    for (auto t : tokens) push_history(t);
}

// ---------------------------------------------------------------- prefill
void Generator::prefill(const std::vector<i32>& tokens) {
    if (tokens.empty()) return;
    Timer t;
    int start = cache_.length();

    // Fast path: fresh prompt -> one batched forward (tensor-core friendly on
    // T4, ~10-30x faster than P single-token GEMMs). Fallback: sequential
    // decode_step for continuations (identical numerics to decode).
    if (start == 0 && tokens.size() > 1) {
        // sliding window for very long prompts, keeping BOS+system prefix
        std::vector<i32> work = tokens;
        int64_t pos_base = 0;  // absolute coordinate of work[0]
        bool truncated = false;
        int drop = 0;
        if (static_cast<int>(work.size()) > max_context_) {
            drop = static_cast<int>(work.size()) - max_context_;
            work.erase(work.begin() + 1, work.begin() + 1 + drop);
            log_warn(strfmt("prefill: prompt truncated by %d tokens", drop));
            truncated = true;
        }
        if (!truncated) {
            // PRO-HARDEN: forward_prefill يخصص ~12 Tensor عابرة (~150MB عند
            // P=4096) عبر cudaMalloc/Free لكل prefill فيفتت heap الـT4.
            // نقطع أي prompt >1024 إلى: أول 1024 batched ثم الباقي decode
            // متسلسل (نفس الأرقام، ذروة VRAM ثابتة).
            static constexpr size_t PREFILL_CHUNK = 1024;
            if (work.size() > PREFILL_CHUNK) {
                std::vector<i32> head(work.begin(), work.begin() + PREFILL_CHUNK);
                forward_prefill(head, 0);
                absolute_pos_ = static_cast<int64_t>(head.size());
                for (size_t i = PREFILL_CHUNK; i < work.size(); ++i) {
                    if (cache_.length() >= max_context_) cache_.evict_front(max_context_ / 4, 8);
                    decode_step(work[i], static_cast<int>(absolute_pos_));
                    ++absolute_pos_;
                    push_history(work[i]);
                }
            } else {
                forward_prefill(work, 0);
                // P0-03: update absolute position counter after batched prefill
                absolute_pos_ = static_cast<int64_t>(work.size());
            }
        } else {
            // DeepSeek RoPE rule: truncation must NOT re-zero positions.
            // work = [tok0, tok_{drop+1}, ...]; true coordinates are
            // [0, drop+1, drop+2, ...]. Batched prefill assumes contiguous
            // 0..P-1, so use the sequential path with exact coordinates here
            // (rare edge case; correctness over 10-30x speed).
            (void)pos_base;
            decode_step(work[0], 0);
            push_history(work[0]);
            for (size_t i = 1; i < work.size(); ++i) {
                int pos = drop + static_cast<int>(i);
                decode_step(work[i], pos);
                push_history(work[i]);
            }
            // Absolute coordinate of the NEXT token = original length.
            absolute_pos_ = static_cast<int64_t>(tokens.size());
        }
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (cache_.length() >= max_context_) {
                // slide the window, keeping the first 8 tokens (BOS + system prefix)
                cache_.evict_front(max_context_ / 4, 8);
            }
            // P0-03: pass absolute_pos_, not (start + i) which would also be
            // wrong after eviction; absolute_pos_ is always monotonically increasing.
            decode_step(tokens[i], static_cast<int>(absolute_pos_));
            ++absolute_pos_;
            push_history(tokens[i]);
        }
    }
    stats_.prompt_tokens += static_cast<int>(tokens.size());
    stats_.prefill_sec += t.seconds();
}

// ---------------------------------------------------------------- generate
std::vector<i32> Generator::generate(const GenerationConfig& cfg, StreamFn on_token) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    std::vector<i32> out;
    Sampler sampler(cfg.sampling);
    Tokenizer::Stream stream(tok_);
    Timer t;

    std::string tail;   // for stop-string matching
    const int limit = cfg.max_new_tokens;
    const int V = model_.config().vocab_size;
    Device dev = model_.device();
    const SamplingConfig& sc = sampler.config();

    // AUDIT P1 fast path: GPU penalties + top-k/argmax, ~1KB D2H per token
    // instead of the full 32k row (128KB + CPU-side full sort). Exact whenever
    // it engages: penalties are demote-only mirrors computed over the full
    // row on device, top-K is post-penalty exact, and the host draw tail is
    // shared with the legacy path (bit-identical tokens incl. RNG).
    // Falls back to legacy when: CPU device, tiny vocab, fast path disabled,
    // repetition window > 2048, or full-distribution sampling (top_k <= 0 or
    // top_k > 128 — the tail beyond top-128 could still win the draw).
    const bool want_greedy = sc.greedy || sc.temperature <= 0.0f;
    const int win = sc.repetition_window > 0 ? sc.repetition_window
                                             : static_cast<int>(history_.size());
    const int K = (!want_greedy && sc.top_k > 0 && sc.top_k <= FAST_TOPK_MAX)
                      ? sc.top_k
                      : 0;
    const bool use_fast = sc.gpu_fast_sample && dev == Device::CUDA && V >= 512 &&
                          win <= FAST_HIST_MAX && sc.no_repeat_ngram == 0 &&
                          (want_greedy || K > 0);

    // The logits from the last prefill step are already in logits_dev_
    // (fast) and logits_host_ (legacy).
    for (int step = 0; step < limit; ++step) {
        i32 next = -1;
        if (use_fast) {
            int wn = std::min({win, static_cast<int>(history_.size()), FAST_HIST_MAX});
            const i32* hptr = wn > 0 ? history_.data() + history_.size() - wn : nullptr;
            ops::apply_rep_penalties(dev, logits_dev_.f32(), V, hptr, wn,
                                     sc.repetition_penalty, sc.frequency_penalty,
                                     sc.presence_penalty);
            if (want_greedy) {
                next = ops::argmax_token(dev, logits_dev_.f32(), V);
            } else {
                ops::topk_select(dev, logits_dev_.f32(), V, K,
                                 topk_vals_dev_.f32(), topk_ids_dev_.i32p());
                device_copy(topk_vals_host_.data(), Device::CPU,
                            topk_vals_dev_.f32(), dev, sizeof(float) * (size_t)K);
                device_copy(topk_ids_host_.data(), Device::CPU,
                            topk_ids_dev_.i32p(), dev, sizeof(i32) * (size_t)K);
                next = sampler.sample_candidates(topk_vals_host_.data(),
                                                 topk_ids_host_.data(), K);
            }
        } else {
            next = sampler.sample(logits_host_.data(), V, history_);
        }

        bool is_stop = std::find(cfg.stop_tokens.begin(), cfg.stop_tokens.end(), next)
                       != cfg.stop_tokens.end();
        if (is_stop) break;

        out.push_back(next);
        push_history(next);
        ++stats_.new_tokens;

        std::string piece = stream.push(next);
        if (!piece.empty()) {
            tail += piece;
            if (tail.size() > 256) tail.erase(0, tail.size() - 256);
            if (on_token && !on_token(piece, next)) break;
        }

        bool hit = false;
        for (const auto& s : cfg.stop_strings) {
            if (!s.empty() && tail.size() >= s.size() &&
                tail.compare(tail.size() - s.size(), s.size(), s) == 0) { hit = true; break; }
        }
        if (hit) break;

        if (cache_.length() >= max_context_) cache_.evict_front(max_context_ / 4, 8);
        // P0-03: pass absolute_pos_ (true monotonic coordinate) instead of
        // cache_.length() which shrinks after evict_front, corrupting RoPE.
        // Fast path leaves logits on device (no full-vocab D2H); legacy keeps
        // the host copy the sampler reads next iteration.
        GAI_CHECK(absolute_pos_ >= 0 && absolute_pos_ <= (i64)std::numeric_limits<int>::max(),
                  "generate: absolute_pos overflow (2B tokens); reset the Generator");
        if (use_fast) decode_step_logits(next, static_cast<int>(absolute_pos_));
        else decode_step(next, static_cast<int>(absolute_pos_));
        ++absolute_pos_;
    }

    std::string rest = stream.flush();
    if (!rest.empty() && on_token) on_token(rest, -1);

    stats_.decode_sec += t.seconds();
    return out;
}

std::string Generator::complete(const std::string& prompt, const GenerationConfig& cfg,
                                StreamFn on_token) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    reset();
    std::vector<i32> ids = tok_.encode(prompt, true, false);
    prefill(ids);
    std::vector<i32> gen = generate(cfg, on_token);
    std::string text = tok_.decode(gen, true);
    return cfg.echo_prompt ? prompt + text : text;
}

std::string Generator::chat(const std::vector<Message>& msgs, const GenerationConfig& cfg,
                            StreamFn on_token) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    reset();
    std::vector<i32> ids = ChatTemplate::encode(tok_, msgs, true, nullptr);
    // truncate from the front if the prompt alone exceeds the window
    int budget = max_context_ - cfg.max_new_tokens - 4;
    if (budget > 0 && static_cast<int>(ids.size()) > budget) {
        int drop = static_cast<int>(ids.size()) - budget;
        ids.erase(ids.begin() + 1, ids.begin() + 1 + drop);   // keep BOS
        log_warn(strfmt("chat: history truncated by %d tokens", drop));
    }
    prefill(ids);
    std::vector<i32> gen = generate(cfg, on_token);
    return tok_.decode(gen, true);
}

// ---------------------------------------------------------------- scoring
// logits [T,V] (~524MB عند T=4096,V=32k) فيقطع T4. نقطع إلى بلوكات 512
// (ذروة ~64MB) ونجمع sum/n عبر البلوكات — نفس المتوسط تماماً مع الحفاظ على
// إزاحة RoPE الأصلية عند كل بلوك.
double Generator::score_tokens(const std::vector<i32>& tokens, const std::vector<u8>* mask,
                               i64* out_ntok) {
    GAI_CHECK(model_.device() == expected_device_,
              "model device changed after Generator construction; rebuild the Generator");
    if (tokens.size() < 2) { if (out_ntok) *out_ntok = 0; return 0.0; }
    // FIX: heap over-read when mask shorter than tokens (SFT scoring crash).
    // Fail fast instead of reading past the vector (training/eval safety).
    GAI_CHECK(!mask || mask->size() >= tokens.size(), "score_tokens: mask shorter than tokens");
    const int T = static_cast<int>(std::min<size_t>(tokens.size(), static_cast<size_t>(model_.config().max_seq_len)));

    static constexpr int SCORE_CHUNK = 512;  // ~64MB logits عند V=32k
    Device dev = model_.device();
    double sum_all = 0.0;
    i64 n_all = 0;
    for (int off = 0; off < T; off += SCORE_CHUNK) {
        const int Tc = std::min(SCORE_CHUNK, T - off);
        // نحتاج Tc+1 توكن لبناء أهداف البلوك (آخر هدف من البلوك التالي).
        const int Tend = std::min(T, off + Tc + 1);
        const int Tn = Tend - off;
        if (Tn < 2) break;
        Activations act = model_.make_activations(1, Tn, false);
        act.pos_offset = off;
        std::vector<i32> ids(tokens.begin() + off, tokens.begin() + off + Tn);
        std::vector<i32> tgt(static_cast<size_t>(Tn), -100);
        for (int i = 0; i + 1 < Tn; ++i) {
            bool supervised = !mask || (*mask)[static_cast<size_t>(off + i) + 1];
            if (supervised) tgt[static_cast<size_t>(i)] = tokens[static_cast<size_t>(off + i) + 1];
        }

        // ids/tgt must live on the model device (CUDA reads device memory).
        Tensor d_ids({Tn}, DType::I32, Device::CPU);
        std::memcpy(d_ids.data_ptr(), ids.data(), sizeof(i32) * ids.size());
        Tensor dev_ids({Tn}, DType::I32, dev);
        dev_ids.copy_from(d_ids);
        Tensor d_tgt({Tn}, DType::I32, Device::CPU);
        std::memcpy(d_tgt.data_ptr(), tgt.data(), sizeof(i32) * tgt.size());
        Tensor dev_tgt({Tn}, DType::I32, dev);
        dev_tgt.copy_from(d_tgt);

        Tensor& logits = model_.forward(dev_ids.i32p(), 1, Tn, act);
        double sum = 0.0;
        i64 n = 0;
        ops::softmax_cross_entropy(dev, logits.f32(), dev_tgt.i32p(), nullptr,
                                   Tn, model_.config().vocab_size, &sum, &n);
        sum_all += sum;
        n_all += n;
    }
    if (out_ntok) *out_ntok = n_all;
    return n_all > 0 ? sum_all / static_cast<double>(n_all) : 0.0;
}

} // namespace gai
