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

    {
        size_t need = 2ULL * static_cast<size_t>(c.num_layers) *
                      static_cast<size_t>(max_context_) *
                      static_cast<size_t>(c.kv_dim()) * sizeof(float);
        if (dev == Device::CUDA && cuda_available()) {

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

    cache_.set_pinned_prefix(std::max(0, std::min(8, max_context_ - 1)));

    const int d = c.hidden_size;
    x_          = Tensor::empty({d}, DType::F32, dev);
    xb_         = Tensor::empty({d}, DType::F32, dev);
    q_          = Tensor::empty({c.q_dim()}, DType::F32, dev);
    k_          = Tensor::empty({c.kv_dim()}, DType::F32, dev);
    v_          = Tensor::empty({c.kv_dim()}, DType::F32, dev);
    if (model_.fp16_weight_cache_enabled()) {
        qkv_ = Tensor::empty({c.q_dim() + 2 * c.kv_dim()}, DType::F32, dev);
    }
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
    absolute_pos_ = 0;
}

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

    device_copy(tok_dev_.data_ptr(), dev, &token, Device::CPU, sizeof(i32));
    ops::embedding_forward(dev, tok_dev_.i32p(), model_.tok_embeddings().w.f32(), x_.f32(), 1, d, V);

    const int slot = cache_.allocate_slot();

    for (int l = 0; l < c.num_layers; ++l) {
        LayerParams& L = model_.layer(l);

        ops::rmsnorm_forward(dev, x_.f32(), L.attn_norm.w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);

        float* qp = q_.f32();
        float* kp = k_.f32();
        float* vp = v_.f32();
        if (qkv_.defined()) {
            ops::linear_forward_fp16(dev, xb_.f32(), model_.fused_qkv_ptr(l),
                                     qkv_.f32(), 1, d, qd + 2 * kvd);
            qp = qkv_.f32();
            kp = qp + qd;
            vp = kp + kvd;
        } else {
            ops::linear_forward(dev, xb_.f32(), L.wq.w.f32(), qp, 1, d, qd);
            ops::linear_forward(dev, xb_.f32(), L.wk.w.f32(), kp, 1, d, kvd);
            ops::linear_forward(dev, xb_.f32(), L.wv.w.f32(), vp, 1, d, kvd);
        }

        if (c.use_qk_norm && L.qk_qnorm.w.defined()) {
            ops::rmsnorm_forward(dev, qp, L.qk_qnorm.w.f32(), qp,
                                 nullptr, H, hd, c.rms_eps);
            ops::rmsnorm_forward(dev, kp, L.qk_knorm.w.f32(), kp,
                                 nullptr, KV, hd, c.rms_eps);
        }

        const float* rope_freq = model_.rope_inv_freq_ptr();
        GAI_CHECK(rope_freq != nullptr, "decode: RoPE frequency cache is missing");

        if (l == 0)
            device_copy(pos_dev_.data_ptr(), dev, &position, Device::CPU, sizeof(i32));
        ops::rope_forward_cached(dev, qp, kp, pos_dev_.i32p(), rope_freq, 1, H, KV, hd,
                                 c.rope_type);

        {
            size_t bytes = sizeof(float) * static_cast<size_t>(kvd);
            char* kdst = reinterpret_cast<char*>(cache_.k(l)) + static_cast<size_t>(slot) * bytes;
            char* vdst = reinterpret_cast<char*>(cache_.v(l)) + static_cast<size_t>(slot) * bytes;
            device_copy(kdst, dev, kp, dev, bytes);
            device_copy(vdst, dev, vp, dev, bytes);
        }

        ops::attention_decode_ring(dev, q_.f32(), cache_.k(l), cache_.v(l), attn_.f32(),
                                   H, KV, hd, cache_.ring_start(), cache_.pinned_prefix(),
                                   cache_.length(), cache_.capacity(), scale, scores_.f32(),
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

    ops::rmsnorm_forward(dev, x_.f32(), model_.final_norm().w.f32(), xb_.f32(), nullptr, 1, d, c.rms_eps);
    ops::linear_forward(dev, xb_.f32(), model_.lm_head().w.f32(), logits_dev_.f32(), 1, d, V);
    return logits_dev_.f32();
}

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

    if (tokens.empty() || cache_.length() != 0 || start_pos != 0) {
        for (size_t i = 0; i < tokens.size(); ++i) {
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

    Tensor ids({P}, DType::I32, Device::CPU);
    std::memcpy(ids.data_ptr(), tokens.data(), sizeof(i32) * tokens.size());
    Tensor d_ids({P}, DType::I32, dev);
    d_ids.copy_from(ids);
    Tensor pos({P}, DType::I32, Device::CPU);
    for (int i = 0; i < P; ++i) pos.i32p()[i] = start_pos + i;
    Tensor d_pos({P}, DType::I32, dev);
    d_pos.copy_from(pos);

    Tensor x({(i64)P * d}, DType::F32, dev);
    Tensor xb({(i64)P * d}, DType::F32, dev);
    Tensor q({(i64)P * qd}, DType::F32, dev);
    Tensor k({(i64)P * kvd}, DType::F32, dev);
    Tensor v({(i64)P * kvd}, DType::F32, dev);
    Tensor qkv;
    if (model_.fp16_weight_cache_enabled()) {
        qkv = Tensor({(i64)P * (qd + 2 * kvd)}, DType::F32, dev);
    }
    Tensor att({(i64)P * qd}, DType::F32, dev);
    Tensor proj({(i64)P * d}, DType::F32, dev);

    const i64 nke = (i64)c.moe_top_k * c.moe_expert_dim;
    Tensor mg({(i64)P * nke}, DType::F32, dev);
    Tensor mu({(i64)P * nke}, DType::F32, dev);
    Tensor ma({(i64)P * nke}, DType::F32, dev);
    Tensor gtmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);
    Tensor utmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);
    Tensor atmp({(i64)c.intermediate_size * (c.use_moe ? 1 : P)}, DType::F32, dev);

    ops::embedding_forward(dev, d_ids.i32p(), model_.tok_embeddings().w.f32(), x.f32(), P, d, c.vocab_size);
    const float* rope_freq = model_.rope_inv_freq_ptr();
    GAI_CHECK(rope_freq != nullptr, "prefill: RoPE frequency cache is missing");
    for (int l = 0; l < c.num_layers; ++l) {
        LayerParams& L = model_.layer(l);
        ops::rmsnorm_forward(dev, x.f32(), L.attn_norm.w.f32(), xb.f32(), nullptr, P, d, c.rms_eps);
        float* qp = q.f32();
        float* kp = k.f32();
        float* vp = v.f32();
        if (qkv.defined()) {
            ops::linear_forward_fp16(dev, xb.f32(), model_.fused_qkv_ptr(l),
                                     qkv.f32(), P, d, qd + 2 * kvd);
            ops::split_qkv(dev, qkv.f32(), q.f32(), k.f32(), v.f32(), P, qd, kvd);
            qp = q.f32();
            kp = k.f32();
            vp = v.f32();
        } else {
            ops::linear_forward(dev, xb.f32(), L.wq.w.f32(), qp, P, d, qd);
            ops::linear_forward(dev, xb.f32(), L.wk.w.f32(), kp, P, d, kvd);
            ops::linear_forward(dev, xb.f32(), L.wv.w.f32(), vp, P, d, kvd);
        }

        if (c.use_qk_norm && L.qk_qnorm.w.defined()) {
            ops::rmsnorm_forward(dev, qp, L.qk_qnorm.w.f32(), qp,
                                 nullptr, (i64)P * H, hd, c.rms_eps);
            ops::rmsnorm_forward(dev, kp, L.qk_knorm.w.f32(), kp,
                                 nullptr, (i64)P * KV, hd, c.rms_eps);
        }
        ops::rope_forward_cached(dev, qp, kp, d_pos.i32p(), rope_freq, P, H, KV, hd,
                                 c.rope_type);

        device_copy(cache_.k(l), dev, kp, dev, sizeof(float) * (size_t)P * kvd);
        device_copy(cache_.v(l), dev, vp, dev, sizeof(float) * (size_t)P * kvd);
        ops::attention_forward_ex(dev, qp, kp, vp, att.f32(), nullptr,
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

    ops::rmsnorm_forward(dev, x.f32() + (size_t)(P - 1) * d,
                         model_.final_norm().w.f32(), hlast_.f32(),
                         nullptr, 1, d, c.rms_eps);
    ops::linear_forward(dev, hlast_.f32(), model_.lm_head().w.f32(),
                        logits_dev_.f32(), 1, d, c.vocab_size);
    device_copy(logits_host_.data(), Device::CPU,
                logits_dev_.f32(), dev,
                sizeof(float) * (size_t)c.vocab_size);

    device_copy(x_.f32(), dev, x.f32() + (size_t)(P - 1) * d, dev, sizeof(float) * (size_t)d);
    for (auto t : tokens) push_history(t);
}

void Generator::prefill(const std::vector<i32>& tokens, int front_drop) {
    if (tokens.empty()) return;
    GAI_CHECK(front_drop >= 0, "prefill: front_drop must be >= 0");
    Timer t;
    int start = cache_.length();

    int processed = static_cast<int>(tokens.size());
    if (start == 0 && tokens.size() > 1) {

        std::vector<i32> work = tokens;
        bool truncated = false;
        int drop = 0;
        if (static_cast<int>(work.size()) > max_context_) {
            drop = static_cast<int>(work.size()) - max_context_;
            work.erase(work.begin() + 1, work.begin() + 1 + drop);
            log_warn(strfmt("prefill: prompt truncated by %d tokens", drop));
            truncated = true;
            processed = static_cast<int>(work.size());
        }

        if (!truncated && front_drop == 0) {

            static constexpr size_t PREFILL_CHUNK = 1024;
            if (work.size() > PREFILL_CHUNK) {
                std::vector<i32> head(work.begin(), work.begin() + PREFILL_CHUNK);
                forward_prefill(head, 0);
                absolute_pos_ = static_cast<int64_t>(head.size());
                for (size_t i = PREFILL_CHUNK; i < work.size(); ++i) {
                    decode_step(work[i], static_cast<int>(absolute_pos_));
                    ++absolute_pos_;
                    push_history(work[i]);
                }
            } else {
                forward_prefill(work, 0);

                absolute_pos_ = static_cast<int64_t>(work.size());
            }
        } else {

            const int base = drop + front_drop;
            decode_step(work[0], 0);
            push_history(work[0]);
            for (size_t i = 1; i < work.size(); ++i) {
                int pos = base + static_cast<int>(i);
                decode_step(work[i], pos);
                push_history(work[i]);
            }

            absolute_pos_ = static_cast<int64_t>(front_drop) + static_cast<int64_t>(tokens.size());
        }
    } else {
        for (size_t i = 0; i < tokens.size(); ++i) {

            decode_step(tokens[i], static_cast<int>(absolute_pos_));
            ++absolute_pos_;
            push_history(tokens[i]);
        }
    }
    stats_.prompt_tokens += processed;
    stats_.prefill_sec += t.seconds();
}

std::vector<i32> Generator::generate(const GenerationConfig& cfg, StreamFn on_token) {
    GAI_CHECK(cfg.max_context == 0 || cfg.max_context == max_context_,
              "GenerationConfig::max_context must match the Generator context");
    std::vector<i32> out;
    Sampler sampler(cfg.sampling);
    Tokenizer::Stream stream(tok_);
    Timer t;

    std::string tail;
    const int limit = cfg.max_new_tokens;
    const int V = model_.config().vocab_size;
    Device dev = model_.device();
    const SamplingConfig& sc = sampler.config();

    const bool want_greedy = sc.greedy || sc.temperature <= 0.0f || sc.top_k == 1;
    const int K = (!want_greedy && sc.top_k > 0 && sc.top_k <= FAST_TOPK_MAX)
                      ? sc.top_k
                      : 0;

    size_t tail_cap = 256;
    for (const auto& s : cfg.stop_strings)
        tail_cap = std::max(tail_cap, s.size());
    auto use_fast_for = [&](int hist_size) {
        const int w = sc.repetition_window > 0 ? sc.repetition_window : hist_size;

        const bool topk_ok = want_greedy || (K > 0 && V >= 64 * K && V <= 64 * 512);
        return sc.gpu_fast_sample && dev == Device::CUDA && V >= 512 &&
               w <= FAST_HIST_MAX && sc.no_repeat_ngram == 0 && topk_ok;
    };

    bool saw_fast = false;
    for (int step = 0; step < limit; ++step) {

        const int win_now = sc.repetition_window > 0 ? sc.repetition_window
                                                     : static_cast<int>(history_.size());
        const bool use_fast = use_fast_for(static_cast<int>(history_.size()));
        if (use_fast) saw_fast = true;
        i32 next = -1;
        if (use_fast) {
            int wn = std::min({win_now, static_cast<int>(history_.size()), FAST_HIST_MAX});
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
            if (tail.size() > tail_cap) tail.erase(0, tail.size() - tail_cap);
            if (on_token && !on_token(piece, next)) break;
        }

        bool hit = false;
        for (const auto& s : cfg.stop_strings) {
            if (!s.empty() && tail.size() >= s.size() &&
                tail.compare(tail.size() - s.size(), s.size(), s) == 0) { hit = true; break; }
        }
        if (hit) break;

        GAI_CHECK(absolute_pos_ >= 0 && absolute_pos_ <= (i64)std::numeric_limits<int>::max(),
                  "generate: absolute_pos overflow (2B tokens); reset the Generator");
        if (use_fast) decode_step_logits(next, static_cast<int>(absolute_pos_));
        else decode_step(next, static_cast<int>(absolute_pos_));
        ++absolute_pos_;
    }

    std::string rest = stream.flush();
    if (!rest.empty() && on_token) on_token(rest, -1);

    if (saw_fast && dev == Device::CUDA && static_cast<int>(logits_host_.size()) == V) {
        device_copy(logits_host_.data(), Device::CPU, logits_dev_.f32(), dev,
                    sizeof(float) * static_cast<size_t>(V));
    }
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

    int budget = max_context_ - cfg.max_new_tokens - 4;
    int drop = 0;
    if (budget > 0 && static_cast<int>(ids.size()) > budget) {
        drop = static_cast<int>(ids.size()) - budget;
        ids.erase(ids.begin() + 1, ids.begin() + 1 + drop);
        log_warn(strfmt("chat: history truncated by %d tokens", drop));
    }
    prefill(ids, drop);
    std::vector<i32> gen = generate(cfg, on_token);
    return tok_.decode(gen, true);
}

double Generator::score_tokens(const std::vector<i32>& tokens, const std::vector<u8>* mask,
                               i64* out_ntok, const std::vector<i32>* segment_ids) {
    GAI_CHECK(model_.device() == expected_device_,
              "model device changed after Generator construction; rebuild the Generator");
    if (tokens.size() < 2) { if (out_ntok) *out_ntok = 0; return 0.0; }
    const int T = static_cast<int>(std::min<size_t>(tokens.size(), static_cast<size_t>(model_.config().max_seq_len)));
    if (static_cast<int>(tokens.size()) > T) {
        log_warn(strfmt("score_tokens: input truncated to max_seq_len (%d -> %d tokens); "
                        "perplexity covers the head window only",
                        static_cast<int>(tokens.size()), T));
    }

    GAI_CHECK(!mask || mask->size() >= static_cast<size_t>(T), "score_tokens: mask shorter than scored tokens");
    GAI_CHECK(!segment_ids || segment_ids->size() >= static_cast<size_t>(T),
              "score_tokens: segment_ids shorter than scored tokens");

    static constexpr int SCORE_CHUNK = 512;
    Device dev = model_.device();
    double sum_all = 0.0;
    i64 n_all = 0;
    for (int off = 0; off < T; off += SCORE_CHUNK) {
        const int Tc = std::min(SCORE_CHUNK, T - off);

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

        Tensor d_ids({Tn}, DType::I32, Device::CPU);
        std::memcpy(d_ids.data_ptr(), ids.data(), sizeof(i32) * ids.size());
        Tensor dev_ids({Tn}, DType::I32, dev);
        dev_ids.copy_from(d_ids);
        Tensor d_tgt({Tn}, DType::I32, Device::CPU);
        std::memcpy(d_tgt.data_ptr(), tgt.data(), sizeof(i32) * tgt.size());
        Tensor dev_tgt({Tn}, DType::I32, dev);
        dev_tgt.copy_from(d_tgt);
        Tensor dev_segments;
        if (segment_ids) {
            Tensor h_segments({Tn}, DType::I32, Device::CPU);
            for (int i = 0; i < Tn; ++i) {
                h_segments.i32p()[i] = (*segment_ids)[static_cast<size_t>(off + i)];
            }
            dev_segments = Tensor({Tn}, DType::I32, dev);
            dev_segments.copy_from(h_segments);
        }

        Tensor& logits = model_.forward(dev_ids.i32p(), 1, Tn, act,
                                        segment_ids ? dev_segments.i32p() : nullptr);
        double sum = 0.0;
        i64 n = 0;

        ops::softmax_cross_entropy(dev, logits.f32(), dev_tgt.i32p(), nullptr,
                                   Tn, model_.config().vocab_size, &sum, &n,
                                   model_.config().z_loss_scale);
        sum_all += sum;
        n_all += n;
    }
    if (out_ntok) *out_ntok = n_all;
    return n_all > 0 ? sum_all / static_cast<double>(n_all) : 0.0;
}

}
