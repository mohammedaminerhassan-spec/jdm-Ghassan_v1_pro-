#pragma once

#include "core/tensor.h"
#include "core/config.h"
#include "core/rng.h"
#include <vector>
#include <string>
#include <memory>

namespace gai {

// ---------------------------------------------------------------- config
// Ghassan v1 Flash: decoder-only pre-norm Transformer with Mixture-of-Experts
// (DeepSeek-style top-k routing + shared expert) in every FFN block.
// Defaults below ARE the shipped Flash architecture (~467M total, ~191M active).
struct ModelConfig {
    int   vocab_size        = 16000;
    int   hidden_size       = 768;
    int   num_layers        = 26;
    int   num_heads         = 12;
    int   num_kv_heads      = 4;
    int   intermediate_size = 2240;   // dense-FFN fallback only (use_moe=false)
    int   max_seq_len       = 4096;
    float rope_theta        = 10000.0f;
    float rms_eps           = 1e-5f;
    bool  tie_embeddings    = true;
    float init_std          = 0.02f;
    // ---- MoE (used when use_moe=true)
    bool  use_moe           = true;
    int   num_experts       = 8;      // routed experts per layer
    int   moe_top_k         = 2;      // experts active per token
    int   moe_expert_dim    = 768;    // SwiGLU hidden size per expert
    bool  moe_shared        = true;   // +1 always-on shared expert
    float moe_aux_scale     = 0.01f;  // load-balance loss weight (0 disables)
    float moe_jitter        = 0.0f;   // DeepSeek-V2 router jitter (train-only
                                      // multiplicative noise on router logits,
                                      // 0.01 typical; 0 = off/deterministic)
    bool  moe_allow_dense   = false;  // research hatch: allow top_k > ne/2
                                      // (dense routing). Default false keeps
                                      // the T4-safe sparse contract; set true
                                      // only for short research runs (OOMs T4).
    // ---- stability / context additions (backward-compatible defaults)
    bool  use_qk_norm       = false;  // per-head QK RMSNorm (MoE stability at 1B)
    float z_loss_scale      = 0.0f;   // router/logit penalty: loss += scale*logZ^2
    float rope_scale        = 1.0f;   // 1.0 off; >1 extends ctx via NTK theta scaling
    float rope_yarn_mscale  = 0.0f;   // YaRN attention-scale (0=auto from rope_scale,
                                      // DeepSeek long-ctx: 0.1*ln(scale)+1)
    // ---- Ghassan v1 Pro (DeepSeek-V3 / GLM-4 class; all default OFF so كل
    // checkpoint موجود يبقى bit-identical. فعلها فقط لتدريب Pro جديد).
    bool  moe_aux_free      = false;  // aux-loss-free routing (V3 §3.2): عند true
                                      // يتوقف moe_aux_loss عن إضافة aux grad
                                      // (bias update المنفصل خطوة لاحقة؛ الآن
                                      // يعني توازنا عبر jitter+shared فقط)
    float rope_yarn_low     = 1.0f;   // YaRN ramp: الأبعاد < low تبقى خطية
    float rope_yarn_high    = 32.0f;  // الأبعاد > high تُستكمل NTK كاملة
    int   sliding_window    = 0;      // 0=off (full causal). >0 sliding window
                                      // attention (Mistral/SWA; routed via
                                      // extended attention API)
    int   rope_type         = 0;      // 0=interleaved (legacy Ghassan v1),
                                      // 1=neox half-rotate (HF/Llama/Qwen/DS;
                                      // routed via extended RoPE API)

    int head_dim() const { return hidden_size / num_heads; }
    int kv_dim()   const { return num_kv_heads * head_dim(); }
    int q_dim()    const { return num_heads * head_dim(); }

    void validate() const;
    static ModelConfig from_config(const Config& c, const std::string& prefix = "model");
    std::string summary() const;
};

// Effective RoPE theta with NTK scaling (rope_scale==1 -> base theta).
float rope_theta_eff(const ModelConfig& m);
// YaRN attention-temperature rescale (1.0 when rope_scale==1).
float rope_mscale(const ModelConfig& m);

// ---------------------------------------------------------------- parameters
// A named parameter: weight + (optionally) gradient, both flat f32 buffers.
struct Parameter {
    std::string      name;
    std::vector<i64> shape;
    Tensor           w;
    Tensor           g;
    Tensor           fp16_cache;
    bool             decay = true;   // AdamW weight decay applies?
    bool             frozen = false; // AdamW skips frozen params entirely

    i64 numel() const { return w.numel(); }
};

struct LayerParams {
    Parameter attn_norm;   // [d]
    Parameter wq;          // [q_dim, d]
    Parameter wk;          // [kv_dim, d]
    Parameter wv;          // [kv_dim, d]
    Tensor    wqkv_fp16;   // [q_dim + 2*kv_dim, d]
    Parameter wo;          // [d, q_dim]
    Parameter ffn_norm;    // [d]
    Parameter qk_qnorm;    // [hd] per-head Q gain (only when use_qk_norm)
    Parameter qk_knorm;    // [hd] per-head K gain (only when use_qk_norm)
    // ---- MoE FFN (DeepSeek-style). Flat layout [ne*E, d] / [ne*d, E] keeps
    // each weight matrix contiguous for cuBLAS and for GGUF export.
    Parameter router;      // [ne, d]            routing logits
    Parameter moe_gate;    // [ne*E, d]          routed experts, gate proj
    Parameter moe_up;      // [ne*E, d]          routed experts, up proj
    Parameter moe_down;    // [ne*d, E]          routed experts, down proj ([ne,d,E])
    Parameter sh_gate;     // [E, d]             shared expert (always on)
    Parameter sh_up;       // [E, d]
    Parameter sh_down;     // [d, E]
    // ---- dense fallback (only allocated when use_moe=false)
    Parameter w_gate;      // [ffn, d]
    Parameter w_up;        // [ffn, d]
    Parameter w_down;      // [d, ffn]

    bool has_moe() const { return moe_gate.w.defined(); }
};

// ---------------------------------------------------------------- param report
struct ParamGroupCount {
    std::string name;
    i64         count = 0;
    i64         per_unit = 0;
    int         units = 1;
};

struct ParamReport {
    std::vector<ParamGroupCount> groups;
    i64 total = 0;
    i64 embedding = 0;
    i64 non_embedding = 0;
    std::string to_string(const ModelConfig& cfg) const;
};

// ---------------------------------------------------------------- activations
// All intermediate buffers for one forward/backward pass, allocated once.
struct Activations {
    int B = 0, T = 0;
    bool with_grad = false;

    Tensor x;          // [B*T, d]  residual stream
    Tensor xb;         // [B*T, d]  normed input (attn)
    Tensor xb2;        // [B*T, d]  normed input (ffn)
    Tensor q, k, v;    // [B*T, *]
    Tensor qkv;        // [B*T, q_dim + 2*kv_dim]
    Tensor att_out;    // [B*T, q_dim]
    Tensor proj;       // [B*T, d]
    Tensor gate, up, act;   // dense FFN only: [B*T, ffn]
    // MoE scratch, reusable across layers: [N, K, E] + routing [N, ne]/[N, K]
    Tensor moe_gate, moe_up, moe_act;   // [N, K, E]
    Tensor moe_probs;                   // [N, ne] f32
    Tensor moe_idx;                     // [N, K]  i32
    Tensor moe_w;                       // [N, K]  f32
    Tensor moe_dact;                    // [N, K, E] backward scratch
    Tensor moe_auxfrac;                 // [ne] f32 (device) load-balance fractions
    Tensor ffn_out;    // [B*T, d]
    // Logits scratch. Full [N,V] for forward()/eval; compact [Cc,V] row-block
    // scratch when built for chunked training (ce_chunks > 1), in which case
    // only forward_backward() may use these activations (forward() refuses).
    // (dlogits in the gradient scratch below follows the same sizing.)
    Tensor logits;     // [N or Cc, V]
    Tensor pos;        // [B*T] int32 positions
    // PERF: positions for a given (B,T) are deterministic ([t] tiled over b),
    // and training reuses one (B,T) for 128 micros/step. Rebuilding + H2D on
    // every forward is pure overhead, so cache validity here: forward_body
    // refills act.pos only when (B,T) changed since the last fill.
    int pos_cached_B = -1;
    int pos_cached_T = -1;
    int pos_offset = 0;
    int pos_cached_offset = -1;
    int ce_chunks = 1; // loss chunking used by forward_backward (>=1)
    i64 ce_rows = 0;   // rows per chunk (== N when ce_chunks == 1)

    // saved per layer for backward
    // NOTE: attention probs are NOT stored per layer (O(T^2) x L = 2.6GB at
    // B=2,T=1024). They are recomputed per layer in backward into the single
    // transient attn_probs_tmp below (100MB), classic activation checkpointing.
    std::vector<Tensor> saved_x;        // residual input to each layer
    std::vector<Tensor> saved_xb;       // attn-normed
    std::vector<Tensor> saved_rrms1;
    std::vector<Tensor> saved_q, saved_k, saved_v;
    std::vector<Tensor> saved_qk_rms_q; // [N*H] QK-norm rrms (only when use_qk_norm)
    std::vector<Tensor> saved_qk_rms_k; // [N*KV] QK-norm rrms (only when use_qk_norm)
    std::vector<Tensor> saved_qk_raw_q; // [N, qd] pre-norm Q (only when use_qk_norm)
    std::vector<Tensor> saved_qk_raw_k; // [N, kvd] pre-norm K (only when use_qk_norm)
    Tensor attn_probs_tmp;              // [B*H*T*T] transient recompute buffer
    std::vector<Tensor> saved_attout;
    std::vector<Tensor> saved_xmid;     // x after attention residual
    std::vector<Tensor> saved_xb2;
    std::vector<Tensor> saved_rrms2;
    std::vector<Tensor> saved_gate, saved_up, saved_act;   // dense OR [N,K,E] MoE
    std::vector<Tensor> saved_moe_probs;  // [N, ne] router probs (MoE only)
    std::vector<Tensor> saved_moe_idx;    // [N, K]  i32 (MoE only)
    std::vector<Tensor> saved_moe_w;      // [N, K]  f32 (MoE only)
    Tensor saved_xfinal;                // input to the final norm
    Tensor saved_rrms_final;
    Tensor saved_hnorm;                 // output of the final norm

    // gradient scratch
    Tensor dx, dxb, dq, dk, dv, dattout, dproj, dgate, dup, dact, dffn, dlogits, dtmp;

    size_t bytes = 0;
};

// ---------------------------------------------------------------- model
class Model {
public:
    explicit Model(ModelConfig cfg, Device dev = Device::CPU);
    ~Model();

    const ModelConfig& config() const { return cfg_; }
    Device device() const { return device_; }

    // ---- parameters
    std::vector<Parameter*>&       parameters()       { return params_; }
    const std::vector<Parameter*>& parameters() const { return params_; }
    Parameter* find_parameter(const std::string& name);
    i64  num_parameters() const;
    i64  num_parameters_non_embedding() const;
    ParamReport parameter_report() const;
    void print_parameter_report() const;

    void init_weights(u64 seed = 42);
    void enable_fp16_weight_cache(bool on);
    void mark_weights_dirty();
    bool fp16_weight_cache_enabled() const { return fp16_weight_cache_; }
    size_t fp16_weight_cache_bytes() const;
    void enable_grad(bool on);
    bool grad_enabled() const { return grad_enabled_; }
    void zero_grad();
    void to(Device dev);
    void set_rope_runtime(float scale, float yarn_mscale);
    const float* rope_inv_freq_ptr() const;

    // ---- forward / backward (training path, dense over B*T)
    // ids [B,T] int32 on device. Returns logits [B*T, V] (view into activations).
    Tensor& forward(const i32* ids, int B, int T, Activations& act,
                    const i32* segment_ids = nullptr);
    // targets [B*T] with -100 for ignored positions.
    // dout_scale amplifies dlogits for FP16 loss scaling (grads come out scaled
    // by the same factor; the caller unscales via the optimizer grad_scale).
    double  forward_backward(const i32* ids, const i32* targets, int B, int T,
                              Activations& act, i64* out_ntok = nullptr,
                              float dout_scale = 1.0f, bool want_aux_stats = false,
                              const i32* segment_ids = nullptr,
                              // F-10: optional HOST mirror of `targets` (the
                              // trainer's batch.targets). Used ONLY to skip
                              // fully-masked CE blocks on the host without a
                              // device sync; never dereferenced on device and
                              // never affects numerics. Null = always compute.
                              const i32* host_targets = nullptr);


    // ce_chunks: row-blocks for the chunked loss in forward_backward
    // (1 = legacy full [N,V] logits; >1 = compact [Cc,V] scratch where
    // Cc = ceil(N/chunks)). Callers of forward() must keep 1.
    Activations make_activations(int B, int T, bool with_grad, int ce_chunks = 1) const;
    size_t estimate_activation_bytes(int B, int T, bool with_grad, int ce_chunks = 1) const;

    // ---- arithmetic-only accounting (no tensor is allocated) --------------
    // F-23: `gai_train --dry-run` used to construct the full Model first, so a
    // 1B CPU dry-run died (OOM) before printing the estimate it exists to
    // print. These mirror the constructor's allocation formulas and the fp16
    // cache, using pure arithmetic, so any machine can price any recipe.
    // tests/test_memory_plan.cpp cross-checks them against a real Model for the
    // shipped configs, which is what keeps the duplication honest.
    struct MemoryPlan {
        u64    params = 0;             // total parameter elements
        u64    params_no_embedding = 0;
        size_t weights = 0;            // fp32 master weights
        size_t grads = 0;              // fp32 gradients (training only)
        size_t fp16_cache = 0;         // persistent fp16 weight cache (F-14)
        size_t activations = 0;        // peak activation arena for (B,T)
        size_t static_total = 0;       // weights + grads + fp16_cache
        size_t total = 0;              // static_total + activations
    };
    static u64    count_parameters(const ModelConfig& cfg);
    // F-14: the per-layer FUSED qkv fp16 cache (wqkv_fp16) was missing from
    // the runtime accounting, under-reporting persistent VRAM by
    // L*(qd+2*kvd)*d*2 bytes.
    static size_t count_fp16_cache_bytes(const ModelConfig& cfg);
    static MemoryPlan plan_memory(const ModelConfig& cfg, int B, int T,
                                  bool with_grad, int ce_chunks,
                                  bool fp16_weight_cache);
    static size_t estimate_activation_bytes_for(const ModelConfig& cfg,
                                                bool fp16_weight_cache_on,
                                                int B, int T, bool with_grad,
                                                int ce_chunks);
    // F-16: worst-case CUDA workspace sizes for (B,T), pure arithmetic. The
    // trainer pre-sizes both pools from this before the first step so the run
    // never pays a cudaFree+cudaMalloc resize stall mid-training. Values mirror
    // the moe_workspace()/workspace() call sites (forward + backward) with
    // headroom; over-estimating is safe (monotonic pools), under-estimating
    // just falls back to the historical grow-on-demand path.
    struct WorkspacePlan { size_t gemm_bytes = 0; size_t moe_bytes = 0; };
    static WorkspacePlan workspace_plan(const ModelConfig& cfg, int B, int T);


    // Raw (unscaled) DeepSeek-style load-balance aux loss summed over layers,
    // measured from the routing caches of the last forward() call. Refills the
    // shared fractions buffer (also consumed by the backward pass).
    double moe_aux_loss(Activations& act, int B, int T);

    // Expert utilization report from accumulated routing stats (empty for
    // dense models). Call periodically to detect router collapse.
    std::string moe_balance_report() const;
    void moe_balance_reset();
    // Aux-loss-free bias (DeepSeek-V3 §3.2): per-layer [ne] steering bias,
    // updated by EMA on host (no grad). nullptr when aux_free is off.
    void ensure_moe_bias();
    const float* moe_bias_ptr(int layer) const;
    float* moe_bias_ptr_mut(int layer);
    void update_moe_bias(int layer, const float* frac_host, int ne);
    void set_moe_bias(int layer, const float* values, size_t count);
    const std::vector<std::vector<float>>& moe_bias_all() const { return moe_bias_; }

    // Two-phase aux-free bias update (DDP-safe, once-per-optimizer-step):
    //
    // Phase 1 (per microbatch): accumulate_moe_bias_fracs() adds RAW routed
    // slot counts into moe_bias_acc_ [L*ne]. Counts, not fractions: fractions
    // cannot be summed across microbatches/ranks without re-weighting.
    //
    // Phase 2 (once per optimizer step, after the DDP all-reduce):
    // apply_moe_bias_step() converts the globally summed counts to per-layer
    // fractions and runs the EMA update exactly once.
    //
    // F-12 (population contract): the denominator is DERIVED from the same
    // counts (row_sum == routed tokens * K) instead of an externally supplied
    // token count, so the load fraction is counts[e] / row_sum. Every routed
    // token contributes exactly top_k slots, so numerator and denominator can
    // never describe different populations (the old ntok_global denominator
    // mixed supervised loss tokens with all-token routing counts, which
    // silently mis-scaled the bias under SFT masks).
    void accumulate_moe_bias_fracs(Activations& act, int B, int T);
    // `global_count_sum`: DDP-summed [L*ne] slot counts for this step.
    // `apply=false` discards the accumulator WITHOUT applying the EMA update
    // (F-11: the optimizer skipped this step on a non-finite grad norm, so no
    // optimizer-step-coupled control state may move). The accumulator is always
    // cleared so counts can never leak into a later step.
    void apply_moe_bias_step(const float* global_count_sum, bool apply);
    // Raw [L * ne] slot-count accumulator for Trainer to all-reduce.
    std::vector<float>& moe_bias_acc_host() { return moe_bias_acc_; }
    // F-02: device-side twin of the accumulator. On CUDA the per-microbatch
    // counts stay on device (moe_count_slots, zero D2H) and only the [L*ne]
    // summary crosses the host once per optimizer step. Null/undefined on CPU.
    Tensor& moe_bias_acc_dev() { return moe_bias_acc_dev_; }

    // ---- weights io (raw f32 dump; the .gai format lives in format/)
    void save_raw(const std::string& path) const;
    bool load_raw(const std::string& path);

    LayerParams&       layer(int i)       { return layers_[static_cast<size_t>(i)]; }
    const LayerParams& layer(int i) const { return layers_[static_cast<size_t>(i)]; }
    const u16* fused_qkv_ptr(int i) const;
    Parameter& tok_embeddings() { return tok_emb_; }
    Parameter& final_norm()     { return final_norm_; }
    Parameter& lm_head()        { return cfg_.tie_embeddings ? tok_emb_ : lm_head_; }
    const Parameter& lm_head() const { return cfg_.tie_embeddings ? tok_emb_ : lm_head_; }

private:
    void alloc_param(Parameter& p, const std::string& name, std::vector<i64> shape, bool decay);
    void rebuild_rope_cache();
    // Device move. announce=true warns that live Trainer/Generator scratch must
    // be rebuilt; init_weights() passes false for its intentional CPU round-trip.
    void move_to_device(Device dev, bool announce);

    // per-layer aux-loss helper (also accumulates routing stats below).
    // want_stats=false (training hot path): no host copies at all; the raw
    // scalar folds into d_raw_accum on device (read once per microbatch).
    // want_stats=true (or null accum): full host raw + balance stats.
    double moe_layer_aux(Device dev, const float* probs, const i32* idx,
                         float* auxfrac_dev, int layer, i64 N, int K, int ne,
                         bool want_stats = true, double* d_raw_accum = nullptr);

    // Body of forward() up to (and including) the final norm; the lm_head
    // GEMM is left to the caller so forward_backward() can chunk it.
    void forward_body(const i32* ids, int B, int T, Activations& act,
                      const i32* segment_ids);

    ModelConfig cfg_;
    // routing stats: [layer][expert] tokens routed (for balance reporting)
    mutable std::vector<std::vector<double>> moe_tok_acc_;
    mutable u64 moe_aux_batches_ = 0;
    // aux-loss-free steering bias [layer][expert] (host master; mirrored to
    // device on demand in forward). Empty unless moe_aux_free is on.
    std::vector<std::vector<float>> moe_bias_;
    std::vector<Tensor>             moe_bias_dev_;
    float                           moe_bias_lr_ = 0.001f;
    // Per-step accumulator for DDP-safe two-phase aux-free bias update:
    // [L * ne] flat; zeroed at start of each optimizer step.
    std::vector<float>              moe_bias_acc_;
    // F-02: device-side twin, [L * ne] f32 on the model device. Undefined
    // unless the CUDA counting path populated it this step.
    Tensor                          moe_bias_acc_dev_;
    Device      device_ = Device::CPU;
    bool        grad_enabled_ = false;
    bool        fp16_weight_cache_ = false;
    bool        fp16_weights_dirty_ = false;
    Tensor      rope_inv_freq_;

    Parameter                tok_emb_;
    Parameter                lm_head_;     // only used when !tie_embeddings
    Parameter                final_norm_;
    std::vector<LayerParams> layers_;
    std::vector<Parameter*>  params_;
};

} // namespace gai
