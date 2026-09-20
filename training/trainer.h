#pragma once

#include "model/model.h"
#include "training/optimizer.h"
#include "training/scheduler.h"
#include "training/dataloader.h"
#include "training/checkpoint.h"
#include "training/distributed.h"
#include <map>
#include <memory>

namespace gai {

struct TrainerConfig {
    // data
    std::string data_dir       = "artifacts/shards";
    std::string train_prefix   = "train";
    std::string val_prefix     = "val";
    // weighted domain mixture: domain -> weight, matched to
    // train_<domain>_*.gbin shards (see data_pipeline --domain).
    // Empty = legacy uniform sampling over train_*.gbin.
    std::map<std::string, double> mix;
    // schedule — EXPLICIT, never ambiguous (see from_config validation):
    //   steps mode : max_steps > 0, epochs == 0  -> run exactly max_steps
    //   epochs mode: epochs > 0, max_steps == 0  -> steps_per_epoch * epochs,
    //                where an "epoch" = one full pass over shard tokens
    //                (random-window sampling; approximate coverage, see DataLoader)
    //   both set or both zero -> configuration ERROR (fail fast, no guessing)
    int   batch_size    = 8;
    int   seq_len       = 1024;
    int   grad_accum    = 16;
    i64   max_steps     = 0;
    int   epochs        = 0;
    float learning_rate = 3e-4f;
    float min_lr_ratio  = 0.1f;
    i64   warmup_steps  = 2000;
    // optimizer
    // adamw | lion | muon. lion = ~50% opt memory (T4 saver). muon =
    // orthogonalized momentum for matrices (fastest per-step progress on
    // small models; takes learning_rate directly, use ~0.01-0.03).
    std::string optimizer = "adamw";
    float weight_decay  = 0.1f;
    float beta1         = 0.9f;
    float beta2         = 0.95f;   // lion default overridden to 0.99 when optimizer==lion
    float eps           = 1e-8f;   // adamw only
    float grad_clip     = 1.0f;
    // scheduler
    std::string scheduler = "cosine";  // cosine | wsd
    float sched_decay_frac = 0.2f;     // wsd: last 20% linearly decays
    // precision
    DType param_dtype   = DType::F32;   // F32, BF16, F16 (label; compute is fp32/ff16-GEMM)
    bool  gemm_fp16     = true;         // CUDA: large GEMMs on FP16 tensor cores
    double loss_scale_init   = 65536.0; // dynamic loss-scaler start (0 disables scaling)
    int    loss_scale_window = 2000;    // clean steps before doubling the scale
    // T4 memory saver: gradient/activation checkpointing via micro-batch
    // splitting. When true, each (B,T) micro-batch is run as `ckpt_segments`
    // sequential forward/backward slices along the batch dim (e.g. B=2,S=2 ->
    // 2x [1,T]), accumulating grads. Peak activation memory drops ~Sx while
    // the math stays identical (grads sum linearly). No effect when B==1
    // (use grad_accum instead); seq-split is intentionally NOT done because
    // causal attention would change the math.
    bool  activation_checkpointing = false;
    int   ckpt_segments = 2;   // 1..8, clamped
    // Chunked cross-entropy (roadmap item 6): row-blocks for the lm-head
    // loss in forward_backward. 1 (or 0) = legacy full [N,V] logits+dlogits.
    // 4 shrinks them to [N/4,V] each (~393MB saved at B=2,T=1024,V=32k) with
    // mathematically identical grads (sums, not means, per block). 1..32.
    int   ce_chunks = 4;
    // distributed training
    bool  ddp = false;         // enable multi-GPU DDP (auto-detected if >1 GPU)
    // io / cadence
    i64   log_every   = 10;
    i64   eval_every  = 500;
    i64   eval_batches= 20;
    i64   save_every  = 1000;
    std::string checkpoint_dir = "artifacts/checkpoints/200m";
    std::string resume = "auto";
    u64   seed = 42;
    std::string device = "auto";
    // sft-specific
    std::string pretrained_checkpoint;
    // false (default): FAIL loudly if the pretrained checkpoint is missing.
    // true: allow fresh SFT from random weights (testing only).
    bool  allow_no_pretrained  = false;
    bool  freeze_embeddings    = false;
    // false (default): FAIL loudly when the resume checkpoint was saved with
    // different MATH fields (rope_theta/scale/yarn, rms_eps, max_seq_len) —
    // same weights would otherwise define a different model function.
    // Loss weights (aux/z/jitter) stay warn-only: retuning them on resume is
    // legitimate. true: explicit opt-out for research (--allow-recipe-drift).
    bool  allow_recipe_drift   = false;

    // "pretrain" = LM training on raw shards (masks, if present, still apply).
    // "sft"/"cpt" = instruction tuning on chat shards (requires loss masks).
    // The stage NEVER changes implicitly. In particular, setting epochs with
    // stage=pretrain runs epoch-budgeted PRETRAINING (not SFT).
    std::string stage = "pretrain";

    // strict=true upgrades unknown/dead config keys from warnings to a
    // fail-fast error (P2-5; see --strict-config).
    static TrainerConfig from_config(const Config& c, bool strict = false);
    // Per-rank micro throughput (one process). Global throughput multiplies by
    // world_size — DeepSeek budgeting rule: scheduler/steps must use GLOBAL.
    i64 tokens_per_step() const {
        return static_cast<i64>(batch_size) * seq_len * grad_accum;
    }
    i64 tokens_per_step_global(int world_size) const {
        if (world_size < 1) world_size = 1;
        return tokens_per_step() * static_cast<i64>(world_size);
    }
    bool is_sft() const { return stage == "sft" || stage == "cpt"; }
    // total planned optimizer steps given the corpus size (epochs mode).
    // world_size divides the budget: 4 GPUs consume 4x tokens per step.
    i64 epoch_steps(u64 total_tokens, int world_size = 1) const;
    std::string precision_name() const {
        if (param_dtype == DType::F16) return "fp16";
        if (param_dtype == DType::BF16) return "bf16";
        return "fp32";
    }
};

class Trainer {
public:
    Trainer(Model& model, TrainerConfig cfg);

    void run();
    double evaluate(i64 max_batches);

    const TrainState& state() const { return state_; }

private:
    void run_pretrain();
    void run_sft();
    // planned total steps for the active schedule (scheduler + ETA)
    i64 planned_total() const;
    void log_step(double loss, float lr, double gnorm, double dt, i64 ntok);
    void save(const std::string& name);

    Model&        model_;
    TrainerConfig cfg_;
    // Only one optimizer is ever allocated (T4 memory): adamw (m+v),
    // lion (m) or muon (m + tiny v + NS scratch).
    std::unique_ptr<AdamW> opt_adam_;
    std::unique_ptr<Lion>  opt_lion_;
    std::unique_ptr<Muon>  opt_muon_;
    bool use_lion_ = false;
    bool use_muon_ = false;
    double opt_step(float lr, float grad_scale);
    size_t opt_state_bytes() const;
    void   opt_set_step(i64 t);
    LrScheduler   sched_;
    DataLoader    train_loader_;
    DataLoader    val_loader_;
    TrainState    state_;
    i64           total_steps_ = 0;   // set in run(), before the loop starts
    Activations   act_;
    Activations   eval_act_;
    Timer         wall_;
    double        ema_loss_ = 0.0;
    bool          have_val_ = false;

    Tensor dev_ids_;
    Tensor dev_targets_;
    Parameter* frozen_emb_ = nullptr;   // set when freeze_embeddings (grads re-zeroed)

    // activation-checkpointing scratch (allocated only when enabled and B>1)
    Activations ckpt_act_;
    Tensor ckpt_ids_;
    Tensor ckpt_targets_;
    bool use_ckpt_ = false;
    // Persistent CPU staging for ckpt batch-slicing (grows monotonically,
    // never per-segment malloc). Old code allocated 2 vectors per segment per
    // micro per step -> allocator churn that starved the T4 GPU.
    std::vector<i32> ckpt_staging_ids_;
    std::vector<i32> ckpt_staging_tgt_;
    // one micro-batch (possibly split into segments); returns ntok-weighted loss
    double forward_backward_micro(const Batch& batch, float dscale, i64* out_ntok);

    // dynamic loss scaler state (FP16 training safety net)
    double loss_scale_ = 65536.0;
    int    clean_steps_ = 0;
    float  scaler_for_step();                 // current scale (1.0 when disabled)
    void   scaler_update(double gnorm);       // shrink on overflow, grow when clean

    // ---- distributed training ----
    std::unique_ptr<DistributedContext> dist_;
    void init_distributed();
    void sync_gradients();  // fused bucketed all-reduce SUM (token-weighted; no /world_size)
    void sync_model();      // broadcast model from rank 0 (for initialization/resume)
    i64 sync_ntok_sum(i64 local); // exact global supervised count (sum over ranks)
    bool is_main_rank() const;    // rank 0 or single-GPU (logs/writes checkpoints)
    // Persistent fused DDP staging (grows monotonically, never per-step alloc).
    Tensor dist_fused_;
    Tensor dist_ntok_; // persistent 1-float device buffer for ntok sync (no per-step alloc)
};

} // namespace gai
