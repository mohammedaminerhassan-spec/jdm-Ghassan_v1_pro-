#pragma once

#include "model/model.h"
#include "training/optimizer.h"
#include "training/scheduler.h"
#include "training/dataloader.h"
#include "training/checkpoint.h"
#include "training/distributed.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

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
    // Muon-only knobs. ns_steps tunes the Newton-Schulz cost directly;
    // muon_min_ns_dim gates small matrices out of NS (0 = NS on every decay
    // matrix, historical). None of the shipped T4 recipes enable muon.
    int   muon_ns_steps   = 5;    // 1..10, clamped by the Muon ctor
    int   muon_min_ns_dim = 0;    // >=0
    float muon_vec_ratio  = 0.1f; // non-matrix LR = lr*ratio (Lion 0.1x rule)
    // scheduler
    std::string scheduler = "cosine";  // cosine | wsd
    float sched_decay_frac = 0.2f;     // wsd: last 20% linearly decays
    // precision
    DType param_dtype   = DType::F32;   // F32, BF16, F16 (label; compute is fp32/ff16-GEMM)
    bool  gemm_fp16     = true;         // CUDA: large GEMMs on FP16 tensor cores
    bool  fp16_weight_cache = false;
    double loss_scale_init   = 65536.0; // dynamic loss-scaler start (0 disables scaling)
    int    loss_scale_window = 2000;    // clean steps before doubling the scale
    // Growth ceiling (0 = fp16-safe hard max 16384). Pin BELOW a level measured
    // to overflow on the target GPU so the scaler never cycles into it
    // (T4 recipes use 8192: 16384 overflows there). Never above the hard max.
    double loss_scale_max    = 0.0;
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
    // Phase 1A — Segment-masked sequence packing: pack multiple short docs
    // end-to-end into one [B,T] row with a block-causal mask so no PAD tokens
    // are wasted. Recovers 20-40% throughput vs the one-doc-per-row baseline.
    bool  pack_sequences = false;
    // distributed training
    bool  ddp = false;
    bool  ddp_grad_compression = false;
    // io / cadence
    i64   log_every   = 10;
    i64   eval_every  = 500;
    i64   eval_batches= 20;
    i64   save_every  = 1000;
    std::string checkpoint_dir = "artifacts/checkpoints/200m";
    std::string resume = "auto";
    // Content fingerprint (FNV-1a) of the .gtok actually used, persisted in
    // the checkpoint (v11) and re-checked on resume. 0 = unknown -> the check
    // is skipped (never blocks). Set by the CLI from tokenizer.path; can be
    // pinned in yaml as training.tok_fingerprint.
    u64   tok_fingerprint = 0;
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
    // Output-quota gate (0 = off). Kaggle SAVES everything under /kaggle/working
    // and caps that at ~20 GB, which is NOT the same limit as free disk space
    // (~57 GB): a run can fit on the filesystem and still blow the saved-output
    // quota with repo + build + shards + checkpoints + GGUF — and the failure
    // only appears at "Save Version", long after the training. Project it up
    // front instead (--output-budget-mb).
    i64   output_budget_mb    = 0;
    // Resume contract.
    //   "migrate" (default): forgiving behavior — a math drift warns, a
    //                      drift warns, a missing parameter keeps its fresh
    //                      init, an optimizer-kind switch restarts moments at
    //                      t_=0. Always LOGS exactly what was reset.
    //   "exact": fail closed on parameter set/shape, tokenizer vocab, model
    //             math fields, optimizer kind/state, scheduler recipe and DDP
    //             world size. Use it for production runs where a silently
    //             different model is worse than a failed start.
    // --allow-recipe-drift stays the explicit escape hatch in both modes.
    std::string resume_mode = "migrate";
    bool resume_exact() const { return resume_mode == "exact"; }

    // "pretrain" = LM training on raw shards (masks, if present, still apply).
    // "sft"/"cpt" = instruction tuning on chat shards (requires loss masks).
    // The stage NEVER changes implicitly. In particular, setting epochs with
    // stage=pretrain runs epoch-budgeted PRETRAINING (not SFT).
    std::string stage = "pretrain";

    // strict=true upgrades unknown/dead config keys from warnings to a
    // fail-fast error (see --strict-config).
    static TrainerConfig from_config(const Config& c, bool strict = false);
    // Per-rank micro throughput (one process). Global throughput multiplies by
    // world_size — DeepSeek budgeting rule: scheduler/steps must use GLOBAL.
    static i64 checked_schedule_mul(i64 a, i64 b, const char* what) {
        if (a > 0 && b > 0 && a > std::numeric_limits<i64>::max() / b) {
            GAI_FAIL(std::string("training schedule overflow in ") + what);
        }
        if (a < 0 || b < 0) {
            GAI_FAIL(std::string("training schedule negative value in ") + what);
        }
        return a * b;
    }
    i64 tokens_per_step() const {
        return checked_schedule_mul(
            checked_schedule_mul(static_cast<i64>(batch_size), static_cast<i64>(seq_len),
                                 "tokens_per_step"),
            static_cast<i64>(grad_accum), "tokens_per_step");
    }
    i64 tokens_per_step_global(int world_size) const {
        if (world_size < 1) world_size = 1;
        return checked_schedule_mul(tokens_per_step(), static_cast<i64>(world_size),
                                    "tokens_per_step_global");
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

// Single source of truth for "what does this recipe cost?".
//
// `gai_train --dry-run` (the Kaggle preflight gate) and
// tests/test_recipe_gates.cpp BOTH call this. The duplication that used to
// exist between them is exactly the rot that lets a config ship a change the
// preflight would have rejected: the gate and the unit test must price a
// recipe the same way or the test proves nothing. Pure arithmetic — nothing is
// allocated, so a 1B recipe can be priced on any machine.
struct RecipeCost {
    u64    params           = 0;  // parameter elements
    size_t weights          = 0;  // fp32 masters
    size_t grads            = 0;  // fp32 gradients
    size_t opt_state        = 0;  // adamw m+v / lion m / muon m+v+NS scratch
    size_t muon_scratch     = 0;  // the NS part of opt_state (0 unless muon)
    size_t fp16_cache       = 0;  // persistent fp16 weight cache (0 when off)
    size_t activations      = 0;  // peak training arena at (B,T)
    size_t eval_extra       = 0;  // rank-0 eval arena estimate
    size_t gemm_ws          = 0;  // CUDA GEMM/SCE pool
    size_t moe_ws           = 0;  // CUDA MoE pool
    size_t nccl             = 0;  // NCCL comms + staging
    size_t total            = 0;  // predicted per-GPU peak VRAM
    size_t snapshot         = 0;  // one checkpoint (weights + moments, NO grads)
    size_t gguf             = 0;  // rough final GGUF (q4-ish) size
    size_t output_projection = 0; // 2 snapshots + gguf (the quota projection)
};
RecipeCost price_recipe(const ModelConfig& m, const TrainerConfig& t);

class Trainer {
public:
    Trainer(Model& model, TrainerConfig cfg);
    ~Trainer();

    void run();
    double evaluate(i64 max_batches);

    const TrainState& state() const { return state_; }

    // Public for tests + preflight scripts (pure arithmetic over host RAM).
    static size_t ckpt_ram_budget_public() { return ckpt_ram_budget(); }

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
    Tensor dev_segments_;
    Parameter* frozen_emb_ = nullptr;   // set when freeze_embeddings (grads re-zeroed)

    // activation-checkpointing scratch (allocated only when enabled and B>1)
    Activations ckpt_act_;
    Tensor ckpt_ids_;
    Tensor ckpt_targets_;
    Tensor ckpt_segments_;
    bool use_ckpt_ = false;
    // Persistent CPU staging for ckpt batch-slicing (grows monotonically,
    // never per-segment malloc).
    std::vector<i32> ckpt_staging_ids_;
    std::vector<i32> ckpt_staging_tgt_;
    // one micro-batch (possibly split into segments); returns ntok-weighted loss
    double forward_backward_micro(const Batch& batch, float dscale, i64* out_ntok);

    // dynamic loss scaler state (FP16 training safety net)
    double loss_scale_ = 65536.0;
    int    clean_steps_ = 0;
    // Overflow accounting: a couple of skipped steps while the scaler finds its
    // level is normal, a persistent stream of them means fp16 gradients are
    // unstable and the run is quietly wasting GPU hours. Counted, reported at
    // the end of every stage, and warned about past a threshold.
    i64    scaler_overflows_ = 0;
    i64    skipped_steps_ = 0;
    float  scaler_for_step();                 // current scale (1.0 when disabled)
    void   scaler_update(double gnorm);       // shrink on overflow, grow when clean
    void   log_scaler_summary() const;
    // Effective growth ceiling: min(yaml max, fp16-safe hard max). Keeps the
    // scaler from revisiting a level measured to overflow on this hardware
    // (T4 recipes pin 8192; 16384 overflows there on step 0-1).
    double loss_scale_cap() const;

    // ---- distributed training ----
    std::unique_ptr<DistributedContext> dist_;
    void init_distributed();    void sync_gradients();  // fused bucketed all-reduce SUM (token-weighted; no /world_size)
    void sync_model();      // broadcast model from rank 0 (for initialization/resume)
    i64 sync_ntok_sum(i64 local); // exact global supervised count (sum over ranks)
    // Aux-free router bias is optimizer-step-coupled control state, so it
    // only moves when the optimizer actually applied an update. All-reduces the
    // [L*ne] slot-count accumulator across ranks, then applies the EMA once.
    void sync_moe_bias(bool opt_step_applied);
    bool is_main_rank() const;    // rank 0 or single-GPU (logs/writes checkpoints)
    // Collective helpers. Every rank must call these unconditionally so
    // the NCCL collective order stays identical across ranks; they are no-ops
    // for single-process runs.
    void broadcast_from_main(void* buf, size_t numel, int dtype_size);
    // Agree on the rank-0-only "new best?" decision so every rank enters
    // save() (which contains collectives) the same number of times. RETURNS the
    // agreed decision and callers MUST use the return value: the argument is
    // rank-local (only the main rank evaluates), so keeping a local copy makes
    // rank 0 save while the others skip it, desyncing the collective order —
    // which NCCL surfaces as an illegal memory access mid-run.
    bool sync_eval_best(bool is_main, bool is_best);
    // Surface a fatal background-checkpoint failure at the next safe
    // training boundary instead of letting thousands of steps burn GPU hours
    // on an unusable disk.
    void check_ckpt_health();
    // SIGINT/SIGTERM asked us to stop: set the flag, and the loops break out at
    // the next step boundary (never mid-step) so the final save is clean.
    void check_stop_requested();
    bool stopped_by_signal_ = false;
    // Persistent fused DDP staging (grows monotonically, never per-step alloc).
    Tensor dist_fused_;
    Tensor dist_ntok_; // persistent 1-i64 device buffer for ntok sync (no per-step alloc)

    // ---- Phase 1D: background checkpoint writer ----
    // Serialization runs on a dedicated thread so the training loop never
    // stalls on disk I/O. The mutex+cv protect pending_saves_ (immutable
    // snapshots captured by value at the call site). save_async() returns
    // immediately; wait_for_save() drains before run() exits, and a new save
    // for the same destination supersedes the queued one.
    //
    // ckpt_ram_budget() bounds every LIVE snapshot reference (queued +
    // in-flight + retained cache). Exceeding it applies backpressure to the
    // training thread rather than aborting the run.
    //
    // 2xT4 production: the queue holds at most 2 UNIQUE snapshots
    // (one being written + one queued; same-step best+last share one pointer
    // via snapshot_cache_, deduped by unique_snapshot_bytes). The old 35%
    // budget (10GB on 29GB Kaggle) rejected a valid 480M run (5.76GB x2 =
    // 11.5GB) and any 1B run deterministically. New rule: reserve 8GB for
    // OS + training heap, give the rest to the checkpoint queue, clamped to
    // [8GB, 32GB]. On 29GB Kaggle -> 21GB: 2x 8GB Lion-1B snapshots (16GB)
    // progress with margin; on tiny boxes a too-large model still fails fast
    // via the single-copy gate in save_async (not via spinning backpressure).
    // Gradients are NEVER in the snapshot (bytes() = weights + optimizer
    // moments only), so the budget never pays for grads.
    static size_t ckpt_ram_budget() {
        static const size_t kBudget = [] {
            const size_t kCap = 32ULL << 30;                 // historical ceiling
            const size_t kFloor = 8ULL << 30;                // >= one 480M snapshot
            const size_t kReserve = 8ULL << 30;              // OS + runtime heap
            const size_t phys = physical_ram_bytes();
            if (phys == 0) return kCap;                      // unknown: keep old behaviour
            if (phys <= kReserve + kFloor) return (kFloor < phys) ? kFloor : phys;
            size_t b = phys - kReserve;
            if (b > kCap) b = kCap;
            return b;
        }();
        return kBudget;
    }
    std::thread            ckpt_writer_thread_;
    std::mutex             ckpt_mutex_;
    std::condition_variable ckpt_cv_;
    struct QueuedSave {
        std::string path;
        i64 step = -1;   // model step, so a second name can reuse the file
        std::shared_ptr<CheckpointSnapshot> snapshot;
    };
    std::vector<QueuedSave> pending_saves_;
    bool                   ckpt_stop_ = false;
    bool                   ckpt_busy_ = false;
    bool                   ckpt_failed_ = false;
    std::string            ckpt_error_;
    // Snapshot currently being serialized (counted in the RAM budget).
    std::shared_ptr<CheckpointSnapshot> ckpt_active_;
    // Committed step -> path, bounded to the newest few.
    std::map<i64, std::string>          ckpt_committed_;
    std::shared_ptr<CheckpointSnapshot> snapshot_cache_;
    i64                    snapshot_step_ = -1;
    void start_ckpt_writer();
    void stop_ckpt_writer();
    void save_async(const std::string& path, std::shared_ptr<CheckpointSnapshot> snapshot);  // enqueue; drops superseded writes
    void wait_for_save();                       // block until queue is empty

    // ---- Phase 1C: prefetch batch ----
    // A background thread calls train_loader_.next() into prefetch_batch_
    // while the GPU is computing the current step. next_batch() swaps the
    // ready buffer in and starts the next prefetch immediately.
    std::thread             prefetch_thread_;
    std::mutex              prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    Batch                   prefetch_batch_;
    bool                    prefetch_ready_ = false;
    bool                    prefetch_stop_  = false;
    bool                    prefetch_pause_ = false;
    bool                    prefetch_in_io_ = false;
    bool                    prefetch_running_ = false;
    // Preserve the real dataloader error (was swallowed -> generic message).
    std::exception_ptr      prefetch_error_;
    // Checkpoint the last batch actually handed to the trainer, NOT the
    // speculative batch the prefetch thread may already have generated.
    // Saving train_loader_.get_state() at save() time could resume one batch
    // ahead and silently skip training data, breaking exact resume.
    DataLoader::State       last_consumed_loader_state_{};
    void start_prefetch();
    void stop_prefetch();
    void quiesce_prefetch();
    void resume_prefetch();
    bool next_train_batch(Batch& out);
};

} // namespace gai
