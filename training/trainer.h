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

    std::string data_dir       = "artifacts/shards";
    std::string train_prefix   = "train";
    std::string val_prefix     = "val";

    std::map<std::string, double> mix;

    int   batch_size    = 8;
    int   seq_len       = 1024;
    int   grad_accum    = 16;
    i64   max_steps     = 0;
    int   epochs        = 0;
    float learning_rate = 3e-4f;
    float min_lr_ratio  = 0.1f;
    i64   warmup_steps  = 2000;

    std::string optimizer = "adamw";
    float weight_decay  = 0.1f;
    float beta1         = 0.9f;
    float beta2         = 0.95f;
    float eps           = 1e-8f;
    float grad_clip     = 1.0f;

    int   muon_ns_steps   = 5;
    int   muon_min_ns_dim = 0;
    float muon_vec_ratio  = 0.1f;

    std::string scheduler = "cosine";
    float sched_decay_frac = 0.2f;

    DType param_dtype   = DType::F32;
    bool  gemm_fp16     = true;
    bool  fp16_weight_cache = false;
    double loss_scale_init   = 65536.0;
    int    loss_scale_window = 2000;

    double loss_scale_max    = 0.0;

    bool  activation_checkpointing = false;
    int   ckpt_segments = 2;

    int   ce_chunks = 4;

    bool  pack_sequences = false;

    bool  ddp = false;
    bool  ddp_grad_compression = false;

    i64   log_every   = 10;
    i64   eval_every  = 500;
    i64   eval_batches= 20;
    i64   save_every  = 1000;
    std::string checkpoint_dir = "artifacts/checkpoints/200m";
    std::string resume = "auto";

    u64   tok_fingerprint = 0;
    u64   seed = 42;
    std::string device = "auto";

    std::string pretrained_checkpoint;

    bool  allow_no_pretrained  = false;
    bool  freeze_embeddings    = false;

    bool  allow_recipe_drift   = false;

    i64   output_budget_mb    = 0;

    std::string resume_mode = "migrate";
    bool resume_exact() const { return resume_mode == "exact"; }

    std::string stage = "pretrain";

    static TrainerConfig from_config(const Config& c, bool strict = false);

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

    i64 epoch_steps(u64 total_tokens, int world_size = 1) const;
    std::string precision_name() const {
        if (param_dtype == DType::F16) return "fp16";
        if (param_dtype == DType::BF16) return "bf16";
        return "fp32";
    }
};

struct RecipeCost {
    u64    params           = 0;
    size_t weights          = 0;
    size_t grads            = 0;
    size_t opt_state        = 0;
    size_t muon_scratch     = 0;
    size_t fp16_cache       = 0;
    size_t activations      = 0;
    size_t eval_extra       = 0;
    size_t gemm_ws          = 0;
    size_t moe_ws           = 0;
    size_t nccl             = 0;
    size_t total            = 0;
    size_t snapshot         = 0;
    size_t gguf             = 0;
    size_t output_projection = 0;
};
RecipeCost price_recipe(const ModelConfig& m, const TrainerConfig& t);

class Trainer {
public:
    Trainer(Model& model, TrainerConfig cfg);
    ~Trainer();

    void run();
    double evaluate(i64 max_batches);

    const TrainState& state() const { return state_; }

    static size_t ckpt_ram_budget_public() { return ckpt_ram_budget(); }

private:
    void run_pretrain();
    void run_sft();

    i64 planned_total() const;
    void log_step(double loss, float lr, double gnorm, double dt, i64 ntok);
    void save(const std::string& name);

    Model&        model_;
    TrainerConfig cfg_;

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
    i64           total_steps_ = 0;
    Activations   act_;
    Activations   eval_act_;
    Timer         wall_;
    double        ema_loss_ = 0.0;
    bool          have_val_ = false;

    Tensor dev_ids_;
    Tensor dev_targets_;
    Tensor dev_segments_;
    Parameter* frozen_emb_ = nullptr;

    Activations ckpt_act_;
    Tensor ckpt_ids_;
    Tensor ckpt_targets_;
    Tensor ckpt_segments_;
    bool use_ckpt_ = false;

    double forward_backward_micro(const Batch& batch, float dscale, i64* out_ntok);

    double loss_scale_ = 65536.0;
    int    clean_steps_ = 0;

    i64    scaler_overflows_ = 0;
    i64    skipped_steps_ = 0;
    float  scaler_for_step();
    void   scaler_update(double gnorm);
    void   log_scaler_summary() const;

    double loss_scale_cap() const;

    std::unique_ptr<DistributedContext> dist_;
    void init_distributed();    void sync_gradients();
    void sync_model();
    i64 sync_ntok_sum(i64 local);

    void sync_moe_bias(bool opt_step_applied);
    bool is_main_rank() const;

    void broadcast_from_main(void* buf, size_t numel, int dtype_size);

    bool sync_eval_best(bool is_main, bool is_best);

    void check_ckpt_health();

    void check_stop_requested();
    bool stopped_by_signal_ = false;

    Tensor dist_fused_;
    Tensor dist_ntok_;

    static size_t ckpt_ram_budget() {
        static const size_t kBudget = [] {
            const size_t kCap = 32ULL << 30;
            const size_t kFloor = 8ULL << 30;
            const size_t kReserve = 8ULL << 30;
            const size_t phys = physical_ram_bytes();
            if (phys == 0) return kCap;
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
        i64 step = -1;
        std::shared_ptr<CheckpointSnapshot> snapshot;
    };
    std::vector<QueuedSave> pending_saves_;
    bool                   ckpt_stop_ = false;
    bool                   ckpt_busy_ = false;
    bool                   ckpt_failed_ = false;
    std::string            ckpt_error_;

    std::shared_ptr<CheckpointSnapshot> ckpt_active_;

    std::map<i64, std::string>          ckpt_committed_;
    std::shared_ptr<CheckpointSnapshot> snapshot_cache_;
    i64                    snapshot_step_ = -1;
    void start_ckpt_writer();
    void stop_ckpt_writer();
    void save_async(const std::string& path, std::shared_ptr<CheckpointSnapshot> snapshot);
    void wait_for_save();

    std::thread             prefetch_thread_;
    std::mutex              prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    Batch                   prefetch_batch_;
    bool                    prefetch_ready_ = false;
    bool                    prefetch_stop_  = false;
    bool                    prefetch_pause_ = false;
    bool                    prefetch_in_io_ = false;
    bool                    prefetch_running_ = false;

    std::exception_ptr      prefetch_error_;

    DataLoader::State       last_consumed_loader_state_{};
    void start_prefetch();
    void stop_prefetch();
    void quiesce_prefetch();
    void resume_prefetch();
    bool next_train_batch(Batch& out);
};

}
