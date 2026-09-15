#include "training/trainer.h"
#include "core/ops.h"
#include "core/device.h"
#include "core/common.h"
#ifdef GAI_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_ops.h"
#endif

#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

namespace gai {

TrainerConfig TrainerConfig::from_config(const Config& c) {
    TrainerConfig t;
    t.data_dir       = c.get_str ("training.data_dir", t.data_dir);
    t.batch_size     = static_cast<int>(c.get_int("training.batch_size", t.batch_size));
    t.seq_len        = static_cast<int>(c.get_int("training.seq_len", t.seq_len));
    t.grad_accum     = static_cast<int>(c.get_int("training.grad_accum", t.grad_accum));
    t.max_steps      = c.get_int("training.max_steps", 0);
    t.epochs         = static_cast<int>(c.get_int("training.epochs", 0));
    // explicit schedule, no silent conversion: exactly one of the two set
    if (t.max_steps > 0 && t.epochs > 0)
        GAI_FAIL("ambiguous schedule: set EITHER training.max_steps OR training.epochs, not both");
    if (t.max_steps <= 0 && t.epochs <= 0)
        GAI_FAIL("no schedule: set training.max_steps (>0) or training.epochs (>0)");
    t.learning_rate  = c.get_f32("training.learning_rate", t.learning_rate);
    t.min_lr_ratio   = c.get_f32("training.min_lr_ratio", t.min_lr_ratio);
    t.warmup_steps   = c.get_int("training.warmup_steps", t.warmup_steps);
    t.optimizer      = c.get_str("training.optimizer", t.optimizer);
    if (t.optimizer != "adamw" && t.optimizer != "lion")
        GAI_FAIL("unknown training.optimizer '" + t.optimizer + "' (adamw|lion)");
    t.scheduler        = c.get_str("training.scheduler", t.scheduler);
    if (t.scheduler != "cosine" && t.scheduler != "wsd")
        GAI_FAIL("unknown training.scheduler '" + t.scheduler + "' (cosine|wsd)");
    t.sched_decay_frac = c.get_f32("training.sched_decay_frac", t.sched_decay_frac);
    t.weight_decay   = c.get_f32("training.weight_decay", t.weight_decay);
    t.beta1          = c.get_f32("training.beta1", t.beta1);
    // lion paper default b2=0.99; keep adamw 0.95 unless explicitly set.
    // Heuristic: if optimizer==lion and beta2 still holds the adamw default,
    // switch to 0.99 (explicit yaml value always wins).
    t.beta2          = c.get_f32("training.beta2", t.optimizer == "lion" ? 0.99f : t.beta2);
    t.eps            = c.get_f32("training.eps", t.eps);
    t.grad_clip      = c.get_f32("training.grad_clip", t.grad_clip);
    // precision: fp32 (default), bf16, fp16
    std::string precision = c.get_str("training.precision", "fp32");
    if (precision == "bf16") t.param_dtype = DType::BF16;
    else if (precision == "fp16") t.param_dtype = DType::F16;
    else t.param_dtype = DType::F32;
    // compute GEMMs in fp16 on CUDA tensor cores (weights/grads stay fp32)
    t.gemm_fp16          = c.get_bool("training.gemm_fp16", t.gemm_fp16);
    t.loss_scale_init    = static_cast<double>(c.get_f32("training.loss_scale_init",
                                                         static_cast<float>(t.loss_scale_init)));
    t.loss_scale_window  = static_cast<int>(c.get_int("training.loss_scale_window",
                                                      t.loss_scale_window));
    t.log_every      = c.get_int("training.log_every", t.log_every);
    t.eval_every     = c.get_int("training.eval_every", t.eval_every);
    t.eval_batches   = c.get_int("training.eval_batches", t.eval_batches);
    t.save_every     = c.get_int("training.save_every", t.save_every);
    t.checkpoint_dir = c.get_str("training.checkpoint_dir", t.checkpoint_dir);
    t.resume         = c.get_str("training.resume", t.resume);
    t.seed           = static_cast<u64>(c.get_int("training.seed", static_cast<i64>(t.seed)));
    t.device         = c.get_str("training.device", t.device);
    t.stage          = c.get_str("training.stage", t.stage);
    if (t.stage != "pretrain" && t.stage != "sft" && t.stage != "cpt")
        GAI_FAIL("unknown training.stage '" + t.stage + "' (pretrain|sft|cpt)");
    // NOTE: the stage NEVER changes implicitly. epochs+pretrain = epoch-budgeted
    // pretraining, NOT sft. Masks in shards always apply when present.
    t.pretrained_checkpoint = c.get_str("training.pretrained_checkpoint", t.pretrained_checkpoint);
    t.allow_no_pretrained   = c.get_bool("training.allow_no_pretrained", t.allow_no_pretrained);
    t.freeze_embeddings    = c.get_bool("training.freeze_embeddings", t.freeze_embeddings);
    // Weighted domain mixture (data.mix.<domain>: <weight>). Keys map to
    // train_<domain>_*.gbin shards. Empty/absent = legacy uniform sampling.
    for (const auto& [k, v] : c.flat()) {
        const std::string pre = "data.mix.";
        if (k.size() > pre.size() && k.compare(0, pre.size(), pre) == 0) {
            std::string domain = k.substr(pre.size());
            try {
                double w = std::stod(v);
                if (w > 0.0) t.mix[domain] = w;
            } catch (...) { /* ignore malformed weights */ }
        }
    }
    if (c.get_bool("training.activation_checkpointing", false)) {
        t.activation_checkpointing = true;
        t.ckpt_segments = static_cast<int>(c.get_int("training.ckpt_segments", 2));
        if (t.ckpt_segments < 1) t.ckpt_segments = 1;
        if (t.ckpt_segments > 8) t.ckpt_segments = 8;
    }
    t.ddp = c.get_bool("training.ddp", false);
    return t;
}

i64 TrainerConfig::epoch_steps(u64 total_tokens) const {
    i64 tps = tokens_per_step();
    if (tps <= 0 || total_tokens == 0) return 0;
    i64 per_epoch = static_cast<i64>((total_tokens + static_cast<u64>(tps) - 1) /
                                     static_cast<u64>(tps));
    return per_epoch * epochs;
}

static AdamWConfig make_adam(const TrainerConfig& c) {
    AdamWConfig a;
    a.lr           = c.learning_rate;
    a.beta1        = c.beta1;
    a.beta2        = c.beta2;
    a.eps          = c.eps;
    a.weight_decay = c.weight_decay;
    a.grad_clip    = c.grad_clip;
    return a;
}

static LionConfig make_lion(const TrainerConfig& c) {
    LionConfig l;
    // Lion updates are sign-based (large); default to ~0.1x AdamW peak unless
    // the yaml sets an explicit LR for lion.
    l.lr           = c.learning_rate * 0.1f;
    l.beta1        = c.beta1;
    l.beta2        = c.beta2;
    l.weight_decay = c.weight_decay;
    l.grad_clip    = c.grad_clip;
    return l;
}

Trainer::Trainer(Model& model, TrainerConfig cfg)
    : model_(model), cfg_(std::move(cfg)) {
    // Only the selected optimizer allocates moments (T4: lion saves ~1.9GB).
    use_lion_ = (cfg_.optimizer == "lion");
    if (use_lion_) opt_lion_ = std::make_unique<Lion>(model_, make_lion(cfg_));
    else            opt_adam_ = std::make_unique<AdamW>(model_, make_adam(cfg_));
    // NOTE: sched_ is (re)built in run() once the exact planned total is
    // known (steps mode and epochs mode differ). Never trust a default here.

    state_.seed = cfg_.seed;

    // mixed-precision compute: fp16/bf16 GEMMs on CUDA, fp32 everywhere else.
    // CPU ignores the flag (always fp32); the loss scaler is harmless there.
    // FIX: BF16 tensor cores exist only on Ampere+ (cc>=8). T4 (sm75) has FP16
    // cores only, so a bf16 request on T4 must fall back to fp32+fp16, never to
    // a slow non-tensor BF16 path. TF32 is opt-out via GAI_TF32=0.
    loss_scale_ = cfg_.loss_scale_init > 0.0 ? cfg_.loss_scale_init : 1.0;
    clean_steps_ = 0;
    bool is_cuda = model_.device() == Device::CUDA;
    const DeviceInfo& di_prec = device_info();
    bool hw_bf16 = is_cuda && di_prec.cuda_available && di_prec.supports_bf16;
    bool want_bf16 = (cfg_.param_dtype == DType::BF16) && is_cuda;
    if (want_bf16 && !hw_bf16) {
        log_warn("[prec] bf16 requested but GPU has no BF16 tensor cores (need Ampere+); "
                 "falling back to fp32 masters + fp16 GEMMs");
        want_bf16 = false;
    }
    ops::set_gemm_fp16(cfg_.gemm_fp16 && is_cuda && !want_bf16);
#ifdef GAI_CUDA
    ops::set_gemm_bf16(want_bf16);
#endif
    log_info(strfmt("[prec] gemm_fp16=%s gemm_bf16=%s loss_scale=%.0f",
                    (cfg_.gemm_fp16 && is_cuda && !want_bf16) ? "on" : "off",
                    want_bf16 ? "on" : "off",
                    loss_scale_));
    // P0-01 + §37 contract: surface the effective loss-scaling state so a
    // silent 1/65536 suppression can never hide in logs again.
    {
        bool scaling = (cfg_.loss_scale_init > 0.0) && ops::gemm_fp16_enabled();
        log_info(strfmt("[prec] loss_scaling=%s (init %.0f, scaler now %.0f)",
                        scaling ? "ENABLED (fp16 GEMMs active)" : "DISABLED (pure fp32 path)",
                        cfg_.loss_scale_init, scaling ? loss_scale_ : 1.0));
    }

    // ---- explicit pretrained policy: SFT must never silently start from scratch
    if (cfg_.is_sft()) {
        if (!cfg_.pretrained_checkpoint.empty()) {
            TrainState pretrained_state;
            GAI_CHECK(Checkpoint::load(cfg_.pretrained_checkpoint, model_, static_cast<AdamW*>(nullptr), pretrained_state),
                      "cannot load pretrained checkpoint for SFT: " + cfg_.pretrained_checkpoint);
            log_info(strfmt("[sft ] loaded pretrained weights from %s (pretrain step %lld, val %.4f)",
                            cfg_.pretrained_checkpoint.c_str(),
                            static_cast<long long>(pretrained_state.step),
                            pretrained_state.best_val));
        } else if (!cfg_.allow_no_pretrained) {
            GAI_FAIL("SFT with no pretrained checkpoint: set training.pretrained_checkpoint "
                     "or training.allow_no_pretrained=true (testing only)");
        } else {
            log_warn("[sft ] starting from RANDOM weights (allow_no_pretrained=true)");
        }
    }

    if (cfg_.freeze_embeddings) {
        Parameter* tok_emb = model_.find_parameter("tok_embeddings");
        GAI_CHECK(tok_emb != nullptr, "freeze_embeddings: tok_embeddings not found");
        tok_emb->frozen = true;   // AdamW skips frozen params; grads zeroed each step
        frozen_emb_ = tok_emb;
        log_info("[sft ] embeddings FROZEN (no updates at all)");
    } else {
        frozen_emb_ = nullptr;
    }

    // ---- distributed training (multi-GPU) ----
    init_distributed();

    // P0-03 FIX: rank-aware sampler seeds. Old code used cfg_.seed on every
    // rank, so 4xT4 DDP processed 4x DUPLICATE batches (wasted compute, wrong
    // world-size scaling). Now: train stream is salted by global_rank
    // (disjoint streams, still deterministic); val stream stays identical on
    // all ranks so every rank evaluates the same batches (or rank0-only eval).
    u64 train_seed = cfg_.seed;
    u64 val_seed = cfg_.seed + 1;
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) {
        train_seed = cfg_.seed + static_cast<u64>(dist_->global_rank()) * 1000003ULL;
        log_info(strfmt("[data] DDP rank %d/%d: train sampler seed %llu (base %llu)",
                        dist_->global_rank(), dist_->world_size(),
                        (unsigned long long)train_seed, (unsigned long long)cfg_.seed));
    }
#endif
    BatchSpec spec{cfg_.batch_size, cfg_.seq_len};
    bool opened = false;
    if (!cfg_.mix.empty()) {
        // Prefer domain-weighted sampling when train_<domain>_*.gbin exists.
        // Falls back to legacy uniform sampling when no domain shards match.
        opened = train_loader_.open_mix(cfg_.data_dir, cfg_.mix, spec, train_seed);
        if (!opened)
            log_warn("[data] mix configured but no train_<domain>_*.gbin found; using uniform sampling");
    }
    if (!opened) {
        if (!train_loader_.open_glob(cfg_.data_dir, cfg_.train_prefix, spec, train_seed)) {
            GAI_FAIL("no training shards found in " + cfg_.data_dir +
                     " (expected " + cfg_.train_prefix + "*.gbin)");
        }
    }
    have_val_ = val_loader_.open_glob(cfg_.data_dir, cfg_.val_prefix, spec, val_seed);

    log_info(strfmt("[data] train: %d shards, %s tokens%s", train_loader_.num_shards(),
                    human_count(train_loader_.total_tokens()).c_str(),
                    train_loader_.using_mix() ? (" | " + train_loader_.mix_report()).c_str() : " (uniform)"));
    if (have_val_) {
        log_info(strfmt("[data] val  : %d shards, %s tokens", val_loader_.num_shards(),
                        human_count(val_loader_.total_tokens()).c_str()));
    } else {
        log_warn("[data] no validation shards found; validation is disabled");
    }
    // §38 startup contract (§10 sampling semantics made explicit): one block
    // so a wrong recipe is obvious BEFORE GPU hours are spent.
    {
        int ws = 1, rank = 0;
#ifdef GAI_CUDA
        if (dist_ && dist_->world_size() > 1) { ws = dist_->world_size(); rank = dist_->global_rank(); }
#endif
        bool scaling = (cfg_.loss_scale_init > 0.0) && ops::gemm_fp16_enabled();
        log_info("---------------- run contract ----------------");
        log_info(strfmt("  model: %s", model_.config().summary().c_str()));
        log_info(strfmt("  precision: %s | gemm_fp16=%s | loss_scaling=%s",
                        cfg_.precision_name().c_str(),
                        ops::gemm_fp16_enabled() ? "on" : "off",
                        scaling ? "on" : "off"));
        log_info(strfmt("  distributed: world=%d rank=%d | sampler=train_seed %llu (rank-salted) val_seed %llu | ckpt_writer=rank0",
                        ws, rank, (unsigned long long)train_seed, (unsigned long long)val_seed));
        log_info(strfmt("  data: sampling=stochastic_with_replacement (random windows; 'epochs' = token budget, NOT classic full passes)"));
        log_info(strfmt("  sft: stage=%s | grads=supervised_token_weighted (sum/ntok_global) | pretrained=%s",
                        cfg_.stage.c_str(),
                        cfg_.is_sft() ? (cfg_.pretrained_checkpoint.empty() ? "RANDOM (allow_no_pretrained!)" : cfg_.pretrained_checkpoint.c_str()) : "n/a (pretrain)"));
        log_info("----------------------------------------------");
    }

    act_      = model_.make_activations(cfg_.batch_size, cfg_.seq_len, true);
    // §17 T² pressure (training needs the B*H*T*T probs recompute buffer):
    // T=1024/H=12/B=2 ~= 100MB transient (fine); T=4096 ~= 1.6GB (1B/T4 killer).
    // Forward is flash-tiled O(T); backward materializes the row into the ONE
    // shared buffer (not per layer). Keep T=512 for 1B/T4; longer ctx needs a
    // full blockwise FlashAttention backward (future upgrade, not this fix).
    {
        double probs_mb = (double)cfg_.batch_size * model_.config().num_heads *
                          cfg_.seq_len * cfg_.seq_len * 4.0 / (1024.0 * 1024.0);
        if (probs_mb > 800.0)
            GAI_FAIL(strfmt("OOM guard: attention T² buffer %.0fMB (B=%d H=%d T=%d) will OOM T4 16GB. "
                            "Halve seq_len (512 for 1B) or batch_size before training.",
                            probs_mb, cfg_.batch_size, model_.config().num_heads, cfg_.seq_len));
        if (probs_mb > 400.0)
            log_warn(strfmt("[mem ] attention T² buffer %.0fMB (B=%d H=%d T=%d): halve seq_len for 1B/T4",
                            probs_mb, cfg_.batch_size, model_.config().num_heads, cfg_.seq_len));
    }
    // Eval activations are lazily sized: no val shards => zero bytes resident.
    // Saves ~0.25-0.4GB on 1B/T4 runs where every MB counts.
    if (have_val_)
        eval_act_ = model_.make_activations(cfg_.batch_size, cfg_.seq_len, false);
    else
        eval_act_ = Activations{};

    dev_ids_ = Tensor::empty({static_cast<i64>(cfg_.batch_size) * cfg_.seq_len}, DType::I32, model_.device());
    dev_targets_ = Tensor::empty({static_cast<i64>(cfg_.batch_size) * cfg_.seq_len}, DType::I32, model_.device());

    // ---- activation checkpointing via batch-splitting (T4 path)
    use_ckpt_ = cfg_.activation_checkpointing && cfg_.ckpt_segments > 1 && cfg_.batch_size > 1;
    if (cfg_.activation_checkpointing && !use_ckpt_) {
        log_warn("[ckpt-act] activation_checkpointing on but batch_size==1 or segments==1: no effect "
                 "(increase batch_size>=2 or grad_accum instead)");
    }
    if (use_ckpt_) {
        int seg = cfg_.ckpt_segments;
        if (seg > cfg_.batch_size) seg = cfg_.batch_size;
        int segB = (cfg_.batch_size + seg - 1) / seg;
        ckpt_act_ = model_.make_activations(segB, cfg_.seq_len, true);
        ckpt_ids_ = Tensor::empty({static_cast<i64>(segB) * cfg_.seq_len}, DType::I32, model_.device());
        ckpt_targets_ = Tensor::empty({static_cast<i64>(segB) * cfg_.seq_len}, DType::I32, model_.device());
        log_info(strfmt("[ckpt-act] ON: micro-batch %d split into segments of B=%d (peak act ~1/%d, grads identical)",
                        cfg_.batch_size, segB, seg));
    }

    log_info(strfmt("[mem ] activations train %s | eval %s | optimizer %s",
                    human_bytes(act_.bytes).c_str(),
                    human_bytes(eval_act_.bytes).c_str(),
                    human_bytes(opt_state_bytes()).c_str()));
    // ---- T4/Kaggle fail-fast memory guard (no silent OOM mid-run)
    // FIX (10/10): old estimate forgot CUDA ctx (~400MB with 64MB-chunk
    // workspaces, down from ~600MB) + cuBLAS/MoE workspaces (~150MB, monotonic
    // pools) + NCCL buffers (~300MB with DDP) + fragmentation slack (~400MB).
    // Old 1.5GB slack + 88% fail threshold wrongly rejected the flagship
    // 1B/B=1/T=512/Lion config (93% with old math). Re-measured: core math is
    // exact, slack 1GB is enough with the new pools, fail at 95% / warn at 85%.
    {
        u64 params = static_cast<u64>(model_.num_parameters());
        size_t peak_act = use_ckpt_ ? ckpt_act_.bytes : act_.bytes;
        size_t need_core = params * 8 + opt_state_bytes() + peak_act + eval_act_.bytes;
        // params*8 = weights+grads fp32; opt = m+v (adamw) or m (lion); + train/eval acts.
        // Note: with DDP, each GPU has its own full copy of weights/grads/opt state.
        size_t slack = (1024ull << 20); // 1GB: ctx + workspaces + frag (pools are tight now)
        if (dist_ && dist_->world_size() > 1) slack += (320ull << 20); // NCCL per-GPU
        size_t need = need_core + slack;
        const DeviceInfo& di = device_info();
        if (model_.device() == Device::CUDA && di.cuda_available && di.total_mem > 0) {
            double frac = (double)need / (double)di.total_mem;
            double frac_core = (double)need_core / (double)di.total_mem;
            log_info(strfmt("[mem ] est core %s + slack %s = total %s / GPU %s (%.0f%%)%s",
                            human_bytes(need_core).c_str(), human_bytes(slack).c_str(),
                            human_bytes(need).c_str(), human_bytes(di.total_mem).c_str(),
                            frac * 100.0,
                            dist_ ? " [DDP: per-GPU]" : ""));
            if (frac > 0.95)
                GAI_FAIL(strfmt("OOM guard: need %s (core %s) but GPU has %s (%.0f%% with slack). Lower batch_size/seq_len "
                                "(pilot uses 1x256) or switch optimizer lion before training.",
                                human_bytes(need).c_str(), human_bytes(need_core).c_str(),
                                human_bytes(di.total_mem).c_str(), frac * 100.0));
            else if (frac > 0.85)
                log_warn(strfmt("[mem ] over 85%% of GPU memory (core %.0f%%): watch nvidia-smi; reduce batch/seq_len if unstable",
                                frac_core * 100.0));
        } else {
            log_info(strfmt("[mem ] est total %s (weights+grads+opt+acts+slack)%s",
                            human_bytes(need).c_str(), dist_ ? " [DDP: per-GPU]" : ""));
        }
    }

    // ---- resume
    std::string resume_path;
    if (cfg_.resume == "auto")      resume_path = Checkpoint::latest_in(cfg_.checkpoint_dir);
    else if (cfg_.resume != "none") resume_path = cfg_.resume;

    if (!resume_path.empty() && fs::exists(resume_path)) {
        bool ok = use_lion_ ? Checkpoint::load(resume_path, model_, opt_lion_.get(), state_)
                            : Checkpoint::load(resume_path, model_, opt_adam_.get(), state_);
        if (ok) {
            opt_set_step(state_.step);
            train_loader_.set_state(state_.loader);
            // restore the dynamic loss scaler exactly (v3 checkpoint fields)
            if (cfg_.loss_scale_init > 0.0 && state_.loss_scale >= 1.0) {
                loss_scale_ = state_.loss_scale;
                clean_steps_ = state_.clean_steps;
            }
            log_info(strfmt("[ckpt] resumed from %s at step %lld (%s tokens seen, scaler %.0f)",
                            resume_path.c_str(), static_cast<long long>(state_.step),
                            human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                            loss_scale_));
        } else {
            log_warn("[ckpt] failed to load " + resume_path + "; starting from scratch");
        }
    }
}

float Trainer::scaler_for_step() {
    // P0-01 FIX: loss scaling only makes sense with FP16 GEMMs (compute-only
    // mixed precision). Old code scaled optimizer by 1/loss_scale even when
    // gemm_fp16 was OFF (pure FP32), so effective grads were ~1/65536.
    // Single source of truth: scaling enabled iff user configured it AND
    // the FP16 tensor-core path is actually active. scaler_update() already
    // no-ops when loss_scale_init<=0; this guards the FP16-OFF case.
    if (cfg_.loss_scale_init <= 0.0) return 1.0f;
    if (!ops::gemm_fp16_enabled()) return 1.0f;
    return static_cast<float>(loss_scale_);
}

void Trainer::scaler_update(double gnorm) {
    if (cfg_.loss_scale_init <= 0.0) return;
    if (!ops::gemm_fp16_enabled()) return; // P0-01: scaler parked when FP32-only
    if (!std::isfinite(gnorm)) {
        // overflow: the AdamW step was already skipped internally; shrink fast.
        loss_scale_ = std::max(1.0, loss_scale_ * 0.5);
        clean_steps_ = 0;
        log_warn(strfmt("[scaler] overflow at step %lld, scale -> %.0f",
                        static_cast<long long>(state_.step), loss_scale_));
    } else if (++clean_steps_ >= cfg_.loss_scale_window) {
        clean_steps_ = 0;
        double grown = std::min(16777216.0, loss_scale_ * 2.0);
        if (grown != loss_scale_) {
            loss_scale_ = grown;
            log_info(strfmt("[scaler] %d clean steps, scale -> %.0f",
                            cfg_.loss_scale_window, loss_scale_));
        }
    }
}

double Trainer::forward_backward_micro(const Batch& batch, float dscale, i64* out_ntok) {
    if (!use_ckpt_) {
        i64 N = static_cast<i64>(batch.B) * batch.T;
        device_copy(dev_ids_.data_ptr(), model_.device(), batch.ids.data(), Device::CPU, N * sizeof(i32));
        device_copy(dev_targets_.data_ptr(), model_.device(), batch.targets.data(), Device::CPU, N * sizeof(i32));
        i64 ntok = 0;
        double l = model_.forward_backward(dev_ids_.i32p(), dev_targets_.i32p(),
                                           batch.B, batch.T, act_, &ntok, dscale);
        if (frozen_emb_ && frozen_emb_->g.defined()) frozen_emb_->g.zero_();
        if (out_ntok) *out_ntok = ntok;
        return l;
    }
    // ---- checkpointed path: split along B, grads accumulate identically ----
    const int B = batch.B, T = batch.T;
    const int segB = ckpt_act_.B;   // precomputed ceil(B/segments)
    double loss_num = 0.0;
    i64 ntok_tot = 0;
    for (int b0 = 0; b0 < B; b0 += segB) {
        int curB = std::min(segB, B - b0);
        i64 curN = static_cast<i64>(curB) * T;
        // gather slice [b0, b0+curB) from CPU batch into contiguous CPU staging
        // (batch layout is [B,T] row-major, so each row is contiguous)
        std::vector<i32> slice_ids(static_cast<size_t>(curN));
        std::vector<i32> slice_tgt(static_cast<size_t>(curN));
        for (int b = 0; b < curB; ++b) {
            size_t src = static_cast<size_t>(b0 + b) * T;
            size_t dst = static_cast<size_t>(b) * T;
            std::memcpy(slice_ids.data() + dst, batch.ids.data() + src, sizeof(i32) * T);
            std::memcpy(slice_tgt.data() + dst, batch.targets.data() + src, sizeof(i32) * T);
        }
        device_copy(ckpt_ids_.data_ptr(), model_.device(), slice_ids.data(), Device::CPU, curN * sizeof(i32));
        device_copy(ckpt_targets_.data_ptr(), model_.device(), slice_tgt.data(), Device::CPU, curN * sizeof(i32));
        i64 ntok = 0;
        double l = model_.forward_backward(ckpt_ids_.i32p(), ckpt_targets_.i32p(),
                                           curB, T, ckpt_act_, &ntok, dscale);
        if (frozen_emb_ && frozen_emb_->g.defined()) frozen_emb_->g.zero_();
        loss_num += l * static_cast<double>(ntok);
        ntok_tot += ntok;
    }
    if (out_ntok) *out_ntok = ntok_tot;
    return ntok_tot > 0 ? loss_num / static_cast<double>(ntok_tot) : 0.0;
}

double Trainer::evaluate(i64 max_batches) {
    if (!have_val_) return 0.0;
    double total = 0.0;
    i64    ntok_total = 0;
    Batch batch;
    for (i64 i = 0; i < max_batches; ++i) {
        if (!val_loader_.next(batch)) break;
        
        i64 N = static_cast<i64>(batch.B) * batch.T;
        device_copy(dev_ids_.data_ptr(), model_.device(), batch.ids.data(), Device::CPU, N * sizeof(i32));
        device_copy(dev_targets_.data_ptr(), model_.device(), batch.targets.data(), Device::CPU, N * sizeof(i32));
        
        Tensor& logits = model_.forward(dev_ids_.i32p(), batch.B, batch.T, eval_act_);
        double sum = 0.0;
        i64 n = 0;
        ops::softmax_cross_entropy(model_.device(), logits.f32(), dev_targets_.i32p(),
                                   nullptr, N, model_.config().vocab_size, &sum, &n);
        total += sum;
        ntok_total += n;
    }
    return ntok_total > 0 ? total / static_cast<double>(ntok_total) : 0.0;
}

void Trainer::log_step(double loss, float lr, double gnorm, double dt, i64 ntok) {
    double tps = dt > 0 ? static_cast<double>(ntok) / dt : 0.0;
    i64 remaining = total_steps_ - state_.step;
    double eta = remaining > 0 ? remaining * dt : 0.0;
    log_info(strfmt("step %6lld | loss %7.4f | ema %7.4f | ppl %8.2f | lr %.3e | gnorm %6.3f "
                    "| %7.0f tok/s | %s | eta %s",
                    static_cast<long long>(state_.step), loss, ema_loss_,
                    std::exp(std::min(20.0, ema_loss_)), lr, gnorm, tps,
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(eta).c_str()));
}

double Trainer::opt_step(float lr, float grad_scale) {
    if (use_lion_) return opt_lion_->step(lr, grad_scale);
    return opt_adam_->step(lr, grad_scale);
}

size_t Trainer::opt_state_bytes() const {
    if (use_lion_) return opt_lion_->state_bytes();
    return opt_adam_->state_bytes();
}

void Trainer::opt_set_step(i64 t) {
    if (use_lion_) opt_lion_->set_step_count(t);
    else opt_adam_->set_step_count(t);
}

void Trainer::save(const std::string& name) {
    // P0-04 FIX: rank-0-only writes. Old code let 4 ranks race on the same
    // last.ckpt.tmp -> last.ckpt (corruption/rename failures). Now only the
    // main rank writes; others barrier (their weights are bit-identical after
    // synced optimizer steps, and loader state is rank-specific so resume is
    // defined as rank0's stream — documented in logs).
    // NOTE: loader state saved is the MAIN rank's stream. Resuming multi-GPU
    // continues each rank from base_seed+rank salt (deterministic), not from
    // an exact 4-way offset — global ordering stays disjoint, exact cross-
    // session bit-continuation of all 4 streams is not claimed (see §10).
    if (!is_main_rank()) {
#ifdef GAI_CUDA
        if (dist_ && dist_->world_size() > 1) dist_->barrier();
#endif
        return;
    }
    state_.loader = train_loader_.get_state();
    state_.loss_scale = loss_scale_;
    state_.clean_steps = clean_steps_;
    state_.tok_vocab = model_.config().vocab_size;
    std::string path = (fs::path(cfg_.checkpoint_dir) / name).string();
    Timer t;
    if (use_lion_) Checkpoint::save(path, model_, *opt_lion_, state_);
    else Checkpoint::save(path, model_, *opt_adam_, state_);
    log_info(strfmt("[ckpt] saved %s (%s)", path.c_str(), human_duration(t.seconds()).c_str()));
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) dist_->barrier();
#endif
}

i64 Trainer::planned_total() const { return total_steps_; }

void Trainer::run() {
    // ---- resolve the explicit schedule into an exact step budget
    if (cfg_.max_steps > 0) {
        total_steps_ = cfg_.max_steps;
        log_info(strfmt("[sched] steps mode: %lld optimizer steps",
                        static_cast<long long>(total_steps_)));
    } else {
        total_steps_ = cfg_.epoch_steps(train_loader_.total_tokens());
        GAI_CHECK(total_steps_ > 0,
                  "epochs mode needs data: no training tokens found in " + cfg_.data_dir);
        // §10 honesty: random-window sampling with replacement (see DataLoader),
        // so 'epochs' is a stochastic token BUDGET (coverage approximate), not
        // classic full-pass epochs. Global ordering across DDP ranks is disjoint
        // by rank salt, but not a single deterministic permutation.
        log_info(strfmt("[sched] epochs mode: %d epochs x %s tokens = %lld steps (stochastic windows, not classic passes)",
                        cfg_.epochs,
                        human_count(train_loader_.total_tokens()).c_str(),
                        static_cast<long long>(total_steps_)));
    }
    // scheduler is built HERE from the true total (never from a default)
    // lion uses its own effective LR (0.1x); scheduler shapes it identically.
    float sched_peak = use_lion_ ? make_lion(cfg_).lr : cfg_.learning_rate;
    sched_ = LrScheduler(sched_peak, cfg_.warmup_steps,
                         total_steps_, cfg_.min_lr_ratio,
                         cfg_.scheduler, cfg_.sched_decay_frac);
    log_info(strfmt("[sched] %s warmup %lld total %lld min %.0f%% decay_frac %.2f",
                    cfg_.scheduler.c_str(), static_cast<long long>(cfg_.warmup_steps),
                    static_cast<long long>(total_steps_),
                    cfg_.min_lr_ratio * 100.0, cfg_.sched_decay_frac));
    if (state_.step >= total_steps_)
        log_warn(strfmt("[sched] resumed at step %lld which already reaches the planned total %lld",
                        static_cast<long long>(state_.step),
                        static_cast<long long>(total_steps_)));

    if (cfg_.is_sft()) {
        run_sft();
    } else {
        run_pretrain();
    }
}

void Trainer::run_pretrain() {
    log_info("---------------- pretraining ----------------");
    log_info(strfmt("  steps %lld (%s) | micro-batch %d x %d tok | accum %d | %s tok/step",
                    static_cast<long long>(total_steps_),
                    cfg_.max_steps > 0 ? "steps mode" : "epochs mode",
                    cfg_.batch_size, cfg_.seq_len,
                    cfg_.grad_accum, human_count(static_cast<u64>(cfg_.tokens_per_step())).c_str()));
    log_info(strfmt("  opt %s | sched %s | peak lr %.2e | warmup %lld | %s to %.0f%% | wd %.2f | clip %.1f",
                    cfg_.optimizer.c_str(), cfg_.scheduler.c_str(),
                    sched_.peak(), static_cast<long long>(cfg_.warmup_steps),
                    cfg_.scheduler.c_str(),
                    cfg_.min_lr_ratio * 100.0, cfg_.weight_decay, cfg_.grad_clip));
    // T4-ONLY precision reporting (fp32 masters + FP16 tensor cores).
    log_info(strfmt("  precision : %s", cfg_.precision_name().c_str()));

    Batch batch;
    Timer step_timer;

    while (state_.step < total_steps_) {
        step_timer.reset();
        model_.zero_grad();
        const float dscale = scaler_for_step();   // frozen within the step
        // P2-02 jitter entropy: same position gets different noise each step
        // and each rank (reproducible from base seed + step + rank).
        {
            int rank = 0;
#ifdef GAI_CUDA
            if (dist_ && dist_->world_size() > 1) rank = dist_->global_rank();
#endif
            u64 ctx = cfg_.seed ^ (static_cast<u64>(state_.step) * 0x9E3779B97F4A7C15ULL)
                                ^ (static_cast<u64>(rank) * 0xBF58476D1CE4E5B9ULL);
            ops::set_moe_jitter_seed(ctx);
        }

        // P0-05 token-weighted grads: Model::forward_backward now emits SUM
        // grads (mean×ntok, still ×dscale). Micros accumulate as sums; below
        // we divide ONCE by the exact supervised total (local + DDP-global).
        // Loss reporting was already ntok-weighted and stays a MEAN.
        double loss_num = 0.0;
        i64    ntok_step = 0;
        int    micro_done = 0;

        for (int micro = 0; micro < cfg_.grad_accum; ++micro) {
            // NOTE: the loader samples random windows indefinitely (with
            // replacement); next() only fails when NO shards exist at all.
            // See §10: this is a stochastic token budget, not classic epochs.
            if (!train_loader_.next(batch)) {
                GAI_FAIL("dataloader has no shards mid-run in " + cfg_.data_dir);
            }

            i64 ntok = 0;
            double l = forward_backward_micro(batch, dscale, &ntok);
            loss_num += l * static_cast<double>(ntok);
            ntok_step += ntok;
            ++micro_done;
        }
        if (micro_done == 0) break;

        double loss = ntok_step > 0 ? loss_num / static_cast<double>(ntok_step) : 0.0;
        ema_loss_ = (state_.step == 0 && ema_loss_ == 0.0) ? loss : 0.98 * ema_loss_ + 0.02 * loss;

        float lr = sched_.lr_at(state_.step);

        // ---- distributed: SUM gradients + exact global ntok across GPUs ----
        sync_gradients();
        i64 ntok_global = sync_ntok_sum(ntok_step);

        // Unscale (dscale) + average over supervised tokens (global for DDP).
        // Guard ntok==0 (all-masked step): grads are aux-only/zero; keep the
        // old accum divisor so the step is a harmless near-no-op, never NaN.
        float grad_scale;
        if (ntok_global > 0) grad_scale = 1.0f / (static_cast<float>(ntok_global) * dscale);
        else grad_scale = (1.0f / static_cast<float>(cfg_.grad_accum)) / dscale;

        double gnorm = opt_step(lr, grad_scale);
        scaler_update(gnorm);

        ++state_.step;
        state_.tokens_seen += ntok_step;   // local supervised tokens (per-rank honest accounting)
        state_.last_loss = loss;

        double dt = step_timer.seconds();
        bool main = is_main_rank();
        if (main && cfg_.log_every > 0 && state_.step % cfg_.log_every == 0) {
            log_step(loss, lr, gnorm, dt, ntok_step);
        }

        if (have_val_ && cfg_.eval_every > 0 && state_.step % cfg_.eval_every == 0) {
            double vl = evaluate(cfg_.eval_batches);
            if (main) {
                std::string bal = model_.moe_balance_report();
                if (!bal.empty()) log_info("  >> " + bal);
                model_.moe_balance_reset();
                log_info(strfmt("  >> val loss %.4f  (ppl %.2f)  [best %.4f]",
                                vl, std::exp(std::min(20.0, vl)), state_.best_val));
                if (vl < state_.best_val) {
                    state_.best_val = vl;
                    save("best.ckpt");
                }
            } else {
                model_.moe_balance_reset();
            }
        }

        if (cfg_.save_every > 0 && state_.step % cfg_.save_every == 0) {
            save("last.ckpt");
        }
    }

    save("last.ckpt");
    log_info(strfmt("---------------- pretrain done: %lld steps, %s tokens, %s --------------------",
                    static_cast<long long>(state_.step),
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(wall_.seconds()).c_str()));
}

void Trainer::run_sft() {
    // total_steps_ and sched_ were resolved in run(); nothing is estimated here.
    log_info("---------------- SFT (instruction tuning) ----------------");
    log_info(strfmt("  stage: %s | steps: %lld (%s)", cfg_.stage.c_str(),
                    static_cast<long long>(total_steps_),
                    cfg_.max_steps > 0 ? "steps mode" : "epochs mode"));
    log_info(strfmt("  opt %s | sched %s | micro-batch %d x %d tok | accum %d | %s tok/step | lr %.2e",
                    cfg_.optimizer.c_str(), cfg_.scheduler.c_str(),
                    cfg_.batch_size, cfg_.seq_len, cfg_.grad_accum,
                    human_count(static_cast<u64>(cfg_.tokens_per_step())).c_str(), sched_.peak()));
    log_info(strfmt("  pretrained: %s | freeze_embeddings: %s",
                    cfg_.pretrained_checkpoint.empty() ? "none (fresh)" : cfg_.pretrained_checkpoint.c_str(),
                    cfg_.freeze_embeddings ? "yes" : "no"));
    log_info(strfmt("  precision : %s (T4 FP16 tensor cores)", cfg_.precision_name().c_str()));

    Batch batch;
    Timer step_timer;

    while (state_.step < total_steps_) {
        step_timer.reset();
        model_.zero_grad();
        const float dscale = scaler_for_step();   // frozen within the step

        // P0-05 token-weighted grads (same as pretrain; critical for SFT where
        // masks make supervised counts vary strongly per micro).
        double loss_num = 0.0;
        i64    ntok_step = 0;
        int    micro_done = 0;

        for (int micro = 0; micro < cfg_.grad_accum; ++micro) {
            if (!train_loader_.next(batch)) {
                GAI_FAIL("dataloader has no shards mid-run in " + cfg_.data_dir);
            }

            i64 ntok = 0;
            double l = forward_backward_micro(batch, dscale, &ntok);
            loss_num += l * static_cast<double>(ntok);
            ntok_step += ntok;
            ++micro_done;
        }
        if (micro_done == 0) break;

        double loss = ntok_step > 0 ? loss_num / static_cast<double>(ntok_step) : 0.0;
        ema_loss_ = (state_.step == 0 && ema_loss_ == 0.0) ? loss : 0.98 * ema_loss_ + 0.02 * loss;

        float lr = sched_.lr_at(state_.step);

        // ---- distributed: SUM gradients + exact global ntok across GPUs ----
        sync_gradients();
        i64 ntok_global = sync_ntok_sum(ntok_step);

        float grad_scale;
        if (ntok_global > 0) grad_scale = 1.0f / (static_cast<float>(ntok_global) * dscale);
        else grad_scale = (1.0f / static_cast<float>(cfg_.grad_accum)) / dscale;

        double gnorm = opt_step(lr, grad_scale);
        scaler_update(gnorm);

        ++state_.step;
        state_.tokens_seen += ntok_step;   // local supervised tokens (per-rank honest accounting)
        state_.last_loss = loss;

        double dt = step_timer.seconds();
        bool main = is_main_rank();
        if (main && cfg_.log_every > 0 && state_.step % cfg_.log_every == 0) {
            log_step(loss, lr, gnorm, dt, ntok_step);
        }

        if (have_val_ && cfg_.eval_every > 0 && state_.step % cfg_.eval_every == 0) {
            double vl = evaluate(cfg_.eval_batches);
            if (main) {
                std::string bal = model_.moe_balance_report();
                if (!bal.empty()) log_info("  >> " + bal);
                model_.moe_balance_reset();
                log_info(strfmt("  >> val loss %.4f  (ppl %.2f)  [best %.4f]",
                                vl, std::exp(std::min(20.0, vl)), state_.best_val));
                if (vl < state_.best_val) {
                    state_.best_val = vl;
                    save("best.ckpt");
                }
            } else {
                model_.moe_balance_reset();
            }
        }

        if (cfg_.save_every > 0 && state_.step % cfg_.save_every == 0) {
            save("last.ckpt");
        }
    }

    save("last.ckpt");
    log_info(strfmt("---------------- SFT done: %lld steps, %s tokens, %s --------------------",
                    static_cast<long long>(state_.step),
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(wall_.seconds()).c_str()));
}

// ---- distributed training ----
void Trainer::init_distributed() {
#ifdef GAI_CUDA
    if (model_.device() != Device::CUDA) return;

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus <= 1) return;

    DistributedContext::Config dcfg = distributed_config_from_env(num_gpus);
    if (dcfg.world_size <= 1) return;

    dist_ = std::make_unique<DistributedContext>();
    bool ok = dist_->init(dcfg);
    if (!ok) {
        log_warn("[dist] Failed to initialize distributed context, falling back to single GPU");
        dist_.reset();
        return;
    }

    // Set CUDA device for this rank
    cudaSetDevice(dcfg.local_rank);

    // Broadcast model from rank 0 to all ranks (ensures identical initialization)
    if (dist_->global_rank() != 0) {
        sync_model();
    }
    log_info(strfmt("[dist] Training on %d GPUs (rank %d/%d)",
                    dist_->world_size(), dist_->global_rank(), dist_->local_rank()));
#else
    (void)model_; // suppress unused warning
#endif
}

void Trainer::sync_gradients() {
#ifdef GAI_CUDA
    if (!dist_ || dist_->world_size() <= 1) return;

    // Fused bucketed all-reduce SUM (P0-05 token-weighted, P0-03 rank-aware).
    // Old code averaged (sum/world_size) assuming equal work per rank, which
    // breaks token-weighted grads when SFT masks give different ntok per rank.
    // Now: pure SUM over ranks (no division); the caller divides once by the
    // exact global ntok (sync_ntok_sum). ~200 params => ~200 NCCL launches was
    // the dominant DDP overhead; large tensors (>=1M) go individually,
    // small packed into one persistent staging buffer, reduced once.
    constexpr size_t kLarge = 1 << 20;
    struct Item { float* ptr; size_t n; };
    std::vector<Item> small;
    small.reserve(64);
    size_t small_total = 0;
    for (Parameter* p : model_.parameters()) {
        if (!p->g.defined() || p->frozen) continue;
        if (p->g.device() != Device::CUDA) continue;
        size_t numel = static_cast<size_t>(p->g.numel());
        if (numel == 0) continue;
        if (numel >= kLarge) {
            dist_->all_reduce_sum(static_cast<float*>(p->g.data_ptr()), numel);
        } else {
            small.push_back({static_cast<float*>(p->g.data_ptr()), numel});
            small_total += numel;
        }
    }
    if (!small.empty() && small_total > 0) {
        // Grow-once staging (monotonic, no per-step cudaMalloc).
        if (!dist_fused_.defined() ||
            static_cast<size_t>(dist_fused_.numel()) < (i64)small_total) {
            size_t want = small_total + small_total / 8 + 1024;
            dist_fused_ = Tensor::empty({(i64)want}, DType::F32, Device::CUDA);
        }
        float* stage = dist_fused_.f32();
        size_t off = 0;
        for (auto& it : small) {
            device_copy(stage + off, Device::CUDA, it.ptr, Device::CUDA,
                        it.n * sizeof(float));
            off += it.n;
        }
        dist_->all_reduce_sum(stage, small_total);
        off = 0;
        for (auto& it : small) {
            device_copy(it.ptr, Device::CUDA, stage + off, Device::CUDA,
                        it.n * sizeof(float));
            off += it.n;
        }
    }

    dist_->barrier();
#else
    (void)dist_; (void)model_; // suppress unused warning
#endif
}

void Trainer::sync_model() {
#ifdef GAI_CUDA
    if (!dist_ || dist_->world_size() <= 1) return;

    for (Parameter* p : model_.parameters()) {
        if (!p->w.defined()) continue;
        if (p->w.device() != Device::CUDA) continue;

        size_t numel = static_cast<size_t>(p->w.numel());
        if (numel == 0) continue;

        dist_->broadcast(p->w.data_ptr(), numel, sizeof(float), 0);
    }
    dist_->barrier();
#else
    (void)dist_; (void)model_; // suppress unused warning
#endif
}

bool Trainer::is_main_rank() const {
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) return dist_->global_rank() == 0;
#endif
    return true;
}

i64 Trainer::sync_ntok_sum(i64 local) {
    // Exact global supervised-token count for token-weighted DDP (P0-05).
    // NCCL sums floats; ntok/step (<1M) is exactly representable in fp32.
    // Persistent 1-float buffer: no per-step alloc. Single tiny sync.
#ifdef GAI_CUDA
    if (!dist_ || dist_->world_size() <= 1) return local;
    if (model_.device() != Device::CUDA) {
        // CPU fallback (tests): sum not available without NCCL; ranks are
        // independent processes here, so approximate with local*world_size.
        // Exact path is CUDA+NCCL (production).
        return local * (i64)dist_->world_size();
    }
    if (!dist_ntok_.defined())
        dist_ntok_ = Tensor::empty({1}, DType::F32, Device::CUDA);
    float h = static_cast<float>(local);
    device_copy(dist_ntok_.data_ptr(), Device::CUDA, &h, Device::CPU, sizeof(float));
    dist_->all_reduce_sum(static_cast<float*>(dist_ntok_.data_ptr()), 1);
    float out = 0.0f;
    device_copy(&out, Device::CPU, dist_ntok_.data_ptr(), Device::CUDA, sizeof(float));
    return static_cast<i64>(std::llround(out));
#else
    (void)dist_;
    return local;
#endif
}

} // namespace gai
