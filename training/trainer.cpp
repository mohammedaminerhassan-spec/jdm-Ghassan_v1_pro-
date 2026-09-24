#include "training/trainer.h"
#include "core/ops.h"
#include "core/device.h"
#include "core/common.h"
#ifdef GAI_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_ops.h"
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

namespace gai {

// Largest loss scale that keeps |dlogits| inside the fp16 normal range.
// Peak |dlogits| == the scale itself (see Model::forward_backward), fp16 max
// is 65504, so 16384 leaves 4x headroom for the GEMM's internal accumulation.
static constexpr double kLossScaleMax = 16384.0;

// Splits a "<dir>/<prefix>*.gbin" shard glob (the data.train_glob shape)
// into its directory + prefix halves. Returns false unless the value has
// exactly that shape.
static bool split_shard_glob(const std::string& glob, std::string& dir, std::string& prefix) {
    if (glob.empty()) return false;
    const size_t slash = glob.find_last_of("/\\");
    const std::string d = (slash == std::string::npos) ? "." : glob.substr(0, slash);
    const std::string f = (slash == std::string::npos) ? glob : glob.substr(slash + 1);
    const std::string tail = "*.gbin";
    if (f.size() <= tail.size() || f.compare(f.size() - tail.size(), tail.size(), tail) != 0)
        return false;
    const std::string p = f.substr(0, f.size() - tail.size());
    if (p.empty()) return false;
    dir = d;
    prefix = p;
    return true;
}

TrainerConfig TrainerConfig::from_config(const Config& c, bool strict) {
    TrainerConfig t;
    t.data_dir       = c.get_str ("training.data_dir", t.data_dir);
    // P2-5: these were silently dead (the loader used hardcoded prefixes).
    t.train_prefix   = c.get_str("data.train_prefix", t.train_prefix);
    t.val_prefix     = c.get_str("data.val_prefix", t.val_prefix);
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
    if (t.optimizer != "adamw" && t.optimizer != "lion" && t.optimizer != "muon")
        GAI_FAIL("unknown training.optimizer '" + t.optimizer + "' (adamw|lion|muon)");
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
    t.fp16_weight_cache  = c.get_bool("training.fp16_weight_cache", t.fp16_weight_cache);
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
    t.allow_recipe_drift   = c.get_bool("training.allow_recipe_drift", t.allow_recipe_drift);
    // Weighted domain mixture (data.mix.<domain>: <weight>). Keys map to
    // train_<domain>_*.gbin shards. Empty/absent = legacy uniform sampling.
    for (const auto& [k, v] : c.flat()) {
        const std::string pre = "data.mix.";
        if (k.size() > pre.size() && k.compare(0, pre.size(), pre) == 0) {
            std::string domain = k.substr(pre.size());
            // DeepSeek data rule: a typo'd weight must never silently reshape
            // the mixture. Parse strictly (full consume, w > 0) and warn loudly.
            try {
                // trim ASCII whitespace for the pos check
                size_t a = v.find_first_not_of(" \t\r\n");
                size_t b = v.find_last_not_of(" \t\r\n");
                std::string tstr = (a == std::string::npos) ? "" : v.substr(a, b - a + 1);
                size_t pos = 0;
                double w = std::stod(tstr, &pos);
                if (pos != tstr.size() || !std::isfinite(w) || !(w > 0.0)) {
                    log_warn("config: ignoring malformed data.mix weight for '" + k +
                             "': '" + v + "' (want positive number)");
                    continue;
                }
                t.mix[domain] = w;
            } catch (...) {
                log_warn("config: ignoring malformed data.mix weight for '" + k +
                         "': '" + v + "' (want positive number)");
            }
        }
    }
    if (c.get_bool("training.activation_checkpointing", false)) {
        t.activation_checkpointing = true;
        t.ckpt_segments = static_cast<int>(c.get_int("training.ckpt_segments", 2));
        if (t.ckpt_segments < 1) t.ckpt_segments = 1;
        if (t.ckpt_segments > 8) t.ckpt_segments = 8;
    }
    t.ce_chunks = static_cast<int>(c.get_int("training.ce_chunks", t.ce_chunks));
    if (t.ce_chunks <= 0) t.ce_chunks = 1;
    if (t.ce_chunks > 32) t.ce_chunks = 32;
    // Phase 1A: sequence packing (default off — safe for existing shards).
    t.pack_sequences = c.get_bool("training.pack_sequences", false);
    t.ddp = c.get_bool("training.ddp", false);

    // P2-5: data.train_glob/val_glob ("<dir>/<prefix>*.gbin") used to be
    // silently DEAD while training "worked by accident" off training.data_dir.
    // They are honored now as overrides: an explicit dir/prefix that disagrees
    // with them is a fail-fast conflict, never a silent coin flip.
    if (c.has("data.train_glob") || c.has("data.val_glob")) {
        std::string tdir, tpre, vdir, vpre;
        const bool has_tg = c.has("data.train_glob");
        const bool has_vg = c.has("data.val_glob");
        if (has_tg && !split_shard_glob(c.get_str("data.train_glob"), tdir, tpre)) {
            GAI_FAIL("cannot parse data.train_glob (want <dir>/<prefix>*.gbin): " +
                     c.get_str("data.train_glob"));
        }
        if (has_vg && !split_shard_glob(c.get_str("data.val_glob"), vdir, vpre)) {
            GAI_FAIL("cannot parse data.val_glob (want <dir>/<prefix>*.gbin): " +
                     c.get_str("data.val_glob"));
        }
        if (has_tg && has_vg && tdir != vdir) {
            GAI_FAIL("data.train_glob and data.val_glob span different shard dirs (" +
                     tdir + " vs " + vdir + "): a run uses a single training.data_dir");
        }
        const std::string gdir = has_tg ? tdir : vdir;
        if (c.has("training.data_dir") && c.get_str("training.data_dir") != gdir) {
            GAI_FAIL("training.data_dir '" + c.get_str("training.data_dir") +
                     "' conflicts with the shard glob dir '" + gdir + "'");
        }
        t.data_dir = gdir;
        if (has_tg) {
            if (c.has("data.train_prefix") && c.get_str("data.train_prefix") != tpre) {
                GAI_FAIL("data.train_prefix '" + c.get_str("data.train_prefix") +
                         "' conflicts with data.train_glob (prefix '" + tpre + "')");
            }
            t.train_prefix = tpre;
        }
        if (has_vg) {
            if (c.has("data.val_prefix") && c.get_str("data.val_prefix") != vpre) {
                GAI_FAIL("data.val_prefix '" + c.get_str("data.val_prefix") +
                         "' conflicts with data.val_glob (prefix '" + vpre + "')");
            }
            t.val_prefix = vpre;
        }
        log_info(strfmt("[data] shard globs honored: dir=%s train_prefix=%s val_prefix=%s",
                        t.data_dir.c_str(), t.train_prefix.c_str(), t.val_prefix.c_str()));
    }

    // DeepSeek config rule: data.data_dir is a legacy duplicate of
    // training.data_dir (shipped SFT yamls carry both). It is accepted for
    // backward compat but must AGREE — a conflict fails fast, never a coin flip.
    if (c.has("data.data_dir")) {
        const std::string dd = c.get_str("data.data_dir");
        if (!dd.empty() && dd != t.data_dir) {
            GAI_FAIL("data.data_dir '" + dd + "' conflicts with training.data_dir '" +
                     t.data_dir + "' (single shard dir per run; delete the duplicate)");
        }
        if (!dd.empty())
            log_warn("[cfg ] data.data_dir duplicates training.data_dir (legacy; ignored)");
    }
    // P2-5 typo catcher over the whole file: every key must be consumed
    // somewhere (model.*, training.*, tokenizer.*, data.* below). Unknown keys
    // warn by default; --strict-config fails fast (a typo'd hidden_size must
    // never silently train the default).
    {
        static const std::vector<std::string> kExact = {
            "model.vocab_size", "model.hidden_size", "model.num_layers",
            "model.num_heads", "model.num_kv_heads", "model.intermediate_size",
            "model.max_seq_len", "model.rope_theta", "model.rms_eps",
            "model.tie_embeddings", "model.init_std", "model.use_moe",
            "model.num_experts", "model.moe_top_k", "model.moe_expert_dim",
            "model.moe_shared", "model.moe_aux_scale", "model.moe_jitter",
            "model.moe_allow_dense", "model.moe_aux_free",
            "model.use_qk_norm", "model.z_loss_scale",
            "model.rope_scale", "model.rope_yarn_mscale",
            "model.rope_yarn_low", "model.rope_yarn_high",
            "model.sliding_window", "model.rope_type",
            "tokenizer.path", "tokenizer.vocab_size",
            "training.data_dir", "training.batch_size", "training.seq_len",
            "training.grad_accum", "training.max_steps", "training.epochs",
            "training.learning_rate", "training.min_lr_ratio",
            "training.warmup_steps", "training.optimizer", "training.scheduler",
            "training.sched_decay_frac", "training.weight_decay", "training.beta1",
            "training.beta2", "training.eps", "training.grad_clip",
             "training.precision", "training.gemm_fp16", "training.fp16_weight_cache",
             "training.loss_scale_init",
            "training.loss_scale_window", "training.log_every", "training.eval_every",
            "training.eval_batches", "training.save_every", "training.checkpoint_dir",
            "training.resume", "training.seed", "training.device", "training.stage",
            "training.pretrained_checkpoint", "training.allow_no_pretrained",
            "training.allow_recipe_drift",
            "training.freeze_embeddings", "training.activation_checkpointing",
             "training.ckpt_segments", "training.ce_chunks", "training.pack_sequences",
             "training.ddp",
            "data.data_dir",
            "data.train_glob", "data.val_glob", "data.train_prefix", "data.val_prefix",
        };
        static const std::vector<std::string> kPrefixes = {"data.mix."};
        c.check_known(kExact, kPrefixes, strict);
    }
    return t;
}

i64 TrainerConfig::epoch_steps(u64 total_tokens, int world_size) const {
    if (world_size < 1) world_size = 1;
    i64 tps = tokens_per_step_global(world_size);
    if (tps <= 0 || total_tokens == 0) return 0;
    if (total_tokens > static_cast<u64>(std::numeric_limits<i64>::max())) {
        GAI_FAIL("training schedule overflow in epoch_steps");
    }
    i64 per_epoch = static_cast<i64>((total_tokens + static_cast<u64>(tps) - 1) /
                                     static_cast<u64>(tps));
    return checked_schedule_mul(per_epoch, static_cast<i64>(epochs), "epoch_steps");
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

static MuonConfig make_muon(const TrainerConfig& c) {
    MuonConfig m;
    // Muon takes learning_rate DIRECTLY (orthogonal updates are inherently
    // ~Adam-scaled; no 0.1x rule). Set 0.01-0.03 in the yaml for muon runs.
    m.lr           = c.learning_rate;
    m.beta1        = c.beta1;
    m.beta2        = c.beta2;
    m.eps          = c.eps;
    m.weight_decay = c.weight_decay;
    m.grad_clip    = c.grad_clip;
    return m;
}

Trainer::Trainer(Model& model, TrainerConfig cfg)
    : model_(model), cfg_(std::move(cfg)) {
    // Strong-model contract: fail fast on illogical shapes (never mid-run).
    GAI_CHECK((cfg_.max_steps > 0) != (cfg_.epochs > 0),
              "training schedule must set exactly one of max_steps > 0 or epochs > 0");
    GAI_CHECK(cfg_.batch_size > 0, "training.batch_size must be > 0");
    GAI_CHECK(cfg_.seq_len > 0, "training.seq_len must be > 0");
    GAI_CHECK(cfg_.grad_accum > 0, "training.grad_accum must be > 0");
    GAI_CHECK(std::isfinite(cfg_.learning_rate) && cfg_.learning_rate > 0.0f,
              "training.learning_rate must be finite and > 0");
    GAI_CHECK(std::isfinite(cfg_.min_lr_ratio) && cfg_.min_lr_ratio >= 0.0f && cfg_.min_lr_ratio <= 1.0f,
              "training.min_lr_ratio must be finite and in [0,1]");
    GAI_CHECK(std::isfinite(cfg_.sched_decay_frac) && cfg_.sched_decay_frac > 0.0f && cfg_.sched_decay_frac <= 1.0f,
              "training.sched_decay_frac must be finite and in (0,1]");
    GAI_CHECK(std::isfinite(cfg_.weight_decay) && cfg_.weight_decay >= 0.0f,
              "training.weight_decay must be finite and >= 0");
    GAI_CHECK(std::isfinite(cfg_.beta1) && cfg_.beta1 >= 0.0f && cfg_.beta1 < 1.0f,
              "training.beta1 must be finite and in [0,1)");
    GAI_CHECK(std::isfinite(cfg_.beta2) && cfg_.beta2 >= 0.0f && cfg_.beta2 < 1.0f,
              "training.beta2 must be finite and in [0,1)");
    GAI_CHECK(std::isfinite(cfg_.eps) && cfg_.eps > 0.0f,
              "training.eps must be finite and > 0");
    GAI_CHECK(std::isfinite(cfg_.grad_clip) && cfg_.grad_clip >= 0.0f,
              "training.grad_clip must be finite and >= 0");
    GAI_CHECK(std::isfinite(cfg_.loss_scale_init) && cfg_.loss_scale_init >= 0.0,
              "training.loss_scale_init must be finite and >= 0");
    GAI_CHECK(cfg_.loss_scale_window > 0, "training.loss_scale_window must be > 0");
    GAI_CHECK(cfg_.ce_chunks >= 1 && cfg_.ce_chunks <= 32,
              "training.ce_chunks must be in [1,32]");
    GAI_CHECK(cfg_.log_every >= 0 && cfg_.eval_every >= 0 && cfg_.eval_batches >= 0 && cfg_.save_every >= 0,
              "training cadence values must be >= 0");
    GAI_CHECK(cfg_.seq_len <= model_.config().max_seq_len,
              "training.seq_len exceeds model.max_seq_len (shorten seq_len or raise max_seq_len)");
    // Only the selected optimizer allocates moments (T4: lion saves ~1.9GB).
    // OPTIMAL-POST GUARD (AdamW vs Lion at full power):
    //   - 1B+ models on T4 16GB MUST use lion (AdamW = +4GB moments -> OOM).
    //     AdamW here is not a warning, it is a guaranteed mid-run killer.
    //   - <=500M models SHOULD use adamw for best quality (proven convergence);
    //     lion is allowed but logged as memory-saver mode.
    // Each technique runs where it is strongest, never as a weak afterthought.
    {
        const i64 total_params = model_.num_parameters();
        if (cfg_.optimizer == "adamw" && total_params > 800000000LL &&
            model_.device() == Device::CUDA) {
            GAI_FAIL(strfmt("optimizer mismatch: %s params with AdamW needs ~%.1fGB extra moments vs Lion and OOMs T4 16GB. "
                            "Use optimizer: lion for 1B models (see configs/t4_1b.yaml, ultra_1b.yaml). "
                            "AdamW is optimal only for <=500M models (flash/pro).",
                            human_count(static_cast<u64>(total_params)).c_str(),
                            static_cast<double>(total_params) * 4.0 / 1e9));
        }
        if (cfg_.optimizer == "lion" && total_params <= 500000000LL) {
            log_info("[opt ] lion on a <=500M model: memory-saver mode (AdamW gives slightly better final loss here)");
        }
        // 1B T4 RECIPE GUARD (14.7/16GB: any B/T increase = instant OOM).
        // The generic OOM guard below would also fire, but this message names
        // the exact known-good recipe so a typo'd yaml fails in seconds with
        // an actionable fix instead of dying 20min into a Kaggle session.
        if (total_params > 800000000LL && model_.device() == Device::CUDA) {
            if (cfg_.batch_size > 1) {
                GAI_FAIL(strfmt("1B/T4 recipe: batch_size=%d OOMs 16GB (measured 14.7GB at B=1/T=512/Lion). "
                                "Use batch_size=1 + grad_accum for throughput (see configs/t4_1b.yaml).",
                                cfg_.batch_size));
            }
            if (cfg_.seq_len > 512) {
                GAI_FAIL(strfmt("1B/T4 recipe: seq_len=%d OOMs 16GB (activations+logits scale with T; T=512 fits, T=1024 kills). "
                                "Use seq_len=512 for 1B on T4 (see configs/t4_1b.yaml).",
                                cfg_.seq_len));
            }
        }
    }
    // Freeze FIRST (P2-3): the optimizer skips moments for frozen parameters,
    // so the flags must be set before it is constructed. Previously the
    // freeze ran ~70 lines later and the embedding m+v (196MB) was allocated
    // and saved needlessly on every frozen SFT run.
    if (cfg_.freeze_embeddings) {
        Parameter* tok_emb_early = model_.find_parameter("tok_embeddings");
        GAI_CHECK(tok_emb_early != nullptr, "freeze_embeddings: tok_embeddings not found");
        tok_emb_early->frozen = true;
        frozen_emb_ = tok_emb_early;
    } else {
        frozen_emb_ = nullptr;
    }
    use_lion_ = (cfg_.optimizer == "lion");
    use_muon_ = (cfg_.optimizer == "muon");
    if (use_muon_)       opt_muon_ = std::make_unique<Muon>(model_, make_muon(cfg_));
    else if (use_lion_) opt_lion_ = std::make_unique<Lion>(model_, make_lion(cfg_));
    else                opt_adam_ = std::make_unique<AdamW>(model_, make_adam(cfg_));
    if (use_muon_ && make_muon(cfg_).lr < 0.005f) {
        log_warn("[opt ] muon with lr < 0.005 learns very slowly (orthogonal steps "
                 "want ~0.01-0.03); consider raising training.learning_rate");
    }
    // NOTE: sched_ is (re)built in run() once the exact planned total is
    // known (steps mode and epochs mode differ). Never trust a default here.

    state_.seed = cfg_.seed;

    // mixed-precision compute: fp16/bf16 GEMMs on CUDA, fp32 everywhere else.
    // CPU ignores the flag (always fp32); the loss scaler is harmless there.
    // FIX: BF16 tensor cores exist only on Ampere+ (cc>=8). T4 (sm75) has FP16
    // cores only, so a bf16 request on T4 must fall back to fp32+fp16, never to
    // a slow non-tensor BF16 path. TF32 is opt-out via GAI_TF32=0.
    // FIX (P0-2c): max|dlogits| after the model's sum-conversion is exactly
    // eff_scale (SCE emits p/ntok, peak 1/ntok; the model multiplies by
    // eff_scale*ntok). fp16 max normal is 65504, so the shipped default of
    // 65536 overflowed to inf on step 1 and the scaler backed off blindly.
    // Clamp here — the one place that both scales AND unscales.
    loss_scale_ = cfg_.loss_scale_init > 0.0 ? cfg_.loss_scale_init : 1.0;
    if (loss_scale_ > kLossScaleMax) {
        log_warn(strfmt("[scaler] loss_scale_init %.0f exceeds the fp16-safe max %.0f "
                        "(|dlogits| peaks at the scale itself); clamped",
                        loss_scale_, kLossScaleMax));
        loss_scale_ = kLossScaleMax;
    }
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

    // (Freeze flags were applied above, before optimizer construction, so
    // frozen moments are never allocated. This only logs the policy now.)
    if (cfg_.freeze_embeddings) {
        GAI_CHECK(frozen_emb_ != nullptr, "freeze_embeddings: internal error");
        log_info("[sft ] embeddings FROZEN (no updates at all, no moments stored)");
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
    BatchSpec spec{cfg_.batch_size, cfg_.seq_len, cfg_.pack_sequences};
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
    if (cfg_.is_sft()) {
        GAI_CHECK(train_loader_.all_shards_have_mask(),
                  "SFT training shards must contain an assistant-span loss mask");
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

    if (cfg_.ce_chunks > 1)
        log_info(strfmt("[mem ] chunked CE x%d: [N,V] logits+dlogits shrink to row-block scratch", cfg_.ce_chunks));
    // ---- activation checkpointing mode resolved BEFORE arenas (T4-P1-15):
    // the segmented path never touches act_, so allocating the full arena
    // alongside ckpt_act_ wastes exactly the memory checkpointing should save.
    use_ckpt_ = cfg_.activation_checkpointing && cfg_.ckpt_segments > 1 && cfg_.batch_size > 1;
    if (use_ckpt_)
        act_ = Activations{};  // empty: forward_backward_micro uses ckpt_act_ only
    else
        act_ = model_.make_activations(cfg_.batch_size, cfg_.seq_len, true, cfg_.ce_chunks);
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
    dev_segments_ = Tensor::empty({static_cast<i64>(cfg_.batch_size) * cfg_.seq_len}, DType::I32, model_.device());

    // ---- activation checkpointing via batch-splitting (T4 path)
    // (use_ckpt_ resolved above, before arena allocation)
    if (cfg_.activation_checkpointing && !use_ckpt_) {
        log_warn("[ckpt-act] activation_checkpointing on but batch_size==1 or segments==1: no effect "
                 "(increase batch_size>=2 or grad_accum instead)");
    }
    if (use_ckpt_) {
        int seg = cfg_.ckpt_segments;
        if (seg > cfg_.batch_size) seg = cfg_.batch_size;
        int segB = (cfg_.batch_size + seg - 1) / seg;
        ckpt_act_ = model_.make_activations(segB, cfg_.seq_len, true, cfg_.ce_chunks);
        ckpt_ids_ = Tensor::empty({static_cast<i64>(segB) * cfg_.seq_len}, DType::I32, model_.device());
        ckpt_targets_ = Tensor::empty({static_cast<i64>(segB) * cfg_.seq_len}, DType::I32, model_.device());
        ckpt_segments_ = Tensor::empty({static_cast<i64>(segB) * cfg_.seq_len}, DType::I32, model_.device());
        log_info(strfmt("[ckpt-act] ON: micro-batch %d split into segments of B=%d (peak act ~1/%d, grads identical)",
                        cfg_.batch_size, segB, seg));
    }

    log_info(strfmt("[mem ] activations train %s%s | eval %s | optimizer %s",
                    human_bytes(act_.bytes + (use_ckpt_ ? ckpt_act_.bytes : 0)).c_str(),
                    use_ckpt_ ? " (segmented; full arena not allocated)" : "",
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
        // Exactly one training arena is resident now (T4-P1-15: the unused
        // full arena is no longer allocated in checkpoint mode).
        size_t live_train_act = act_.bytes + (use_ckpt_ ? ckpt_act_.bytes : 0);
        size_t peak_act = live_train_act;
        size_t fp16_cache = cfg_.fp16_weight_cache ? static_cast<size_t>(params * 2) : 0;
        size_t need_core = params * 8 + fp16_cache + opt_state_bytes() + peak_act + eval_act_.bytes;
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
        // P1-36: fail closed on MATH recipe drift (same weights would define
        // a different model function). Peek is a header-only read, so this
        // gate runs before a single weight loads. Loss weights (aux/z/jitter)
        // stay warn-only inside arch_match: retuning them on resume is legit.
        {
            ModelConfig ckpt_cfg;
            TrainState peek_st;
            if (Checkpoint::peek(resume_path, ckpt_cfg, peek_st)) {
                const ModelConfig& cur = model_.config();
                // PRO: yarn_low/high و rope_type يغيران الدالة الرياضية أيضا
                // (ramp/sincos) فيدخلان بوابة math_same الصارمة مثل theta/scale.
                const bool math_same =
                    ckpt_cfg.rope_theta == cur.rope_theta &&
                    ckpt_cfg.rope_scale == cur.rope_scale &&
                    ckpt_cfg.rope_yarn_mscale == cur.rope_yarn_mscale &&
                    ckpt_cfg.rope_yarn_low == cur.rope_yarn_low &&
                    ckpt_cfg.rope_yarn_high == cur.rope_yarn_high &&
                    ckpt_cfg.rope_type == cur.rope_type &&
                    ckpt_cfg.rms_eps == cur.rms_eps &&
                    ckpt_cfg.max_seq_len == cur.max_seq_len;
                if (!math_same && !cfg_.allow_recipe_drift) {
                    GAI_FAIL("resume recipe drift: checkpoint was saved with different math "
                             "(rope_theta/scale/yarn, rms_eps, max_seq_len). Refusing resume: same "
                             "weights would compute a different model. Restore the original recipe "
                             "or pass --allow-recipe-drift (research only)");
                }
                if (!math_same)
                    log_warn("[ckpt] resuming WITH recipe drift (--allow-recipe-drift): numerics differ from save time");
            }
        }
        bool moments_restored = false;
        bool ok = use_muon_ ? Checkpoint::load(resume_path, model_, opt_muon_.get(), state_, &moments_restored)
                    : use_lion_ ? Checkpoint::load(resume_path, model_, opt_lion_.get(), state_, &moments_restored)
                                : Checkpoint::load(resume_path, model_, opt_adam_.get(), state_, &moments_restored);
        if (ok) {
            // DeepSeek resume rule: bias-correction t_ must match moments.
            // Fresh moments + t_=N => m_hat≈(1-b)*g (~10x too small first
            // steps). Only carry t_=N when moments actually restored.
            if (moments_restored) {
                opt_set_step(state_.step);
            } else {
                opt_set_step(0);
                log_warn(strfmt("[ckpt] moments NOT restored (kind switch/corrupt/legacy); "
                                "optimizer t_=0 with weights at step %lld (loud, not silent)",
                                (long long)state_.step));
            }
            train_loader_.set_state(state_.loader);
            // restore the dynamic loss scaler exactly (v3 checkpoint fields)
            if (cfg_.loss_scale_init > 0.0 && state_.loss_scale >= 1.0) {
                loss_scale_ = state_.loss_scale;
                clean_steps_ = state_.clean_steps;
            }
            // DeepSeek DDP-resume rule (bit-exact): the checkpoint holds
            // rank0's stream, but every rank's stream is deterministic from
            // (base_seed + rank salt + batches consumed). Rank0 restores via
            // set_state() above (exact, zero cost). Other ranks rebuild via
            // reseed(rank_seed)+skip(saved_batches) — same count, own seed —
            // so streams stay disjoint across sessions. Resuming with a
            // different world_size fails fast (data would repeat/shrink).
#ifdef GAI_CUDA
            if (dist_ && dist_->world_size() > 1) {
                const int cur_ws = dist_->world_size();
                const int saved_ws = state_.ddp_world > 0 ? state_.ddp_world : 1;
                if (saved_ws != cur_ws) {
                    GAI_FAIL(strfmt("DDP resume world mismatch: checkpoint saved with %d ranks but now %d "
                                    "(per-rank RNG streams would repeat/shrink; restart with matching WORLD_SIZE "
                                    "or --resume none)", saved_ws, cur_ws));
                }
                const int rank = dist_->global_rank();
                if (rank > 0) {
                    const i64 saved_batches = state_.loader.batches;
                    const u64 rank_seed = cfg_.seed + static_cast<u64>(rank) * 1000003ULL;
                    train_loader_.reseed(rank_seed);
                    if (saved_batches > 0) train_loader_.skip_batches(saved_batches);
                    log_info(strfmt("[ckpt] DDP resume: rank %d rebuilt exact stream (seed %llu + %lld batches)",
                                    rank, (unsigned long long)rank_seed, (long long)saved_batches));
                }
            }
#else
            (void)state_;
#endif
            log_info(strfmt("[ckpt] resumed from %s at step %lld (%s tokens seen, scaler %.0f)",
                            resume_path.c_str(), static_cast<long long>(state_.step),
                            human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                            loss_scale_));
        } else {
            model_.init_weights(cfg_.seed);
            log_warn("[ckpt] failed to load " + resume_path + "; model reset before scratch run");
        }
    }
    if (cfg_.fp16_weight_cache) {
        model_.enable_fp16_weight_cache(true);
        log_info(strfmt("[prec] persistent fp16 weight cache: %s",
                        human_bytes(model_.fp16_weight_cache_bytes()).c_str()));
    }
}

Trainer::~Trainer() {
    stop_prefetch();
    stop_ckpt_writer();
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
    // Clamp to fp16-safe max: |dlogits| peaks at the scale itself, fp16 max is 65504.
    // 16384 leaves 4x headroom for GEMM internal accumulation.
    static constexpr double kLossScaleMax = 16384.0;
    double scale = std::min(loss_scale_, kLossScaleMax);
    return static_cast<float>(scale);
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
        double grown = std::min(kLossScaleMax, loss_scale_ * 2.0);
        if (grown != loss_scale_) {
            loss_scale_ = grown;
            log_info(strfmt("[scaler] %d clean steps, scale -> %.0f",
                            cfg_.loss_scale_window, loss_scale_));
        }
    }
}

double Trainer::forward_backward_micro(const Batch& batch, float dscale, i64* out_ntok) {
    // Balance stats refresh ONLY on steps that will evaluate (the report is
    // read at eval cadence): anywhere else the MoE aux path is fully
    // device-resident (audit P1: zero per-layer syncs).
    const bool want_aux_stats = have_val_ && cfg_.eval_every > 0 &&
                                ((state_.step + 1) % cfg_.eval_every == 0);
    if (!use_ckpt_) {
        i64 N = static_cast<i64>(batch.B) * batch.T;
        device_copy(dev_ids_.data_ptr(), model_.device(), batch.ids.data(), Device::CPU, N * sizeof(i32));
        device_copy(dev_targets_.data_ptr(), model_.device(), batch.targets.data(), Device::CPU, N * sizeof(i32));
        device_copy(dev_segments_.data_ptr(), model_.device(), batch.segment_ids.data(), Device::CPU, N * sizeof(i32));
        i64 ntok = 0;
        double l = model_.forward_backward(dev_ids_.i32p(), dev_targets_.i32p(),
                                           batch.B, batch.T, act_, &ntok, dscale,
                                           want_aux_stats, dev_segments_.i32p());
        if (frozen_emb_ && frozen_emb_->g.defined()) frozen_emb_->g.zero_();
        if (out_ntok) *out_ntok = ntok;
        return l;
    }
    // ---- checkpointed path: split along B, grads accumulate identically ----
    // FIX: old code allocated 2 vectors per segment per micro per step.
    // Reuse persistent monotonic staging (see trainer.h) -> zero per-step mallocs.
    const int B = batch.B, T = batch.T;
    const int segB = ckpt_act_.B;   // precomputed ceil(B/segments)
    double loss_num = 0.0;
    i64 ntok_tot = 0;
    for (int b0 = 0; b0 < B; b0 += segB) {
        int curB = std::min(segB, B - b0);
        i64 curN = static_cast<i64>(curB) * T;
        size_t src = static_cast<size_t>(b0) * static_cast<size_t>(T);
        device_copy(ckpt_ids_.data_ptr(), model_.device(), batch.ids.data() + src, Device::CPU,
                    static_cast<size_t>(curN) * sizeof(i32));
        device_copy(ckpt_targets_.data_ptr(), model_.device(), batch.targets.data() + src, Device::CPU,
                    static_cast<size_t>(curN) * sizeof(i32));
        device_copy(ckpt_segments_.data_ptr(), model_.device(), batch.segment_ids.data() + src, Device::CPU,
                    static_cast<size_t>(curN) * sizeof(i32));
        i64 ntok = 0;
        double l = model_.forward_backward(ckpt_ids_.i32p(), ckpt_targets_.i32p(),
                                           curB, T, ckpt_act_, &ntok, dscale,
                                           want_aux_stats, ckpt_segments_.i32p());
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
        device_copy(dev_segments_.data_ptr(), model_.device(), batch.segment_ids.data(), Device::CPU, N * sizeof(i32));

        Tensor& logits = model_.forward(dev_ids_.i32p(), batch.B, batch.T, eval_act_, dev_segments_.i32p());
        double sum = 0.0;
        i64 n = 0;
        // DeepSeek eval parity: train adds z_loss*logZ^2 (1e-4 in all MoE
        // yamls). Omitting it made val systematically lower and let
        // logit-explosion win best.ckpt. Include it so train/val/best are
        // comparable; aux is train-only regularization and stays excluded.
        ops::softmax_cross_entropy(model_.device(), logits.f32(), dev_targets_.i32p(),
                                   nullptr, N, model_.config().vocab_size, &sum, &n,
                                   model_.config().z_loss_scale);
        total += sum;
        ntok_total += n;
    }
    return ntok_total > 0 ? total / static_cast<double>(ntok_total) : 0.0;
}

void Trainer::log_step(double loss, float lr, double gnorm, double dt, i64 ntok) {
    double tps = dt > 0 ? static_cast<double>(ntok) / dt : 0.0;
    i64 remaining = total_steps_ - state_.step;
    double eta = remaining > 0 ? remaining * dt : 0.0;
    // PERF telemetry (audit P2-2): launch/transfer totals since run start.
    // Deltas between log lines / grad_accum = per-step launch cost — the
    // number to drive down alongside tok/s (Nsight on T4 for hotspots).
    log_info(strfmt("step %6lld | loss %7.4f | ema %7.4f | ppl %8.2f | lr %.3e | gnorm %6.3f "
                    "| %7.0f tok/s | %s | eta %s | perf [%s]",
                    static_cast<long long>(state_.step), loss, ema_loss_,
                    std::exp(std::min(20.0, ema_loss_)), lr, gnorm, tps,
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(eta).c_str(),
                    ops::perf_report().c_str()));
}

double Trainer::opt_step(float lr, float grad_scale) {
    if (use_muon_) return opt_muon_->step(lr, grad_scale);
    if (use_lion_) return opt_lion_->step(lr, grad_scale);
    return opt_adam_->step(lr, grad_scale);
}

size_t Trainer::opt_state_bytes() const {
    if (use_muon_) return opt_muon_->state_bytes();
    if (use_lion_) return opt_lion_->state_bytes();
    return opt_adam_->state_bytes();
}

void Trainer::opt_set_step(i64 t) {
    if (use_muon_) opt_muon_->set_step_count(t);
    else if (use_lion_) opt_lion_->set_step_count(t);
    else opt_adam_->set_step_count(t);
}

void Trainer::save(const std::string& name) {
    if (!is_main_rank()) {
#ifdef GAI_CUDA
        if (dist_ && dist_->world_size() > 1) dist_->barrier();
#endif
        return;
    }

    quiesce_prefetch();
    try {
        state_.loader = train_loader_.get_state();
        state_.loss_scale = loss_scale_;
        state_.clean_steps = clean_steps_;
        state_.tok_vocab = model_.config().vocab_size;
        state_.sched_total = total_steps_ > 0 ? total_steps_ : sched_.total();
        state_.sched_warmup = sched_.warmup();
        state_.sched_peak = sched_.peak();
        state_.sched_min_ratio = cfg_.min_lr_ratio;
        state_.sched_decay_frac = cfg_.sched_decay_frac;
        state_.sched_kind = (cfg_.scheduler == "wsd") ? 1 : 0;
        {
            int ws = 1;
#ifdef GAI_CUDA
            if (dist_ && dist_->world_size() > 1) ws = dist_->world_size();
#endif
            state_.ddp_world = ws;
        }
        if (!snapshot_cache_ || snapshot_step_ != state_.step) {
            if (use_muon_) snapshot_cache_ = std::make_shared<CheckpointSnapshot>(
                Checkpoint::capture(model_, *opt_muon_, state_));
            else if (use_lion_) snapshot_cache_ = std::make_shared<CheckpointSnapshot>(
                Checkpoint::capture(model_, *opt_lion_, state_));
            else snapshot_cache_ = std::make_shared<CheckpointSnapshot>(
                Checkpoint::capture(model_, *opt_adam_, state_));
            snapshot_step_ = state_.step;
        }
    } catch (...) {
        resume_prefetch();
        throw;
    }
    resume_prefetch();

    const std::string path = (fs::path(cfg_.checkpoint_dir) / name).string();
    std::shared_ptr<CheckpointSnapshot> snapshot = snapshot_cache_;
    save_async([path, snapshot]() {
        Timer t;
        Checkpoint::save(*snapshot, path);
        log_info(strfmt("[ckpt] saved %s (%s)", path.c_str(),
                        human_duration(t.seconds()).c_str()));
    });
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) dist_->barrier();
#endif
}

i64 Trainer::planned_total() const { return total_steps_; }

void Trainer::run() {
    // DeepSeek DDP budgeting: global consumption = per-rank * world_size.
    // Old code planned steps from per-rank throughput, so 4xGPU ran 4x too
    // many steps with a 4x-stretched scheduler. Resolve ws once here.
    int world_sz = 1;
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) world_sz = dist_->world_size();
#endif
    // ---- resolve the explicit schedule into an exact step budget
    if (cfg_.max_steps > 0) {
        total_steps_ = cfg_.max_steps;
        log_info(strfmt("[sched] steps mode: %lld optimizer steps (per-rank %s tok/step, global %s tok/step x%d)",
                        static_cast<long long>(total_steps_),
                        human_count(static_cast<u64>(cfg_.tokens_per_step())).c_str(),
                        human_count(static_cast<u64>(cfg_.tokens_per_step_global(world_sz))).c_str(),
                        world_sz));
    } else {
        total_steps_ = cfg_.epoch_steps(train_loader_.total_tokens(), world_sz);
        GAI_CHECK(total_steps_ > 0,
                  "epochs mode needs data: no training tokens found in " + cfg_.data_dir);
        // §10 honesty: random-window sampling with replacement (see DataLoader),
        // so 'epochs' is a stochastic token BUDGET (coverage approximate), not
        // classic full-pass epochs. Global ordering across DDP ranks is disjoint
        // by rank salt, but not a single deterministic permutation.
        log_info(strfmt("[sched] epochs mode: %d epochs x %s tokens = %lld steps (per-rank %s, global %s x%d; stochastic windows, not classic passes)",
                        cfg_.epochs,
                        human_count(train_loader_.total_tokens()).c_str(),
                        static_cast<long long>(total_steps_),
                        human_count(static_cast<u64>(cfg_.tokens_per_step())).c_str(),
                        human_count(static_cast<u64>(cfg_.tokens_per_step_global(world_sz))).c_str(),
                        world_sz));
    }
    // DeepSeek schedule guard: warmup >= total pins the entire run in the
    // warmup branch (<2% peak) and decay_frac<=0 disables decay silently.
    GAI_CHECK(cfg_.warmup_steps >= 0 && cfg_.warmup_steps < total_steps_,
              strfmt("training.warmup_steps=%lld must be in [0,total=%lld)",
                     (long long)cfg_.warmup_steps, (long long)total_steps_));
    GAI_CHECK(cfg_.sched_decay_frac > 0.0f && cfg_.sched_decay_frac <= 1.0f,
              "training.sched_decay_frac must be in (0,1]");
    // scheduler is built HERE from the true total (never from a default).
    // lion uses its own effective LR (0.1x), muon its direct LR; the
    // scheduler shapes whichever peak identically.
    float sched_peak = use_muon_ ? make_muon(cfg_).lr
                     : use_lion_ ? make_lion(cfg_).lr
                                 : cfg_.learning_rate;
    sched_ = LrScheduler(sched_peak, cfg_.warmup_steps,
                         total_steps_, cfg_.min_lr_ratio,
                         cfg_.scheduler, cfg_.sched_decay_frac);
    log_info(strfmt("[sched] %s warmup %lld total %lld min %.0f%% decay_frac %.2f",
                    cfg_.scheduler.c_str(), static_cast<long long>(cfg_.warmup_steps),
                    static_cast<long long>(total_steps_),
                    cfg_.min_lr_ratio * 100.0, cfg_.sched_decay_frac));
    // v8 resume guard: changing total/warmup/peak/type retroactively moves
    // lr_at(N) for ALREADY-RUN steps (non-monotonic jump). Warn loudly.
    if (state_.sched_total > 0 && state_.sched_total != total_steps_) {
        log_warn(strfmt("[sched] RESUME MISMATCH: checkpoint planned total %lld but now %lld "
                        "(max_steps/epochs/data changed) — past LR curve reshaped; "
                        "keep the original schedule to stay bit-consistent",
                        (long long)state_.sched_total, (long long)total_steps_));
    }
    if (state_.sched_warmup > 0 && state_.sched_warmup != cfg_.warmup_steps) {
        log_warn(strfmt("[sched] RESUME MISMATCH: checkpoint warmup %lld but now %lld",
                        (long long)state_.sched_warmup, (long long)cfg_.warmup_steps));
    }
    if (state_.sched_peak > 0.0f && std::fabs(state_.sched_peak - sched_peak) > 1e-9f) {
        log_warn(strfmt("[sched] RESUME MISMATCH: checkpoint peak %.3e but now %.3e",
                        (double)state_.sched_peak, (double)sched_peak));
    }
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

    // Phase 1C: start async prefetch; Phase 1D: start background ckpt writer.
    start_prefetch();
    start_ckpt_writer();
    // Prime the first prefetch (prefetch thread already called next() into slot).

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
            // Phase 1C: next_train_batch() returns the prefetched batch and
            // immediately triggers the background thread to load the next one,
            // so the GPU is never idle waiting for disk I/O.
            if (!next_train_batch(batch)) {
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
        // DeepSeek rule for ntok==0 (all-masked SFT step): grads are pure
        // aux (∝N) so ANY divisor is dimensionally wrong — old code divided
        // by grad_accum leaving a ~100x phantom step. True no-op instead:
        // skip the optimizer, keep scheduler advancing, never NaN.
        double gnorm = 0.0;
        if (ntok_global > 0) {
            float grad_scale = 1.0f / (static_cast<float>(ntok_global) * dscale);
            gnorm = opt_step(lr, grad_scale);
            model_.mark_weights_dirty();
        } else {
            static int warned_empty = 0;
            if (warned_empty++ < 3)
                log_warn("[train] all-masked step (ntok=0): optimizer skipped (aux-only grads)");
            gnorm = 0.0;
            for (Parameter* p : model_.parameters()) {
                if (p->g.defined()) p->g.zero_();
            }
        }
        scaler_update(gnorm);

        ++state_.step;
        // DeepSeek accounting: tokens_seen tracks GLOBAL supervised tokens
        // (ntok_global == local on single GPU, sum across ranks on DDP) so
        // budgets/ETA stay honest at any world_size.
        state_.tokens_seen += ntok_global;
        state_.last_loss = loss;

        double dt = step_timer.seconds();
        bool main = is_main_rank();
        if (main && cfg_.log_every > 0 && state_.step % cfg_.log_every == 0) {
            log_step(loss, lr, gnorm, dt, ntok_global);
        }

        // T4-P1-23: validation forwards run on rank 0 ONLY. Every rank was
        // executing identical eval passes while only rank 0 logged the
        // result — pure duplicated GPU work each eval cadence. Non-main
        // ranks still reset MoE balance stats (they accumulate locally in
        // training forwards). Skipping evaluate() elsewhere is safe: each
        // rank owns an independent val_loader whose state isn't checkpointed.
        if (have_val_ && cfg_.eval_every > 0 && state_.step % cfg_.eval_every == 0) {
            if (main) {
                double vl = evaluate(cfg_.eval_batches);
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

    // Phase 1C: stop prefetch thread before final save.
    stop_prefetch();
    // FIX P1-6: skip the unconditional final save when the loop just saved
    // (worst case was a double ~8-12GB write on the same step: best.ckpt
    // inside eval + last.ckpt on cadence + last.ckpt here).
    if (!(cfg_.save_every > 0 && state_.step % cfg_.save_every == 0))
        save("last.ckpt");
    // Phase 1D: drain the async writer before printing the done line.
    wait_for_save();
    stop_ckpt_writer();
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

    // Phase 1C+1D: start prefetch + background ckpt writer for SFT loop.
    start_prefetch();
    start_ckpt_writer();

    while (state_.step < total_steps_) {
        step_timer.reset();
        model_.zero_grad();
        const float dscale = scaler_for_step();   // frozen within the step
        {
            int rank = 0;
#ifdef GAI_CUDA
            if (dist_ && dist_->world_size() > 1) rank = dist_->global_rank();
#endif
            u64 ctx = cfg_.seed ^ (static_cast<u64>(state_.step) * 0x9E3779B97F4A7C15ULL)
                                ^ (static_cast<u64>(rank) * 0xBF58476D1CE4E5B9ULL);
            ops::set_moe_jitter_seed(ctx);
        }

        // P0-05 token-weighted grads (same as pretrain; critical for SFT where
        // masks make supervised counts vary strongly per micro).
        double loss_num = 0.0;
        i64    ntok_step = 0;
        int    micro_done = 0;

        for (int micro = 0; micro < cfg_.grad_accum; ++micro) {
            if (!next_train_batch(batch)) {
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

        // Same all-masked no-op rule as pretrain (see above): never divide
        // aux-only grads by grad_accum.
        double gnorm = 0.0;
        if (ntok_global > 0) {
            float grad_scale = 1.0f / (static_cast<float>(ntok_global) * dscale);
            gnorm = opt_step(lr, grad_scale);
            model_.mark_weights_dirty();
        } else {
            static int warned_empty_sft = 0;
            if (warned_empty_sft++ < 3)
                log_warn("[sft] all-masked step (ntok=0): optimizer skipped (aux-only grads)");
            gnorm = 0.0;
            for (Parameter* p : model_.parameters()) {
                if (p->g.defined()) p->g.zero_();
            }
        }
        scaler_update(gnorm);

        ++state_.step;
        state_.tokens_seen += ntok_global;
        state_.last_loss = loss;

        double dt = step_timer.seconds();
        bool main = is_main_rank();
        if (main && cfg_.log_every > 0 && state_.step % cfg_.log_every == 0) {
            log_step(loss, lr, gnorm, dt, ntok_global);
        }

        // T4-P1-23: rank-0-only validation (see pretrain loop for rationale).
        if (have_val_ && cfg_.eval_every > 0 && state_.step % cfg_.eval_every == 0) {
            if (main) {
                double vl = evaluate(cfg_.eval_batches);
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

    // Phase 1C: stop prefetch thread before final save.
    stop_prefetch();
    // FIX P1-6 (SFT mirror): skip redundant final save (see pretrain loop).
    if (!(cfg_.save_every > 0 && state_.step % cfg_.save_every == 0))
        save("last.ckpt");
    // Phase 1D: drain async writer.
    wait_for_save();
    stop_ckpt_writer();
    log_info(strfmt("---------------- SFT done: %lld steps, %s tokens, %s --------------------",
                    static_cast<long long>(state_.step),
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(wall_.seconds()).c_str()));
}

// ---- distributed training ----
void Trainer::init_distributed() {
#ifdef GAI_CUDA
    if (model_.device() != Device::CUDA) return;
    // DeepSeek DDP contract: training.ddp is the master switch.
    // ddp:false on multi-GPU = intentional single-GPU run (no NCCL even if
    // WORLD_SIZE>1 from a stale env). ddp:true on 1 GPU fails fast instead
    // of silently running single. WORLD_SIZE>1 env forces DDP on (torchrun).
    {
        const char* ws = std::getenv("WORLD_SIZE");
        int env_ws = ws ? std::atoi(ws) : 1;
        if (!cfg_.ddp && env_ws <= 1) {
            log_info("[dist] ddp=off (yaml) single-GPU run");
            return;
        }
        if (cfg_.ddp) log_info("[dist] ddp=on (yaml) requesting multi-GPU");
        else log_info("[dist] WORLD_SIZE>1 forces DDP on (yaml ddp=off overridden)");
    }

    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus <= 1) {
        if (cfg_.ddp) GAI_FAIL("training.ddp=true but only 1 GPU visible (WORLD_SIZE mismatch?)");
        return;
    }

    DistributedContext::Config dcfg = distributed_config_from_env(num_gpus);
    if (dcfg.world_size <= 1) {
        if (cfg_.ddp) GAI_FAIL("training.ddp=true but world_size==1 (launch 2+ ranks or set ddp=false)");
        return;
    }

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
    // FIX: broadcast+barrier are collective — ALL ranks must enter sync_model(),
    // including rank 0. Old code ran it only on rank!=0 so ranks 1..3 hung
    // forever waiting for rank 0 (4xT4 startup deadlock, Kaggle training hang).
    sync_model();
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
        // FIX P2 (silent DDP divergence): a CPU-resident grad was skipped
        // from the NCCL SUM yet the step still divided by global ntok,
        // so ranks diverged silently. Fail fast instead.
        if (p->g.device() != Device::CUDA)
            GAI_FAIL("DDP requires all trainable grads on CUDA (param '" + p->name +
                     "' is CPU); move the model to CUDA or disable ddp");
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
            static_cast<size_t>(dist_fused_.numel()) < small_total) {
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
        if (p->w.device() != Device::CUDA)
            GAI_FAIL("DDP broadcast requires all weights on CUDA (param '" + p->name +
                     "' is CPU); move the model to CUDA or disable ddp");

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
    i64 res = static_cast<i64>(std::llround(out));
    return res > 0 ? res : 0;
#else
    (void)dist_;
    return local > 0 ? local : 0;
#endif
}

// ============================================================
// Phase 1D — Background Checkpoint Writer
// ============================================================
// save_async() captures the current TrainState + loader/optimizer snapshot by
// value inside a closure and enqueues it. If a previous write is still queued
// (not yet started), it is replaced by the newer one (we always want the
// freshest checkpoint). The writer thread blocks until it has work, executes
// the closure, then sleeps again. wait_for_save() drains the queue before the
// training run exits so no data is lost on Kaggle preemption.

void Trainer::start_ckpt_writer() {
    if (ckpt_writer_thread_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(ckpt_mutex_);
        ckpt_stop_ = false;
        ckpt_failed_ = false;
        ckpt_error_.clear();
    }
    ckpt_writer_thread_ = std::thread([this]() {
        while (true) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> lk(ckpt_mutex_);
                ckpt_cv_.wait(lk, [this] { return !pending_saves_.empty() || ckpt_stop_; });
                if (pending_saves_.empty() && ckpt_stop_) break;
                fn = std::move(pending_saves_.front());
                pending_saves_.pop_front();
                ckpt_busy_ = true;
            }
            if (fn) {
                try {
                    fn();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(ckpt_mutex_);
                    ckpt_failed_ = true;
                    ckpt_error_ = e.what();
                    log_warn(std::string("[ckpt-async] write failed: ") + e.what());
                } catch (...) {
                    std::lock_guard<std::mutex> lk(ckpt_mutex_);
                    ckpt_failed_ = true;
                    ckpt_error_ = "unknown checkpoint write failure";
                    log_warn("[ckpt-async] write failed with unknown exception");
                }
            }
            {
                std::lock_guard<std::mutex> lk(ckpt_mutex_);
                ckpt_busy_ = false;
            }
            ckpt_cv_.notify_all();
        }
    });
}

void Trainer::stop_ckpt_writer() {
    {
        std::lock_guard<std::mutex> lk(ckpt_mutex_);
        ckpt_stop_ = true;
    }
    ckpt_cv_.notify_all();
    if (ckpt_writer_thread_.joinable()) ckpt_writer_thread_.join();
}

void Trainer::save_async(std::function<void()> fn) {
    GAI_CHECK(ckpt_writer_thread_.joinable(), "checkpoint writer is not running");
    {
        std::lock_guard<std::mutex> lk(ckpt_mutex_);
        pending_saves_.push_back(std::move(fn));
    }
    ckpt_cv_.notify_one();
}

void Trainer::wait_for_save() {
    std::unique_lock<std::mutex> lk(ckpt_mutex_);
    ckpt_cv_.wait(lk, [this]{ return pending_saves_.empty() && !ckpt_busy_; });
    if (ckpt_failed_) GAI_FAIL("checkpoint save failed: " + ckpt_error_);
}

// ============================================================
// Phase 1C — Async Prefetch DataLoader
// ============================================================
// start_prefetch() spawns a thread that calls train_loader_.next() in the
// background and signals prefetch_ready_ when done. next_train_batch() swaps
// in the ready buffer and immediately enqueues the next prefetch so the GPU
// is never idle waiting for disk I/O. Thread-safety: train_loader_ is accessed
// only by this thread after training starts (the main thread uses
// next_train_batch instead of train_loader_.next directly).

void Trainer::start_prefetch() {
    {
        std::lock_guard<std::mutex> lk(prefetch_mutex_);
        if (prefetch_running_) return;
        prefetch_ready_ = false;
        prefetch_stop_ = false;
        prefetch_pause_ = false;
        prefetch_in_io_ = false;
        prefetch_running_ = true;
    }
    prefetch_thread_ = std::thread([this]() {
        while (true) {
            std::unique_lock<std::mutex> lk(prefetch_mutex_);
            prefetch_cv_.wait(lk, [this] {
                return prefetch_stop_ || (!prefetch_pause_ && !prefetch_ready_);
            });
            if (prefetch_stop_) break;
            prefetch_in_io_ = true;
            lk.unlock();

            bool ok = false;
            try {
                ok = train_loader_.next(prefetch_batch_);
            } catch (const std::exception&) {
                std::lock_guard<std::mutex> fail_lock(prefetch_mutex_);
                prefetch_stop_ = true;
                prefetch_ready_ = false;
                break;
            }

            lk.lock();
            prefetch_in_io_ = false;
            if (prefetch_stop_) break;
            prefetch_ready_ = ok;
            lk.unlock();
            prefetch_cv_.notify_all();

            lk.lock();
            prefetch_cv_.wait(lk, [this] {
                return prefetch_stop_ || prefetch_pause_ || !prefetch_ready_;
            });
            if (prefetch_stop_) break;
        }
        std::lock_guard<std::mutex> done_lock(prefetch_mutex_);
        prefetch_in_io_ = false;
        prefetch_running_ = false;
    });
}

void Trainer::stop_prefetch() {
    {
        std::lock_guard<std::mutex> lk(prefetch_mutex_);
        if (!prefetch_running_ && !prefetch_thread_.joinable()) return;
        prefetch_stop_ = true;
        prefetch_pause_ = false;
        prefetch_ready_ = false;
    }
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable()) prefetch_thread_.join();
    std::lock_guard<std::mutex> lk(prefetch_mutex_);
    prefetch_running_ = false;
}

void Trainer::quiesce_prefetch() {
    std::unique_lock<std::mutex> lk(prefetch_mutex_);
    if (!prefetch_running_) return;
    prefetch_pause_ = true;
    prefetch_cv_.wait(lk, [this] { return !prefetch_in_io_; });
}

void Trainer::resume_prefetch() {
    {
        std::lock_guard<std::mutex> lk(prefetch_mutex_);
        if (!prefetch_running_) return;
        prefetch_pause_ = false;
    }
    prefetch_cv_.notify_all();
}

bool Trainer::next_train_batch(Batch& out) {
    {
        std::unique_lock<std::mutex> lk(prefetch_mutex_);
        if (!prefetch_running_) {
            lk.unlock();
            return train_loader_.next(out);
        }
        prefetch_cv_.wait(lk, [this] { return prefetch_ready_ || prefetch_stop_; });
        if (!prefetch_ready_) GAI_FAIL("dataloader has no shards mid-run in " + cfg_.data_dir);
        out = std::move(prefetch_batch_);
        prefetch_ready_ = false;
    }
    prefetch_cv_.notify_all();
    return true;
}

} // namespace gai
