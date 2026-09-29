#include "training/trainer.h"
#include "core/ops.h"
#include "core/device.h"
#include "core/common.h"
#include "core/signals.h"
#ifdef GAI_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <filesystem>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

namespace gai {

static constexpr double kLossScaleMax = 16384.0;

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

RecipeCost price_recipe(const ModelConfig& m, const TrainerConfig& t) {
    RecipeCost c;
    const Model::MemoryPlan plan = Model::plan_memory(
        m, t.batch_size, t.seq_len, true, t.ce_chunks,
        t.fp16_weight_cache);
    c.params     = plan.params;
    c.weights    = plan.weights;
    c.grads      = plan.grads;
    c.fp16_cache = plan.fp16_cache;

    size_t act = plan.activations;

    if (t.activation_checkpointing && t.batch_size > 1 && t.ckpt_segments > 1) {
        int seg = t.ckpt_segments;
        if (seg > t.batch_size) seg = t.batch_size;
        act = (act + static_cast<size_t>(seg) - 1) / static_cast<size_t>(seg);
    }
    c.activations = act;

    const bool lion  = (t.optimizer == "lion");
    const bool muon  = (t.optimizer == "muon");
    size_t opt_b = (lion || muon) ? static_cast<size_t>(plan.params) * 4
                                   : static_cast<size_t>(plan.params) * 8;
    if (t.freeze_embeddings) {
        const size_t per = static_cast<size_t>(m.vocab_size) *
                           static_cast<size_t>(m.hidden_size) *
                           static_cast<size_t>((lion || muon) ? 4 : 8);
        opt_b = (opt_b >= per) ? opt_b - per : 0;
    }
    if (muon) {

        const size_t d  = static_cast<size_t>(m.hidden_size);
        const size_t E  = static_cast<size_t>(m.moe_expert_dim);
        const size_t Vd = static_cast<size_t>(m.vocab_size) * d;
        const size_t Fd = static_cast<size_t>(m.intermediate_size) * d;
        size_t rc = std::max(d * d, E * d);
        rc = std::max(rc, Vd);
        rc = std::max(rc, Fd);
        size_t cc = std::max(d * d, E * E);
        cc = std::max(cc, static_cast<size_t>(m.intermediate_size) * static_cast<size_t>(m.intermediate_size));
        c.muon_scratch = (rc * 2 + cc) * sizeof(float);
        opt_b += c.muon_scratch;
    }
    c.opt_state = opt_b;

    c.eval_extra = Model::estimate_activation_bytes_for(
                       m, t.fp16_weight_cache, t.batch_size, t.seq_len,
                       false, 1) / 3;

    const Model::WorkspacePlan wsp =
        Model::workspace_plan(m, t.batch_size, t.seq_len);
    c.gemm_ws = wsp.gemm_bytes;
    c.moe_ws  = wsp.moe_bytes;
    c.nccl    = t.ddp ? (384ull << 20) : 0;

    c.total = c.weights + c.grads + c.fp16_cache + c.opt_state + c.activations +
              c.eval_extra + c.gemm_ws + c.moe_ws + c.nccl;

    c.snapshot = static_cast<size_t>(plan.params) * 4 + opt_b;
    c.gguf     = static_cast<size_t>(plan.params) + c.snapshot / 8;
    c.output_projection = c.snapshot * 2 + c.gguf;
    return c;
}

TrainerConfig TrainerConfig::from_config(const Config& c, bool strict) {
    TrainerConfig t;
    t.data_dir       = c.get_str ("training.data_dir", t.data_dir);

    t.train_prefix   = c.get_str("data.train_prefix", t.train_prefix);
    t.val_prefix     = c.get_str("data.val_prefix", t.val_prefix);
    t.batch_size     = static_cast<int>(c.get_int("training.batch_size", t.batch_size));
    t.seq_len        = static_cast<int>(c.get_int("training.seq_len", t.seq_len));
    t.grad_accum     = static_cast<int>(c.get_int("training.grad_accum", t.grad_accum));
    t.max_steps      = c.get_int("training.max_steps", 0);
    t.epochs         = static_cast<int>(c.get_int("training.epochs", 0));

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

    t.beta2          = c.get_f32("training.beta2", t.optimizer == "lion" ? 0.99f : t.beta2);
    t.eps            = c.get_f32("training.eps", t.eps);
    t.grad_clip      = c.get_f32("training.grad_clip", t.grad_clip);

    t.muon_ns_steps   = static_cast<int>(c.get_int("training.ns_steps", t.muon_ns_steps));
    t.muon_min_ns_dim = static_cast<int>(c.get_int("training.muon_min_ns_dim", t.muon_min_ns_dim));
    t.muon_vec_ratio  = c.get_f32("training.muon_vec_ratio", t.muon_vec_ratio);
    if (t.muon_ns_steps < 1 || t.muon_ns_steps > 10)
        GAI_FAIL("training.ns_steps must be in 1..10");
    if (t.muon_min_ns_dim < 0)
        GAI_FAIL("training.muon_min_ns_dim must be >= 0");
    if (!(t.muon_vec_ratio > 0.0f && t.muon_vec_ratio <= 1.0f))
        GAI_FAIL("training.muon_vec_ratio must be in (0,1]");

    std::string precision = c.get_str("training.precision", "fp32");
    if (precision == "bf16") t.param_dtype = DType::BF16;
    else if (precision == "fp16") t.param_dtype = DType::F16;
    else if (precision == "fp32") t.param_dtype = DType::F32;
    else {

        GAI_FAIL("training.precision must be one of: fp32, fp16, bf16 (got '" + precision + "')");
    }

    t.gemm_fp16          = c.get_bool("training.gemm_fp16", t.gemm_fp16);
    t.fp16_weight_cache  = c.get_bool("training.fp16_weight_cache", t.fp16_weight_cache);
    t.loss_scale_init    = static_cast<double>(c.get_f32("training.loss_scale_init",
                                                         static_cast<float>(t.loss_scale_init)));
    t.loss_scale_max     = static_cast<double>(c.get_f32("training.loss_scale_max",
                                                         static_cast<float>(t.loss_scale_max)));
    if (!(t.loss_scale_max >= 0.0) || !std::isfinite(t.loss_scale_max))
        GAI_FAIL("training.loss_scale_max must be finite and >= 0 (0 = fp16-safe max)");
    t.loss_scale_window  = static_cast<int>(c.get_int("training.loss_scale_window",
                                                      t.loss_scale_window));
    t.log_every      = c.get_int("training.log_every", t.log_every);
    t.eval_every     = c.get_int("training.eval_every", t.eval_every);
    t.eval_batches   = c.get_int("training.eval_batches", t.eval_batches);
    t.save_every     = c.get_int("training.save_every", t.save_every);
    t.checkpoint_dir = c.get_str("training.checkpoint_dir", t.checkpoint_dir);
    t.resume         = c.get_str("training.resume", t.resume);

    t.tok_fingerprint = static_cast<u64>(c.get_int("training.tok_fingerprint",
                                                   static_cast<i64>(t.tok_fingerprint)));
    t.seed           = static_cast<u64>(c.get_int("training.seed", static_cast<i64>(t.seed)));

    t.output_budget_mb = c.get_int("training.output_budget_mb", t.output_budget_mb);
    t.device         = c.get_str("training.device", t.device);
    t.stage          = c.get_str("training.stage", t.stage);
    if (t.stage != "pretrain" && t.stage != "sft" && t.stage != "cpt")
        GAI_FAIL("unknown training.stage '" + t.stage + "' (pretrain|sft|cpt)");

    t.pretrained_checkpoint = c.get_str("training.pretrained_checkpoint", t.pretrained_checkpoint);
    t.allow_no_pretrained   = c.get_bool("training.allow_no_pretrained", t.allow_no_pretrained);
    t.freeze_embeddings    = c.get_bool("training.freeze_embeddings", t.freeze_embeddings);
    t.allow_recipe_drift   = c.get_bool("training.allow_recipe_drift", t.allow_recipe_drift);

    t.resume_mode = c.get_str("training.resume_mode", t.resume_mode);
    if (t.resume_mode != "migrate" && t.resume_mode != "exact")
        GAI_FAIL("unknown training.resume_mode '" + t.resume_mode + "' (exact|migrate)");

    for (const auto& [k, v] : c.flat()) {
        const std::string pre = "data.mix.";
        if (k.size() > pre.size() && k.compare(0, pre.size(), pre) == 0) {
            std::string domain = k.substr(pre.size());

            try {

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

    t.pack_sequences = c.get_bool("training.pack_sequences", false);
    t.ddp = c.get_bool("training.ddp", false);
    t.ddp_grad_compression = c.get_bool("training.ddp_grad_compression", false);

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

    if (c.has("data.data_dir")) {
        const std::string dd = c.get_str("data.data_dir");
        if (!dd.empty() && dd != t.data_dir) {
            GAI_FAIL("data.data_dir '" + dd + "' conflicts with training.data_dir '" +
                     t.data_dir + "' (single shard dir per run; delete the duplicate)");
        }
        if (!dd.empty())
            log_warn("[cfg ] data.data_dir duplicates training.data_dir (legacy; ignored)");
    }

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
              "training.ns_steps", "training.muon_min_ns_dim",
             "training.muon_vec_ratio", "training.output_budget_mb",
             "training.tok_fingerprint",

             "training.precision", "training.gemm_fp16", "training.fp16_weight_cache",
             "training.loss_scale_init",
             "training.loss_scale_max", "training.loss_scale_window", "training.log_every", "training.eval_every",
            "training.eval_batches", "training.save_every", "training.checkpoint_dir",
            "training.resume", "training.seed", "training.device", "training.stage",
            "training.pretrained_checkpoint", "training.allow_no_pretrained",
             "training.allow_recipe_drift", "training.resume_mode",

             "training.freeze_embeddings", "training.activation_checkpointing",
              "training.ckpt_segments", "training.ce_chunks", "training.pack_sequences",
              "training.ddp", "training.ddp_grad_compression",
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

    l.lr           = c.learning_rate * 0.1f;
    l.beta1        = c.beta1;
    l.beta2        = c.beta2;
    l.weight_decay = c.weight_decay;
    l.grad_clip    = c.grad_clip;
    return l;
}

static MuonConfig make_muon(const TrainerConfig& c) {
    MuonConfig m;

    m.lr           = c.learning_rate;
    m.vec_lr_ratio = c.muon_vec_ratio;
    m.beta1        = c.beta1;
    m.beta2        = c.beta2;
    m.eps          = c.eps;
    m.weight_decay = c.weight_decay;
    m.grad_clip    = c.grad_clip;

    m.ns_steps     = c.muon_ns_steps;
    m.min_ns_dim   = c.muon_min_ns_dim;
    return m;
}

static float effective_peak_lr(const TrainerConfig& c) {
    const bool muon = (c.optimizer == "muon");
    const bool lion = !muon && (c.optimizer == "lion");
    return muon ? make_muon(c).lr : lion ? make_lion(c).lr : c.learning_rate;
}

Trainer::Trainer(Model& model, TrainerConfig cfg)
    : model_(model), cfg_(std::move(cfg)) {

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

    {
        const i64 total_params = model_.num_parameters();
        if (cfg_.optimizer == "adamw" && total_params > 800000000LL &&
            model_.device() == Device::CUDA) {
            GAI_FAIL(strfmt("optimizer mismatch: %s params with AdamW needs ~%.1fGB extra moments vs Lion and OOMs T4 16GB. "
                            "Use optimizer: lion for 1B models (see configs/pro_1b_4xt4_legacy.yaml, pro_1b_single.yaml). "
                            "AdamW is optimal only for <=500M models (flash/pro).",
                            human_count(static_cast<u64>(total_params)).c_str(),
                            static_cast<double>(total_params) * 4.0 / 1e9));
        }
        if (cfg_.optimizer == "lion" && total_params <= 500000000LL) {
            log_info("[opt ] lion on a <=500M model: memory-saver mode (AdamW gives slightly better final loss here)");
        }

        if (total_params > 800000000LL && model_.device() == Device::CUDA) {
            if (cfg_.batch_size > 1) {
                GAI_FAIL(strfmt("1B/T4 recipe: batch_size=%d OOMs 16GB (measured 14.7GB at B=1/T=512/Lion). "
                                "Use batch_size=1 + grad_accum for throughput (see configs/pro_1b_4xt4_legacy.yaml).",
                                cfg_.batch_size));
            }
            if (cfg_.seq_len > 512) {
                GAI_FAIL(strfmt("1B/T4 recipe: seq_len=%d OOMs 16GB (activations+logits scale with T; T=512 fits, T=1024 kills). "
                                "Use seq_len=512 for 1B on T4 (see configs/pro_1b_4xt4_legacy.yaml).",
                                cfg_.seq_len));
            }
        }
    }

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

    state_.seed = cfg_.seed;

    loss_scale_ = cfg_.loss_scale_init > 0.0 ? cfg_.loss_scale_init : 1.0;
    if (loss_scale_ > loss_scale_cap()) {
        log_warn(strfmt("[scaler] loss_scale_init %.0f exceeds the cap %.0f "
                        "(|dlogits| peaks at the scale itself); clamped",
                        loss_scale_, loss_scale_cap()));
        loss_scale_ = loss_scale_cap();
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

    {
        bool scaling = (cfg_.loss_scale_init > 0.0) && ops::gemm_fp16_enabled();
        log_info(strfmt("[prec] loss_scaling=%s (init %.0f, scaler now %.0f)",
                        scaling ? "ENABLED (fp16 GEMMs active)" : "DISABLED (pure fp32 path)",
                        cfg_.loss_scale_init, scaling ? loss_scale_ : 1.0));
    }

    if (cfg_.is_sft()) {
        if (!cfg_.pretrained_checkpoint.empty()) {
            {
                ModelConfig pt_cfg;
                TrainState pt_peek;
                GAI_CHECK(Checkpoint::peek(cfg_.pretrained_checkpoint, pt_cfg, pt_peek),
                          "cannot peek pretrained checkpoint for SFT: " + cfg_.pretrained_checkpoint);
                std::string why;
                if (!pt_cfg.same_architecture_as(model_.config(), &why) && !cfg_.allow_recipe_drift) {
                    GAI_FAIL("SFT architecture mismatch: " + why +
                             " (pretrain checkpoint vs SFT config define different model functions; " +
                             "SFT would silently train the wrong architecture. Align model.* blocks or pass " +
                             "--allow-recipe-drift for research). pretrain=" + pt_cfg.arch_identity() +
                             " sft=" + model_.config().arch_identity());
                }
                if (!pt_cfg.same_architecture_as(model_.config(), &why))
                    log_warn("[sft ] proceeding WITH architecture drift (--allow-recipe-drift): " + why);
                else
                    log_info("[sft ] architecture parity OK: " + model_.config().arch_identity());
            }
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
        GAI_CHECK(frozen_emb_ != nullptr, "freeze_embeddings: internal error");
        log_info("[sft ] embeddings FROZEN (no updates at all, no moments stored)");
    }

    init_distributed();

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
    train_loader_.set_vocab_size(model_.config().vocab_size);
    val_loader_.set_vocab_size(model_.config().vocab_size);
    bool opened = false;
    if (!cfg_.mix.empty()) {

        opened = train_loader_.open_mix(cfg_.data_dir, cfg_.mix, spec, train_seed);
        if (!opened)
            GAI_FAIL("data.mix configured but no train_<domain>_*.gbin found in " +
                     cfg_.data_dir + " (fix domains or clear data.mix; uniform fallback removed)");
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

    if (cfg_.fp16_weight_cache) {
        model_.enable_fp16_weight_cache(true);
        log_info(strfmt("[prec] persistent fp16 weight cache: %s (enabled before arenas)",
                        human_bytes(model_.fp16_weight_cache_bytes()).c_str()));
    }
#ifdef GAI_CUDA

    if (model_.device() == Device::CUDA) {
        const Model::WorkspacePlan wsp =
            Model::workspace_plan(model_.config(), cfg_.batch_size, cfg_.seq_len);
        cuda_ops::reserve_workspaces(wsp.gemm_bytes, wsp.moe_bytes);
        log_info(strfmt("[mem ] cuda workspaces pre-sized: gemm %s + moe %s",
                        human_bytes(wsp.gemm_bytes).c_str(),
                        human_bytes(wsp.moe_bytes).c_str()));
    }
#endif

    use_ckpt_ = cfg_.activation_checkpointing && cfg_.ckpt_segments > 1 && cfg_.batch_size > 1;
    if (use_ckpt_)
        act_ = Activations{};
    else
        act_ = model_.make_activations(cfg_.batch_size, cfg_.seq_len, true, cfg_.ce_chunks);

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

    if (have_val_)
        eval_act_ = model_.make_activations(cfg_.batch_size, cfg_.seq_len, false);
    else
        eval_act_ = Activations{};

    dev_ids_ = Tensor::empty({static_cast<i64>(cfg_.batch_size) * cfg_.seq_len}, DType::I32, model_.device());
    dev_targets_ = Tensor::empty({static_cast<i64>(cfg_.batch_size) * cfg_.seq_len}, DType::I32, model_.device());
    dev_segments_ = Tensor::empty({static_cast<i64>(cfg_.batch_size) * cfg_.seq_len}, DType::I32, model_.device());

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

    {
        u64 params = static_cast<u64>(model_.num_parameters());

        size_t live_train_act = act_.bytes + (use_ckpt_ ? ckpt_act_.bytes : 0);
        size_t peak_act = live_train_act;

        size_t fp16_cache = cfg_.fp16_weight_cache ? model_.fp16_weight_cache_bytes() : 0;
        size_t need_core = params * 8 + fp16_cache + opt_state_bytes() + peak_act + eval_act_.bytes;

        size_t slack = (1024ull << 20);
        if (dist_ && dist_->world_size() > 1) slack += (320ull << 20);

        size_t dist_extra = 0;
        if (dist_ && dist_->world_size() > 1) {

            dist_extra = (64ull << 20);
        }
        size_t need = need_core + slack + dist_extra;
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
#ifdef GAI_CUDA

            {
                size_t live_free = cuda::free_bytes_live();
                if (live_free > 0 && need_core > live_free) {
                    GAI_FAIL(strfmt("OOM guard (live): need core %s but only %s free live "
                                    "(pools/fragmentation). Lower batch_size/seq_len or disable "
                                    "fp16_weight_cache before training.",
                                    human_bytes(need_core).c_str(),
                                    human_bytes(live_free).c_str()));
                }
            }
#endif
        } else {
            log_info(strfmt("[mem ] est total %s (weights+grads+opt+acts+slack)%s",
                            human_bytes(need).c_str(), dist_ ? " [DDP: per-GPU]" : ""));
        }
    }

    {
        const u64 params = static_cast<u64>(model_.num_parameters());

        const size_t per_snapshot = static_cast<size_t>(params) * 4 + opt_state_bytes();

        const size_t ram = physical_ram_bytes();
        const size_t ram_budget = ckpt_ram_budget();
        if (ram > 0) {
            log_info(strfmt("[mem ] host RAM %s total, checkpoint queue budget %s (%.0f%%), one snapshot %s",
                            human_bytes(ram).c_str(), human_bytes(ram_budget).c_str(),
                            100.0 * static_cast<double>(ram_budget) / static_cast<double>(ram),
                            human_bytes(per_snapshot).c_str()));

            const size_t need = per_snapshot * 2 + (512ull << 20);
            if (ram < need) {
                GAI_FAIL(strfmt("host RAM guard: this run needs ~%s (two %s checkpoint "
                                "snapshots + 512 MB) but the machine has only %s. "
                                "Reduce the model, or free RAM; training here would be "
                                "OOM-killed mid-save.",
                                human_bytes(need).c_str(), human_bytes(per_snapshot).c_str(),
                                human_bytes(ram).c_str()));
            }
        }

        const size_t free_disk = free_disk_bytes(cfg_.checkpoint_dir);
        if (free_disk > 0) {

            size_t published = 0;
            {
                std::error_code ec1, ec2;
                const auto pb = fs::path(cfg_.checkpoint_dir) / "best.ckpt";
                const auto pl = fs::path(cfg_.checkpoint_dir) / "last.ckpt";
                const bool hb = fs::exists(pb, ec1), hl = fs::exists(pl, ec2);
                if (hb) ++published;
                if (hl) ++published;
                if (hb && hl && !ec1 && !ec2 && fs::equivalent(pb, pl, ec1) && !ec1)
                    --published;
            }
            const size_t need_disk = per_snapshot * (published + 1) + per_snapshot / 4;
            log_info(strfmt("[disk] %s free at %s (%zu checkpoint(s) published); "
                            "a save transiently needs ~%s",
                            human_bytes(free_disk).c_str(), cfg_.checkpoint_dir.c_str(),
                            published, human_bytes(need_disk).c_str()));
            if (free_disk < need_disk) {
                GAI_FAIL(strfmt("disk guard: %s free at %s but a checkpoint save needs "
                                "~%s (%zu published + one .tmp + 25%% margin). Free space "
                                "or point --checkpoint-dir at a bigger volume; ENOSPC at "
                                "hour 6 would lose the run.",
                                human_bytes(free_disk).c_str(), cfg_.checkpoint_dir.c_str(),
                                human_bytes(need_disk).c_str(), published));
            }
        } else {
            log_warn("[disk] could not stat free space for " + cfg_.checkpoint_dir +
                     " — cannot pre-flight the checkpoint budget");
        }

        if (cfg_.output_budget_mb > 0) {
            const size_t budget = static_cast<size_t>(cfg_.output_budget_mb) * (1024u * 1024u);

            fs::path root = fs::absolute(fs::path(cfg_.checkpoint_dir));
            for (int i = 0; i < 4 && root.has_parent_path() &&
                            root.filename().string() != "working"; ++i)
                root = root.parent_path();
            if (root.filename().string() != "working") root = fs::absolute(".");

            const std::vector<std::string> skip_dirs = {
                "build", "build_test", "build_cpu", "out", ".git", ".cache", "ccache"
            };
            size_t unique_files = 0;
            size_t used = tree_size_bytes_excluding(root.string(), skip_dirs, &unique_files);

            size_t freed_pretrained = 0;
            if (cfg_.is_sft() && !cfg_.pretrained_checkpoint.empty()) {
                std::error_code fec;
                const auto psize = fs::file_size(fs::path(cfg_.pretrained_checkpoint), fec);
                if (!fec && psize > 0 && psize <= used) {
                    freed_pretrained = static_cast<size_t>(psize);
                    used -= freed_pretrained;
                }
            }
            size_t ckpt_files = 0;
            const size_t ckpt_now = tree_size_bytes(cfg_.checkpoint_dir, &ckpt_files);

            const size_t transient = per_snapshot * 2;

            const size_t gguf = static_cast<size_t>(params) + per_snapshot / 8;
            const size_t projected = used + transient + gguf;
            log_info(strfmt("[quota] %s budget | %s already in %s (%zu files, incl. %s of checkpoints)",
                            human_bytes(budget).c_str(), human_bytes(used).c_str(),
                            root.string().c_str(), unique_files,
                            human_bytes(ckpt_now).c_str()));
            if (freed_pretrained > 0) {
                log_info(strfmt("[quota] excluding %s of pretrain checkpoint (loaded into "
                                "memory at construction, not kept by this run)",
                                human_bytes(freed_pretrained).c_str()));
            }
            log_info(strfmt("[quota] projected peak = %s (current %s + save transient %s + gguf %s)",
                            human_bytes(projected).c_str(), human_bytes(used).c_str(),
                            human_bytes(transient).c_str(), human_bytes(gguf).c_str()));
            if (projected > budget) {
                GAI_FAIL(strfmt("output quota guard: this run projects %s of saved output but "
                                "the budget is %s (over by %s, %s of it already-published "
                                "checkpoints/data). The quota bites at Save Version, long "
                                "after training. Delete stale checkpoints or shards under %s, "
                                "or lower --output-budget-mb knowingly, or shrink the model.",
                                human_bytes(projected).c_str(), human_bytes(budget).c_str(),
                                human_bytes(projected - budget).c_str(), human_bytes(used).c_str(),
                                root.string().c_str()));
            }
        }
    }

    std::string resume_path;
    if (cfg_.resume == "auto")      resume_path = Checkpoint::latest_in(cfg_.checkpoint_dir);
    else if (cfg_.resume != "none") resume_path = cfg_.resume;
    if (!resume_path.empty() && fs::exists(resume_path)) {
        const bool exact = cfg_.resume_exact();
        log_info(strfmt("[ckpt] resume_mode=%s from %s", cfg_.resume_mode.c_str(),
                        resume_path.c_str()));

        {
            ModelConfig ckpt_cfg;
            TrainState peek_st;
            if (Checkpoint::peek(resume_path, ckpt_cfg, peek_st)) {
                const ModelConfig& cur = model_.config();

                const bool math_same =
                    ckpt_cfg.rope_theta == cur.rope_theta &&
                    ckpt_cfg.rope_scale == cur.rope_scale &&
                    ckpt_cfg.rope_yarn_mscale == cur.rope_yarn_mscale &&
                    ckpt_cfg.rope_yarn_low == cur.rope_yarn_low &&
                    ckpt_cfg.rope_yarn_high == cur.rope_yarn_high &&
                    ckpt_cfg.rope_type == cur.rope_type &&
                    ckpt_cfg.rms_eps == cur.rms_eps &&
                    ckpt_cfg.max_seq_len == cur.max_seq_len &&
                    ckpt_cfg.sliding_window == cur.sliding_window &&
                    ckpt_cfg.moe_aux_free == cur.moe_aux_free;
                if (!math_same && !cfg_.allow_recipe_drift) {
                    GAI_FAIL("resume recipe drift: checkpoint was saved with different math "
                             "(rope_theta/scale/yarn/type, sliding_window, aux_free, rms_eps, max_seq_len). Refusing resume: same "
                             "weights would compute a different model. Restore the original recipe "
                             "or pass --allow-recipe-drift (research only)");
                }
                if (!math_same)
                    log_warn("[ckpt] resuming WITH recipe drift (--allow-recipe-drift): numerics differ from save time");

                {
                    std::string why;
                    const bool arch_same = ckpt_cfg.same_architecture_as(cur, &why);
                    if (!arch_same && exact) {
                        GAI_FAIL("exact resume refused: checkpoint architecture differs: " + why +
                                 ". Restore the original recipe, or use resume_mode: migrate "
                                 "to accept a controlled reset");
                    }
                    if (!arch_same)
                        log_warn("[ckpt] resuming WITH architecture drift (migrate): " + why);
                }

                if (peek_st.tok_fingerprint != 0 && cfg_.tok_fingerprint != 0 &&
                    peek_st.tok_fingerprint != cfg_.tok_fingerprint) {
                    GAI_FAIL(strfmt(
                        "tokenizer mismatch: checkpoint was trained with .gtok fingerprint %s "
                        "but the current tokenizer is %s. Same vocab_size does NOT mean same "
                        "merges -> different token ids -> corrupted embeddings. Ship the ORIGINAL "
                        "tokenizer file (artifacts/tokenizer/english32k.gtok) with the checkpoint, "
                        "or start a fresh run (--resume none).",
                        fingerprint_hex(peek_st.tok_fingerprint).c_str(),
                        fingerprint_hex(cfg_.tok_fingerprint).c_str()));
                }

                if (peek_st.tok_fingerprint == 0 && cfg_.tok_fingerprint != 0) {
                    if (exact) {
                        GAI_FAIL("exact resume refused: checkpoint predates tokenizer "
                                 "fingerprints (v10 or older), so the tokenizer identity "
                                 "cannot be verified. Use resume_mode: migrate to accept it, "
                                 "or start a fresh run (--resume none).");
                    }
                    log_warn("[ckpt] checkpoint predates tokenizer fingerprints (v10 or older): "
                             "cannot verify the tokenizer identity on this resume");
                }

                if (peek_st.sched_total > 0) {
                    const float want_peak = effective_peak_lr(cfg_);
                    const bool sched_same =
                        peek_st.sched_kind == ((cfg_.scheduler == "wsd") ? 1 : 0) &&
                        std::fabs(peek_st.sched_peak - want_peak) <=
                            1e-6f * std::fabs(peek_st.sched_peak) + 1e-30f;
                    if (!sched_same) {
                        const std::string why =
                            "scheduler recipe differs (kind or peak lr) from the checkpoint";
                        if (exact)
                            GAI_FAIL("exact resume: " + why +
                                     ". Restore the original recipe, use resume_mode: migrate, "
                                     "or pass --allow-recipe-drift");
                        log_warn("[ckpt] resuming WITH scheduler drift: " + why +
                                 " (past lr_at(N) values are unchanged; the rest is not)");
                    }
                }
            }
        }
        bool moments_restored = false;
        const bool strict_load = exact;

        const TrainState pre_resume_state = state_;
        bool ok = use_muon_ ? Checkpoint::load(resume_path, model_, opt_muon_.get(), state_, &moments_restored, strict_load)
                    : use_lion_ ? Checkpoint::load(resume_path, model_, opt_lion_.get(),  state_, &moments_restored, strict_load)
                                : Checkpoint::load(resume_path, model_, opt_adam_.get(),  state_, &moments_restored, strict_load);
        if (!ok && exact) {
            GAI_FAIL("exact resume rejected: " + resume_path +
                     " is not an exact continuation of this recipe "
                     "(parameter set, tokenizer vocab, optimizer state or architecture differ). "
                     "Fix the recipe, or use resume_mode: migrate to accept a controlled reset");
        }
        if (ok) {

            if (moments_restored) {
                opt_set_step(state_.step);
            } else {
                opt_set_step(0);
                log_warn(strfmt("[ckpt] moments NOT restored (kind switch/corrupt/legacy); "
                                "optimizer t_=0 with weights at step %lld (loud, not silent)",
                                (long long)state_.step));
            }

            train_loader_.set_state(state_.loader);

            last_consumed_loader_state_ = state_.loader;

            if (have_val_ && state_.val_loader.batches >= 0) {
                val_loader_.set_state(state_.val_loader);
                log_info(strfmt("[ckpt] val stream restored at %lld batches",
                                static_cast<long long>(state_.val_loader.batches)));
            }

            if (cfg_.loss_scale_init > 0.0 && state_.loss_scale >= 1.0) {
                loss_scale_ = state_.loss_scale;
                clean_steps_ = state_.clean_steps;
            }

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

                    last_consumed_loader_state_ = train_loader_.get_state();
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

            state_ = pre_resume_state;
            state_.step = 0;
            state_.tokens_seen = 0;
            state_.sched_total = 0;
            state_.sched_warmup = 0;
            state_.sched_peak = 0.0f;
            opt_set_step(0);
            model_.init_weights(cfg_.seed);
            log_warn("[ckpt] failed to load " + resume_path + "; model reset before scratch run");
        }
    }

    if (cfg_.fp16_weight_cache) model_.mark_weights_dirty();
}

Trainer::~Trainer() {

    try {
        stop_prefetch();
    } catch (const std::exception& e) {
        log_error(std::string("[cleanup] prefetch stop failed: ") + e.what());
    } catch (...) {
        log_error("[cleanup] prefetch stop failed with unknown exception");
    }
    try {
        stop_ckpt_writer();
    } catch (const std::exception& e) {
        log_error(std::string("[cleanup] ckpt-writer stop failed: ") + e.what());
    } catch (...) {
        log_error("[cleanup] ckpt-writer stop failed with unknown exception");
    }
}

double Trainer::loss_scale_cap() const {
    if (!(cfg_.loss_scale_max > 0.0)) return kLossScaleMax;
    return std::min(cfg_.loss_scale_max, kLossScaleMax);
}

float Trainer::scaler_for_step() {

    if (cfg_.loss_scale_init <= 0.0) return 1.0f;
    if (!ops::gemm_fp16_enabled()) return 1.0f;

    double scale = std::min(loss_scale_, loss_scale_cap());
    return static_cast<float>(scale);
}

void Trainer::scaler_update(double gnorm) {
    if (cfg_.loss_scale_init <= 0.0) return;
    if (!ops::gemm_fp16_enabled()) return;
    if (!std::isfinite(gnorm)) {

        loss_scale_ = std::max(1.0, loss_scale_ * 0.5);
        clean_steps_ = 0;
        ++scaler_overflows_;
        log_warn(strfmt("[scaler] overflow at step %lld, scale -> %.0f",
                        static_cast<long long>(state_.step), loss_scale_));
    } else if (++clean_steps_ >= cfg_.loss_scale_window) {
        clean_steps_ = 0;
        double grown = std::min(loss_scale_cap(), loss_scale_ * 2.0);
        if (grown != loss_scale_) {
            loss_scale_ = grown;
            log_info(strfmt("[scaler] %d clean steps, scale -> %.0f",
                            cfg_.loss_scale_window, loss_scale_));
        }
    }
}

void Trainer::log_scaler_summary() const {
    if (cfg_.loss_scale_init <= 0.0 || !ops::gemm_fp16_enabled()) return;
    if (state_.step <= 0) return;
    const double ratio = static_cast<double>(skipped_steps_) / static_cast<double>(state_.step);
    log_info(strfmt("[scaler] %lld/%lld optimizer steps skipped (%.2f%%), %lld overflow(s), final scale %.0f",
                    static_cast<long long>(skipped_steps_),
                    static_cast<long long>(state_.step), ratio * 100.0,
                    static_cast<long long>(scaler_overflows_), loss_scale_));
    if (ratio > 0.02)
        log_warn(strfmt("[scaler] %.2f%% of steps were skipped for non-finite grads: "
                        "fp16 gradients are unstable at loss_scale_init=%.0f. Lower "
                        "training.loss_scale_init (or disable gemm_fp16) — the run is "
                        "burning GPU time on rejected steps.",
                        ratio * 100.0, cfg_.loss_scale_init));
}

double Trainer::forward_backward_micro(const Batch& batch, float dscale, i64* out_ntok) {

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
                                           want_aux_stats, dev_segments_.i32p(),
                                           batch.targets.data());
        if (frozen_emb_ && frozen_emb_->g.defined()) frozen_emb_->g.zero_();
        if (out_ntok) *out_ntok = ntok;
        return l;
    }

    const int B = batch.B, T = batch.T;
    const int segB = ckpt_act_.B;
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
                                           want_aux_stats, ckpt_segments_.i32p(),
                                           batch.targets.data() + src);
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

    const float jit_saved = ops::moe_jitter();
    if (jit_saved != 0.0f) ops::set_moe_jitter(0.0f);
    for (i64 i = 0; i < max_batches; ++i) {
        if (!val_loader_.next(batch)) break;

        i64 N = static_cast<i64>(batch.B) * batch.T;
        device_copy(dev_ids_.data_ptr(), model_.device(), batch.ids.data(), Device::CPU, N * sizeof(i32));
        device_copy(dev_targets_.data_ptr(), model_.device(), batch.targets.data(), Device::CPU, N * sizeof(i32));
        device_copy(dev_segments_.data_ptr(), model_.device(), batch.segment_ids.data(), Device::CPU, N * sizeof(i32));

        Tensor& logits = model_.forward(dev_ids_.i32p(), batch.B, batch.T, eval_act_, dev_segments_.i32p());
        double sum = 0.0;
        i64 n = 0;

        ops::softmax_cross_entropy(model_.device(), logits.f32(), dev_targets_.i32p(),
                                   nullptr, N, model_.config().vocab_size, &sum, &n,
                                   model_.config().z_loss_scale);
        total += sum;
        ntok_total += n;
    }
    if (jit_saved != 0.0f) ops::set_moe_jitter(jit_saved);
    return ntok_total > 0 ? total / static_cast<double>(ntok_total) : 0.0;
}

void Trainer::log_step(double loss, float lr, double gnorm, double dt, i64 ntok) {
    double tps = dt > 0 ? static_cast<double>(ntok) / dt : 0.0;
    i64 remaining = total_steps_ - state_.step;
    double eta = remaining > 0 ? remaining * dt : 0.0;

    double sup_frac = 0.0;
    {
        const i64 dense = cfg_.tokens_per_step_global(dist_ ? dist_->world_size() : 1);
        if (dense > 0) sup_frac = static_cast<double>(ntok) / static_cast<double>(dense);
    }

    size_t ckpt_depth = 0, ckpt_bytes = 0;
    {
        std::lock_guard<std::mutex> lk(ckpt_mutex_);
        ckpt_depth = pending_saves_.size();
        for (const auto& qs : pending_saves_)
            if (qs.snapshot) ckpt_bytes += qs.snapshot->bytes();
        if (ckpt_active_) ckpt_bytes += ckpt_active_->bytes();
    }
    log_info(strfmt("step %6lld | loss %7.4f | ema %7.4f | ppl %8.2f | lr %.3e | gnorm %6.3f "
                    "| %7.0f tok/s | %s | eta %s | sup %4.1f%% | ckptq %zu (%s) | perf [%s]",
                    static_cast<long long>(state_.step), loss, ema_loss_,
                    std::exp(std::min(20.0, ema_loss_)), lr, gnorm, tps,
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(eta).c_str(), sup_frac * 100.0, ckpt_depth,
                    human_bytes(ckpt_bytes).c_str(),
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
    quiesce_prefetch();
    u8 capture_ok = 1;
    if (is_main_rank()) {
        try {

            state_.loader = last_consumed_loader_state_;

            if (have_val_) {
                state_.val_loader = val_loader_.get_state();
            } else {
                state_.val_loader = DataLoader::State{};
                state_.val_loader.batches = -1;
            }
            state_.loss_scale = loss_scale_;
            state_.clean_steps = clean_steps_;
            state_.tok_vocab = model_.config().vocab_size;
            state_.tok_fingerprint = cfg_.tok_fingerprint;
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
        } catch (const std::exception& e) {
            log_error(std::string("Checkpoint capture failed: ") + e.what());
            capture_ok = 0;
        } catch (...) {
            log_error("Checkpoint capture failed with unknown exception");
            capture_ok = 0;
        }
    }

#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) {
        dist_->broadcast(&capture_ok, 1, sizeof(u8), 0);
    }
#endif

    if (!capture_ok) {
        resume_prefetch();
        GAI_FAIL("Checkpoint capture failed on rank 0");
    }

    resume_prefetch();

    if (!is_main_rank()) {
#ifdef GAI_CUDA
        if (dist_ && dist_->world_size() > 1) dist_->barrier();
#endif
        return;
    }

    const std::string path = (fs::path(cfg_.checkpoint_dir) / name).string();
    std::shared_ptr<CheckpointSnapshot> snapshot = snapshot_cache_;
    try {
        save_async(path, snapshot);
    } catch (...) {
        resume_prefetch();
#ifdef GAI_CUDA
        if (dist_ && dist_->world_size() > 1) dist_->barrier();
#endif
        throw;
    }
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) dist_->barrier();
#endif
}

i64 Trainer::planned_total() const { return total_steps_; }

void Trainer::run() {

    signals::install_stop_handlers();

    int world_sz = 1;
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) world_sz = dist_->world_size();
#endif

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

        log_info(strfmt("[sched] epochs mode: %d epochs x %s tokens = %lld steps (per-rank %s, global %s x%d; stochastic windows, not classic passes)",
                        cfg_.epochs,
                        human_count(train_loader_.total_tokens()).c_str(),
                        static_cast<long long>(total_steps_),
                        human_count(static_cast<u64>(cfg_.tokens_per_step())).c_str(),
                        human_count(static_cast<u64>(cfg_.tokens_per_step_global(world_sz))).c_str(),
                        world_sz));
    }

    if (state_.sched_total > 0 && state_.sched_total > total_steps_) {
        log_warn(strfmt("[sched] continuation: checkpoint planned %lld steps, this session "
                        "budgeted %lld — KEEPING the original %lld-step plan so the LR curve "
                        "of the steps already run stays valid. This session simply stops "
                        "early; resume again to continue.",
                        (long long)state_.sched_total, (long long)total_steps_,
                        (long long)state_.sched_total));
        total_steps_ = state_.sched_total;
        if (!(cfg_.warmup_steps < total_steps_)) {
            log_warn(strfmt("[sched] warmup %lld >= kept total %lld: clamping to total/10",
                            (long long)cfg_.warmup_steps, (long long)total_steps_));
            cfg_.warmup_steps = std::max<i64>(1, total_steps_ / 10);
        }
    }

    GAI_CHECK(cfg_.warmup_steps >= 0 && cfg_.warmup_steps < total_steps_,
              strfmt("training.warmup_steps=%lld must be in [0,total=%lld)",
                     (long long)cfg_.warmup_steps, (long long)total_steps_));
    GAI_CHECK(cfg_.sched_decay_frac > 0.0f && cfg_.sched_decay_frac <= 1.0f,
              "training.sched_decay_frac must be in (0,1]");

    float sched_peak = effective_peak_lr(cfg_);
    sched_ = LrScheduler(sched_peak, cfg_.warmup_steps,
                         total_steps_, cfg_.min_lr_ratio,
                         cfg_.scheduler, cfg_.sched_decay_frac);
    log_info(strfmt("[sched] %s warmup %lld total %lld min %.0f%% decay_frac %.2f",
                    cfg_.scheduler.c_str(), static_cast<long long>(cfg_.warmup_steps),
                    static_cast<long long>(total_steps_),
                    cfg_.min_lr_ratio * 100.0, cfg_.sched_decay_frac));

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

    log_info(strfmt("  precision : %s", cfg_.precision_name().c_str()));

    Batch batch;
    Timer step_timer;

    start_prefetch();
    start_ckpt_writer();

    while (state_.step < total_steps_) {
        step_timer.reset();
        model_.zero_grad();
        const float dscale = scaler_for_step();

        {
            int rank = 0;
#ifdef GAI_CUDA
            if (dist_ && dist_->world_size() > 1) rank = dist_->global_rank();
#endif
            u64 ctx = cfg_.seed ^ (static_cast<u64>(state_.step) * 0x9E3779B97F4A7C15ULL)
                                ^ (static_cast<u64>(rank) * 0xBF58476D1CE4E5B9ULL);
            ops::set_moe_jitter_seed(ctx);
        }

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

        sync_gradients();
        i64 ntok_global = sync_ntok_sum(ntok_step);

        double gnorm = 0.0;
        bool opt_applied = false;
        if (ntok_global > 0) {
            float grad_scale = 1.0f / (static_cast<float>(ntok_global) * dscale);
            gnorm = opt_step(lr, grad_scale);

            opt_applied = std::isfinite(gnorm);
            if (opt_applied) model_.mark_weights_dirty();
            else ++skipped_steps_;
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

        sync_moe_bias(opt_applied);

        ++state_.step;

        state_.tokens_seen += ntok_global;
        state_.last_loss = loss;

        double dt = step_timer.seconds();
        bool main = is_main_rank();
        if (main && cfg_.log_every > 0 && state_.step % cfg_.log_every == 0) {
            log_step(loss, lr, gnorm, dt, ntok_global);

            check_ckpt_health();
        }

        if (!main) log_debug(strfmt("[step] rank %d/%d step %lld loss %.4f ntok %lld",
                                    dist_ ? dist_->global_rank() : 0,
                                    dist_ ? dist_->world_size() : 1,
                                    static_cast<long long>(state_.step), loss,
                                    static_cast<long long>(ntok_global)));

        if (have_val_ && cfg_.eval_every > 0 && state_.step % cfg_.eval_every == 0) {
            bool is_best = false;
            if (main) {
                double vl = evaluate(cfg_.eval_batches);
                std::string bal = model_.moe_balance_report();
                if (!bal.empty()) log_info("  >> " + bal);
                model_.moe_balance_reset();

                log_info(strfmt("  >> val loss %.4f  (ppl %.2f)  [best %s]",
                                vl, std::exp(std::min(20.0, vl)),
                                state_.best_val < 1e29 ? strfmt("%.4f", state_.best_val).c_str()
                                                       : "-"));
                is_best = (vl < state_.best_val);
                if (is_best) state_.best_val = vl;
            } else {
                model_.moe_balance_reset();
            }
            is_best = sync_eval_best(main, is_best);
            if (is_best) save("best.ckpt");
        }

        if (cfg_.save_every > 0 && state_.step % cfg_.save_every == 0) {
            save("last.ckpt");
        }

        check_ckpt_health();
        check_stop_requested();

        if (main) snapshot_cache_.reset();
        if (stopped_by_signal_) break;
    }

    stop_prefetch();
    if (stopped_by_signal_) {

        log_warn(strfmt("[stop] %s received at step %lld/%lld: saving last.ckpt and exiting "
                        "cleanly (--resume auto continues from here)",
                        signals::stop_reason(), static_cast<long long>(state_.step),
                        static_cast<long long>(total_steps_)));
        save("last.ckpt");
    } else if (!(cfg_.save_every > 0 && state_.step % cfg_.save_every == 0)) {

        save("last.ckpt");
    }

    wait_for_save();
    stop_ckpt_writer();
    log_info(strfmt("---------------- pretrain done: %lld steps, %s tokens, %s --------------------",
                    static_cast<long long>(state_.step),
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(wall_.seconds()).c_str()));
    log_scaler_summary();
}

void Trainer::run_sft() {

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

    start_prefetch();
    start_ckpt_writer();

    while (state_.step < total_steps_) {
        step_timer.reset();
        model_.zero_grad();
        const float dscale = scaler_for_step();
        {
            int rank = 0;
#ifdef GAI_CUDA
            if (dist_ && dist_->world_size() > 1) rank = dist_->global_rank();
#endif
            u64 ctx = cfg_.seed ^ (static_cast<u64>(state_.step) * 0x9E3779B97F4A7C15ULL)
                                ^ (static_cast<u64>(rank) * 0xBF58476D1CE4E5B9ULL);
            ops::set_moe_jitter_seed(ctx);
        }

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

        sync_gradients();
        i64 ntok_global = sync_ntok_sum(ntok_step);

        double gnorm = 0.0;
        bool opt_applied = false;
        if (ntok_global > 0) {
            float grad_scale = 1.0f / (static_cast<float>(ntok_global) * dscale);
            gnorm = opt_step(lr, grad_scale);
            opt_applied = std::isfinite(gnorm);
            if (opt_applied) model_.mark_weights_dirty();
            else ++skipped_steps_;
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

        sync_moe_bias(opt_applied);

        ++state_.step;
        state_.tokens_seen += ntok_global;
        state_.last_loss = loss;

        double dt = step_timer.seconds();
        bool main = is_main_rank();
        if (main && cfg_.log_every > 0 && state_.step % cfg_.log_every == 0) {
            log_step(loss, lr, gnorm, dt, ntok_global);

            check_ckpt_health();
        }

        if (have_val_ && cfg_.eval_every > 0 && state_.step % cfg_.eval_every == 0) {
            bool is_best = false;
            if (main) {
                double vl = evaluate(cfg_.eval_batches);
                std::string bal = model_.moe_balance_report();
                if (!bal.empty()) log_info("  >> " + bal);
                model_.moe_balance_reset();

                log_info(strfmt("  >> val loss %.4f  (ppl %.2f)  [best %s]",
                                vl, std::exp(std::min(20.0, vl)),
                                state_.best_val < 1e29 ? strfmt("%.4f", state_.best_val).c_str()
                                                       : "-"));
                is_best = (vl < state_.best_val);
                if (is_best) state_.best_val = vl;
            } else {
                model_.moe_balance_reset();
            }
            is_best = sync_eval_best(main, is_best);
            if (is_best) save("best.ckpt");
        }

        if (cfg_.save_every > 0 && state_.step % cfg_.save_every == 0) {
            save("last.ckpt");
        }
        check_ckpt_health();
        check_stop_requested();

        if (main) snapshot_cache_.reset();
        if (stopped_by_signal_) break;
    }

    stop_prefetch();
    if (stopped_by_signal_) {
        log_warn(strfmt("[stop] %s received at step %lld/%lld: saving last.ckpt and exiting "
                        "cleanly (--resume auto continues from here)",
                        signals::stop_reason(), static_cast<long long>(state_.step),
                        static_cast<long long>(total_steps_)));
        save("last.ckpt");
    } else if (!(cfg_.save_every > 0 && state_.step % cfg_.save_every == 0)) {

        save("last.ckpt");
    }

    wait_for_save();
    stop_ckpt_writer();
    log_info(strfmt("---------------- SFT done: %lld steps, %s tokens, %s --------------------",
                    static_cast<long long>(state_.step),
                    human_count(static_cast<u64>(state_.tokens_seen)).c_str(),
                    human_duration(wall_.seconds()).c_str()));
    log_scaler_summary();
}

void Trainer::init_distributed() {
#ifdef GAI_CUDA
    if (model_.device() != Device::CUDA) return;

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
    dcfg.grad_compression = cfg_.ddp_grad_compression;
    if (dcfg.world_size <= 1) {
        if (cfg_.ddp) GAI_FAIL("training.ddp=true but world_size==1 (launch 2+ ranks or set ddp=false)");
        return;
    }

    dist_ = std::make_unique<DistributedContext>();
    bool ok = dist_->init(dcfg);
    if (!ok) {
        GAI_FAIL("[dist] Failed to initialize distributed context; WORLD_SIZE>1 requires successful NCCL init.");
    }

    cudaSetDevice(dcfg.local_rank);

    sync_model();
    log_info(strfmt("[dist] Training on %d GPUs (rank %d/%d)",
                    dist_->world_size(), dist_->global_rank(), dist_->local_rank()));
#else

    if (cfg_.ddp) {
        GAI_FAIL("training.ddp=true but this binary was built without CUDA+NCCL "
                 "(GAI_CUDA off). Rebuild with CUDA+NCCL (kaggle/setup.sh on a GPU "
                 "session) or set training.ddp=false for single-GPU/CPU.");
    }
    {
        const char* ws = std::getenv("WORLD_SIZE");
        int env_ws = ws ? std::atoi(ws) : 1;
        if (env_ws > 1) {
            GAI_FAIL("WORLD_SIZE>1 but this binary was built without CUDA+NCCL. "
                     "DDP needs NCCL; rebuild with CUDA+NCCL or run a single rank.");
        }
    }
    (void)model_;
#endif
}

void Trainer::sync_gradients() {
#ifdef GAI_CUDA
    if (!dist_ || dist_->world_size() <= 1) return;

    constexpr size_t kLarge = 1 << 20;
    struct Item { float* ptr; size_t n; };
    std::vector<Item> large;
    std::vector<Item> small;
    large.reserve(16);
    small.reserve(64);
    size_t small_total = 0;
    for (Parameter* p : model_.parameters()) {
        if (!p->g.defined() || p->frozen) continue;

        if (p->g.device() != Device::CUDA)
            GAI_FAIL("DDP requires all trainable grads on CUDA (param '" + p->name +
                     "' is CPU); move the model to CUDA or disable ddp");
        size_t numel = static_cast<size_t>(p->g.numel());
        if (numel == 0) continue;
        if (numel >= kLarge) {
            large.push_back({static_cast<float*>(p->g.data_ptr()), numel});
        } else {
            small.push_back({static_cast<float*>(p->g.data_ptr()), numel});
            small_total += numel;
        }
    }

    float* stage = nullptr;
    if (!small.empty() && small_total > 0) {

        if (!dist_fused_.defined() ||
            static_cast<size_t>(dist_fused_.numel()) < small_total) {
            size_t want = small_total + small_total / 8 + 1024;
            dist_fused_ = Tensor::empty({(i64)want}, DType::F32, Device::CUDA);
        }
        stage = dist_fused_.f32();
        size_t off = 0;
        for (auto& it : small) {
            device_copy(stage + off, Device::CUDA, it.ptr, Device::CUDA,
                        it.n * sizeof(float));
            off += it.n;
        }
    }

    dist_->begin_group();
    for (auto& it : large)
        dist_->all_reduce_sum_nosync(it.ptr, it.n, sizeof(float));
    if (stage)
        dist_->all_reduce_sum_nosync(stage, small_total, sizeof(float));
    dist_->end_group();
    dist_->sync_stream();
    if (stage) {
        size_t off = 0;
        for (auto& it : small) {
            device_copy(it.ptr, Device::CUDA, stage + off, Device::CUDA,
                        it.n * sizeof(float));
            off += it.n;
        }
    }

#else
    (void)dist_; (void)model_;
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
    (void)dist_; (void)model_;
#endif
}

bool Trainer::is_main_rank() const {
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1) return dist_->global_rank() == 0;
#endif
    return true;
}

i64 Trainer::sync_ntok_sum(i64 local) {

#ifdef GAI_CUDA
    if (!dist_ || dist_->world_size() <= 1) return local;
    if (model_.device() != Device::CUDA) {

        static bool warned_ntok_approx = false;
        if (!warned_ntok_approx) {
            warned_ntok_approx = true;
            log_warn("[ddp ] CPU ntok sync is approximate (local*world); "
                     "exact token-weighted grads need CUDA+NCCL");
        }
        return local * (i64)dist_->world_size();
    }
    if (!dist_ntok_.defined() || dist_ntok_.dtype() != DType::I64)
        dist_ntok_ = Tensor::empty({1}, DType::I64, Device::CUDA);
    i64 h = local;
    device_copy(dist_ntok_.data_ptr(), Device::CUDA, &h, Device::CPU, sizeof(i64));
    dist_->all_reduce_sum_i64(static_cast<int64_t*>(dist_ntok_.data_ptr()), 1);
    i64 out = 0;
    device_copy(&out, Device::CPU, dist_ntok_.data_ptr(), Device::CUDA, sizeof(i64));
    return out > 0 ? out : 0;
#else
    (void)dist_;
    return local > 0 ? local : 0;
#endif
}

void Trainer::sync_moe_bias(bool opt_step_applied) {

    Tensor& acc_dev = model_.moe_bias_acc_dev();
    auto& acc = model_.moe_bias_acc_host();
    if (acc.empty() && !acc_dev.defined()) {

        return;
    }
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1 && model_.device() == Device::CUDA && acc_dev.defined()) {

        dist_->all_reduce_sum(acc_dev.f32(), static_cast<size_t>(acc_dev.numel()));
        const size_t n = static_cast<size_t>(acc_dev.numel());
        acc.assign(n, 0.0f);
        device_copy(acc.data(), Device::CPU, acc_dev.data_ptr(), Device::CUDA,
                    n * sizeof(float));
    } else if (dist_ && dist_->world_size() > 1 && !acc.empty()) {

    }
    if (!dist_ || dist_->world_size() <= 1) {
        if (acc_dev.defined() && acc.empty()) {

            const size_t n = static_cast<size_t>(acc_dev.numel());
            acc.assign(n, 0.0f);
            device_copy(acc.data(), Device::CPU, acc_dev.data_ptr(), Device::CUDA,
                        n * sizeof(float));
        }
    }
#else
    (void)acc_dev;
#endif
    model_.apply_moe_bias_step(acc.empty() ? nullptr : acc.data(), opt_step_applied);
}

void Trainer::broadcast_from_main(void* buf, size_t numel, int dtype_size) {
#ifdef GAI_CUDA
    if (dist_ && dist_->world_size() > 1)
        dist_->broadcast(buf, numel, dtype_size, 0);
#else
    (void)buf; (void)numel; (void)dtype_size;
#endif
}

bool Trainer::sync_eval_best(bool is_main, bool is_best) {

    i64 payload[2] = {is_best ? 1 : 0, 0};
    if (is_main) {
        double bv = state_.best_val;
        std::memcpy(&payload[1], &bv, sizeof(double));
    }
    broadcast_from_main(payload, 2, static_cast<int>(sizeof(i64)));
    const bool agreed_is_best = payload[0] != 0;
    if (is_main) {
        double bv = 0.0;
        std::memcpy(&bv, &payload[1], sizeof(double));
        state_.best_val = bv;
    }

    return agreed_is_best;
}

void Trainer::check_stop_requested() {
    if (stopped_by_signal_ || !signals::stop_requested) return;
    stopped_by_signal_ = true;
}

void Trainer::check_ckpt_health() {

    bool failed = false;
    std::string err;
    {
        std::lock_guard<std::mutex> lk(ckpt_mutex_);
        failed = ckpt_failed_;
        err = ckpt_error_;
    }
    if (failed)
        GAI_FAIL("background checkpoint write failed: " + err +
                 " (stopping now instead of training into a dead disk)");
}

static bool link_or_copy_committed(const std::string& src, const std::string& dst) {
    std::error_code ec;
    if (!fs::exists(src, ec) || ec) return false;
    const std::string tmp = dst + ".lnk";
    fs::remove(tmp, ec);
    ec.clear();
    fs::create_hard_link(src, tmp, ec);
    if (ec) {
        ec.clear();
        fs::copy_file(src, tmp, fs::copy_options::overwrite_existing, ec);
    }
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    ec.clear();
    fs::rename(tmp, dst, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

void Trainer::start_ckpt_writer() {
    if (ckpt_writer_thread_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(ckpt_mutex_);
        ckpt_stop_ = false;
        ckpt_failed_ = false;
        ckpt_error_.clear();
        ckpt_committed_.clear();
        ckpt_active_.reset();
    }
    ckpt_writer_thread_ = std::thread([this]() {
        while (true) {
            QueuedSave qs;
            {
                std::unique_lock<std::mutex> lk(ckpt_mutex_);
                ckpt_cv_.wait(lk, [this] { return !pending_saves_.empty() || ckpt_stop_; });
                if (pending_saves_.empty() && ckpt_stop_) break;
                qs = std::move(pending_saves_.front());
                pending_saves_.erase(pending_saves_.begin());
                ckpt_busy_ = true;
                ckpt_active_ = qs.snapshot;
            }
            if (qs.snapshot) {
                try {
                    Timer t;

                    std::string reuse;
                    {
                        std::lock_guard<std::mutex> lk(ckpt_mutex_);
                        auto it = ckpt_committed_.find(qs.step);
                        if (it != ckpt_committed_.end() && it->second != qs.path)
                            reuse = it->second;
                    }
                    if (!reuse.empty() && link_or_copy_committed(reuse, qs.path)) {
                        log_info(strfmt("[ckpt] published %s from %s (same step %lld, no re-serialize)",
                                        qs.path.c_str(), reuse.c_str(), static_cast<long long>(qs.step)));
                        {
                            std::lock_guard<std::mutex> lk(ckpt_mutex_);
                            ckpt_committed_[qs.step] = qs.path;
                        }
                    } else {
                        Checkpoint::save(*qs.snapshot, qs.path);
                        {
                            std::lock_guard<std::mutex> lk(ckpt_mutex_);
                            ckpt_committed_[qs.step] = qs.path;

                            while (ckpt_committed_.size() > 8) ckpt_committed_.erase(ckpt_committed_.begin());
                        }
                        log_info(strfmt("[ckpt] saved %s (%s)", qs.path.c_str(),
                                        human_duration(t.seconds()).c_str()));
                    }
                    qs.snapshot.reset();
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
                ckpt_active_.reset();
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

void Trainer::save_async(const std::string& path, std::shared_ptr<CheckpointSnapshot> snapshot) {
    GAI_CHECK(ckpt_writer_thread_.joinable(), "checkpoint writer is not running");
    GAI_CHECK(snapshot != nullptr, "save_async: null checkpoint snapshot");
    const i64 step = snapshot_step_;
    const size_t single = snapshot->bytes();

    std::unique_lock<std::mutex> lk(ckpt_mutex_);

    {
        const size_t budget = ckpt_ram_budget();
        if (single > budget) {
            const size_t phys = physical_ram_bytes();
            GAI_FAIL(strfmt("host RAM cannot checkpoint this model: one snapshot is %s, "
                            "but the queue budget is %s "
                            "(machine has %s). Even a single copy does not fit. Use a "
                            "smaller model/optimizer, or free host RAM.",
                            human_bytes(single).c_str(),
                            human_bytes(budget).c_str(),
                            phys ? human_bytes(phys).c_str() : "unknown"));
        }
    }

    for (int spins = 0; ; ++spins) {

        auto it = std::remove_if(pending_saves_.begin(), pending_saves_.end(),
                                 [&](const QueuedSave& p) { return p.path == path; });
        pending_saves_.erase(it, pending_saves_.end());

        std::vector<std::shared_ptr<CheckpointSnapshot>> refs;
        refs.reserve(pending_saves_.size() + 3);
        refs.push_back(snapshot);
        if (ckpt_active_) refs.push_back(ckpt_active_);
        if (snapshot_cache_) refs.push_back(snapshot_cache_);
        for (const auto& qs : pending_saves_) refs.push_back(qs.snapshot);
        const size_t live = unique_snapshot_bytes(refs);
        const size_t budget = ckpt_ram_budget();

        if (live <= budget || ckpt_failed_ || ckpt_stop_) {
            if (live > budget) {
                GAI_FAIL("checkpoint backlog exceeds the host-RAM budget and the writer "
                         "cannot drain (see ckpt_error_)");
            }
            break;
        }
        if (spins == 0)
            log_warn(strfmt("[ckpt] host-RAM budget reached (%.0f MB live of %.0f MB): applying backpressure",
                            static_cast<double>(live) / (1024.0 * 1024.0),
                            static_cast<double>(budget) / (1024.0 * 1024.0)));
        ckpt_cv_.wait_for(lk, std::chrono::milliseconds(50));
    }

    pending_saves_.push_back({path, step, std::move(snapshot)});
    log_info(strfmt("[ckpt] queued %s (snapshot %.0f MB, step %lld)", path.c_str(),
                    static_cast<double>(single) / (1024.0 * 1024.0),
                    static_cast<long long>(step)));
    lk.unlock();
    ckpt_cv_.notify_one();
}

void Trainer::wait_for_save() {
    std::unique_lock<std::mutex> lk(ckpt_mutex_);
    ckpt_cv_.wait(lk, [this]{ return pending_saves_.empty() && !ckpt_busy_; });
    if (ckpt_failed_) GAI_FAIL("checkpoint save failed: " + ckpt_error_);
}

void Trainer::start_prefetch() {
    {
        std::lock_guard<std::mutex> lk(prefetch_mutex_);
        if (prefetch_running_) return;
        prefetch_ready_ = false;
        prefetch_stop_ = false;
        prefetch_pause_ = false;
        prefetch_in_io_ = false;
        prefetch_running_ = true;
        prefetch_error_ = nullptr;
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
            } catch (...) {

                std::lock_guard<std::mutex> fail_lock(prefetch_mutex_);
                prefetch_error_ = std::current_exception();
                prefetch_stop_ = true;
                prefetch_ready_ = false;
                prefetch_cv_.notify_all();
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
            const bool ok = train_loader_.next(out);

            if (ok) last_consumed_loader_state_ = train_loader_.get_state();
            return ok;
        }
        prefetch_cv_.wait(lk, [this] { return prefetch_ready_ || prefetch_stop_; });
        if (prefetch_error_) {
            auto e = prefetch_error_;
            lk.unlock();
            std::rethrow_exception(e);
        }
        if (!prefetch_ready_) GAI_FAIL("dataloader has no shards mid-run in " + cfg_.data_dir);

        out.ids.swap(prefetch_batch_.ids);
        out.targets.swap(prefetch_batch_.targets);
        out.segment_ids.swap(prefetch_batch_.segment_ids);
        out.B = prefetch_batch_.B;
        out.T = prefetch_batch_.T;
        out.tokens_supervised = prefetch_batch_.tokens_supervised;

        last_consumed_loader_state_ = train_loader_.get_state();
        prefetch_ready_ = false;
    }
    prefetch_cv_.notify_all();
    return true;
}

}
