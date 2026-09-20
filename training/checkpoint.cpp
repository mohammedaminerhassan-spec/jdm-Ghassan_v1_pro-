#include "training/checkpoint.h"

#include <fstream>
#include <filesystem>
#include <algorithm>
#include <unordered_set>

namespace fs = std::filesystem;

namespace gai {

static constexpr u32 CKPT_MAGIC   = 0x54504B47u;   // "GKPT"
// v3: TrainState carries loss_scale + clean_steps + tok_vocab (v2 rejected)
// v4: adds u8 opt_kind after has_opt (0=adamw, 1=lion).
// v5: field-wise ModelConfig (no compiler padding) + field-wise loader RNG
//     state incl. Box-Muller spare. Loader accepts v3+v4+v5.
// v6 (10/10): adds moe_jitter + rope_yarn_mscale (DeepSeek long-ctx + jitter).
//     Loader accepts v3+v4+v5+v6; v5 files get jitter=0/mscale=0 defaults.
// v7: optimizer state blob is versioned field-wise data with per-parameter
//     presence flags (frozen params store nothing; P2-3/P2-4). Loader accepts
//     v3..v7; v<=6 optimizer blobs use the legacy raw-struct layout.
// v8: TrainState carries scheduler snapshot (total/warmup/peak/min/decay/kind)
//     so resume with a different schedule warns instead of silently reshaping
//     past lr_at(N). Loader accepts v3..v8; v<8 gets sched_total=0 (unknown).
// v9 (Pro): persists moe_aux_free + rope_yarn_low/high + sliding_window +
//     rope_type. Loader accepts v3..v9; v<=8 files get Pro defaults (off).
static constexpr u32 CKPT_VERSION = 9u;
static constexpr u8 OPT_ADAMW = 0u;
static constexpr u8 OPT_LION  = 1u;
static constexpr u8 OPT_MUON  = 2u;

template <typename T> static void wr(std::ostream& o, const T& v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T> static bool rd(std::istream& i, T& v) {
    return static_cast<bool>(i.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

// Field-wise config IO: stable across compilers (no struct padding).
static void wr_config(std::ostream& o, const ModelConfig& c) {
    wr(o, c.vocab_size); wr(o, c.hidden_size); wr(o, c.num_layers);
    wr(o, c.num_heads); wr(o, c.num_kv_heads); wr(o, c.intermediate_size);
    wr(o, c.max_seq_len);
    wr(o, c.rope_theta); wr(o, c.rms_eps); wr(o, c.init_std);
    wr(o, c.moe_aux_scale); wr(o, c.z_loss_scale); wr(o, c.rope_scale);
    u8 tie = c.tie_embeddings ? 1 : 0, moe = c.use_moe ? 1 : 0;
    u8 sh = c.moe_shared ? 1 : 0, qk = c.use_qk_norm ? 1 : 0;
    wr(o, tie); wr(o, moe); wr(o, sh); wr(o, qk);
    wr(o, c.num_experts); wr(o, c.moe_top_k); wr(o, c.moe_expert_dim);
    wr(o, c.moe_jitter); wr(o, c.rope_yarn_mscale); // v6 additions
    u8 auxfree = c.moe_aux_free ? 1 : 0;            // v9 Pro additions
    wr(o, auxfree);
    wr(o, c.rope_yarn_low); wr(o, c.rope_yarn_high);
    wr(o, c.sliding_window); wr(o, c.rope_type);
}
static bool rd_config_v6(std::istream& i, ModelConfig& c) {
    u8 tie = 0, moe = 0, sh = 0, qk = 0;
    if (!rd(i, c.vocab_size)) return false;
    if (!rd(i, c.hidden_size)) return false;
    if (!rd(i, c.num_layers)) return false;
    if (!rd(i, c.num_heads)) return false;
    if (!rd(i, c.num_kv_heads)) return false;
    if (!rd(i, c.intermediate_size)) return false;
    if (!rd(i, c.max_seq_len)) return false;
    if (!rd(i, c.rope_theta)) return false;
    if (!rd(i, c.rms_eps)) return false;
    if (!rd(i, c.init_std)) return false;
    if (!rd(i, c.moe_aux_scale)) return false;
    if (!rd(i, c.z_loss_scale)) return false;
    if (!rd(i, c.rope_scale)) return false;
    if (!rd(i, tie)) return false;
    if (!rd(i, moe)) return false;
    if (!rd(i, sh)) return false;
    if (!rd(i, qk)) return false;
    if (!rd(i, c.num_experts)) return false;
    if (!rd(i, c.moe_top_k)) return false;
    if (!rd(i, c.moe_expert_dim)) return false;
    if (!rd(i, c.moe_jitter)) return false;
    if (!rd(i, c.rope_yarn_mscale)) return false;
    c.tie_embeddings = tie != 0; c.use_moe = moe != 0;
    c.moe_shared = sh != 0; c.use_qk_norm = qk != 0;
    return true;
}
// v9 Pro tail: v6 payload + aux_free/yarn_low/high/sliding/rope_type.
// v<=8 files never had these bytes: caller must set Pro defaults first.
static bool rd_config_v9(std::istream& i, ModelConfig& c) {
    if (!rd_config_v6(i, c)) return false;
    u8 auxfree = 0;
    if (!rd(i, auxfree)) return false;
    if (!rd(i, c.rope_yarn_low)) return false;
    if (!rd(i, c.rope_yarn_high)) return false;
    if (!rd(i, c.sliding_window)) return false;
    if (!rd(i, c.rope_type)) return false;
    c.moe_aux_free = auxfree != 0;
    return true;
}
static bool rd_config_v5(std::istream& i, ModelConfig& c) {
    u8 tie = 0, moe = 0, sh = 0, qk = 0;
    if (!rd(i, c.vocab_size)) return false;
    if (!rd(i, c.hidden_size)) return false;
    if (!rd(i, c.num_layers)) return false;
    if (!rd(i, c.num_heads)) return false;
    if (!rd(i, c.num_kv_heads)) return false;
    if (!rd(i, c.intermediate_size)) return false;
    if (!rd(i, c.max_seq_len)) return false;
    if (!rd(i, c.rope_theta)) return false;
    if (!rd(i, c.rms_eps)) return false;
    if (!rd(i, c.init_std)) return false;
    if (!rd(i, c.moe_aux_scale)) return false;
    if (!rd(i, c.z_loss_scale)) return false;
    if (!rd(i, c.rope_scale)) return false;
    if (!rd(i, tie)) return false;
    if (!rd(i, moe)) return false;
    if (!rd(i, sh)) return false;
    if (!rd(i, qk)) return false;
    if (!rd(i, c.num_experts)) return false;
    if (!rd(i, c.moe_top_k)) return false;
    if (!rd(i, c.moe_expert_dim)) return false;
    c.tie_embeddings = tie != 0; c.use_moe = moe != 0;
    c.moe_shared = sh != 0; c.use_qk_norm = qk != 0;
    c.moe_jitter = 0.0f; c.rope_yarn_mscale = 0.0f; // v5 defaults
    return true;
}
static void wr_loader_v5(std::ostream& o, const DataLoader::State& s) {
    for (int k = 0; k < 4; ++k) wr(o, s.rng[k]);
    wr(o, s.batches);
    wr(o, s.rng_spare);
    wr(o, s.rng_has_spare);
}
// v8 scheduler snapshot (fixed-size, no padding): total/warmup/peak/min/decay/kind + ddp world.
static void wr_sched_v8(std::ostream& o, const TrainState& s) {
    wr(o, s.sched_total);
    wr(o, s.sched_warmup);
    wr(o, s.sched_peak);
    wr(o, s.sched_min_ratio);
    wr(o, s.sched_decay_frac);
    wr(o, s.sched_kind);
    wr(o, s.ddp_world);
}
static bool rd_sched_v8(std::istream& i, TrainState& s) {
    if (!rd(i, s.sched_total)) return false;
    if (!rd(i, s.sched_warmup)) return false;
    if (!rd(i, s.sched_peak)) return false;
    if (!rd(i, s.sched_min_ratio)) return false;
    if (!rd(i, s.sched_decay_frac)) return false;
    if (!rd(i, s.sched_kind)) return false;
    if (!rd(i, s.ddp_world)) return false;
    return true;
}
static bool rd_loader_v5(std::istream& i, DataLoader::State& s) {
    for (int k = 0; k < 4; ++k) if (!rd(i, s.rng[k])) return false;
    if (!rd(i, s.batches)) return false;
    if (!rd(i, s.rng_spare)) return false;
    if (!rd(i, s.rng_has_spare)) return false;
    return true;
}
// Legacy loader state (v3/v4): u64[4] + i64, no spare.
static bool rd_loader_legacy(std::istream& i, DataLoader::State& s) {
    s = DataLoader::State{};
    for (int k = 0; k < 4; ++k) if (!rd(i, s.rng[k])) return false;
    if (!rd(i, s.batches)) return false;
    return true;
}
static bool arch_match(const ModelConfig& a, const ModelConfig& b) {
    // FIX (10/10): only SHAPE-affecting fields block resume (wrong numel).
    // Loss/recipe fields (aux/z/rope/jitter/mscale/eps/init) only warn — old
    // code failed resume when changing aux_scale, which is just a loss weight.
    // qk_norm DOES change shapes (adds 2xHD params/layer) so it must match.
    // max_seq_len changes KV-cache only (no weights), so warn, don't fail.
    if (!(a.vocab_size == b.vocab_size && a.hidden_size == b.hidden_size &&
          a.num_layers == b.num_layers && a.num_heads == b.num_heads &&
          a.num_kv_heads == b.num_kv_heads && a.intermediate_size == b.intermediate_size &&
          a.tie_embeddings == b.tie_embeddings && a.use_moe == b.use_moe &&
          a.num_experts == b.num_experts && a.moe_top_k == b.moe_top_k &&
          a.moe_expert_dim == b.moe_expert_dim && a.moe_shared == b.moe_shared &&
          a.use_qk_norm == b.use_qk_norm))
        return false;
    if (a.max_seq_len != b.max_seq_len || a.rope_theta != b.rope_theta ||
        a.rope_scale != b.rope_scale || a.rope_yarn_mscale != b.rope_yarn_mscale ||
        a.rope_yarn_low != b.rope_yarn_low || a.rope_yarn_high != b.rope_yarn_high ||
        a.sliding_window != b.sliding_window || a.rope_type != b.rope_type ||
        a.moe_aux_free != b.moe_aux_free ||
        a.rms_eps != b.rms_eps || a.z_loss_scale != b.z_loss_scale ||
        a.moe_aux_scale != b.moe_aux_scale || a.moe_jitter != b.moe_jitter)
        log_warn("checkpoint recipe differs (rope/eps/z/aux/jitter/pro); weights match, continuing");
    return true;
}

void Checkpoint::save(const std::string& path, const Model& model,
                      const AdamW& opt, const TrainState& state) {
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    // write to a temp file then rename: a crash mid-write never corrupts the
    // last good checkpoint.
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        GAI_CHECK(f.good(), "cannot write checkpoint: " + tmp);

        wr(f, CKPT_MAGIC);
        wr(f, CKPT_VERSION);
        wr_config(f, model.config());
        wr(f, state.step);
        wr(f, state.tokens_seen);
        wr(f, state.best_val);
        wr(f, state.last_loss);
        wr(f, state.seed);
        wr_loader_v5(f, state.loader);
        wr(f, state.loss_scale);
        wr(f, state.clean_steps);
        wr(f, state.tok_vocab);
        wr_sched_v8(f, state);

        // weights
        const auto& params = const_cast<Model&>(model).parameters();
        u64 n = static_cast<u64>(params.size());
        wr(f, n);
        for (const Parameter* pp : params) {
            u32 len = static_cast<u32>(pp->name.size());
            wr(f, len);
            f.write(pp->name.data(), len);
            u64 ne = static_cast<u64>(pp->numel());
            wr(f, ne);
            Tensor cpu = pp->w.to(Device::CPU);
            f.write(reinterpret_cast<const char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()));
        }

        u8 has_opt = 1;
        wr(f, has_opt);
        u8 kind = OPT_ADAMW;
        wr(f, kind);
        opt.save_state(f);

        GAI_CHECK(f.good(), "checkpoint write failed");
        f.flush();
        // NOTE: f closes here (RAII) BEFORE rename below — data is on disk.
    }
    // FIX: atomic publish. Old code did remove(path)+rename(tmp,path): a
    // SIGKILL/preemption (Kaggle) in that window DELETED last.ckpt with no
    // replacement (*.tmp is ignored by latest_in) -> hours of training lost.
    // POSIX rename() overwrites atomically, so never unlink first. On Windows
    // rename fails if dst exists -> fallback to remove+rename only there.
    {
        std::error_code ec;
        fs::rename(tmp, path, ec);
        if (ec) {
#if defined(_WIN32)
            fs::remove(path, ec);
            ec.clear();
            fs::rename(tmp, path, ec);
#endif
            GAI_CHECK(!ec, "cannot finalise checkpoint: " + ec.message());
        }
    }
}

void Checkpoint::save(const std::string& path, const Model& model,
                      const Lion& opt, const TrainState& state) {
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        GAI_CHECK(f.good(), "cannot write checkpoint: " + tmp);

        wr(f, CKPT_MAGIC);
        wr(f, CKPT_VERSION);
        wr_config(f, model.config());
        wr(f, state.step);
        wr(f, state.tokens_seen);
        wr(f, state.best_val);
        wr(f, state.last_loss);
        wr(f, state.seed);
        wr_loader_v5(f, state.loader);
        wr(f, state.loss_scale);
        wr(f, state.clean_steps);
        wr(f, state.tok_vocab);
        wr_sched_v8(f, state);

        const auto& params = const_cast<Model&>(model).parameters();
        u64 n = static_cast<u64>(params.size());
        wr(f, n);
        for (const Parameter* pp : params) {
            u32 len = static_cast<u32>(pp->name.size());
            wr(f, len);
            f.write(pp->name.data(), len);
            u64 ne = static_cast<u64>(pp->numel());
            wr(f, ne);
            Tensor cpu = pp->w.to(Device::CPU);
            f.write(reinterpret_cast<const char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()));
        }

        u8 has_opt = 1;
        wr(f, has_opt);
        u8 kind = OPT_LION;
        wr(f, kind);
        opt.save_state(f);

        GAI_CHECK(f.good(), "checkpoint write failed");
        f.flush();
    }
    // Same atomic-publish contract as the AdamW overload (see above).
    {
        std::error_code ec;
        fs::rename(tmp, path, ec);
        if (ec) {
#if defined(_WIN32)
            fs::remove(path, ec);
            ec.clear();
            fs::rename(tmp, path, ec);
#endif
            GAI_CHECK(!ec, "cannot finalise checkpoint: " + ec.message());
        }
    }
}

void Checkpoint::save(const std::string& path, const Model& model,
                      const Muon& opt, const TrainState& state) {
    // Same atomic-publish contract as the AdamW/Lion overloads; only the
    // kind tag and the moments payload differ.
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        GAI_CHECK(f.good(), "cannot write checkpoint: " + tmp);

        wr(f, CKPT_MAGIC);
        wr(f, CKPT_VERSION);
        wr_config(f, model.config());
        wr(f, state.step);
        wr(f, state.tokens_seen);
        wr(f, state.best_val);
        wr(f, state.last_loss);
        wr(f, state.seed);
        wr_loader_v5(f, state.loader);
        wr(f, state.loss_scale);
        wr(f, state.clean_steps);
        wr(f, state.tok_vocab);
        wr_sched_v8(f, state);

        const auto& params = const_cast<Model&>(model).parameters();
        u64 n = static_cast<u64>(params.size());
        wr(f, n);
        for (const Parameter* pp : params) {
            u32 len = static_cast<u32>(pp->name.size());
            wr(f, len);
            f.write(pp->name.data(), len);
            u64 ne = static_cast<u64>(pp->numel());
            wr(f, ne);
            Tensor cpu = pp->w.to(Device::CPU);
            f.write(reinterpret_cast<const char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()));
        }

        u8 has_opt = 1;
        wr(f, has_opt);
        u8 kind = OPT_MUON;
        wr(f, kind);
        opt.save_state(f);

        GAI_CHECK(f.good(), "checkpoint write failed");
        f.flush();
    }
    {
        std::error_code ec;
        fs::rename(tmp, path, ec);
        if (ec) {
#if defined(_WIN32)
            fs::remove(path, ec);
            ec.clear();
            fs::rename(tmp, path, ec);
#endif
            GAI_CHECK(!ec, "cannot finalise checkpoint: " + ec.message());
        }
    }
}

bool Checkpoint::load(const std::string& path, Model& model, AdamW* opt, TrainState& state,
                      bool* out_moments_restored) {
    if (out_moments_restored) *out_moments_restored = false;
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;
    if (!rd(f, version) ||
        (version != 9u && version != 8u && version != 7u && version != 6u && version != 5u && version != 4u && version != 3u))
        return false;

    ModelConfig cfg{};
    if (version >= 9u) {
        if (!rd_config_v9(f, cfg)) return false;
    } else if (version >= 6u) {
        if (!rd_config_v6(f, cfg)) return false;
    } else if (version == 5u) {
        if (!rd_config_v5(f, cfg)) return false;
    } else {
        if (!rd(f, cfg)) return false;
    }
    const ModelConfig& mc = model.config();
    if (!arch_match(cfg, mc)) {
        log_error("checkpoint architecture does not match the current model config");
        return false;
    }

    if (!rd(f, state.step)) return false;
    if (!rd(f, state.tokens_seen)) return false;
    if (!rd(f, state.best_val)) return false;
    if (!rd(f, state.last_loss)) return false;
    if (!rd(f, state.seed)) return false;
    if (version >= 5u) {
        if (!rd_loader_v5(f, state.loader)) return false;
    } else {
        if (!rd_loader_legacy(f, state.loader)) return false;
    }
    if (!rd(f, state.loss_scale)) return false;
    if (!rd(f, state.clean_steps)) return false;
    if (!rd(f, state.tok_vocab)) return false;
    if (version >= 8u) {
        if (!rd_sched_v8(f, state)) return false;
    } else {
        state.sched_total = 0; state.sched_warmup = 0; state.sched_peak = 0.0f; state.ddp_world = 1;
        state.sched_min_ratio = 0.1f; state.sched_decay_frac = 0.2f; state.sched_kind = 0;
    }
    // tokenizer identity: resuming with a different vocab silently corrupts
    // every embedding row. tok_vocab==0 means "unknown" (never for v3 files).
    if (state.tok_vocab != 0 && state.tok_vocab != model.config().vocab_size) {
        log_error(strfmt("checkpoint vocab %d != model vocab %d; refusing resume",
                         state.tok_vocab, model.config().vocab_size));
        return false;
    }

    u64 n = 0;
    if (!rd(f, n)) return false;
    // PRO-HARDEN: ملف فاسد قد يحمل n=1e12 فيدور CPU في حلقة DoS قبل أول rd
    // يفشل. النماذج هنا <10000 بارامتر؛ نفشل فورا فوقها.
    if (n > 10000) { log_error("checkpoint corrupt: param count implausible"); return false; }
    std::unordered_set<std::string> seen;
    for (u64 i = 0; i < n; ++i) {
        u32 len = 0;
        if (!rd(f, len) || len > 512) return false;
        std::string name(len, '\0');
        if (!f.read(name.data(), len)) return false;
        u64 ne = 0;
        if (!rd(f, ne)) return false;
        Parameter* pp = model.find_parameter(name);
        if (!pp || static_cast<u64>(pp->numel()) != ne) {
            log_error("checkpoint tensor mismatch: " + name);
            return false;
        }
        Tensor cpu(pp->shape, DType::F32, Device::CPU);
        if (!f.read(reinterpret_cast<char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()))) return false;
        pp->w.copy_from(cpu);
        seen.insert(name);
    }
    // New params (e.g. qk_qnorm/qk_knorm) missing from old checkpoints keep
    // their fresh init instead of failing the whole resume.
    for (Parameter* pp : model.parameters()) {
        if (seen.find(pp->name) == seen.end())
            log_warn("checkpoint lacks param " + pp->name + "; keeping fresh init");
    }

    u8 has_opt = 0;
    if (!rd(f, has_opt)) return true;   // weights-only checkpoint is valid
    if (version == 3u) {
        // legacy v3: payload is always AdamW
        if (has_opt && opt) {
            if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
                log_warn("optimizer state could not be restored; continuing with fresh moments");
            } else if (out_moments_restored) *out_moments_restored = true;
        }
        return true;
    }
    u8 kind = OPT_ADAMW;
    if (has_opt && !rd(f, kind)) return true;
    if (has_opt && opt) {
        if (kind != OPT_ADAMW) {
            log_warn("checkpoint holds lion moments but trainer uses adamw; starting fresh moments");
            return true;
        }
        if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
            log_warn("optimizer state could not be restored; continuing with fresh moments");
        } else if (out_moments_restored) *out_moments_restored = true;
    }
    return true;
}

bool Checkpoint::load(const std::string& path, Model& model, Lion* opt, TrainState& state,
                      bool* out_moments_restored) {
    if (out_moments_restored) *out_moments_restored = false;
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;
    // Accept v3..v8: weights + schedule always restore; moments restart fresh
    // with a warning when the optimizer kind differs, rather than restarting step 0.
    if (!rd(f, version) ||
        (version != 9u && version != 8u && version != 7u && version != 6u && version != 5u && version != 4u && version != 3u))
        return false;
    bool is_legacy = (version != 8u && version != 7u && version != 6u && version != 5u);

    ModelConfig cfg{};
    if (version >= 9u) {
        if (!rd_config_v9(f, cfg)) return false;
    } else if (version >= 6u) {
        if (!rd_config_v6(f, cfg)) return false;
    } else if (version == 5u) {
        if (!rd_config_v5(f, cfg)) return false;
    } else {
        if (!rd(f, cfg)) return false;
    }
    const ModelConfig& mc = model.config();
    if (!arch_match(cfg, mc)) {
        log_error("checkpoint architecture does not match the current model config");
        return false;
    }

    if (!rd(f, state.step)) return false;
    if (!rd(f, state.tokens_seen)) return false;
    if (!rd(f, state.best_val)) return false;
    if (!rd(f, state.last_loss)) return false;
    if (!rd(f, state.seed)) return false;
    if (version >= 5u) {
        if (!rd_loader_v5(f, state.loader)) return false;
    } else {
        if (!rd_loader_legacy(f, state.loader)) return false;
    }
    if (!rd(f, state.loss_scale)) return false;
    if (!rd(f, state.clean_steps)) return false;
    if (!rd(f, state.tok_vocab)) return false;
    if (version >= 8u) {
        if (!rd_sched_v8(f, state)) return false;
    } else {
        state.sched_total = 0; state.sched_warmup = 0; state.sched_peak = 0.0f; state.ddp_world = 1;
        state.sched_min_ratio = 0.1f; state.sched_decay_frac = 0.2f; state.sched_kind = 0;
    }
    if (state.tok_vocab != 0 && state.tok_vocab != model.config().vocab_size) {
        log_error(strfmt("checkpoint vocab %d != model vocab %d; refusing resume",
                         state.tok_vocab, model.config().vocab_size));
        return false;
    }

    u64 n = 0;
    if (!rd(f, n)) return false;
    // PRO-HARDEN: ملف فاسد قد يحمل n=1e12 فيدور CPU في حلقة DoS قبل أول rd
    // يفشل. النماذج هنا <10000 بارامتر؛ نفشل فورا فوقها.
    if (n > 10000) { log_error("checkpoint corrupt: param count implausible"); return false; }
    std::unordered_set<std::string> seen_lion;
    for (u64 i = 0; i < n; ++i) {
        u32 len = 0;
        if (!rd(f, len) || len > 512) return false;
        std::string name(len, '\0');
        if (!f.read(name.data(), len)) return false;
        u64 ne = 0;
        if (!rd(f, ne)) return false;
        Parameter* pp = model.find_parameter(name);
        if (!pp || static_cast<u64>(pp->numel()) != ne) {
            log_error("checkpoint tensor mismatch: " + name);
            return false;
        }
        Tensor cpu(pp->shape, DType::F32, Device::CPU);
        if (!f.read(reinterpret_cast<char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()))) return false;
        pp->w.copy_from(cpu);
        seen_lion.insert(name);
    }
    for (Parameter* pp : model.parameters()) {
        if (seen_lion.find(pp->name) == seen_lion.end())
            log_warn("checkpoint lacks param " + pp->name + "; keeping fresh init");
    }

    u8 has_opt = 0;
    if (!rd(f, has_opt)) return true;
    if (is_legacy) {
        // v3/v4 payload may be AdamW: weights already restored above,
        // moments can't be reused for Lion -> fresh start for opt only.
        if (has_opt)
            log_warn("checkpoint is legacy (v3/v4) but trainer uses lion; "
                     "weights/schedule restored, moments restart fresh");
        return true;
    }
    u8 kind = OPT_ADAMW;
    if (has_opt && !rd(f, kind)) return true;
    if (has_opt && opt) {
        if (kind != OPT_LION) {
            log_warn("checkpoint holds adamw moments but trainer uses lion; starting fresh moments");
            return true;
        }
        if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
            log_warn("lion state could not be restored; continuing with fresh moments");
        } else if (out_moments_restored) *out_moments_restored = true;
    }
    return true;
}

bool Checkpoint::load(const std::string& path, Model& model, Muon* opt, TrainState& state,
                      bool* out_moments_restored) {
    if (out_moments_restored) *out_moments_restored = false;
    // Mirrors the Lion loader exactly (weights + schedule + kind gate); only
    // the expected kind tag and the moments call differ.
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;
    if (!rd(f, version) ||
        (version != 9u && version != 8u && version != 7u && version != 6u && version != 5u && version != 4u && version != 3u))
        return false;
    bool is_legacy = (version != 8u && version != 7u && version != 6u && version != 5u);

    ModelConfig cfg{};
    if (version >= 9u) {
        if (!rd_config_v9(f, cfg)) return false;
    } else if (version >= 6u) {
        if (!rd_config_v6(f, cfg)) return false;
    } else if (version == 5u) {
        if (!rd_config_v5(f, cfg)) return false;
    } else {
        if (!rd(f, cfg)) return false;
    }
    const ModelConfig& mc = model.config();
    if (!arch_match(cfg, mc)) {
        log_error("checkpoint architecture does not match the current model config");
        return false;
    }

    if (!rd(f, state.step)) return false;
    if (!rd(f, state.tokens_seen)) return false;
    if (!rd(f, state.best_val)) return false;
    if (!rd(f, state.last_loss)) return false;
    if (!rd(f, state.seed)) return false;
    if (version >= 5u) {
        if (!rd_loader_v5(f, state.loader)) return false;
    } else {
        if (!rd_loader_legacy(f, state.loader)) return false;
    }
    if (!rd(f, state.loss_scale)) return false;
    if (!rd(f, state.clean_steps)) return false;
    if (!rd(f, state.tok_vocab)) return false;
    if (version >= 8u) {
        if (!rd_sched_v8(f, state)) return false;
    } else {
        state.sched_total = 0; state.sched_warmup = 0; state.sched_peak = 0.0f; state.ddp_world = 1;
        state.sched_min_ratio = 0.1f; state.sched_decay_frac = 0.2f; state.sched_kind = 0;
    }
    if (state.tok_vocab != 0 && state.tok_vocab != model.config().vocab_size) {
        log_error(strfmt("checkpoint vocab %d != model vocab %d; refusing resume",
                         state.tok_vocab, model.config().vocab_size));
        return false;
    }

    u64 n = 0;
    if (!rd(f, n)) return false;
    // PRO-HARDEN: ملف فاسد قد يحمل n=1e12 فيدور CPU في حلقة DoS قبل أول rd
    // يفشل. النماذج هنا <10000 بارامتر؛ نفشل فورا فوقها.
    if (n > 10000) { log_error("checkpoint corrupt: param count implausible"); return false; }
    std::unordered_set<std::string> seen_muon;
    for (u64 i = 0; i < n; ++i) {
        u32 len = 0;
        if (!rd(f, len) || len > 512) return false;
        std::string name(len, '\0');
        if (!f.read(name.data(), len)) return false;
        u64 ne = 0;
        if (!rd(f, ne)) return false;
        Parameter* pp = model.find_parameter(name);
        if (!pp || static_cast<u64>(pp->numel()) != ne) {
            log_error("checkpoint tensor mismatch: " + name);
            return false;
        }
        Tensor cpu(pp->shape, DType::F32, Device::CPU);
        if (!f.read(reinterpret_cast<char*>(cpu.data_ptr()),
                    static_cast<std::streamsize>(cpu.nbytes()))) return false;
        pp->w.copy_from(cpu);
        seen_muon.insert(name);
    }
    for (Parameter* pp : model.parameters()) {
        if (seen_muon.find(pp->name) == seen_muon.end())
            log_warn("checkpoint lacks param " + pp->name + "; keeping fresh init");
    }

    u8 has_opt = 0;
    if (!rd(f, has_opt)) return true;
    if (is_legacy) {
        // v3/v4 payloads predate Muon: weights already restored above,
        // moments restart fresh for the optimizer only.
        if (has_opt)
            log_warn("checkpoint is legacy but trainer uses muon; "
                     "weights/schedule restored, moments restart fresh");
        return true;
    }
    u8 kind = OPT_ADAMW;
    if (has_opt && !rd(f, kind)) return true;
    if (has_opt && opt) {
        if (kind != OPT_MUON) {
            log_warn("checkpoint holds other-optimizer moments but trainer uses muon; starting fresh moments");
            return true;
        }
        if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
            log_warn("muon state could not be restored; continuing with fresh moments");
        } else if (out_moments_restored) *out_moments_restored = true;
    }
    return true;
}

bool Checkpoint::peek(const std::string& path, ModelConfig& cfg, TrainState& state) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;
    if (!rd(f, version) ||
        (version != 9u && version != 8u && version != 7u && version != 6u && version != 5u && version != 4u && version != 3u))
        return false;
    if (version >= 9u) {
        if (!rd_config_v9(f, cfg)) return false;
    } else if (version >= 6u) {
        if (!rd_config_v6(f, cfg)) return false;
    } else if (version == 5u) {
        if (!rd_config_v5(f, cfg)) return false;
    } else {
        if (!rd(f, cfg)) return false;
    }
    if (!rd(f, state.step)) return false;
    if (!rd(f, state.tokens_seen)) return false;
    if (!rd(f, state.best_val)) return false;
    if (!rd(f, state.last_loss)) return false;
    if (!rd(f, state.seed)) return false;
    return true;
}

std::string Checkpoint::latest_in(const std::string& dir) {
    std::error_code ec;
    if (!fs::exists(dir, ec)) return "";
    std::string last = (fs::path(dir) / "last.ckpt").string();
    if (fs::exists(last, ec)) return last;

    std::vector<std::string> found;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        if (e.path().extension() == ".ckpt") found.push_back(e.path().string());
    }
    if (found.empty()) return "";
    std::sort(found.begin(), found.end());
    return found.back();
}

} // namespace gai
