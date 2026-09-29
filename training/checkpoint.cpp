#include "training/checkpoint.h"

#include <fstream>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <cmath>

namespace fs = std::filesystem;

namespace gai {

static constexpr u32 CKPT_MAGIC   = 0x54504B47u;

static constexpr u32 CKPT_VERSION = 12u;
static constexpr u8 OPT_ADAMW = 0u;
static constexpr u8 OPT_LION  = 1u;
static constexpr u8 OPT_MUON  = 2u;

static bool checkpoint_version_supported(u32 version) {
    return version >= 3u && version <= CKPT_VERSION;
}

static bool rd_tok_fp_v11(std::istream& f, TrainState& state, u32 version);

template <typename T> static void wr(std::ostream& o, const T& v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T> static bool rd(std::istream& i, T& v) {
    return static_cast<bool>(i.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

static bool rd_tok_fp_v11(std::istream& f, TrainState& state, u32 version) {
    if (version >= 11u) {
        if (!rd(f, state.tok_fingerprint)) return false;
    } else {
        state.tok_fingerprint = 0;
    }
    return true;
}

static bool rd_loader_v5(std::istream& i, DataLoader::State& s);
static bool rd_loader_legacy(std::istream& i, DataLoader::State& s);
static bool rd_sched_v8(std::istream& f, TrainState& state);
static bool rd_state_prefix(std::istream& f, TrainState& state, u32 version) {
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
    if (!rd_tok_fp_v11(f, state, version)) return false;
    if (version >= 12u) {
        if (!rd_loader_v5(f, state.val_loader)) return false;
    } else {

        state.val_loader = DataLoader::State{};
        state.val_loader.batches = -1;
    }
    return true;
}

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
    wr(o, c.moe_jitter); wr(o, c.rope_yarn_mscale);
    u8 auxfree = c.moe_aux_free ? 1 : 0;
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
    c.moe_jitter = 0.0f; c.rope_yarn_mscale = 0.0f;
    return true;
}
static void wr_loader_v5(std::ostream& o, const DataLoader::State& s) {
    for (int k = 0; k < 4; ++k) wr(o, s.rng[k]);
    wr(o, s.batches);
    wr(o, s.rng_spare);
    wr(o, s.rng_has_spare);
}

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

static bool rd_loader_legacy(std::istream& i, DataLoader::State& s) {
    s = DataLoader::State{};
    for (int k = 0; k < 4; ++k) if (!rd(i, s.rng[k])) return false;
    if (!rd(i, s.batches)) return false;
    return true;
}
static bool arch_match(const ModelConfig& a, const ModelConfig& b) {

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

static CheckpointSnapshot capture_common(const Model& model, const TrainState& state) {
    CheckpointSnapshot snapshot;
    snapshot.config = model.config();
    snapshot.state = state;
    snapshot.weights.reserve(model.parameters().size());
    snapshot.parameter_names.reserve(model.parameters().size());
    for (const Parameter* p : model.parameters()) {
        Tensor cpu = p->w.device() == Device::CPU ? p->w.clone() : p->w.to(Device::CPU);
        snapshot.weights.push_back(std::move(cpu));
        snapshot.parameter_names.push_back(p->name);
    }
    snapshot.moe_bias = model.moe_bias_all();
    return snapshot;
}

CheckpointSnapshot Checkpoint::capture(const Model& model,
                                       const AdamW& opt,
                                       const TrainState& state) {
    CheckpointSnapshot snapshot = capture_common(model, state);
    snapshot.optimizer = opt.snapshot_state();
    return snapshot;
}

CheckpointSnapshot Checkpoint::capture(const Model& model,
                                       const Lion& opt,
                                       const TrainState& state) {
    CheckpointSnapshot snapshot = capture_common(model, state);
    snapshot.optimizer = opt.snapshot_state();
    return snapshot;
}

CheckpointSnapshot Checkpoint::capture(const Model& model,
                                       const Muon& opt,
                                       const TrainState& state) {
    CheckpointSnapshot snapshot = capture_common(model, state);
    snapshot.optimizer = opt.snapshot_state();
    return snapshot;
}

static void write_optimizer_snapshot(std::ostream& f, const OptimizerStateSnapshot& opt) {
    const u8 fmt = 1;
    const i64 step = opt.step;

    const u64 count = static_cast<u64>(opt.first.size());
    switch (opt.kind) {
        case OptimizerSnapshotKind::AdamW:
            f.write(reinterpret_cast<const char*>(&count), sizeof(count));
            f.write(reinterpret_cast<const char*>(&step), sizeof(step));
            f.write(reinterpret_cast<const char*>(&fmt), sizeof(fmt));
            wr(f, opt.adamw.lr); wr(f, opt.adamw.beta1); wr(f, opt.adamw.beta2);
            wr(f, opt.adamw.eps); wr(f, opt.adamw.weight_decay); wr(f, opt.adamw.grad_clip);
            for (size_t i = 0; i < count; ++i) {
                u8 has = opt.first[i].defined() ? 1 : 0;
                wr(f, has);
                if (!has) continue;
                u64 ne = static_cast<u64>(opt.first[i].numel());
                wr(f, ne);
                f.write(reinterpret_cast<const char*>(opt.first[i].data_ptr()),
                        static_cast<std::streamsize>(opt.first[i].nbytes()));
                f.write(reinterpret_cast<const char*>(opt.second[i].data_ptr()),
                        static_cast<std::streamsize>(opt.second[i].nbytes()));
            }
            break;
        case OptimizerSnapshotKind::Lion:
            f.write(reinterpret_cast<const char*>(&count), sizeof(count));
            f.write(reinterpret_cast<const char*>(&step), sizeof(step));
            f.write(reinterpret_cast<const char*>(&fmt), sizeof(fmt));
            wr(f, opt.lion.lr); wr(f, opt.lion.beta1); wr(f, opt.lion.beta2);
            wr(f, opt.lion.weight_decay); wr(f, opt.lion.grad_clip);
            for (const Tensor& t : opt.first) {
                u8 has = t.defined() ? 1 : 0;
                wr(f, has);
                if (!has) continue;
                u64 ne = static_cast<u64>(t.numel());
                wr(f, ne);
                f.write(reinterpret_cast<const char*>(t.data_ptr()),
                        static_cast<std::streamsize>(t.nbytes()));
            }
            break;
        case OptimizerSnapshotKind::Muon:
            f.write(reinterpret_cast<const char*>(&count), sizeof(count));
            f.write(reinterpret_cast<const char*>(&step), sizeof(step));
            f.write(reinterpret_cast<const char*>(&fmt), sizeof(fmt));
            wr(f, opt.muon.lr); wr(f, opt.muon.vec_lr_ratio); wr(f, opt.muon.beta1);
            wr(f, opt.muon.beta2); wr(f, opt.muon.eps); wr(f, opt.muon.weight_decay);
            wr(f, opt.muon.grad_clip);
            {
                i32 ns = opt.muon.ns_steps;
                wr(f, ns);
            }
            for (size_t i = 0; i < count; ++i) {
                u8 mask = 0;
                if (opt.first[i].defined()) mask |= 1u;
                if (i < opt.second.size() && opt.second[i].defined()) mask |= 2u;
                wr(f, mask);
                if (mask & 1u) {
                    u64 ne = static_cast<u64>(opt.first[i].numel());
                    wr(f, ne);
                    f.write(reinterpret_cast<const char*>(opt.first[i].data_ptr()),
                            static_cast<std::streamsize>(opt.first[i].nbytes()));
                }
                if (mask & 2u) {
                    u64 ne = static_cast<u64>(opt.second[i].numel());
                    wr(f, ne);
                    f.write(reinterpret_cast<const char*>(opt.second[i].data_ptr()),
                            static_cast<std::streamsize>(opt.second[i].nbytes()));
                }
            }
            break;
        case OptimizerSnapshotKind::None:
            break;
    }
}

static void publish_file(const std::string& path, const std::string& tmp) {
    const fs::path target(path);
    const fs::path staged(tmp);
    const fs::path backup(path + ".bak");
    GAI_CHECK(target != staged, "checkpoint temp path aliases destination");
    std::error_code ec;
    if (fs::exists(target, ec) && !ec) {
        GAI_CHECK(!fs::is_directory(target, ec), "checkpoint path is a directory: " + path);
        if (fs::exists(backup, ec) && !ec) {
            fs::remove(backup, ec);
            GAI_CHECK(!ec, "cannot clear stale checkpoint backup: " + ec.message());
        }
        fs::rename(target, backup, ec);
        GAI_CHECK(!ec, "cannot stage previous checkpoint: " + ec.message());
    }
    fs::rename(staged, target, ec);
    if (ec) {
        const std::string reason = ec.message();
        std::error_code restore_ec;
        if (fs::exists(backup, restore_ec) && !restore_ec) fs::rename(backup, target, restore_ec);
        GAI_CHECK(false, "cannot finalise checkpoint: " + reason);
    }

    std::error_code cleanup_ec;
    if (fs::exists(backup, cleanup_ec) && !cleanup_ec) {
        fs::remove(backup, cleanup_ec);
        if (cleanup_ec)
            log_warn("checkpoint backup cleanup failed: " + cleanup_ec.message());
    }
}

size_t unique_snapshot_bytes(const std::vector<std::shared_ptr<CheckpointSnapshot>>& refs) {
    size_t total = 0;
    std::vector<const CheckpointSnapshot*> seen;
    for (const auto& s : refs) {
        if (!s) continue;
        const CheckpointSnapshot* ptr = s.get();
        if (std::find(seen.begin(), seen.end(), ptr) == seen.end()) {
            seen.push_back(ptr);
            total += s->bytes();
        }
    }
    return total;
}

void Checkpoint::save(const std::string& path, const Model& model,
                      const AdamW& opt, const TrainState& state) {
    save(capture(model, opt, state), path);
}

void Checkpoint::save(const std::string& path, const Model& model,
                      const Lion& opt, const TrainState& state) {
    save(capture(model, opt, state), path);
}

void Checkpoint::save(const std::string& path, const Model& model,
                      const Muon& opt, const TrainState& state) {
    save(capture(model, opt, state), path);
}

void Checkpoint::save(const CheckpointSnapshot& snapshot, const std::string& path) {
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::error_code pre_ec;
    if (fs::exists(p, pre_ec) && !pre_ec) {
        GAI_CHECK(!fs::is_directory(p, pre_ec), "checkpoint path is a directory: " + path);
    }
    std::string tmp = path + ".tmp";
    struct TmpGuard {
        std::string path;
        bool keep = false;
        ~TmpGuard() {
            if (!keep && !path.empty()) {
                std::error_code ec;
                fs::remove(path, ec);
            }
        }
    } guard{tmp, false};
    {
        std::ofstream f(tmp, std::ios::binary);
        GAI_CHECK(f.good(), "cannot write checkpoint: " + tmp);
        wr(f, CKPT_MAGIC);
        wr(f, CKPT_VERSION);
        wr_config(f, snapshot.config);
        const TrainState& state = snapshot.state;
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
        wr(f, state.tok_fingerprint);
        wr_loader_v5(f, state.val_loader);

        u64 bias_layers = snapshot.moe_bias.size();
        wr(f, bias_layers);
        for (const auto& bias : snapshot.moe_bias) {
            u32 count = static_cast<u32>(bias.size());
            wr(f, count);
            f.write(reinterpret_cast<const char*>(bias.data()),
                    static_cast<std::streamsize>(bias.size() * sizeof(float)));
        }

        u64 n = snapshot.weights.size();
        GAI_CHECK(n == snapshot.parameter_names.size(), "checkpoint snapshot parameter count mismatch");
        wr(f, n);
        for (size_t i = 0; i < snapshot.weights.size(); ++i) {
            const std::string& name = snapshot.parameter_names[i];
            u32 len = static_cast<u32>(name.size());
            wr(f, len);
            f.write(name.data(), len);
            u64 ne = static_cast<u64>(snapshot.weights[i].numel());
            wr(f, ne);
            f.write(reinterpret_cast<const char*>(snapshot.weights[i].data_ptr()),
                    static_cast<std::streamsize>(snapshot.weights[i].nbytes()));
        }

        u8 has_opt = snapshot.optimizer.kind == OptimizerSnapshotKind::None ? 0 : 1;
        wr(f, has_opt);
        if (has_opt) {
            u8 kind = snapshot.optimizer.kind == OptimizerSnapshotKind::Lion ? OPT_LION :
                      snapshot.optimizer.kind == OptimizerSnapshotKind::Muon ? OPT_MUON : OPT_ADAMW;
            wr(f, kind);
            write_optimizer_snapshot(f, snapshot.optimizer);
        }
        GAI_CHECK(f.good(), "checkpoint write failed");
        f.flush();
    }
    publish_file(path, tmp);
    guard.keep = true;
}

static bool read_moe_bias(std::istream& f, Model& model, bool present) {
    if (!present) return true;
    u64 layers = 0;
    if (!rd(f, layers) || layers > static_cast<u64>(model.config().num_layers)) return false;
    const int ne = model.config().num_experts;
    for (u64 l = 0; l < layers; ++l) {
        u32 count = 0;
        if (!rd(f, count) || count != static_cast<u32>(ne)) return false;
        std::vector<float> bias(count);
        if (count && !f.read(reinterpret_cast<char*>(bias.data()),
                             static_cast<std::streamsize>(count * sizeof(float)))) return false;
        for (float v : bias) if (!std::isfinite(v)) return false;
        model.set_moe_bias(static_cast<int>(l), bias.data(), bias.size());
    }
    return true;
}

bool Checkpoint::load(const std::string& path, Model& model, AdamW* opt, TrainState& state,
                      bool* out_moments_restored, bool strict) {
    if (out_moments_restored) *out_moments_restored = false;
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;
    if (!rd(f, version) || !checkpoint_version_supported(version)) return false;

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

    if (!rd_state_prefix(f, state, version)) return false;

    if (!read_moe_bias(f, model, version >= 10u)) return false;
    if (state.tok_vocab != 0 && state.tok_vocab != model.config().vocab_size) {
        log_error(strfmt("checkpoint vocab %d != model vocab %d; refusing resume",
                         state.tok_vocab, model.config().vocab_size));
        return false;
    }

    u64 n = 0;
    if (!rd(f, n)) return false;

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

    for (Parameter* pp : model.parameters()) {
        if (seen.find(pp->name) == seen.end()) {
            if (strict) {
                log_error("exact resume: checkpoint lacks param " + pp->name);
                return false;
            }
            log_warn("checkpoint lacks param " + pp->name + "; keeping fresh init");
        }
    }

    u8 has_opt = 0;
    if (!rd(f, has_opt)) return false;
    if (version == 3u) {

        if (has_opt && opt) {
            if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
                if (strict) { log_error("exact resume: legacy v3 optimizer state unreadable"); return false; }
                log_warn("optimizer state could not be restored; continuing with fresh moments");
            } else if (out_moments_restored) *out_moments_restored = true;
        }
        return true;
    }
    u8 kind = OPT_ADAMW;
    if (has_opt && !rd(f, kind)) return false;
    if (has_opt && opt) {
        if (kind != OPT_ADAMW) {
            if (strict) {
                log_error("exact resume: checkpoint holds a different optimizer kind");
                return false;
            }
            log_warn("checkpoint holds lion moments but trainer uses adamw; starting fresh moments");
            return true;
        }
        if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
            if (strict) { log_error("exact resume: adamw state unreadable"); return false; }
            log_warn("optimizer state could not be restored; continuing with fresh moments");
        } else if (out_moments_restored) *out_moments_restored = true;
    } else if (has_opt && !opt && strict) {
        log_error("exact resume: checkpoint carries optimizer state but no optimizer was supplied");
        return false;
    }
    return true;
}

bool Checkpoint::load(const std::string& path, Model& model, Lion* opt, TrainState& state,
                      bool* out_moments_restored, bool strict) {
    if (out_moments_restored) *out_moments_restored = false;
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;

    if (!rd(f, version) || !checkpoint_version_supported(version)) return false;
    bool is_legacy = (version < 5u);

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

    if (!rd_state_prefix(f, state, version)) return false;
    if (!read_moe_bias(f, model, version >= 10u)) return false;
    if (state.tok_vocab != 0 && state.tok_vocab != model.config().vocab_size) {
        log_error(strfmt("checkpoint vocab %d != model vocab %d; refusing resume",
                         state.tok_vocab, model.config().vocab_size));
        return false;
    }

    u64 n = 0;
    if (!rd(f, n)) return false;

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
        if (seen_lion.find(pp->name) == seen_lion.end()) {
            if (strict) {
                log_error("exact resume: checkpoint lacks param " + pp->name);
                return false;
            }
            log_warn("checkpoint lacks param " + pp->name + "; keeping fresh init");
        }
    }

    u8 has_opt = 0;
    if (!rd(f, has_opt)) return false;
    if (is_legacy) {

        if (has_opt) {
            if (strict) {
                log_error("exact resume: legacy checkpoint cannot supply lion moments");
                return false;
            }
            log_warn("checkpoint is legacy (v3/v4) but trainer uses lion; "
                     "weights/schedule restored, moments restart fresh");
        }
        return true;
    }
    u8 kind = OPT_ADAMW;
    if (has_opt && !rd(f, kind)) return false;
    if (has_opt && opt) {
        if (kind != OPT_LION) {
            if (strict) {
                log_error("exact resume: checkpoint holds a different optimizer kind");
                return false;
            }
            log_warn("checkpoint holds adamw moments but trainer uses lion; starting fresh moments");
            return true;
        }
        if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
            if (strict) { log_error("exact resume: lion state unreadable"); return false; }
            log_warn("lion state could not be restored; continuing with fresh moments");
        } else if (out_moments_restored) *out_moments_restored = true;
    } else if (has_opt && !opt && strict) {
        log_error("exact resume: checkpoint carries optimizer state but no optimizer was supplied");
        return false;
    }
    return true;
}

bool Checkpoint::load(const std::string& path, Model& model, Muon* opt, TrainState& state,
                      bool* out_moments_restored, bool strict) {
    if (out_moments_restored) *out_moments_restored = false;

    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;
    if (!rd(f, version) || !checkpoint_version_supported(version)) return false;

    bool is_legacy = (version < 5u);

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

    if (!rd_state_prefix(f, state, version)) return false;
    if (!read_moe_bias(f, model, version >= 10u)) return false;
    if (state.tok_vocab != 0 && state.tok_vocab != model.config().vocab_size) {
        log_error(strfmt("checkpoint vocab %d != model vocab %d; refusing resume",
                         state.tok_vocab, model.config().vocab_size));
        return false;
    }

    u64 n = 0;
    if (!rd(f, n)) return false;

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
        if (seen_muon.find(pp->name) == seen_muon.end()) {
            if (strict) {
                log_error("exact resume: checkpoint lacks param " + pp->name);
                return false;
            }
            log_warn("checkpoint lacks param " + pp->name + "; keeping fresh init");
        }
    }

    u8 has_opt = 0;
    if (!rd(f, has_opt)) return false;
    if (is_legacy) {

        if (has_opt) {
            if (strict) {
                log_error("exact resume: legacy checkpoint cannot supply muon moments");
                return false;
            }
            log_warn("checkpoint is legacy but trainer uses muon; "
                     "weights/schedule restored, moments restart fresh");
        }
        return true;
    }
    u8 kind = OPT_ADAMW;
    if (has_opt && !rd(f, kind)) return false;
    if (has_opt && opt) {
        if (kind != OPT_MUON) {
            if (strict) {
                log_error("exact resume: checkpoint holds a different optimizer kind");
                return false;
            }
            log_warn("checkpoint holds other-optimizer moments but trainer uses muon; starting fresh moments");
            return true;
        }
        if (!opt->load_state(f, version >= 7u ? OPT_STATE_CURRENT : OPT_STATE_LEGACY)) {
            if (strict) { log_error("exact resume: muon state unreadable"); return false; }
            log_warn("muon state could not be restored; continuing with fresh moments");
        } else if (out_moments_restored) *out_moments_restored = true;
    } else if (has_opt && !opt && strict) {
        log_error("exact resume: checkpoint carries optimizer state but no optimizer was supplied");
        return false;
    }
    return true;
}

bool Checkpoint::peek(const std::string& path, ModelConfig& cfg, TrainState& state) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    u32 magic = 0, version = 0;
    if (!rd(f, magic) || magic != CKPT_MAGIC) return false;
    if (!rd(f, version) || !checkpoint_version_supported(version)) return false;
    if (version >= 9u) {
        if (!rd_config_v9(f, cfg)) return false;
    } else if (version >= 6u) {
        if (!rd_config_v6(f, cfg)) return false;
    } else if (version == 5u) {
        if (!rd_config_v5(f, cfg)) return false;
    } else {
        if (!rd(f, cfg)) return false;
    }

    if (!rd_state_prefix(f, state, version)) return false;
    return true;
}

std::string Checkpoint::latest_in(const std::string& dir) {
    std::error_code ec;
    if (!fs::exists(dir, ec)) return "";
    std::string last = (fs::path(dir) / "last.ckpt").string();
    if (fs::exists(last, ec)) return last;
    const std::string backup = last + ".bak";
    if (fs::exists(backup, ec)) return backup;

    std::vector<std::string> found;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        std::error_code file_ec;
        if (!e.is_regular_file(file_ec) || file_ec) continue;
        if (e.path().extension() == ".ckpt") found.push_back(e.path().string());
    }
    if (found.empty()) return "";
    std::sort(found.begin(), found.end());
    return found.back();
}

}
