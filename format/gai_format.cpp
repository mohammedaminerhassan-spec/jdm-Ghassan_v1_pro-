#include "format/gai_format.h"
#include "quantization/quantize.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdio>

namespace fs = std::filesystem;

namespace gai {

template <typename T> static void wr(std::ostream& o, const T& v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T> static bool rd(std::istream& i, T& v) {
    return static_cast<bool>(i.read(reinterpret_cast<char*>(&v), sizeof(T)));
}
static void wr_str(std::ostream& o, const std::string& s) {
    u32 n = static_cast<u32>(s.size());
    wr(o, n);
    o.write(s.data(), n);
}
static bool rd_str(std::istream& i, std::string& s, u32 max = 4096) {
    u32 n = 0;
    if (!rd(i, n) || n > max) return false;
    s.assign(n, '\0');
    return n == 0 || static_cast<bool>(i.read(s.data(), n));
}

static u64 align64(u64 x) { return (x + 63ull) & ~63ull; }

// ================================================================ writer
GaiWriter::GaiWriter(const std::string& path) : path_(path) {}

void GaiWriter::set_meta(const std::string& k, const std::string& v) { meta_[k] = v; }

void GaiWriter::set_config(const ModelConfig& c) {
    set_meta("arch", "ghassan");
    set_meta("model_name", "Ghassan v1 Flash");
    set_meta("vocab_size", std::to_string(c.vocab_size));
    set_meta("hidden_size", std::to_string(c.hidden_size));
    set_meta("num_layers", std::to_string(c.num_layers));
    set_meta("num_heads", std::to_string(c.num_heads));
    set_meta("num_kv_heads", std::to_string(c.num_kv_heads));
    set_meta("intermediate_size", std::to_string(c.intermediate_size));
    set_meta("use_moe", c.use_moe ? "1" : "0");
    set_meta("num_experts", std::to_string(c.num_experts));
    set_meta("moe_top_k", std::to_string(c.moe_top_k));
    set_meta("moe_expert_dim", std::to_string(c.moe_expert_dim));
    set_meta("moe_shared", c.moe_shared ? "1" : "0");
    // DeepSeek compat rule: every knob that changes numerics must round-trip.
    // rope_scale/qk_norm/z_loss previously dropped -> silent wrong RoPE/attn
    // on reload. Persist all of them (old files without them load as off/0).
    set_meta("moe_aux_scale", strfmt("%.6f", (double)c.moe_aux_scale));
    set_meta("moe_jitter", strfmt("%.6f", (double)c.moe_jitter));
    set_meta("max_seq_len", std::to_string(c.max_seq_len));
    set_meta("rope_theta", strfmt("%.6f", c.rope_theta));
    set_meta("rope_scale", strfmt("%.6f", (double)c.rope_scale));
    set_meta("rope_yarn_mscale", strfmt("%.6f", (double)c.rope_yarn_mscale));
    // PRO round-trip (old files without them load as defaults).
    set_meta("rope_yarn_low", strfmt("%.6f", (double)c.rope_yarn_low));
    set_meta("rope_yarn_high", strfmt("%.6f", (double)c.rope_yarn_high));
    set_meta("sliding_window", std::to_string(c.sliding_window));
    set_meta("rope_type", std::to_string(c.rope_type));
    set_meta("moe_aux_free", c.moe_aux_free ? "1" : "0");
    set_meta("rms_eps", strfmt("%.9f", c.rms_eps));
    set_meta("use_qk_norm", c.use_qk_norm ? "1" : "0");
    set_meta("z_loss_scale", strfmt("%.9f", (double)c.z_loss_scale));
    set_meta("tie_embeddings", c.tie_embeddings ? "1" : "0");
    set_meta("norm_type", "rmsnorm");
    set_meta("ffn_type", c.use_moe ? "swiglu_moe" : "swiglu");
    set_meta("pos_type", "rope");
    set_meta("attn_type", "gqa_causal");
}

void GaiWriter::set_tokenizer_blob(const std::string& gtok_path) {
    std::ifstream f(gtok_path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot read tokenizer for embedding: " + gtok_path);
    std::ostringstream ss;
    ss << f.rdbuf();
    tok_blob_ = ss.str();
}

void GaiWriter::add_tensor(const std::string& name, const Tensor& t) {
    GAI_CHECK(t.device() == Device::CPU, "gai writer needs CPU tensors");
    tensors_.emplace_back(name, t);
}

void GaiWriter::write() {
    fs::path p(path_);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    // header text block
    std::ostringstream hs;
    for (const auto& [k, v] : meta_) hs << k << "=" << v << "\n";
    std::string header = hs.str();

    std::string tmp = path_ + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        GAI_CHECK(f.good(), "cannot write model: " + tmp);

        wr(f, GAI_MAGIC);
        wr(f, GAI_VERSION);
        u32 hb = static_cast<u32>(header.size());
        wr(f, hb);
        f.write(header.data(), hb);

        u64 count = static_cast<u64>(tensors_.size());
        wr(f, count);

        // Reserve the directory, fill offsets after the data is laid out.
        std::streampos dir_pos = f.tellp();
        std::vector<TensorEntry> entries;
        entries.reserve(tensors_.size());
        for (const auto& [name, t] : tensors_) {
            TensorEntry e;
            e.name   = name;
            e.dtype  = t.dtype();
            e.dims   = t.shape();
            e.nbytes = t.nbytes();
            entries.push_back(std::move(e));
        }
        auto write_dir = [&](std::ostream& os) {
            for (const auto& e : entries) {
                wr_str(os, e.name);
                u32 dt = static_cast<u32>(e.dtype);
                wr(os, dt);
                u32 nd = static_cast<u32>(e.dims.size());
                wr(os, nd);
                for (i64 d : e.dims) wr(os, d);
                wr(os, e.offset);
                wr(os, e.nbytes);
            }
        };
        write_dir(f);

        u64 tok_bytes = static_cast<u64>(tok_blob_.size());
        wr(f, tok_bytes);
        if (tok_bytes) f.write(tok_blob_.data(), static_cast<std::streamsize>(tok_bytes));

        // tensor data, aligned
        for (size_t i = 0; i < tensors_.size(); ++i) {
            u64 here = static_cast<u64>(f.tellp());
            u64 aligned = align64(here);
            for (u64 k = here; k < aligned; ++k) f.put('\0');
            entries[i].offset = aligned;
            const Tensor& t = tensors_[i].second;
            f.write(reinterpret_cast<const char*>(t.data_ptr()),
                    static_cast<std::streamsize>(t.nbytes()));
        }

        // rewrite the directory now that offsets are known
        f.seekp(dir_pos);
        write_dir(f);
        GAI_CHECK(f.good(), "gai write failed");
    }
    std::error_code ec;
    fs::remove(path_, ec);
    fs::rename(tmp, path_, ec);
    GAI_CHECK(!ec, "cannot finalise model file: " + ec.message());
}

// ================================================================ reader
bool GaiReader::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0, hb = 0;
    if (!rd(f, magic) || magic != GAI_MAGIC) return false;
    if (!rd(f, version) || version != GAI_VERSION) return false;
    if (!rd(f, hb) || hb > (1u << 22)) return false;

    std::string header(hb, '\0');
    if (hb && !f.read(header.data(), hb)) return false;

    meta_.clear();
    {
        std::istringstream hs(header);
        std::string line;
        while (std::getline(hs, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            meta_[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }

    u64 count = 0;
    if (!rd(f, count) || count > 1000000ull) return false;
    entries_.clear();
    entries_.reserve(static_cast<size_t>(count));
    for (u64 i = 0; i < count; ++i) {
        TensorEntry e;
        if (!rd_str(f, e.name)) return false;
        u32 dt = 0, nd = 0;
        if (!rd(f, dt) || !rd(f, nd) || nd > 8) return false;
        e.dtype = static_cast<DType>(dt);
        e.dims.resize(nd);
        for (u32 d = 0; d < nd; ++d) if (!rd(f, e.dims[d])) return false;
        if (!rd(f, e.offset) || !rd(f, e.nbytes)) return false;
        entries_.push_back(std::move(e));
    }

    u64 tok_bytes = 0;
    if (!rd(f, tok_bytes)) return false;
    if (tok_bytes > 0 && tok_bytes < (1ull << 32)) {
        tok_blob_.assign(static_cast<size_t>(tok_bytes), '\0');
        if (!f.read(tok_blob_.data(), static_cast<std::streamsize>(tok_bytes))) return false;
    }

    std::error_code ec;
    file_size_ = static_cast<u64>(fs::file_size(path, ec));
    path_ = path;
    return true;
}

std::string GaiReader::meta_str(const std::string& k, const std::string& def) const {
    auto it = meta_.find(k);
    return it == meta_.end() ? def : it->second;
}
i64 GaiReader::meta_int(const std::string& k, i64 def) const {
    auto it = meta_.find(k);
    if (it == meta_.end()) return def;
    try { return std::stoll(it->second); } catch (...) { return def; }
}
float GaiReader::meta_f32(const std::string& k, float def) const {
    auto it = meta_.find(k);
    if (it == meta_.end()) return def;
    try { return std::stof(it->second); } catch (...) { return def; }
}

ModelConfig GaiReader::model_config() const {
    ModelConfig c;
    c.vocab_size        = static_cast<int>(meta_int("vocab_size", c.vocab_size));
    c.hidden_size       = static_cast<int>(meta_int("hidden_size", c.hidden_size));
    c.num_layers        = static_cast<int>(meta_int("num_layers", c.num_layers));
    c.num_heads         = static_cast<int>(meta_int("num_heads", c.num_heads));
    c.num_kv_heads      = static_cast<int>(meta_int("num_kv_heads", c.num_kv_heads));
    c.intermediate_size = static_cast<int>(meta_int("intermediate_size", c.intermediate_size));
    c.use_moe           = meta_int("use_moe", c.use_moe ? 1 : 0) != 0;
    c.num_experts       = static_cast<int>(meta_int("num_experts", c.num_experts));
    c.moe_top_k         = static_cast<int>(meta_int("moe_top_k", c.moe_top_k));
    c.moe_expert_dim    = static_cast<int>(meta_int("moe_expert_dim", c.moe_expert_dim));
    c.moe_shared        = meta_int("moe_shared", c.moe_shared ? 1 : 0) != 0;
    c.moe_aux_scale     = meta_f32("moe_aux_scale", c.moe_aux_scale);
    c.moe_jitter        = meta_f32("moe_jitter", c.moe_jitter);
    c.max_seq_len       = static_cast<int>(meta_int("max_seq_len", c.max_seq_len));
    c.rope_theta        = meta_f32("rope_theta", c.rope_theta);
    c.rope_scale        = meta_f32("rope_scale", c.rope_scale);
    c.rope_yarn_mscale  = meta_f32("rope_yarn_mscale", c.rope_yarn_mscale);
    c.rope_yarn_low     = meta_f32("rope_yarn_low", c.rope_yarn_low);
    c.rope_yarn_high    = meta_f32("rope_yarn_high", c.rope_yarn_high);
    c.sliding_window    = static_cast<int>(meta_int("sliding_window", c.sliding_window));
    c.rope_type         = static_cast<int>(meta_int("rope_type", c.rope_type));
    c.moe_aux_free      = meta_int("moe_aux_free", c.moe_aux_free ? 1 : 0) != 0;
    c.rms_eps           = meta_f32("rms_eps", c.rms_eps);
    c.use_qk_norm       = meta_int("use_qk_norm", c.use_qk_norm ? 1 : 0) != 0;
    c.z_loss_scale      = meta_f32("z_loss_scale", c.z_loss_scale);
    c.tie_embeddings    = meta_int("tie_embeddings", 1) != 0;
    c.validate();
    return c;
}

bool GaiReader::load_tokenizer(Tokenizer& tk) const {
    if (tok_blob_.empty()) return false;
    fs::path tmp = fs::temp_directory_path() /
                   ("gai_tok_" + std::to_string(fnv1a64(tok_blob_.data(), tok_blob_.size())) + ".gtok");
    {
        std::ofstream o(tmp, std::ios::binary);
        if (!o.good()) return false;
        o.write(tok_blob_.data(), static_cast<std::streamsize>(tok_blob_.size()));
    }
    bool ok = tk.load(tmp.string());
    std::error_code ec;
    fs::remove(tmp, ec);
    return ok;
}

const TensorEntry* GaiReader::find(const std::string& name) const {
    for (const auto& e : entries_) if (e.name == name) return &e;
    return nullptr;
}

Tensor GaiReader::read_tensor_raw(const std::string& name) const {
    const TensorEntry* e = find(name);
    GAI_CHECK(e != nullptr, "tensor not found in model file: " + name);
    std::ifstream f(path_, std::ios::binary);
    GAI_CHECK(f.good(), "cannot reopen model file");
    f.seekg(static_cast<std::streamoff>(e->offset));
    Tensor t(e->dims, e->dtype, Device::CPU);
    GAI_CHECK(t.nbytes() == e->nbytes, "tensor size mismatch for " + name);
    GAI_CHECK(static_cast<bool>(f.read(reinterpret_cast<char*>(t.data_ptr()),
                                       static_cast<std::streamsize>(e->nbytes))),
              "short read for tensor " + name);
    t.set_name(name);
    return t;
}

Tensor GaiReader::read_tensor_f32(const std::string& name) const {
    Tensor raw = read_tensor_raw(name);
    if (raw.dtype() == DType::F32) return raw;
    return quant::dequantize(raw, DType::F32);
}

bool GaiReader::map_weights() {
    if (path_.empty()) return false;
    auto m = std::make_shared<MappedFile>();
    if (!m->open(path_)) return false;
    // The directory was parsed from this same file; the mapping must cover it.
    if (m->size() != file_size_ && file_size_ != 0) return false;
    mapping_ = std::move(m);
    return true;
}

Tensor GaiReader::read_tensor_wrapped(const std::string& name) const {
    GAI_CHECK(weights_mapped(), "read_tensor_wrapped needs map_weights() first");
    const TensorEntry* e = find(name);
    GAI_CHECK(e != nullptr, "tensor not found in model file: " + name);
    // Bounds-check BEFORE forming the view: a corrupt offset must fail fast
    // here, never become an out-of-bounds pointer (segfault on first touch).
    const u64 msize = mapping_->size();
    GAI_CHECK(e->offset <= msize && e->nbytes <= msize - e->offset,
              "tensor region outside mapped file (corrupt directory): " + name);
    const size_t expect = dtype_nbytes(e->dtype, static_cast<size_t>(numel_of(e->dims)));
    GAI_CHECK(e->nbytes == expect, "tensor size mismatch for " + name);
    const char* base = static_cast<const char*>(mapping_->data());
    void* view = const_cast<char*>(base + static_cast<size_t>(e->offset));
    Tensor t = Tensor::wrap_external(e->dims, e->dtype, view,
                                     static_cast<size_t>(e->nbytes), mapping_);
    t.set_name(name);
    return t;
}

// ================================================================ model io
bool load_model_from_gai(const std::string& path, Model& model) {
    GaiReader r;
    if (!r.open(path)) {
        log_error("cannot open model file: " + path);
        return false;
    }
    for (Parameter* p : model.parameters()) {
        const TensorEntry* e = r.find(p->name);
        if (!e) {
            log_error("model file is missing tensor: " + p->name);
            return false;
        }
        Tensor t = r.read_tensor_f32(p->name);
        if (t.numel() != p->numel()) {
            log_error(strfmt("tensor %s size mismatch: file %lld vs model %lld",
                             p->name.c_str(), static_cast<long long>(t.numel()),
                             static_cast<long long>(p->numel())));
            return false;
        }
        p->w.copy_from(t);
    }
    return true;
}

bool load_model_from_gai_mmap(const std::string& path, Model& model,
                              int* wrapped_out, int* converted_out) {
    GaiReader r;
    if (!r.open(path)) {
        log_error("cannot open model file: " + path);
        return false;
    }
    if (!r.map_weights()) {
        log_error("cannot memory-map model file: " + path);
        return false;
    }
    int wrapped = 0, converted = 0;
    for (Parameter* p : model.parameters()) {
        const TensorEntry* e = r.find(p->name);
        if (!e) {
            log_error("model file is missing tensor: " + p->name);
            return false;
        }
        if (e->dtype == DType::F32 && e->dims == p->shape) {
            // Zero-copy path: weight memory IS the mapped file (no heap).
            // The mapping stays alive inside the tensor's storage owner.
            Tensor t = r.read_tensor_wrapped(p->name);
            t.set_name(p->name);
            p->w = t;
            ++wrapped;
        } else {
            // Quantized (or otherwise non-F32) entry: CPUs have no Q kernels
            // yet (roadmap item 9), so convert once at load like the normal
            // path. Still correct, just not zero-copy for this tensor.
            Tensor t = r.read_tensor_f32(p->name);
            if (t.numel() != p->numel()) {
                log_error(strfmt("tensor %s size mismatch: file %lld vs model %lld",
                                 p->name.c_str(), static_cast<long long>(t.numel()),
                                 static_cast<long long>(p->numel())));
                return false;
            }
            p->w.copy_from(t);
            ++converted;
        }
    }
    if (wrapped_out) *wrapped_out = wrapped;
    if (converted_out) *converted_out = converted;
    log_info(strfmt("[mmap] %d tensors zero-copy file-backed, %d converted (quantized)",
                    wrapped, converted));
    return true;
}

ExportProfile profile_for(const std::string& name) {
    ExportProfile p;
    p.name = name;
    if (name == "fp32") {
        p.default_dtype = p.embedding_dtype = p.norm_dtype = DType::F32;
    } else if (name == "fp16") {
        p.default_dtype = DType::F16;
        p.embedding_dtype = DType::F16;
        p.norm_dtype = DType::F32;
    } else if (name == "int8") {
        p.default_dtype = DType::Q8_0;
        p.embedding_dtype = DType::Q8_0;
        p.norm_dtype = DType::F32;
    } else if (name == "int4") {
        // Embeddings stay at Q8: they are read by a gather, not a matmul, and are
        // the single most quantization-sensitive tensor in a small model.
        p.default_dtype = DType::Q4_0;
        p.embedding_dtype = DType::Q8_0;
        p.norm_dtype = DType::F32;
    } else {
        GAI_FAIL("unknown quantization profile: " + name + " (fp32|fp16|int8|int4)");
    }
    return p;
}

void export_model_gai(const std::string& path, Model& model,
                      const std::string& tokenizer_path,
                      const ExportProfile& profile,
                      const std::map<std::string, std::string>& extra_meta) {
    GaiWriter w(path);
    w.set_config(model.config());
    w.set_meta("quant_profile", profile.name);
    w.set_meta("default_dtype", dtype_name(profile.default_dtype));
    w.set_meta("embedding_dtype", dtype_name(profile.embedding_dtype));
    w.set_meta("norm_dtype", dtype_name(profile.norm_dtype));
    w.set_meta("param_count", std::to_string(model.num_parameters()));
    for (const auto& [k, v] : extra_meta) w.set_meta(k, v);

    if (!tokenizer_path.empty()) w.set_tokenizer_blob(tokenizer_path);

    u64 total_bytes = 0;
    for (Parameter* p : model.parameters()) {
        bool is_norm = p->name.find("norm") != std::string::npos;
        bool is_emb  = (p->name == "tok_embeddings" || p->name == "lm_head");
        DType dt = is_norm ? profile.norm_dtype
                 : is_emb  ? profile.embedding_dtype
                           : profile.default_dtype;
        // fall back if the tensor cannot be block-quantized
        if (!quant::is_quantizable(p->numel(), dt)) {
            log_warn(strfmt("tensor %s (%lld elems) is not %s-quantizable; storing f16",
                            p->name.c_str(), static_cast<long long>(p->numel()), dtype_name(dt)));
            dt = DType::F16;
        }
        Tensor cpu = p->w.to(Device::CPU);
        Tensor q = quant::quantize(cpu, dt);
        total_bytes += q.nbytes();
        w.add_tensor(p->name, q);
    }
    w.write();

    log_info(strfmt("[export] %s  profile=%s  tensors=%zu  weights=%s",
                    path.c_str(), profile.name.c_str(), model.parameters().size(),
                    human_bytes(total_bytes).c_str()));
}

} // namespace gai
