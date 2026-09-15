// format/gguf_format.cpp
// Full GGUF v3 read/write implementation compatible with llama.cpp spec.
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#include "format/gguf_format.h"
#include "quantization/quantize.h"
#include "core/common.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <cassert>
#include <algorithm>

namespace fs = std::filesystem;

namespace gai {

// ================================================================ helpers
template<typename T>
static void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template<typename T>
static bool read_pod(std::istream& is, T& v) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

static bool read_string(std::istream& is, std::string& out) {
    uint64_t len = 0;
    if (!read_pod(is, len)) return false;
    if (len > (1ull << 24)) return false;   // sanity: max 16 MiB string
    out.assign(static_cast<size_t>(len), '\0');
    if (len && !is.read(out.data(), static_cast<std::streamsize>(len))) return false;
    return true;
}

static uint64_t align_up(uint64_t x, uint64_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

// ================================================================ GGUFWriter
GGUFWriter::GGUFWriter(const std::string& path) : path_(path) {}

// ---- metadata helpers -----------------------------------------------
static std::vector<uint8_t> pack_uint32(uint32_t v) {
    std::vector<uint8_t> d(4);
    memcpy(d.data(), &v, 4);
    return d;
}
static std::vector<uint8_t> pack_uint64(uint64_t v) {
    std::vector<uint8_t> d(8);
    memcpy(d.data(), &v, 8);
    return d;
}
static std::vector<uint8_t> pack_float32(float v) {
    std::vector<uint8_t> d(4);
    memcpy(d.data(), &v, 4);
    return d;
}
static std::vector<uint8_t> pack_bool(bool v) {
    return { static_cast<uint8_t>(v ? 1 : 0) };
}
static std::vector<uint8_t> pack_string(const std::string& s) {
    std::vector<uint8_t> d;
    uint64_t len = s.size();
    d.resize(8 + len);
    memcpy(d.data(), &len, 8);
    memcpy(d.data() + 8, s.data(), len);
    return d;
}

static void write_metadata_value(std::ostream& os, const GGUFMetadataValue& v) {
    uint32_t t = static_cast<uint32_t>(v.type);
    write_pod(os, t);
    if (v.type == GGUFType::STRING) {
        // data holds raw u64+bytes
        os.write(reinterpret_cast<const char*>(v.data.data()),
                 static_cast<std::streamsize>(v.data.size()));
    } else if (v.type == GGUFType::ARRAY) {
        os.write(reinterpret_cast<const char*>(v.data.data()),
                 static_cast<std::streamsize>(v.data.size()));
    } else {
        os.write(reinterpret_cast<const char*>(v.data.data()),
                 static_cast<std::streamsize>(v.data.size()));
    }
}

// ---- public setters -------------------------------------------------
void GGUFWriter::set_arch(const std::string& arch) {
    add_custom_metadata("general.architecture", arch);
}
void GGUFWriter::set_model_name(const std::string& name) {
    add_custom_metadata("general.name", name);
}
void GGUFWriter::set_description(const std::string& desc) {
    add_custom_metadata("general.description", desc);
}
void GGUFWriter::set_vocab_size(uint32_t v) {
    add_custom_metadata("ghassan.vocab_size", static_cast<uint64_t>(v));
}
void GGUFWriter::set_hidden_size(uint32_t h) {
    add_custom_metadata("ghassan.embedding_length", static_cast<uint64_t>(h));
}
void GGUFWriter::set_num_layers(uint32_t l) {
    add_custom_metadata("ghassan.block_count", static_cast<uint64_t>(l));
}
void GGUFWriter::set_num_heads(uint32_t h) {
    add_custom_metadata("ghassan.attention.head_count", static_cast<uint64_t>(h));
}
void GGUFWriter::set_num_kv_heads(uint32_t kv) {
    add_custom_metadata("ghassan.attention.head_count_kv", static_cast<uint64_t>(kv));
}
void GGUFWriter::set_intermediate_size(uint32_t f) {
    add_custom_metadata("ghassan.feed_forward_length", static_cast<uint64_t>(f));
}
void GGUFWriter::set_moe_config(bool use_moe, uint32_t num_experts, uint32_t top_k,
                                uint32_t expert_dim, bool shared) {
    add_custom_metadata("ghassan.use_moe", use_moe);
    add_custom_metadata("ghassan.moe.expert_count", static_cast<uint64_t>(num_experts));
    add_custom_metadata("ghassan.moe.experts_used_count", static_cast<uint64_t>(top_k));
    add_custom_metadata("ghassan.moe.expert_feed_forward_length",
                        static_cast<uint64_t>(expert_dim));
    add_custom_metadata("ghassan.moe.shared_expert", shared);
}
void GGUFWriter::set_max_seq_len(uint32_t ctx) {
    add_custom_metadata("ghassan.context_length", static_cast<uint64_t>(ctx));
}
void GGUFWriter::set_rope_theta(float theta) {
    add_custom_metadata("ghassan.rope.freq_base", theta);
}
void GGUFWriter::set_rope_scaling(const std::map<std::string, std::string>&) {
    // optional: ignored for now
}
void GGUFWriter::set_rms_eps(float eps) {
    add_custom_metadata("ghassan.attention.layer_norm_rms_epsilon", eps);
}
void GGUFWriter::set_tie_embeddings(bool tie) {
    add_custom_metadata("ghassan.tie_word_embeddings", tie);
}
void GGUFWriter::set_norm_type(const std::string& type) {
    add_custom_metadata("ghassan.norm_type", type);
}
void GGUFWriter::set_ffn_type(const std::string& type) {
    add_custom_metadata("ghassan.ffn_type", type);
}
void GGUFWriter::set_attn_type(const std::string& type) {
    add_custom_metadata("ghassan.attention.type", type);
}
void GGUFWriter::set_tokenizer_model(const std::string& model) {
    add_custom_metadata("tokenizer.ggml.model", model);
}
void GGUFWriter::set_tokenizer_tokens(const std::vector<std::string>& tokens) {
    // Pack as ARRAY of STRING
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    // array header: element_type (u32) + count (u64)
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::STRING);
    uint64_t count     = static_cast<uint64_t>(tokens.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (const auto& t : tokens) {
        uint64_t len = t.size();
        push(&len, 8);
        push(t.data(), t.size());
    }
    metadata_["tokenizer.ggml.tokens"] = std::move(v);
}
void GGUFWriter::set_tokenizer_scores(const std::vector<float>& scores) {
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::FLOAT32);
    uint64_t count     = static_cast<uint64_t>(scores.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (float f : scores) push(&f, 4);
    metadata_["tokenizer.ggml.scores"] = std::move(v);
}
void GGUFWriter::set_tokenizer_token_types(const std::vector<int32_t>& types) {
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::INT32);
    uint64_t count     = static_cast<uint64_t>(types.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (int32_t i : types) push(&i, 4);
    metadata_["tokenizer.ggml.token_type"] = std::move(v);
}
void GGUFWriter::set_tokenizer_bos_id(uint32_t id) {
    add_custom_metadata("tokenizer.ggml.bos_token_id", static_cast<uint64_t>(id));
}
void GGUFWriter::set_tokenizer_eos_id(uint32_t id) {
    add_custom_metadata("tokenizer.ggml.eos_token_id", static_cast<uint64_t>(id));
}
void GGUFWriter::set_tokenizer_unk_id(uint32_t id) {
    add_custom_metadata("tokenizer.ggml.unknown_token_id", static_cast<uint64_t>(id));
}
void GGUFWriter::set_tokenizer_pad_id(uint32_t id) {
    add_custom_metadata("tokenizer.ggml.padding_token_id", static_cast<uint64_t>(id));
}
void GGUFWriter::set_tokenizer_add_bos(bool add) {
    add_custom_metadata("tokenizer.ggml.add_bos_token", add);
}
void GGUFWriter::set_tokenizer_add_eos(bool add) {
    add_custom_metadata("tokenizer.ggml.add_eos_token", add);
}
void GGUFWriter::set_quantization_profile(const std::string& profile) {
    add_custom_metadata("general.quantization_version", profile);
}
void GGUFWriter::set_file_type(uint32_t type) {
    add_custom_metadata("general.file_type", static_cast<uint64_t>(type));
}

void GGUFWriter::add_custom_metadata(const std::string& key, const std::string& value) {
    GGUFMetadataValue v;
    v.type = GGUFType::STRING;
    v.data = pack_string(value);
    metadata_[key] = std::move(v);
}
void GGUFWriter::add_custom_metadata(const std::string& key, uint64_t value) {
    GGUFMetadataValue v;
    v.type = GGUFType::UINT64;
    v.data = pack_uint64(value);
    metadata_[key] = std::move(v);
}
void GGUFWriter::add_custom_metadata(const std::string& key, float value) {
    GGUFMetadataValue v;
    v.type = GGUFType::FLOAT32;
    v.data = pack_float32(value);
    metadata_[key] = std::move(v);
}
void GGUFWriter::add_custom_metadata(const std::string& key, bool value) {
    GGUFMetadataValue v;
    v.type = GGUFType::BOOL;
    v.data = pack_bool(value);
    metadata_[key] = std::move(v);
}

// ---- tensor --------------------------------------------------------
void GGUFWriter::add_tensor(const std::string& name, const Tensor& t, GGMLType dtype) {
    tensors_.emplace_back(name, std::make_pair(t, dtype));
}

// ---- write ---------------------------------------------------------
static size_t ggml_type_size(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return 4;
        case GGMLType::F16:  return 2;
        case GGMLType::Q4_0: return 18;    // block_size=32: 2 bytes f16 scale + 16 bytes data
        case GGMLType::Q8_0: return 34;    // block_size=32: 2 bytes f16 scale + 32 bytes data
        default:             return 4;
    }
}

static size_t ggml_blck_size(GGMLType t) {
    switch (t) {
        case GGMLType::Q4_0: return 32;
        case GGMLType::Q8_0: return 32;
        default:             return 1;
    }
}

static size_t ggml_nbytes(GGMLType t, size_t nelems) {
    size_t blk = ggml_blck_size(t);
    size_t bs  = ggml_type_size(t);
    return (nelems + blk - 1) / blk * bs;
}

// Convert our internal DType to GGML-compatible bytes for a tensor
static std::vector<uint8_t> tensor_to_ggml_bytes(const Tensor& t, GGMLType target) {
    // For simplicity we support F32→F32, F32→F16, and Q8_0/Q4_0 via our quantizer
    size_t n = static_cast<size_t>(t.numel());
    const float* src = t.f32();
    GAI_CHECK(src != nullptr, "gguf_format: tensor must be F32 on CPU for export");

    if (target == GGMLType::F32) {
        std::vector<uint8_t> out(n * 4);
        memcpy(out.data(), src, n * 4);
        return out;
    }

    if (target == GGMLType::F16) {
        std::vector<uint8_t> out(n * 2);
        uint16_t* dst = reinterpret_cast<uint16_t*>(out.data());
        for (size_t i = 0; i < n; ++i) {
            // Simple F32→F16 conversion
            float v = src[i];
            uint32_t bits;
            memcpy(&bits, &v, 4);
            uint32_t sign = (bits >> 31) & 1;
            int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
            uint32_t mant = bits & 0x7FFFFF;
            uint16_t h;
            if (exp > 15) {
                h = static_cast<uint16_t>((sign << 15) | 0x7C00); // inf
            } else if (exp < -14) {
                // denorm or zero
                int shift = -14 - exp;
                uint32_t m = (mant | 0x800000) >> (shift + 13);
                h = static_cast<uint16_t>((sign << 15) | m);
            } else {
                h = static_cast<uint16_t>((sign << 15) | ((exp + 15) << 10) | (mant >> 13));
            }
            dst[i] = h;
        }
        return out;
    }

    if (target == GGMLType::Q8_0) {
        // Q8_0: blocks of 32 floats → (f16 absmax, 32 × i8)
        size_t nblocks = (n + 31) / 32;
        std::vector<uint8_t> out(nblocks * 34);
        for (size_t b = 0; b < nblocks; ++b) {
            size_t start = b * 32;
            size_t end   = std::min(start + 32, n);
            // compute absmax
            float amax = 0.f;
            for (size_t i = start; i < end; ++i) amax = std::max(amax, std::abs(src[i]));
            float scale = amax / 127.f;
            float inv   = (scale != 0.f) ? (1.f / scale) : 0.f;
            // write f16 scale
            uint32_t sbits; memcpy(&sbits, &scale, 4);
            uint32_t ssign = (sbits >> 31) & 1;
            int32_t  sexp  = static_cast<int32_t>((sbits >> 23) & 0xFF) - 127;
            uint32_t smant = sbits & 0x7FFFFF;
            uint16_t sh;
            if (sexp > 15) sh = static_cast<uint16_t>((ssign << 15) | 0x7C00);
            else if (sexp < -14) sh = static_cast<uint16_t>(ssign << 15);
            else sh = static_cast<uint16_t>((ssign << 15) | ((sexp + 15) << 10) | (smant >> 13));
            uint8_t* block = out.data() + b * 34;
            memcpy(block, &sh, 2);
            for (size_t i = 0; i < 32; ++i) {
                float fv = (start + i < end) ? src[start + i] : 0.f;
                int8_t q = static_cast<int8_t>(std::max(-127.f, std::min(127.f, fv * inv)));
                block[2 + i] = static_cast<uint8_t>(q);
            }
        }
        return out;
    }

    if (target == GGMLType::Q4_0) {
        // Q4_0: blocks of 32 floats → (f16 absmax, 16 × uint8 packed nibbles)
        size_t nblocks = (n + 31) / 32;
        std::vector<uint8_t> out(nblocks * 18);
        for (size_t b = 0; b < nblocks; ++b) {
            size_t start = b * 32;
            size_t end   = std::min(start + 32, n);
            float amax = 0.f;
            for (size_t i = start; i < end; ++i) amax = std::max(amax, std::abs(src[i]));
            float scale = amax / 7.f;
            float inv   = (scale != 0.f) ? (1.f / scale) : 0.f;
            uint32_t sbits; memcpy(&sbits, &scale, 4);
            uint32_t ssign = (sbits >> 31) & 1;
            int32_t  sexp  = static_cast<int32_t>((sbits >> 23) & 0xFF) - 127;
            uint32_t smant = sbits & 0x7FFFFF;
            uint16_t sh;
            if (sexp > 15) sh = static_cast<uint16_t>((ssign << 15) | 0x7C00);
            else if (sexp < -14) sh = static_cast<uint16_t>(ssign << 15);
            else sh = static_cast<uint16_t>((ssign << 15) | ((sexp + 15) << 10) | (smant >> 13));
            uint8_t* block = out.data() + b * 18;
            memcpy(block, &sh, 2);
            for (size_t i = 0; i < 16; ++i) {
                float f0 = (start + 2*i     < end) ? src[start + 2*i]     : 0.f;
                float f1 = (start + 2*i + 1 < end) ? src[start + 2*i + 1] : 0.f;
                int8_t q0 = static_cast<int8_t>(std::max(-8.f, std::min(7.f, f0 * inv)));
                int8_t q1 = static_cast<int8_t>(std::max(-8.f, std::min(7.f, f1 * inv)));
                block[2 + i] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
            }
        }
        return out;
    }

    // Fallback: treat as F32
    std::vector<uint8_t> out(n * 4);
    memcpy(out.data(), src, n * 4);
    return out;
}

void GGUFWriter::write() {
    fs::path p(path_);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    std::string tmp = path_ + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        GAI_CHECK(f.good(), "gguf_format: cannot open for write: " + tmp);

        // ---- GGUF header
        write_pod(f, GGUF_MAGIC);
        write_pod(f, GGUF_VERSION);

        uint64_t tensor_count   = static_cast<uint64_t>(tensors_.size());
        uint64_t metadata_count = static_cast<uint64_t>(metadata_.size());
        write_pod(f, tensor_count);
        write_pod(f, metadata_count);

        // ---- metadata key-value pairs
        for (const auto& [key, val] : metadata_) {
            write_string(f, key);
            write_metadata_value(f, val);
        }

        // ---- tensor info (name, ndims, dims[], type, offset)
        // We need to compute offsets after writing all tensor info.
        // Strategy: write all tensor data to a temp buffer, compute offsets,
        // then write tensor info with correct offsets.

        // First pass: compute all tensor data
        struct TensorData {
            std::string            name;
            std::vector<uint64_t>  dims;
            GGMLType               type;
            std::vector<uint8_t>   data;
            uint64_t               offset = 0;
        };
        std::vector<TensorData> tdata;
        tdata.reserve(tensors_.size());

        for (auto& [name, pair] : tensors_) {
            const Tensor& t  = pair.first;
            GGMLType dtype   = pair.second;
            TensorData td;
            td.name = name;
            // dims: GGUF stores dimensions in reverse order (innermost first) — same as ggml
            auto shape = t.shape();
            for (auto it = shape.rbegin(); it != shape.rend(); ++it)
                td.dims.push_back(static_cast<uint64_t>(*it));
            td.type = dtype;
            // get CPU F32 tensor
            Tensor cpu = t.to(Device::CPU);
            if (cpu.dtype() != DType::F32) cpu = quant::dequantize(cpu, DType::F32);
            td.data = tensor_to_ggml_bytes(cpu, dtype);
            tdata.push_back(std::move(td));
        }

        // Compute tensor info block size (to know where data starts)
        // Each entry: u64 name_len + name + u32 ndims + ndims*u64 + u32 type + u64 offset
        uint64_t info_size = 0;
        for (const auto& td : tdata) {
            info_size += 8 + td.name.size();   // u64 len + bytes
            info_size += 4;                     // u32 ndims
            info_size += 8 * td.dims.size();    // ndims * u64
            info_size += 4;                     // u32 type
            info_size += 8;                     // u64 offset
        }

        // The tensor data section starts right after the info section (aligned to 32)
        uint64_t data_section_start = static_cast<uint64_t>(f.tellp()) + info_size;
        data_section_start = align_up(data_section_start, 32);

        // Assign offsets
        uint64_t running_offset = 0;
        for (auto& td : tdata) {
            td.offset = running_offset;
            running_offset += static_cast<uint64_t>(td.data.size());
            running_offset = align_up(running_offset, 32);
        }

        // Write tensor info
        for (const auto& td : tdata) {
            write_string(f, td.name);
            uint32_t ndims = static_cast<uint32_t>(td.dims.size());
            write_pod(f, ndims);
            for (uint64_t d : td.dims) write_pod(f, d);
            uint32_t dtype_u32 = static_cast<uint32_t>(td.type);
            write_pod(f, dtype_u32);
            write_pod(f, td.offset);
        }

        // Pad to alignment
        uint64_t cur = static_cast<uint64_t>(f.tellp());
        uint64_t pad_to = align_up(cur, 32);
        for (uint64_t i = cur; i < pad_to; ++i) f.put('\0');

        // Write tensor data
        for (const auto& td : tdata) {
            uint64_t before = static_cast<uint64_t>(f.tellp());
            f.write(reinterpret_cast<const char*>(td.data.data()),
                    static_cast<std::streamsize>(td.data.size()));
            uint64_t after = static_cast<uint64_t>(f.tellp());
            uint64_t padded = align_up(after, 32);
            for (uint64_t i = after; i < padded; ++i) f.put('\0');
            (void)before;
        }

        GAI_CHECK(f.good(), "gguf_format: write failed");
    }

    std::error_code ec;
    fs::remove(path_, ec);
    fs::rename(tmp, path_, ec);
    GAI_CHECK(!ec, "gguf_format: cannot finalise file: " + ec.message());

    log_info(strfmt("[gguf] wrote %s (%zu tensors, %zu metadata entries)",
                    path_.c_str(), tensors_.size(), metadata_.size()));
}

// ================================================================ GGUFReader
bool GGUFReader::open(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    uint32_t magic = 0;
    if (!read_pod(f, magic) || magic != GGUF_MAGIC) return false;

    uint32_t version = 0;
    if (!read_pod(f, version) || (version < 2 || version > 3)) {
        log_warn("gguf_format: unsupported version " + std::to_string(version));
        return false;
    }

    uint64_t tensor_count = 0, metadata_count = 0;
    if (!read_pod(f, tensor_count)) return false;
    if (!read_pod(f, metadata_count)) return false;

    if (!read_metadata(f, metadata_count)) return false;
    if (!read_tensor_info(f, tensor_count)) return false;

    std::error_code ec;
    file_size_ = static_cast<uint64_t>(fs::file_size(path, ec));
    path_ = path;
    return true;
}

// Fixed byte size of GGUF scalar metadata types. 0 = variable-length
// (STRING/ARRAY) or unknown future type: those cannot be skipped blindly
// because their size is unknowable without parsing them.
static size_t gguf_scalar_size(GGUFType t) {
    switch (t) {
        case GGUFType::UINT8:
        case GGUFType::INT8:
        case GGUFType::BOOL:    return 1;
        case GGUFType::UINT16:
        case GGUFType::INT16:   return 2;
        case GGUFType::UINT32:
        case GGUFType::INT32:
        case GGUFType::FLOAT32: return 4;
        case GGUFType::UINT64:
        case GGUFType::INT64:
        case GGUFType::FLOAT64: return 8;
        default: return 0;
    }
}

bool GGUFReader::read_metadata(std::istream& is, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        std::string key;
        if (!read_string(is, key)) return false;
        uint32_t type_u32 = 0;
        if (!read_pod(is, type_u32)) return false;
        GGUFType type = static_cast<GGUFType>(type_u32);

        GGUFMetadataValue val;
        val.type = type;

        auto read_scalar = [&](size_t bytes) {
            val.data.resize(bytes);
            return static_cast<bool>(is.read(reinterpret_cast<char*>(val.data.data()),
                                             static_cast<std::streamsize>(bytes)));
        };

        switch (type) {
            case GGUFType::UINT8:   if (!read_scalar(1)) return false; break;
            case GGUFType::INT8:    if (!read_scalar(1)) return false; break;
            case GGUFType::UINT16:  if (!read_scalar(2)) return false; break;
            case GGUFType::INT16:   if (!read_scalar(2)) return false; break;
            case GGUFType::UINT32:  if (!read_scalar(4)) return false; break;
            case GGUFType::INT32:   if (!read_scalar(4)) return false; break;
            case GGUFType::FLOAT32: if (!read_scalar(4)) return false; break;
            case GGUFType::BOOL:    if (!read_scalar(1)) return false; break;
            case GGUFType::UINT64:  if (!read_scalar(8)) return false; break;
            case GGUFType::INT64:   if (!read_scalar(8)) return false; break;
            case GGUFType::FLOAT64: if (!read_scalar(8)) return false; break;
            case GGUFType::STRING: {
                std::string s;
                if (!read_string(is, s)) return false;
                val.data = pack_string(s);
                break;
            }
            case GGUFType::ARRAY: {
                uint32_t elem_type_u32 = 0;
                uint64_t arr_count     = 0;
                if (!read_pod(is, elem_type_u32)) return false;
                if (!read_pod(is, arr_count))     return false;
                GGUFType elem_type = static_cast<GGUFType>(elem_type_u32);
                // store header
                auto push = [&](const void* d, size_t n) {
                    const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
                    val.data.insert(val.data.end(), p, p + n);
                };
                push(&elem_type_u32, 4);
                push(&arr_count, 8);
                // Forward-compat: fixed-size scalar element types outside the
                // natively understood set (e.g. UINT8/BOOL/UINT64 written by
                // newer exporters) are skipped as raw bytes instead of failing
                // the whole file. Warn once per key, not per element.
                const bool raw_skip = (elem_type != GGUFType::STRING &&
                                       elem_type != GGUFType::FLOAT32 &&
                                       elem_type != GGUFType::INT32 &&
                                       gguf_scalar_size(elem_type) != 0);
                if (raw_skip)
                    log_warn("gguf_format: skipping array '" + key + "' with element type " +
                             std::to_string(elem_type_u32));
                for (uint64_t j = 0; j < arr_count; ++j) {
                    switch (elem_type) {
                        case GGUFType::STRING: {
                            std::string s;
                            if (!read_string(is, s)) return false;
                            uint64_t len = s.size();
                            push(&len, 8);
                            push(s.data(), s.size());
                            if (key == "tokenizer.ggml.tokens") tokenizer_tokens_.push_back(s);
                            break;
                        }
                        case GGUFType::FLOAT32: {
                            float fv = 0;
                            if (!read_pod(is, fv)) return false;
                            push(&fv, 4);
                            if (key == "tokenizer.ggml.scores") tokenizer_scores_.push_back(fv);
                            break;
                        }
                        case GGUFType::INT32: {
                            int32_t iv = 0;
                            if (!read_pod(is, iv)) return false;
                            push(&iv, 4);
                            if (key == "tokenizer.ggml.token_type") tokenizer_token_types_.push_back(iv);
                            break;
                        }
                        default: {
                            // Fixed-size scalars: skip raw bytes (see raw_skip).
                            // Variable/nested types: size is unknowable, so the
                            // stream cannot be resynchronised — fail fast.
                            if (!raw_skip) {
                                log_warn("gguf_format: unsupported array element type " +
                                         std::to_string(elem_type_u32) + " for key '" + key + "'");
                                return false;
                            }
                            const size_t esz = gguf_scalar_size(elem_type);
                            uint8_t buf[8];
                            if (!is.read(reinterpret_cast<char*>(buf),
                                         static_cast<std::streamsize>(esz))) return false;
                            push(buf, esz);
                            break;
                        }
                    }
                }
                break;
            }
            default:
                // Unknown top-level type: its byte size is unknowable, so the
                // stream cannot be resynchronised. Failing here is correct —
                // fail-fast beats silent misalignment corrupting later fields.
                log_warn("gguf_format: unknown metadata type " + std::to_string(type_u32) +
                         " for key '" + key + "'");
                return false;
        }
        metadata_[key] = std::move(val);
    }
    return true;
}

bool GGUFReader::read_tensor_info(std::istream& is, uint64_t count) {
    tensor_infos_.clear();
    tensor_infos_.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        GGUFTensorInfo ti;
        if (!read_string(is, ti.name)) return false;
        uint32_t ndims = 0;
        if (!read_pod(is, ndims) || ndims > 8) return false;
        ti.dimensions.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d)
            if (!read_pod(is, ti.dimensions[d])) return false;
        uint32_t type_u32 = 0;
        if (!read_pod(is, type_u32)) return false;
        ti.type = static_cast<GGMLType>(type_u32);
        if (!read_pod(is, ti.offset)) return false;
        tensor_infos_.push_back(std::move(ti));
    }
    return true;
}

// ---- metadata accessors -------------------------------------------
std::string GGUFReader::get_string(const std::string& key, const std::string& def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end() || it->second.type != GGUFType::STRING) return def;
    const auto& d = it->second.data;
    if (d.size() < 8) return def;
    uint64_t len; memcpy(&len, d.data(), 8);
    if (8 + len > d.size()) return def;
    return std::string(reinterpret_cast<const char*>(d.data() + 8), static_cast<size_t>(len));
}

uint64_t GGUFReader::get_uint(const std::string& key, uint64_t def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& d = it->second.data;
    switch (it->second.type) {
        case GGUFType::UINT32: { uint32_t v; if (d.size() >= 4) { memcpy(&v, d.data(), 4); return v; } break; }
        case GGUFType::UINT64: { uint64_t v; if (d.size() >= 8) { memcpy(&v, d.data(), 8); return v; } break; }
        case GGUFType::INT32:  { int32_t  v; if (d.size() >= 4) { memcpy(&v, d.data(), 4); return static_cast<uint64_t>(v); } break; }
        case GGUFType::INT64:  { int64_t  v; if (d.size() >= 8) { memcpy(&v, d.data(), 8); return static_cast<uint64_t>(v); } break; }
        default: break;
    }
    return def;
}

int64_t GGUFReader::get_int(const std::string& key, int64_t def) const {
    return static_cast<int64_t>(get_uint(key, static_cast<uint64_t>(def)));
}

float GGUFReader::get_float(const std::string& key, float def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& d = it->second.data;
    if (it->second.type == GGUFType::FLOAT32 && d.size() >= 4) {
        float v; memcpy(&v, d.data(), 4); return v;
    }
    return def;
}

bool GGUFReader::get_bool(const std::string& key, bool def) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) return def;
    const auto& d = it->second.data;
    if (it->second.type == GGUFType::BOOL && !d.empty()) return d[0] != 0;
    return def;
}

ModelConfig GGUFReader::model_config() const {
    ModelConfig c;
    c.vocab_size        = static_cast<int>(get_uint("ghassan.vocab_size",        c.vocab_size));
    c.hidden_size       = static_cast<int>(get_uint("ghassan.embedding_length",  c.hidden_size));
    c.num_layers        = static_cast<int>(get_uint("ghassan.block_count",       c.num_layers));
    c.num_heads         = static_cast<int>(get_uint("ghassan.attention.head_count",    c.num_heads));
    c.num_kv_heads      = static_cast<int>(get_uint("ghassan.attention.head_count_kv", c.num_kv_heads));
    c.intermediate_size = static_cast<int>(get_uint("ghassan.feed_forward_length",     c.intermediate_size));
    c.use_moe           = get_bool("ghassan.use_moe", c.use_moe);
    c.num_experts       = static_cast<int>(get_uint("ghassan.moe.expert_count",        c.num_experts));
    c.moe_top_k         = static_cast<int>(get_uint("ghassan.moe.experts_used_count",  c.moe_top_k));
    c.moe_expert_dim    = static_cast<int>(get_uint("ghassan.moe.expert_feed_forward_length", c.moe_expert_dim));
    c.moe_shared        = get_bool("ghassan.moe.shared_expert", c.moe_shared);
    c.max_seq_len       = static_cast<int>(get_uint("ghassan.context_length",    c.max_seq_len));
    c.rope_theta        = get_float("ghassan.rope.freq_base", c.rope_theta);
    c.rms_eps           = get_float("ghassan.attention.layer_norm_rms_epsilon", c.rms_eps);
    c.tie_embeddings    = get_bool("ghassan.tie_word_embeddings", true);
    c.validate();
    return c;
}

bool GGUFReader::load_tokenizer(Tokenizer& tk) const {
    // GGUF embeds tokenizer differently; for now we try to reconstruct a minimal
    // gtok file from the tokenizer arrays if present.
    // If no tokens array: return false so the caller loads from a separate file.
    if (tokenizer_tokens_.empty()) return false;
    // We can't easily reconstruct a BPE tokenizer here without merge rules.
    // Return false and let the caller use the external .gtok file.
    (void)tk;
    return false;
}

const GGUFTensorInfo* GGUFReader::find_tensor(const std::string& name) const {
    for (const auto& ti : tensor_infos_)
        if (ti.name == name) return &ti;
    return nullptr;
}

static float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0)       bits = (sign << 31) | (mant << 13);
    else if (exp == 31) bits = (sign << 31) | 0x7F800000 | (mant << 13);
    else                bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float v; memcpy(&v, &bits, 4); return v;
}

Tensor GGUFReader::read_tensor_raw(const std::string& name) const {
    const GGUFTensorInfo* ti = find_tensor(name);
    GAI_CHECK(ti != nullptr, "gguf_format: tensor not found: " + name);

    // compute nbytes
    size_t nelems = 1;
    // dims are stored innermost-first in GGUF, just multiply them all
    for (uint64_t d : ti->dimensions) nelems *= static_cast<size_t>(d);
    size_t nbytes = ggml_nbytes(ti->type, nelems);

    // find data start: aligned to 32 after all tensor info
    // We need to seek. The offset in GGUFTensorInfo is relative to data section start.
    // data section start = align_up(end_of_tensor_info_block, 32).
    // We stored file offsets as relative to data section start.

    std::ifstream f(path_, std::ios::binary);
    GAI_CHECK(f.good(), "gguf_format: cannot reopen: " + path_);

    // Skip header to find data section start
    uint32_t magic = 0, version = 0;
    uint64_t tensor_count = 0, metadata_count = 0;
    read_pod(f, magic); read_pod(f, version);
    read_pod(f, tensor_count); read_pod(f, metadata_count);

    // Skip metadata
    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key; read_string(f, key);
        uint32_t type_u32 = 0; read_pod(f, type_u32);
        GGUFType type = static_cast<GGUFType>(type_u32);
        switch (type) {
            case GGUFType::UINT8: case GGUFType::INT8: case GGUFType::BOOL: { uint8_t v; read_pod(f, v); break; }
            case GGUFType::UINT16: case GGUFType::INT16: { uint16_t v; read_pod(f, v); break; }
            case GGUFType::UINT32: case GGUFType::INT32: case GGUFType::FLOAT32: { uint32_t v; read_pod(f, v); break; }
            case GGUFType::UINT64: case GGUFType::INT64: case GGUFType::FLOAT64: { uint64_t v; read_pod(f, v); break; }
            case GGUFType::STRING: { std::string s; read_string(f, s); break; }
            case GGUFType::ARRAY: {
                uint32_t et_u32 = 0; uint64_t ac = 0;
                read_pod(f, et_u32); read_pod(f, ac);
                GGUFType et = static_cast<GGUFType>(et_u32);
                for (uint64_t j = 0; j < ac; ++j) {
                    switch (et) {
                        case GGUFType::STRING: { std::string s; read_string(f, s); break; }
                        case GGUFType::FLOAT32: { float fv; read_pod(f, fv); break; }
                        case GGUFType::INT32:   { int32_t iv; read_pod(f, iv); break; }
                        default: break;
                    }
                }
                break;
            }
            default: break;
        }
    }

    // Skip tensor info entries
    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string tname; read_string(f, tname);
        uint32_t ndims = 0; read_pod(f, ndims);
        for (uint32_t d = 0; d < ndims; ++d) { uint64_t dim; read_pod(f, dim); }
        uint32_t t32; read_pod(f, t32);
        uint64_t off; read_pod(f, off);
    }

    // Now align to 32
    uint64_t cur = static_cast<uint64_t>(f.tellg());
    uint64_t data_start = align_up(cur, 32);
    uint64_t abs_offset = data_start + ti->offset;

    f.seekg(static_cast<std::streamoff>(abs_offset));
    GAI_CHECK(f.good(), "gguf_format: seek failed for tensor " + name);

    std::vector<uint8_t> raw(nbytes);
    GAI_CHECK(static_cast<bool>(f.read(reinterpret_cast<char*>(raw.data()),
                                       static_cast<std::streamsize>(nbytes))),
              "gguf_format: short read for tensor " + name);

    // Build a shape from dimensions (reverse back to row-major)
    std::vector<i64> shape;
    for (auto it = ti->dimensions.rbegin(); it != ti->dimensions.rend(); ++it)
        shape.push_back(static_cast<i64>(*it));

    Tensor out(shape, DType::F32, Device::CPU);
    GAI_CHECK(static_cast<size_t>(out.numel()) == nelems, "gguf shape mismatch for " + name);

    float* dst = out.f32();
    if (ti->type == GGMLType::F32) {
        memcpy(dst, raw.data(), nelems * 4);
    } else if (ti->type == GGMLType::F16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < nelems; ++i) dst[i] = fp16_to_f32(src[i]);
    } else if (ti->type == GGMLType::Q8_0) {
        size_t nblocks = (nelems + 31) / 32;
        for (size_t b = 0; b < nblocks; ++b) {
            const uint8_t* block = raw.data() + b * 34;
            float scale = fp16_to_f32(*reinterpret_cast<const uint16_t*>(block));
            for (size_t i = 0; i < 32 && b*32+i < nelems; ++i) {
                dst[b*32+i] = static_cast<int8_t>(block[2+i]) * scale;
            }
        }
    } else if (ti->type == GGMLType::Q4_0) {
        size_t nblocks = (nelems + 31) / 32;
        for (size_t b = 0; b < nblocks; ++b) {
            const uint8_t* block = raw.data() + b * 18;
            float scale = fp16_to_f32(*reinterpret_cast<const uint16_t*>(block));
            for (size_t i = 0; i < 16 && b*32+2*i < nelems; ++i) {
                uint8_t byte = block[2+i];
                int8_t q0 = static_cast<int8_t>((byte & 0xF) - (byte & 0x8 ? 16 : 0));
                int8_t q1 = static_cast<int8_t>((byte >> 4) - (byte & 0x80 ? 16 : 0));
                dst[b*32+2*i]   = q0 * scale;
                if (b*32+2*i+1 < nelems) dst[b*32+2*i+1] = q1 * scale;
            }
        }
    } else {
        GAI_FAIL("gguf_format: unsupported quantization type for reading: " +
                 std::to_string(static_cast<uint32_t>(ti->type)));
    }
    out.set_name(name);
    return out;
}

Tensor GGUFReader::read_tensor_f32(const std::string& name) const {
    return read_tensor_raw(name);  // always returns F32
}

// ================================================================ model io
bool load_model_from_gguf(const std::string& path, Model& model) {
    GGUFReader r;
    if (!r.open(path)) {
        log_error("gguf_format: cannot open: " + path);
        return false;
    }
    for (Parameter* p : model.parameters()) {
        const GGUFTensorInfo* ti = r.find_tensor(p->name);
        if (!ti) {
            log_error("gguf_format: missing tensor: " + p->name);
            return false;
        }
        Tensor t = r.read_tensor_f32(p->name);
        if (t.numel() != p->numel()) {
            log_error(strfmt("gguf tensor size mismatch %s: file=%lld model=%lld",
                             p->name.c_str(),
                             static_cast<long long>(t.numel()),
                             static_cast<long long>(p->numel())));
            return false;
        }
        p->w.copy_from(t);
    }
    return true;
}

ExportProfileGGUF gguf_profile_for(const std::string& name) {
    ExportProfileGGUF p;
    p.name = name;
    if (name == "fp32") {
        p.default_type = p.embedding_type = p.norm_type = p.output_type = GGMLType::F32;
    } else if (name == "fp16") {
        p.default_type   = GGMLType::F16;
        p.embedding_type = GGMLType::F16;
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::F16;
    } else if (name == "q8_k" || name == "int8") {
        p.default_type   = GGMLType::Q8_0;
        p.embedding_type = GGMLType::Q8_0;
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::F16;
    } else if (name == "q4_k" || name == "int4") {
        p.default_type   = GGMLType::Q4_0;
        p.embedding_type = GGMLType::Q8_0;   // embeddings stay at Q8
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::F16;
    } else {
        GAI_FAIL("gguf_format: unknown profile: " + name + " (fp32|fp16|q8_k|q4_k)");
    }
    return p;
}

void export_model_gguf(const std::string& path, Model& model,
                       const std::string& tokenizer_path,
                       const ExportProfileGGUF& profile,
                       const std::map<std::string, std::string>& extra_meta) {

    const ModelConfig& cfg = model.config();
    GGUFWriter w(path);

    // General metadata
    w.set_arch("ghassan");
    w.set_model_name("Ghassan v1 Flash");
    w.set_description("Ghassan v1 Flash - MoE language model for Moroccan Darija");

    // Model architecture metadata
    w.set_vocab_size(static_cast<uint32_t>(cfg.vocab_size));
    w.set_hidden_size(static_cast<uint32_t>(cfg.hidden_size));
    w.set_num_layers(static_cast<uint32_t>(cfg.num_layers));
    w.set_num_heads(static_cast<uint32_t>(cfg.num_heads));
    w.set_num_kv_heads(static_cast<uint32_t>(cfg.num_kv_heads));
    w.set_intermediate_size(static_cast<uint32_t>(cfg.intermediate_size));
    w.set_moe_config(cfg.use_moe, static_cast<uint32_t>(cfg.num_experts),
                     static_cast<uint32_t>(cfg.moe_top_k),
                     static_cast<uint32_t>(cfg.moe_expert_dim), cfg.moe_shared);
    w.set_max_seq_len(static_cast<uint32_t>(cfg.max_seq_len));
    w.set_rope_theta(cfg.rope_theta);
    w.set_rms_eps(cfg.rms_eps);
    w.set_tie_embeddings(cfg.tie_embeddings);
    w.set_norm_type("rmsnorm");
    w.set_ffn_type(cfg.use_moe ? "swiglu_moe" : "swiglu");
    w.set_attn_type("gqa_causal");

    // Tokenizer metadata
    w.set_tokenizer_model("bpe");
    w.set_tokenizer_bos_id(1);
    w.set_tokenizer_eos_id(2);
    w.set_tokenizer_unk_id(3);   // must match special::UNK (was 0/PAD: wrong)
    w.set_tokenizer_add_bos(true);
    w.set_tokenizer_add_eos(true);

    // Quantization info
    w.set_quantization_profile(profile.name);
    uint32_t file_type = 1;  // 0=all_f32, 1=mostly_f16
    if (profile.default_type == GGMLType::Q8_0) file_type = 7;
    if (profile.default_type == GGMLType::Q4_0) file_type = 2;
    w.set_file_type(file_type);

    // Parameter count
    w.add_custom_metadata("general.parameter_count",
                          static_cast<uint64_t>(model.num_parameters()));

    // Extra metadata
    for (const auto& [k, v] : extra_meta) w.add_custom_metadata(k, v);

    // Tensors
    u64 total_bytes = 0;
    for (Parameter* p : model.parameters()) {
        bool is_norm = p->name.find("norm") != std::string::npos;
        bool is_emb  = (p->name == "tok_embeddings" || p->name == "lm_head");
        GGMLType dt = is_norm ? profile.norm_type
                    : is_emb  ? profile.embedding_type
                              : profile.default_type;
        Tensor cpu = p->w.to(Device::CPU);
        if (cpu.dtype() != DType::F32) cpu = quant::dequantize(cpu, DType::F32);
        size_t est = ggml_nbytes(dt, static_cast<size_t>(p->numel()));
        total_bytes += static_cast<u64>(est);
        w.add_tensor(p->name, cpu, dt);
    }

    w.write();

    log_info(strfmt("[gguf] exported %s  profile=%s  tensors=%zu  weights=%s",
                    path.c_str(), profile.name.c_str(),
                    model.parameters().size(),
                    human_bytes(total_bytes).c_str()));

    (void)tokenizer_path;  // GGUF embeds tokenizer vocab; external .gtok is separate
}

} // namespace gai
