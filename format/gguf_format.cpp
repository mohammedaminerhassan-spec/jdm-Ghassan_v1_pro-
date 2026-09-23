// format/gguf_format.cpp
// Full GGUF v3 read/write implementation compatible with llama.cpp spec.
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#include "format/gguf_format.h"
#include "quantization/quantize.h"
#include "core/common.h"
#include "core/rng.h"
#include "core/dtype.h"

#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>

namespace fs = std::filesystem;

namespace gai {

// ================================================================ helpers
template<typename T>
static void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

static inline uint64_t align_up(uint64_t x, uint64_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

static uint64_t checked_stream_pos(std::ostream& os, const std::string& what) {
    if (!os.good()) GAI_FAIL("gguf_format: bad stream before " + what);
    const std::ostream::pos_type p = os.tellp();
    if (p == std::ostream::pos_type(-1)) GAI_FAIL("gguf_format: tell failed before " + what);
    return static_cast<uint64_t>(p);
}

// DeepSeek merge-escaping: pieces may contain ' ' (leading-space gluing), so
// "left right" joined with a bare space is ambiguous (" "+"hello" vs " "+" hello").
// Escape: '\' -> "\\", ' ' -> "\x20" (no literal spaces survive inside a piece,
// so the single separator ' ' is unambiguous). Unescape reverses it. The
// embedded .gtok blob stays the canonical tokenizer; merges are compat only.
static std::string escape_merge_piece(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 4);
    for (unsigned char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == ' ') o += "\\x20";
        else o.push_back(static_cast<char>(c));
    }
    return o;
}
static std::string unescape_merge_piece(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            if (s[i + 1] == '\\') { o.push_back('\\'); i += 2; continue; }
            if (i + 3 < s.size() && s[i + 1] == 'x' && s[i + 2] == '2' && s[i + 3] == '0') {
                o.push_back(' '); i += 4; continue;
            }
        }
        o.push_back(s[i++]);
    }
    return o;
}

// ================================================================ GGUFWriter
GGUFWriter::GGUFWriter(const std::string& path) : path_(path) {}

// ---- metadata helpers -----------------------------------------------
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
    // Spec: token IDs are UINT32 (old code wrote UINT64 -> llama.cpp parse fail).
    set_u32("tokenizer.ggml.bos_token_id", id);
}
void GGUFWriter::set_tokenizer_eos_id(uint32_t id) {
    set_u32("tokenizer.ggml.eos_token_id", id);
}
void GGUFWriter::set_tokenizer_unk_id(uint32_t id) {
    set_u32("tokenizer.ggml.unknown_token_id", id);
}
void GGUFWriter::set_tokenizer_pad_id(uint32_t id) {
    set_u32("tokenizer.ggml.padding_token_id", id);
}
void GGUFWriter::set_tokenizer_merges(const std::vector<std::string>& merges) {
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::STRING);
    uint64_t count     = static_cast<uint64_t>(merges.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    for (const auto& m : merges) {
        uint64_t len = m.size();
        push(&len, 8);
        push(m.data(), m.size());
    }
    metadata_["tokenizer.ggml.merges"] = std::move(v);
}
void GGUFWriter::set_tokenizer_pre(const std::string& pre) {
    add_custom_metadata("tokenizer.ggml.pre", pre);
}
void GGUFWriter::set_tokenizer_chat_template(const std::string& tmpl) {
    add_custom_metadata("tokenizer.chat_template", tmpl);
}
void GGUFWriter::set_u32(const std::string& key, uint32_t v) {
    GGUFMetadataValue m;
    m.type = GGUFType::UINT32;
    m.data.resize(4);
    memcpy(m.data.data(), &v, 4);
    metadata_[key] = std::move(m);
}
void GGUFWriter::set_general_type(const std::string& t) {
    add_custom_metadata("general.type", t);
}
void GGUFWriter::set_quantization_version(uint32_t v) {
    // Spec: general.quantization_version is UINT32 (GGML_QUANT_VERSION = 2).
    set_u32("general.quantization_version", v);
}
void GGUFWriter::set_size_label(const std::string& s) {
    add_custom_metadata("general.size_label", s);
}
void GGUFWriter::set_gtok_blob(const std::vector<uint8_t>& blob) {
    // Exact .gtok image: ARRAY[UINT8]. Unknown keys are skipped by external
    // tools; our reader uses it for bit-perfect tokenizer restore.
    GGUFMetadataValue v;
    v.type = GGUFType::ARRAY;
    uint32_t elem_type = static_cast<uint32_t>(GGUFType::UINT8);
    uint64_t count     = static_cast<uint64_t>(blob.size());
    auto push = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        v.data.insert(v.data.end(), p, p + n);
    };
    push(&elem_type, 4);
    push(&count, 8);
    if (!blob.empty()) push(blob.data(), blob.size());
    metadata_["ghassan.tokenizer.gtok"] = std::move(v);
}
void GGUFWriter::set_normalizer_config(uint32_t bits) {
    set_u32("ghassan.normalizer.config", bits);
}
void GGUFWriter::set_tokenizer_add_bos(bool add) {
    add_custom_metadata("tokenizer.ggml.add_bos_token", add);
}
void GGUFWriter::set_tokenizer_add_eos(bool add) {
    add_custom_metadata("tokenizer.ggml.add_eos_token", add);
}
void GGUFWriter::set_quantization_profile(const std::string& profile) {
    // Spec: general.quantization_version is UINT32 (2). The profile NAME
    // (fp16/q4_k/...) is preserved alongside in our own namespace.
    set_quantization_version(2);
    add_custom_metadata("ghassan.quantization.profile", profile);
}
void GGUFWriter::set_file_type(uint32_t type) {
    // Spec: general.file_type is UINT32 (old code wrote UINT64).
    set_u32("general.file_type", type);
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
            dst[i] = fp32_to_fp16(src[i]);
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
            uint8_t* block = out.data() + b * 34;
            if (!std::isfinite(amax)) {
                uint16_t zero_scale = fp32_to_fp16(0.0f);
                memcpy(block, &zero_scale, 2);
                std::memset(block + 2, 0, 32);
                continue;
            }
            float scale = amax / 127.f;
            float inv   = (scale != 0.f) ? (1.f / scale) : 0.f;
            // write f16 scale
            uint16_t sh = fp32_to_fp16(scale);
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
            uint8_t* block = out.data() + b * 18;
            if (!std::isfinite(amax)) {
                uint16_t zero_scale = fp32_to_fp16(0.0f);
                memcpy(block, &zero_scale, 2);
                std::memset(block + 2, 0, 16);
                continue;
            }
            float scale = amax / 7.f;
            float inv   = (scale != 0.f) ? (1.f / scale) : 0.f;
            uint16_t sh = fp32_to_fp16(scale);
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

        // First pass: tensor descriptors only (NO payload buffering).
        // T4-P2-34: the old code quantized every tensor into RAM up front,
        // so a 1B fp16 export held ~2GB of serialized payloads before
        // writing a byte. Sizes are exactly computable (ggml_nbytes), so
        // offsets are assigned first and each payload is quantized+written
        // streaming below — peak extra RAM is one tensor, not the model.
        struct TensorData {
            std::string            name;
            std::vector<uint64_t>  dims;
            GGMLType               type;
            uint64_t               nbytes = 0;
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
            td.nbytes = ggml_nbytes(dtype, static_cast<size_t>(t.numel()));
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

        // Tensor offsets below are relative to the aligned data section; the
        // reader independently captures that start after the info block.
        const uint64_t info_start = checked_stream_pos(f, "tensor info");

        // Assign offsets
        uint64_t running_offset = 0;
        for (auto& td : tdata) {
            td.offset = running_offset;
            running_offset += td.nbytes;
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
        const uint64_t info_end = checked_stream_pos(f, "tensor info");
        GAI_CHECK(info_end >= info_start && info_end - info_start == info_size,
                  "gguf_format: tensor info size mismatch");

        // Pad to alignment
        uint64_t cur = checked_stream_pos(f, "tensor data padding");
        uint64_t pad_to = align_up(cur, 32);
        for (uint64_t i = cur; i < pad_to; ++i) f.put('\0');

        // Write tensor data, streaming one tensor at a time: quantize the
        // CPU F32 source straight into a scratch payload, write it, pad,
        // release it. The scratch never holds more than the largest tensor.
        for (size_t ti = 0; ti < tdata.size(); ++ti) {
            const auto& td = tdata[ti];
            const Tensor& src = tensors_[ti].second.first;
            Tensor cpu = src.to(Device::CPU);
            if (cpu.dtype() != DType::F32) cpu = quant::dequantize(cpu, DType::F32);
            std::vector<uint8_t> payload = tensor_to_ggml_bytes(cpu, td.type);
            GAI_CHECK(payload.size() == td.nbytes, "gguf_format: payload size drift for " + td.name);
            uint64_t before = checked_stream_pos(f, "tensor write for " + td.name);
            f.write(reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size()));
            uint64_t after = checked_stream_pos(f, "tensor padding for " + td.name);
            uint64_t padded = align_up(after, 32);
            for (uint64_t i = after; i < padded; ++i) f.put('\0');
            (void)before;
        }

        GAI_CHECK(f.good(), "gguf_format: write failed");
        f.flush();
        // NOTE: stream closes here (RAII) BEFORE rename below.
    }

    // Atomic publish (same contract as checkpoints): POSIX rename overwrites
    // atomically, so never unlink first — a kill in remove+rename deleted the
    // previous export with no replacement.
    {
        std::error_code ec;
        fs::rename(tmp, path_, ec);
        if (ec) {
#if defined(_WIN32)
            fs::remove(path_, ec);
            ec.clear();
            fs::rename(tmp, path_, ec);
#endif
            GAI_CHECK(!ec, "gguf_format: cannot finalise file: " + ec.message());
        }
    }

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
    // Corrupt-file guard: counts bound loop iterations AND a reserve() below.
    // Real files carry hundreds of tensors / dozens of metadata entries.
    if (tensor_count > 100000 || metadata_count > 100000) {
        log_warn("gguf_format: absurd header counts (tensors/metadata); refusing file");
        return false;
    }

    if (!read_metadata(f, metadata_count)) return false;
    if (!read_tensor_info(f, tensor_count)) return false;

    std::error_code ec;
    file_size_ = static_cast<uint64_t>(fs::file_size(path, ec));
    if (ec || file_size_ == 0) return false;
    // Capture the data-section start ONCE from this checked parse (see the
    // data_start_ contract in the header). tellg()==-1 means the stream
    // already failed: casting it to u64 would wrap and silently misread.
    const auto end_info = f.tellg();
    if (end_info == std::istream::pos_type(-1)) return false;
    data_start_ = align_up(static_cast<uint64_t>(end_info), 32);
    if (data_start_ > file_size_) return false;
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
                // Exact embedded tokenizer image (our own key): raw bytes, no
                // per-element overhead. Parsed silently (no skip warning).
                if (key == "ghassan.tokenizer.gtok" && elem_type == GGUFType::UINT8) {
                    // Sanity cap: real .gtok images are KBs–MBs; a larger claim
                    // means a corrupt file (fail fast instead of resize-throw).
                    if (arr_count > (1ull << 30)) return false;
                    gtok_blob_.resize(static_cast<size_t>(arr_count));
                    if (arr_count > 0 &&
                        !is.read(reinterpret_cast<char*>(gtok_blob_.data()),
                                 static_cast<std::streamsize>(arr_count)))
                        return false;
                    if (!gtok_blob_.empty())
                        push(gtok_blob_.data(), gtok_blob_.size());
                    metadata_[key] = std::move(val);
                    continue;
                }
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
                            if (key == "tokenizer.ggml.merges") tokenizer_merges_.push_back(s);
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

std::vector<uint8_t> GGUFReader::get_blob(const std::string& key) const {
    std::vector<uint8_t> out;
    auto it = metadata_.find(key);
    if (it == metadata_.end() || it->second.type != GGUFType::ARRAY) return out;
    const auto& d = it->second.data;
    if (d.size() < 12) return out;
    uint32_t et = 0;
    uint64_t n = 0;
    memcpy(&et, d.data(), 4);
    memcpy(&n, d.data() + 4, 8);
    if (et != static_cast<uint32_t>(GGUFType::UINT8)) return out;
    if (12 + n > d.size()) return out;
    out.assign(d.begin() + 12, d.begin() + 12 + static_cast<size_t>(n));
    return out;
}

ModelConfig GGUFReader::model_config() const {
    // Native keys first; llama.cpp-standard keys as fallback so our engine
    // also loads --compat llama files (round-trip + single engine for both).
    const bool has_native = metadata_.count("ghassan.block_count") != 0;
    auto ukey = [&](const char* native, const char* llama) {
        return has_native ? std::string("ghassan.") + native : std::string("llama.") + llama;
    };
    ModelConfig c;
    c.vocab_size        = static_cast<int>(get_uint(ukey("vocab_size", "vocab_size"), static_cast<uint64_t>(c.vocab_size)));
    c.hidden_size       = static_cast<int>(get_uint(ukey("embedding_length", "embedding_length"), static_cast<uint64_t>(c.hidden_size)));
    c.num_layers        = static_cast<int>(get_uint(ukey("block_count", "block_count"), static_cast<uint64_t>(c.num_layers)));
    c.num_heads         = static_cast<int>(get_uint(ukey("attention.head_count", "attention.head_count"), static_cast<uint64_t>(c.num_heads)));
    c.num_kv_heads      = static_cast<int>(get_uint(ukey("attention.head_count_kv", "attention.head_count_kv"), static_cast<uint64_t>(c.num_kv_heads)));
    c.intermediate_size = static_cast<int>(get_uint(ukey("feed_forward_length", "feed_forward_length"), static_cast<uint64_t>(c.intermediate_size)));
    if (has_native) {
        c.use_moe           = get_bool("ghassan.use_moe", c.use_moe);
        c.num_experts       = static_cast<int>(get_uint("ghassan.moe.expert_count", static_cast<uint64_t>(c.num_experts)));
        c.moe_top_k         = static_cast<int>(get_uint("ghassan.moe.experts_used_count", static_cast<uint64_t>(c.moe_top_k)));
        c.moe_expert_dim    = static_cast<int>(get_uint("ghassan.moe.expert_feed_forward_length", static_cast<uint64_t>(c.moe_expert_dim)));
        c.moe_shared        = get_bool("ghassan.moe.shared_expert", c.moe_shared);
        c.moe_aux_scale     = get_float("ghassan.moe.aux_scale", c.moe_aux_scale);
        c.moe_jitter        = get_float("ghassan.moe.jitter", c.moe_jitter);
        c.max_seq_len       = static_cast<int>(get_uint("ghassan.context_length", static_cast<uint64_t>(c.max_seq_len)));
        c.rope_theta        = get_float("ghassan.rope.freq_base", c.rope_theta);
        c.rope_scale        = get_float("ghassan.rope.scale", c.rope_scale);
        c.rope_yarn_mscale  = get_float("ghassan.rope.yarn_mscale", c.rope_yarn_mscale);
        // PRO round-trip: حقول v1 Pro (default OFF = ملفات قديمة بلا هذه المفاتيح
        // ترجع defaults ولا تكسر التحميل).
        c.rope_yarn_low     = get_float("ghassan.rope.yarn_low", c.rope_yarn_low);
        c.rope_yarn_high    = get_float("ghassan.rope.yarn_high", c.rope_yarn_high);
        c.sliding_window    = static_cast<int>(get_uint("ghassan.sliding_window", static_cast<uint64_t>(c.sliding_window)));
        c.rope_type         = static_cast<int>(get_uint("ghassan.rope.type", static_cast<uint64_t>(c.rope_type)));
        c.moe_aux_free      = get_bool("ghassan.moe.aux_free", c.moe_aux_free);
        c.rms_eps           = get_float("ghassan.attention.layer_norm_rms_epsilon", c.rms_eps);
        c.use_qk_norm       = get_bool("ghassan.use_qk_norm", c.use_qk_norm);
        c.z_loss_scale      = get_float("ghassan.z_loss_scale", c.z_loss_scale);
        c.tie_embeddings    = get_bool("ghassan.tie_word_embeddings", true);
    } else {
        // Compat files are dense Llama-equivalents (export refuses MoE here).
        c.use_moe           = false;
        c.max_seq_len       = static_cast<int>(get_uint("llama.context_length", static_cast<uint64_t>(c.max_seq_len)));
        c.rope_theta        = get_float("llama.rope.freq_base", c.rope_theta);
        c.rms_eps           = get_float("llama.attention.layer_norm_rms_epsilon", c.rms_eps);
        c.tie_embeddings    = get_bool("llama.tie_word_embeddings", true);
    }
    c.validate();
    return c;
}

bool GGUFReader::has_tokenizer() const {
    return !gtok_blob_.empty() || !tokenizer_tokens_.empty();
}

bool GGUFReader::load_tokenizer(Tokenizer& tk) const {
    // Path 1 (exact): embedded .gtok image -> temp file -> load.
    // Bit-perfect: merges, normalizer, byte fallback all preserved.
    if (!gtok_blob_.empty()) {
        fs::path tmp = fs::temp_directory_path() /
            ("gguf_tok_" + std::to_string(fnv1a64(gtok_blob_.data(), gtok_blob_.size())) + ".gtok");
        {
            std::ofstream o(tmp, std::ios::binary);
            if (!o.good()) return false;
            o.write(reinterpret_cast<const char*>(gtok_blob_.data()),
                    static_cast<std::streamsize>(gtok_blob_.size()));
            if (!o.good()) return false;
        }
        bool ok = tk.load(tmp.string());
        std::error_code ec;
        fs::remove(tmp, ec);
        // NOTE: no rebuild here on purpose — tk.load() already restores the
        // exact normalizer (nflags in the .gtok header) and merge table.
        // Re-inverting merges could shift ranks when build() filtered pairs,
        // so the blob is used as-is for bit-perfect restore.
        return ok;
    }
    // Path 2 (standard arrays): rebuild BPE from tokens + merges.
    // New files escape pieces (see escape_merge_piece): exactly one literal
    // ' ' separator survives, split on FIRST space + unescape. Old files
    // (bare "left right") fall back to LAST-space best-effort.
    if (tokenizer_tokens_.empty()) return false;
    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(tokenizer_merges_.size());
    for (const auto& m : tokenizer_merges_) {
        size_t sp = m.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= m.size()) continue;
        std::string l = m.substr(0, sp), r = m.substr(sp + 1);
        // Escaped files contain "\\" or "\\x20": unescape both sides. Legacy
        // files contain neither escape nor extra spaces in a way we can prove;
        // unescape is a no-op on them unless they literally hold backslashes.
        bool escaped = (m.find('\\') != std::string::npos);
        if (escaped) {
            merges.emplace_back(unescape_merge_piece(l), unescape_merge_piece(r));
        } else {
            size_t rsp = m.rfind(' ');
            merges.emplace_back(m.substr(0, rsp), m.substr(rsp + 1));
        }
    }
    uint64_t bits = get_uint("ghassan.normalizer.config", 0);
    NormalizerConfig ncfg = (bits != 0) ? NormalizerConfig::unpack(static_cast<u32>(bits))
                                        : NormalizerConfig{};
    tk.build(tokenizer_tokens_, merges, ncfg);
    return tk.vocab_size() > 0;
}

const GGUFTensorInfo* GGUFReader::find_tensor(const std::string& name) const {
    for (const auto& ti : tensor_infos_)
        if (ti.name == name) return &ti;
    return nullptr;
}

std::string llama_tensor_name(const std::string& internal) {
    // Our names: "tok_embeddings" | "final_norm" | "lm_head" |
    // "layers.{L}.{attn_norm|wq|wk|wv|wo|ffn_norm|w_gate|w_up|w_down}".
    // Standard llama.cpp names ("token_embd.weight", "blk.{L}.attn_q.weight"...).
    if (internal == "tok_embeddings") return "token_embd.weight";
    if (internal == "final_norm")     return "output_norm.weight";
    if (internal == "lm_head")        return "output.weight";
    const std::string pre = "layers.";
    if (internal.compare(0, pre.size(), pre) != 0) return {};
    size_t dot = internal.find('.', pre.size());
    if (dot == std::string::npos) return {};
    const std::string layer = internal.substr(pre.size(), dot - pre.size());
    const std::string rest  = internal.substr(dot + 1);
    const std::string blk = "blk." + layer + ".";
    if (rest == "attn_norm") return blk + "attn_norm.weight";
    if (rest == "wq")        return blk + "attn_q.weight";
    if (rest == "wk")        return blk + "attn_k.weight";
    if (rest == "wv")        return blk + "attn_v.weight";
    if (rest == "wo")        return blk + "attn_output.weight";
    if (rest == "ffn_norm")  return blk + "ffn_norm.weight";
    if (rest == "w_gate")    return blk + "ffn_gate.weight";
    if (rest == "w_up")      return blk + "ffn_up.weight";
    if (rest == "w_down")    return blk + "ffn_down.weight";
    return {};
}

const GGUFTensorInfo* GGUFReader::find_weight(const std::string& internal) const {
    if (const GGUFTensorInfo* ti = find_tensor(internal)) return ti;
    const std::string alt = llama_tensor_name(internal);
    if (!alt.empty()) return find_tensor(alt);
    return nullptr;
}

bool llama_compat_ok(const ModelConfig& cfg, std::string* reason) {
    // llama.cpp has no Ghassan backend: only structurally-identical dense
    // nets can be relabeled. Anything else would load-but-compute-wrong.
    auto fail = [&](const std::string& r) {
        if (reason) *reason = r;
        return false;
    };
    if (cfg.use_moe) return fail("MoE has no llama.cpp dense equivalent; "
                                 "export native GGUF for ghassan-ai, --compat llama_moe for MoE Ollama (needs moe_shared=false), "
                                 "or train a dense (use_moe=false) variant for ollama");
    if (cfg.use_qk_norm) return fail("use_qk_norm has no llama-arch equivalent; disable it for --compat llama");
    if (cfg.rope_scale != 1.0f) return fail("rope_scale YaRN has no exact llama mapping here; use rope_scale=1 for --compat llama");
    // PRO fields: أي منها خارج default يعني ملف Pro لا يطابق llama arch.
    if (cfg.moe_aux_free) return fail("moe_aux_free (Pro aux-loss-free) has no llama mapping; use native GGUF");
    if (cfg.sliding_window != 0) return fail("sliding_window (Pro SWA) has no llama mapping here; use 0 for --compat llama");
    if (cfg.rope_type != 0) return fail("rope_type=neox (Pro) mismatches this interleaved kernel build; use 0 for --compat llama");
    return true;
}

bool llama_moe_compat_ok(const ModelConfig& cfg, std::string* reason) {
    auto fail = [&](const std::string& r) {
        if (reason) *reason = r;
        return false;
    };
    if (!cfg.use_moe) return fail("llama_moe needs use_moe=true (use --compat llama for dense)");
    if (cfg.moe_shared) return fail("llama_moe has no shared-expert equivalent in llama.cpp; "
                                    "train with moe_shared=false for Ollama, or use native GGUF (full quality)");
    if (cfg.use_qk_norm) return fail("use_qk_norm has no llama-arch equivalent; disable it for --compat llama_moe");
    if (cfg.rope_scale != 1.0f) return fail("rope_scale YaRN has no exact llama mapping here; use rope_scale=1");
    if (cfg.moe_aux_free) return fail("moe_aux_free has no llama mapping; use native GGUF");
    if (cfg.sliding_window != 0) return fail("sliding_window has no llama mapping here; use 0");
    if (cfg.rope_type != 0) return fail("rope_type=neox mismatches interleaved kernels; use 0");
    return true;
}

std::string llama_moe_tensor_name(const std::string& internal) {
    if (internal == "tok_embeddings") return "token_embd.weight";
    if (internal == "final_norm")     return "output_norm.weight";
    if (internal == "lm_head")        return "output.weight";
    const std::string pre = "layers.";
    if (internal.compare(0, pre.size(), pre) != 0) return {};
    size_t dot = internal.find('.', pre.size());
    if (dot == std::string::npos) return {};
    const std::string layer = internal.substr(pre.size(), dot - pre.size());
    const std::string rest  = internal.substr(dot + 1);
    const std::string blk = "blk." + layer + ".";
    if (rest == "attn_norm") return blk + "attn_norm.weight";
    if (rest == "wq")        return blk + "attn_q.weight";
    if (rest == "wk")        return blk + "attn_k.weight";
    if (rest == "wv")        return blk + "attn_v.weight";
    if (rest == "wo")        return blk + "attn_output.weight";
    if (rest == "ffn_norm")  return blk + "ffn_norm.weight";
    // Mixtral-style MoE (routed experts only, no shared).
    if (rest == "moe_router") return blk + "ffn_gate_inp.weight";
    if (rest == "moe_gate")   return blk + "ffn_gate_exps.weight";
    if (rest == "moe_up")     return blk + "ffn_up_exps.weight";
    if (rest == "moe_down")   return blk + "ffn_down_exps.weight";
    return {};
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

    // Element count with overflow guard: dims come from the file, and an
    // unchecked multiply wraps to a small nbytes (short-read confusion) or a
    // TB-scale vector allocation (host OOM = hard kill). 1T elements is far
    // above any real weight tensor.
    u64 nelems64 = 1;
    for (uint64_t d : ti->dimensions) {
        // dims are stored innermost-first in GGUF, just multiply them all
        GAI_CHECK(d <= (1ull << 40), "gguf_format: absurd tensor dim for " + name);
        GAI_CHECK(nelems64 <= ((1ull << 40) / (d == 0 ? 1 : d)),
                  "gguf_format: tensor element overflow for " + name);
        nelems64 *= d;
    }
    const size_t nelems = static_cast<size_t>(nelems64);
    size_t nbytes = ggml_nbytes(ti->type, nelems);

    // Seek straight to the tensor: data_start_ was captured in open() from
    // the checked parse (no header re-walk, no unchecked reads, O(1) opens).
    // Every bound is validated BEFORE touching the stream or allocating.
    GAI_CHECK(data_start_ <= file_size_, "gguf_format: bad data section for " + name);
    GAI_CHECK(ti->offset <= file_size_ - data_start_, "gguf_format: tensor offset outside file: " + name);
    const uint64_t abs_offset = data_start_ + ti->offset;
    GAI_CHECK(nbytes <= file_size_ - abs_offset, "gguf_format: tensor overruns file: " + name);

    std::ifstream f(path_, std::ios::binary);
    GAI_CHECK(f.good(), "gguf_format: cannot reopen: " + path_);
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
        // Native name first, standard llama.cpp name as fallback (compat files).
        const GGUFTensorInfo* ti = r.find_weight(p->name);
        if (!ti) {
            log_error("gguf_format: missing tensor: " + p->name);
            return false;
        }
        Tensor t = r.read_tensor_f32(ti->name);
        if (t.numel() != p->numel()) {
            log_error(strfmt("gguf tensor size mismatch %s: file=%lld model=%lld",
                             p->name.c_str(),
                             static_cast<long long>(t.numel()),
                             static_cast<long long>(p->numel())));
            return false;
        }
        p->w.copy_from(t);
    }
    // Aux-loss-free bias (optional, native exports only; missing = init 0).
    if (model.config().moe_aux_free) {
        model.ensure_moe_bias();
        for (int l = 0; l < model.config().num_layers; ++l) {
            std::string bn = "layers." + std::to_string(l) + ".moe_bias";
            const GGUFTensorInfo* ti = r.find_tensor(bn);
            if (!ti) continue;
            Tensor t = r.read_tensor_f32(bn);
            float* dst = model.moe_bias_ptr_mut(l);
            if (dst && t.numel() == model.config().num_experts)
                std::memcpy(dst, t.data_ptr(), sizeof(float) * static_cast<size_t>(t.numel()));
        }
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
    } else if (name == "q8_k" || name == "int8" || name == "q6_k" || name == "q5_k_m" || name == "q5_k_s") {
        // GGUF-only lightness: true K-quants (Q6_K/Q5_K) need a K writer (future GPU work).
        // Map to nearest implemented (Q8_0, 8.5 bits) loudly so Ollama-style names keep working.
        if (name == "q6_k" || name == "q5_k_m" || name == "q5_k_s")
            log_warn("[gguf] profile " + name + " mapped to Q8_0 (nearest implemented; true K-quant writer is roadmap)");
        p.name           = "q8_k";
        p.default_type   = GGMLType::Q8_0;
        p.embedding_type = GGMLType::Q8_0;
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::F16;
    } else if (name == "q4_0" || name == "int4" || name == "q4_k_m" || name == "q4_k_s" || name == "q4_k") {
        // Most-requested Ollama profile: Q4_K_M (4.8 bits) -> nearest Q4_0 (4.5 bits).
        // Loud alias keeps `quantize --profile q4_k_m` working everywhere.
        if (name == "q4_k_m" || name == "q4_k_s" || name == "q4_k")
            log_warn("[gguf] profile " + name + " mapped to Q4_0 (nearest implemented; true K-quant writer is roadmap)");
        p.name           = "q4_0";
        p.default_type   = GGMLType::Q4_0;
        p.embedding_type = GGMLType::Q8_0;   // embeddings stay at Q8
        p.norm_type      = GGMLType::F32;
        p.output_type    = GGMLType::F16;
    } else {
        GAI_FAIL("gguf_format: unknown profile: " + name + " (fp32|fp16|q8_k|q4_0|q4_k_m|q6_k)");
    }
    return p;
}

void export_model_gguf(const std::string& path, Model& model,
                       const std::string& tokenizer_path,
                       const ExportProfileGGUF& profile,
                       const std::map<std::string, std::string>& extra_meta,
                       const std::string& compat) {

    const ModelConfig& cfg = model.config();
    const bool want_llama = (compat == "llama");
    const bool want_moe = (compat == "llama_moe");
    if (compat != "native" && !want_llama && !want_moe)
        GAI_FAIL("gguf_format: unknown compat '" + compat + "' (native|llama|llama_moe)");
    if (want_llama) {
        std::string reason;
        GAI_CHECK(llama_compat_ok(cfg, &reason),
                  "gguf_format: --compat llama refused: " + reason);
    }
    if (want_moe) {
        std::string reason;
        GAI_CHECK(llama_moe_compat_ok(cfg, &reason),
                  "gguf_format: --compat llama_moe refused: " + reason);
        log_warn("[gguf] llama_moe is experimental Mixtral-style export; verify with `ghassan-ai logits` vs llama.cpp");
    }
    const std::string arch = (want_llama || want_moe) ? "llama" : "ghassan";

    // ---- tokenizer is REQUIRED: a GGUF without vocabulary is a broken
    // artifact (this is why old files needed a sidecar .gtok to run at all).
    GAI_CHECK(!tokenizer_path.empty(),
              "gguf_format: --tokenizer is required for GGUF export (vocab is embedded)");
    Tokenizer tok;
    GAI_CHECK(tok.load(tokenizer_path),
              "gguf_format: cannot load tokenizer: " + tokenizer_path);
    GAI_CHECK(tok.vocab_size() == cfg.vocab_size,
              strfmt("gguf_format: tokenizer/model vocab mismatch: %d vs %d",
                     tok.vocab_size(), cfg.vocab_size));
    // Exact .gtok image for bit-perfect restore in our engine.
    std::vector<uint8_t> gtok_blob;
    {
        std::ifstream tf(tokenizer_path, std::ios::binary);
        GAI_CHECK(tf.good(), "gguf_format: cannot reopen tokenizer: " + tokenizer_path);
        tf.seekg(0, std::ios::end);
        std::streamsize sz = tf.tellg();
        GAI_CHECK(sz > 0, "gguf_format: empty tokenizer file: " + tokenizer_path);
        tf.seekg(0, std::ios::beg);
        gtok_blob.resize(static_cast<size_t>(sz));
        GAI_CHECK(static_cast<bool>(tf.read(reinterpret_cast<char*>(gtok_blob.data()), sz)),
                  "gguf_format: cannot read tokenizer: " + tokenizer_path);
    }

    GGUFWriter w(path);

    // ---- general metadata (standard keys: parsed by llama.cpp/ollama/LM Studio)
    // P2-1 (audit #21): the name used to be hard-coded "Ghassan v1 Flash" for
    // EVERY config, so 1B-Ultra artifacts were mislabeled. Derive it from the
    // live config + parameter count; extra_meta["model_name"] still overrides.
    w.set_arch(arch);
    {
        std::string name = "Ghassan";
        {
            u64 n = static_cast<u64>(model.num_parameters());
            std::string scale = n >= 1000000000ull
                ? strfmt("%.1fB", static_cast<double>(n) / 1e9)
                : strfmt("%lluM", static_cast<unsigned long long>((n + 500000ull) / 1000000ull));
            std::string net = cfg.use_moe
                ? strfmt("MoE-%d", cfg.num_experts)
                : "dense";
            name += strfmt(" %s %s L%d", scale.c_str(), net.c_str(), cfg.num_layers);
        }
        auto itn = extra_meta.find("model_name");
        if (itn != extra_meta.end() && !itn->second.empty()) name = itn->second;
        w.set_model_name(name);
    }
    w.set_description("Ghassan MoE language model for Moroccan Darija");
    w.set_general_type("model");
    {
        u64 n = static_cast<u64>(model.num_parameters());
        std::string label = n >= 1000000000ull
            ? strfmt("%.1fB", static_cast<double>(n) / 1e9)
            : strfmt("%lluM", static_cast<unsigned long long>((n + 500000ull) / 1000000ull));
        w.set_size_label(label);
        w.add_custom_metadata("general.parameter_count", n);
    }

    // ---- architecture metadata (native ghassan.* or standard llama.*)
    auto akey = [&](const char* k) { return arch + "." + k; };
    auto au64 = [&](const char* k, uint64_t v) { w.add_custom_metadata(akey(k), v); };
    auto af32 = [&](const char* k, float v) { w.add_custom_metadata(akey(k), v); };
    au64("vocab_size", static_cast<uint64_t>(cfg.vocab_size));
    au64("embedding_length", static_cast<uint64_t>(cfg.hidden_size));
    au64("block_count", static_cast<uint64_t>(cfg.num_layers));
    au64("feed_forward_length", static_cast<uint64_t>(cfg.intermediate_size));
    au64("attention.head_count", static_cast<uint64_t>(cfg.num_heads));
    au64("attention.head_count_kv", static_cast<uint64_t>(cfg.num_kv_heads));
    af32("attention.layer_norm_rms_epsilon", cfg.rms_eps);
    au64("rope.dimension_count", static_cast<uint64_t>(cfg.head_dim()));
    af32("rope.freq_base", cfg.rope_theta);
    au64("context_length", static_cast<uint64_t>(cfg.max_seq_len));
    w.add_custom_metadata(akey("tie_word_embeddings"), cfg.tie_embeddings);
    if (want_moe) {
        // Mixtral-style MoE keys expected by llama.cpp/Ollama.
        au64("expert_count", static_cast<uint64_t>(cfg.num_experts));
        au64("expert_used_count", static_cast<uint64_t>(cfg.moe_top_k));
        // feed_forward_length above carries dense fallback; experts use their own dim.
        w.add_custom_metadata(akey("expert_feed_forward_length"),
                              static_cast<uint64_t>(cfg.moe_expert_dim));
    }
    if (!want_llama && !want_moe) {
        // Native-only extras (arch-specific MoE description; the shared keys
        // above already carry vocab/dims/heads/rope/ctx/tie).
        // DeepSeek round-trip rule: every knob that changes numerics must
        // persist (rope_scale/qk_norm/z/aux/jitter were silently dropped).
        w.set_moe_config(cfg.use_moe, static_cast<uint32_t>(cfg.num_experts),
                         static_cast<uint32_t>(cfg.moe_top_k),
                         static_cast<uint32_t>(cfg.moe_expert_dim), cfg.moe_shared);
        af32("rope.scale", cfg.rope_scale);
        af32("rope.yarn_mscale", cfg.rope_yarn_mscale);
        af32("rope.yarn_low", cfg.rope_yarn_low);
        af32("rope.yarn_high", cfg.rope_yarn_high);
        au64("sliding_window", static_cast<uint64_t>(cfg.sliding_window));
        au64("rope.type", static_cast<uint64_t>(cfg.rope_type));
        w.add_custom_metadata(akey("moe.aux_free"), cfg.moe_aux_free);
        af32("moe.aux_scale", cfg.moe_aux_scale);
        af32("moe.jitter", cfg.moe_jitter);
        af32("z_loss_scale", cfg.z_loss_scale);
        w.add_custom_metadata(akey("use_qk_norm"), cfg.use_qk_norm);
        w.set_norm_type("rmsnorm");
        w.set_ffn_type(cfg.use_moe ? "swiglu_moe" : "swiglu");
        w.set_attn_type("gqa_causal");
    }

    // ---- tokenizer metadata (standard keys + exact blob => self-contained)
    // llama.cpp token types: 1=normal 2=unknown 3=control 6=byte.
    w.set_tokenizer_model("gpt2");
    w.set_tokenizer_pre("gpt-2");
    w.set_tokenizer_tokens(tok.vocab_entries());
    {
        std::vector<float> scores(static_cast<size_t>(tok.vocab_size()), 0.0f);
        w.set_tokenizer_scores(scores);
        std::vector<int32_t> types;
        types.reserve(static_cast<size_t>(tok.vocab_size()));
        for (int i = 0; i < tok.vocab_size(); ++i) {
            int32_t t = 1;
            if (i == special::UNK) t = 2;
            else if (i < special::COUNT) t = 3;
            else if (i < special::COUNT + 256) t = 6;
            types.push_back(t);
        }
        w.set_tokenizer_token_types(types);
    }
    {
        std::vector<std::string> merges;
        for (const auto& pr : tok.merge_pairs_ordered())
            merges.push_back(escape_merge_piece(pr.first) + " " + escape_merge_piece(pr.second));
        w.set_tokenizer_merges(merges);
    }
    w.set_tokenizer_bos_id(static_cast<uint32_t>(special::BOS));
    w.set_tokenizer_eos_id(static_cast<uint32_t>(special::EOS));
    w.set_tokenizer_unk_id(static_cast<uint32_t>(special::UNK));
    w.set_tokenizer_pad_id(static_cast<uint32_t>(special::PAD));
    w.set_tokenizer_add_bos(true);
    w.set_tokenizer_add_eos(false);
    w.set_tokenizer_chat_template(
        "{%- for message in messages %}"
        "{%- if message['role'] == 'system' %}"
        "{{- '<s><|system|>' + message['content'] + '<|end|>' }}"
        "{%- elif message['role'] == 'user' %}"
        "{{- '<|user|>' + message['content'] + '<|end|>' }}"
        "{%- elif message['role'] == 'assistant' %}"
        "{{- '<|assistant|>' + message['content'] + '<|end|>' }}"
        "{%- endif %}{%- endfor %}"
        "{%- if add_generation_prompt %}{{- '<|assistant|>' }}{%- endif %}");
    w.set_gtok_blob(gtok_blob);
    w.set_normalizer_config(tok.normalizer_config().pack());
    // Audit #18 (tokenizer interop): the gpt2/gpt-2 labels above are the
    // closest standard GGUF pre-tokenizer tag, but training used a custom
    // Arabic/Darija-aware normalizer + pre-tokenizer. An external runtime
    // that ignores the .gtok blob would tokenize differently (silent
    // train/infer skew). Record the exact semantics alongside the standard
    // keys so any consumer can detect the mismatch instead of guessing.
    w.add_custom_metadata("ghassan.tokenizer.pre_tokenizer",
                          std::string("darija_byte_bpe_v1:leading-space-glue,nl-separator,byte-fallback"));
    w.add_custom_metadata("ghassan.tokenizer.canonical",
                          std::string("gtok-blob:tokenizer.ggml.tokens+merges reconstruct; "
                                      "authoritative=ghassan.gtok_blob"));
    w.add_custom_metadata("ghassan.tokenizer.specials", (uint64_t)special::COUNT);

    // Quantization info
    w.set_quantization_profile(profile.name);
    uint32_t file_type = 1;  // 0=all_f32, 1=mostly_f16
    if (profile.default_type == GGMLType::Q8_0) file_type = 7;
    if (profile.default_type == GGMLType::Q4_0) file_type = 2;
    w.set_file_type(file_type);

    // Extra metadata (model_name was already consumed into general.name above).
    for (const auto& [k, v] : extra_meta) {
        if (k == "model_name") continue;
        w.add_custom_metadata(k, v);
    }

    // Tensors (internal names natively; standard llama.cpp names for compat)
    u64 total_bytes = 0;
    for (Parameter* p : model.parameters()) {
        bool is_norm = p->name.find("norm") != std::string::npos;
        bool is_emb  = (p->name == "tok_embeddings" || p->name == "lm_head");
        GGMLType dt = is_norm ? profile.norm_type
                    : is_emb  ? profile.embedding_type
                              : profile.default_type;
        std::string wname = p->name;
        Tensor cpu = p->w.to(Device::CPU);
        if (cpu.dtype() != DType::F32) cpu = quant::dequantize(cpu, DType::F32);
        if (want_llama) {
            wname = llama_tensor_name(p->name);
            GAI_CHECK(!wname.empty(),
                      "gguf_format: no llama.cpp name for tensor: " + p->name);
        } else if (want_moe) {
            wname = llama_moe_tensor_name(p->name);
            GAI_CHECK(!wname.empty(),
                      "gguf_format: no llama_moe name for tensor: " + p->name +
                      " (shared experts / QK-norm have no Ollama equivalent; use native GGUF)");
            // Router [ne,d] -> llama ffn_gate_inp [d,ne]: transpose row-major.
            if (p->name.find("moe_router") != std::string::npos && cpu.shape().size() == 2) {
                i64 ne = cpu.shape()[0], d = cpu.shape()[1];
                Tensor t({d, ne}, DType::F32, Device::CPU);
                const float* s = cpu.f32();
                float* o = t.f32();
                for (i64 r = 0; r < ne; ++r)
                    for (i64 c = 0; c < d; ++c)
                        o[static_cast<size_t>(c) * static_cast<size_t>(ne) + static_cast<size_t>(r)] =
                            s[static_cast<size_t>(r) * static_cast<size_t>(d) + static_cast<size_t>(c)];
                t.set_name(cpu.name());
                cpu = std::move(t);
            }
        }
        size_t est = ggml_nbytes(dt, static_cast<size_t>(p->numel()));
        total_bytes += static_cast<u64>(est);
        w.add_tensor(wname, cpu, dt);
        // FIX P2 (tied embeddings): parameters() holds ONE tensor when tied,
        // so llama compat exported no output.weight and external llama.cpp
        // failed to find it despite tie_word_embeddings metadata. Duplicate
        // the embedding bytes as output.weight for compat exports (same
        // values, shared content) instead of failing downstream.
        if ((want_llama || want_moe) && p->name == "tok_embeddings" && cfg.tie_embeddings) {
            size_t est2 = ggml_nbytes(dt, static_cast<size_t>(p->numel()));
            total_bytes += static_cast<u64>(est2);
            w.add_tensor("output.weight", cpu, dt);
        }
    }

    // Aux-loss-free bias (native only, F32 exact): layers.{L}.moe_bias [ne].
    // No llama.cpp equivalent — llama/llama_moe exports skip it (they refuse
    // aux_free configs above anyway). Missing on load = init 0 (re-converges).
    if (!want_llama && !want_moe && cfg.moe_aux_free) {
        const auto& all = model.moe_bias_all();
        for (int l = 0; l < cfg.num_layers && l < static_cast<int>(all.size()); ++l) {
            const auto& b = all[static_cast<size_t>(l)];
            if (static_cast<int>(b.size()) != cfg.num_experts) continue;
            Tensor t({(i64)b.size()}, DType::F32, Device::CPU);
            std::memcpy(t.data_ptr(), b.data(), sizeof(float) * b.size());
            t.set_name("layers." + std::to_string(l) + ".moe_bias");
            w.add_tensor(t.name(), t, GGMLType::F32);
        }
    }

    w.write();

    log_info(strfmt("[gguf] exported %s  profile=%s compat=%s self_contained=yes  tensors=%zu  weights=%s",
                    path.c_str(), profile.name.c_str(), compat.c_str(),
                    model.parameters().size(),
                    human_bytes(total_bytes).c_str()));
}

} // namespace gai
