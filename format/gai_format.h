#pragma once

#include "core/tensor.h"
#include "core/mmap.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"
#include <map>

namespace gai {

// ---------------------------------------------------------------- .gai file
//  "GAI1" | u32 version | u32 header_bytes
//  header  : key=value text block (arch config, quant profile, metadata)
//  u64 tensor_count
//  directory: [ u32 name_len, name, u32 dtype, u32 ndim, i64 dims[ndim],
//               u64 offset, u64 nbytes ]
//  u64 vocab_bytes | vocabulary blob (a complete .gtok image)
//  tensor data, each entry 64-byte aligned
//
// The file is fully self-contained: model + tokenizer + config in one artifact.
constexpr u32 GAI_MAGIC   = 0x31494147u;   // "GAI1"
constexpr u32 GAI_VERSION = 1u;

struct TensorEntry {
    std::string      name;
    DType            dtype = DType::F32;
    std::vector<i64> dims;
    u64              offset = 0;
    u64              nbytes = 0;
};

class GaiWriter {
public:
    explicit GaiWriter(const std::string& path);

    void set_meta(const std::string& key, const std::string& value);
    void set_config(const ModelConfig& cfg);
    void set_tokenizer_blob(const std::string& gtok_path);
    void add_tensor(const std::string& name, const Tensor& t);   // CPU tensor
    void write();

private:
    std::string path_;
    std::map<std::string, std::string> meta_;
    std::vector<std::pair<std::string, Tensor>> tensors_;
    std::string tok_blob_;
};

class GaiReader {
public:
    bool open(const std::string& path);

    const std::map<std::string, std::string>& meta() const { return meta_; }
    std::string meta_str(const std::string& k, const std::string& def = "") const;
    i64         meta_int(const std::string& k, i64 def = 0) const;
    float       meta_f32(const std::string& k, float def = 0.0f) const;

    ModelConfig model_config() const;
    bool        has_tokenizer() const { return !tok_blob_.empty(); }
    // writes the embedded tokenizer to a temp path and loads it
    bool        load_tokenizer(Tokenizer& tk) const;

    const std::vector<TensorEntry>& tensors() const { return entries_; }
    const TensorEntry* find(const std::string& name) const;

    // Reads a tensor, dequantizing to f32 if needed.
    Tensor read_tensor_f32(const std::string& name) const;
    // Reads raw storage (keeps the on-disk dtype).
    Tensor read_tensor_raw(const std::string& name) const;

    // Memory-mapped weight access (roadmap item 5, weak-PC inference).
    // map_weights() maps path_ read-only; read_tensor_wrapped() then returns
    // a zero-copy read-only view of the entry's file bytes (bounds-checked).
    // Views hold shared ownership of the mapping, so they stay valid after
    // this reader is destroyed. Requires open() first.
    bool   map_weights();
    bool   weights_mapped() const { return mapping_ && mapping_->valid(); }
    Tensor read_tensor_wrapped(const std::string& name) const;

    const std::string& path() const { return path_; }
    u64 file_size() const { return file_size_; }

private:
    std::string path_;
    std::map<std::string, std::string> meta_;
    std::vector<TensorEntry> entries_;
    std::string tok_blob_;
    u64 file_size_ = 0;
    MappedFilePtr mapping_;
};

// Loads .gai weights into an allocated Model (dequantizing as required).
bool load_model_from_gai(const std::string& path, Model& model);

// Loads .gai weights with mmap-backed zero-copy views wherever the on-disk
// dtype is exactly F32 (no heap commit for those tensors). Other entries fall
// back to converting reads (counted in converted_out). The model stays fully
// usable for CPU inference; training on it is refused (read-only pages).
// Returns false (no partial model) when any tensor is missing or mismatched.
bool load_model_from_gai_mmap(const std::string& path, Model& model,
                              int* wrapped_out = nullptr,
                              int* converted_out = nullptr);

// Exports a Model (+ tokenizer) with the given per-tensor quantization profile.
struct ExportProfile {
    DType default_dtype   = DType::F16;
    DType embedding_dtype = DType::F16;   // embeddings are quantization-sensitive
    DType norm_dtype      = DType::F32;   // norms are tiny; always keep them exact
    std::string name      = "fp16";
};

ExportProfile profile_for(const std::string& name);   // fp32 | fp16 | int8 | int4

void export_model_gai(const std::string& path, Model& model,
                      const std::string& tokenizer_path,
                      const ExportProfile& profile,
                      const std::map<std::string, std::string>& extra_meta = {});

} // namespace gai
