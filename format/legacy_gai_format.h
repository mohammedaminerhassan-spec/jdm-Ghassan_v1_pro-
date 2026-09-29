#pragma once

#include "core/tensor.h"
#include "core/mmap.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"
#include <map>

namespace gai {

constexpr u32 GAI_MAGIC   = 0x31494147u;
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
    void add_tensor(const std::string& name, const Tensor& t);
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

    bool        load_tokenizer(Tokenizer& tk) const;

    const std::vector<TensorEntry>& tensors() const { return entries_; }
    const TensorEntry* find(const std::string& name) const;

    Tensor read_tensor_f32(const std::string& name) const;

    Tensor read_tensor_raw(const std::string& name) const;

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

bool load_model_from_gai(const std::string& path, Model& model);

bool load_model_from_gai_mmap(const std::string& path, Model& model,
                              int* wrapped_out = nullptr,
                              int* converted_out = nullptr);

struct ExportProfile {
    DType default_dtype   = DType::F16;
    DType embedding_dtype = DType::F16;
    DType norm_dtype      = DType::F32;
    std::string name      = "fp16";
};

ExportProfile profile_for(const std::string& name);

void export_model_gai(const std::string& path, Model& model,
                      const std::string& tokenizer_path,
                      const ExportProfile& profile,
                      const std::map<std::string, std::string>& extra_meta = {});

}
