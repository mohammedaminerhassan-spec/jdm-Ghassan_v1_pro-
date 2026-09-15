#pragma once

#include "core/tensor.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"
#include <cstdint>
#include <istream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace gai {

// ================================================================ GGUF Format
// Based on llama.cpp GGUF specification
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

constexpr uint32_t GGUF_MAGIC = 0x46554747u;  // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

enum class GGUFType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// Quantization types (matching GGML)
enum class GGMLType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
};

struct GGUFMetadataValue {
    GGUFType type;
    std::vector<uint8_t> data;  // serialized value
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dimensions;
    GGMLType type = GGMLType::F16;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

class GGUFWriter {
public:
    explicit GGUFWriter(const std::string& path);

    void set_arch(const std::string& arch);           // e.g., "ghassan"
    void set_model_name(const std::string& name);     // e.g., "Ghassan AI 200M"
    void set_vocab_size(uint32_t v);
    void set_hidden_size(uint32_t h);
    void set_num_layers(uint32_t l);
    void set_num_heads(uint32_t h);
    void set_num_kv_heads(uint32_t kv);
    void set_intermediate_size(uint32_t f);
    void set_moe_config(bool use_moe, uint32_t num_experts, uint32_t top_k,
                        uint32_t expert_dim, bool shared);
    void set_max_seq_len(uint32_t ctx);
    void set_rope_theta(float theta);
    void set_rope_scaling(const std::map<std::string, std::string>& scaling);
    void set_rms_eps(float eps);
    void set_tie_embeddings(bool tie);
    void set_norm_type(const std::string& type);      // "rmsnorm"
    void set_ffn_type(const std::string& type);       // "swiglu"
    void set_attn_type(const std::string& type);      // "gqa_causal"
    void set_tokenizer_model(const std::string& model); // "bpe"
    void set_tokenizer_tokens(const std::vector<std::string>& tokens);
    void set_tokenizer_scores(const std::vector<float>& scores);
    void set_tokenizer_token_types(const std::vector<int32_t>& types);
    void set_tokenizer_bos_id(uint32_t id);
    void set_tokenizer_eos_id(uint32_t id);
    void set_tokenizer_unk_id(uint32_t id);
    void set_tokenizer_pad_id(uint32_t id);
    void set_tokenizer_add_bos(bool add);
    void set_tokenizer_add_eos(bool add);
    void set_quantization_profile(const std::string& profile); // "fp16", "q4_k", etc.
    void set_file_type(uint32_t type);                // 1=fp16, 2=q4_0, etc.
    void set_description(const std::string& desc);
    void add_custom_metadata(const std::string& key, const std::string& value);
    void add_custom_metadata(const std::string& key, uint64_t value);
    void add_custom_metadata(const std::string& key, float value);
    void add_custom_metadata(const std::string& key, bool value);

    void add_tensor(const std::string& name, const Tensor& t, GGMLType dtype = GGMLType::F16);
    void write();

private:
    std::string path_;
    std::map<std::string, GGUFMetadataValue> metadata_;
    std::vector<std::pair<std::string, std::pair<Tensor, GGMLType>>> tensors_;
    std::vector<GGUFTensorInfo> tensor_infos_;

    void write_metadata(std::ostream& os);
    void write_tensor_info(std::ostream& os);
    void write_tensor_data(std::ostream& os);
    static uint64_t align_up(uint64_t x, uint64_t alignment = 32);
    template<typename T> static void write_pod(std::ostream& os, const T& v);
    static void write_string(std::ostream& os, const std::string& s);
    static void write_array(std::ostream& os, const std::vector<uint8_t>& data);
};

class GGUFReader {
public:
    bool open(const std::string& path);

    const std::map<std::string, GGUFMetadataValue>& metadata() const { return metadata_; }
    std::string get_string(const std::string& key, const std::string& def = "") const;
    uint64_t get_uint(const std::string& key, uint64_t def = 0) const;
    int64_t get_int(const std::string& key, int64_t def = 0) const;
    float get_float(const std::string& key, float def = 0.0f) const;
    bool get_bool(const std::string& key, bool def = false) const;

    ModelConfig model_config() const;
    bool has_tokenizer() const { return !tokenizer_tokens_.empty(); }
    bool load_tokenizer(Tokenizer& tk) const;

    const std::vector<GGUFTensorInfo>& tensors() const { return tensor_infos_; }
    const GGUFTensorInfo* find_tensor(const std::string& name) const;

    Tensor read_tensor_f32(const std::string& name) const;
    Tensor read_tensor_raw(const std::string& name) const;

    const std::string& path() const { return path_; }
    uint64_t file_size() const { return file_size_; }

private:
    std::string path_;
    std::map<std::string, GGUFMetadataValue> metadata_;
    std::vector<GGUFTensorInfo> tensor_infos_;
    std::vector<std::string> tokenizer_tokens_;
    std::vector<float> tokenizer_scores_;
    std::vector<int32_t> tokenizer_token_types_;
    uint64_t file_size_ = 0;

    bool read_metadata(std::istream& is, uint64_t metadata_count);
    bool read_tensor_info(std::istream& is, uint64_t tensor_count);
    static bool read_string(std::istream& is, std::string& out);
    static bool read_array(std::istream& is, std::vector<uint8_t>& out, uint64_t count);
    template<typename T> static bool read_pod(std::istream& is, T& v);
};

bool load_model_from_gguf(const std::string& path, Model& model);

struct ExportProfileGGUF {
    GGMLType default_type = GGMLType::F16;
    GGMLType embedding_type = GGMLType::F16;
    GGMLType norm_type = GGMLType::F32;
    GGMLType output_type = GGMLType::F16;
    std::string name = "fp16";
};

ExportProfileGGUF gguf_profile_for(const std::string& name);  // fp32 | fp16 | q4_k | q5_k | q8_k

void export_model_gguf(const std::string& path, Model& model,
                      const std::string& tokenizer_path,
                      const ExportProfileGGUF& profile,
                      const std::map<std::string, std::string>& extra_meta = {});

// ================================================================
// Template definitions (must be in header for instantiation)

inline uint64_t GGUFWriter::align_up(uint64_t x, uint64_t alignment) {
    return (x + alignment - 1) & ~(alignment - 1);
}

template<typename T>
void GGUFWriter::write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

inline void GGUFWriter::write_string(std::ostream& os, const std::string& s) {
    uint64_t len = static_cast<uint64_t>(s.size());
    os.write(reinterpret_cast<const char*>(&len), 8);
    os.write(s.data(), static_cast<std::streamsize>(len));
}

inline void GGUFWriter::write_array(std::ostream& os, const std::vector<uint8_t>& data) {
    uint64_t len = static_cast<uint64_t>(data.size());
    os.write(reinterpret_cast<const char*>(&len), 8);
    os.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(len));
}

template<typename T>
bool GGUFReader::read_pod(std::istream& is, T& v) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

inline bool GGUFReader::read_string(std::istream& is, std::string& out) {
    uint64_t len = 0;
    if (!is.read(reinterpret_cast<char*>(&len), 8)) return false;
    out.resize(static_cast<size_t>(len));
    return static_cast<bool>(is.read(out.data(), static_cast<std::streamsize>(len)));
}

inline bool GGUFReader::read_array(std::istream& is, std::vector<uint8_t>& out, uint64_t count) {
    out.resize(static_cast<size_t>(count));
    return static_cast<bool>(is.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count)));
}

} // namespace gai