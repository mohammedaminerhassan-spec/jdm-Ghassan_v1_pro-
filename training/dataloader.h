#pragma once

#include "core/common.h"
#include "core/rng.h"
#include <map>
#include <string>
#include <vector>

namespace gai {

// ---------------------------------------------------------------- shard format
//  .gbin :  magic "GBIN" | u32 version | u32 dtype(0=u16,1=u32) | u32 flags
//           u64 n_tokens | u64 n_docs
//           token stream (n_tokens elements)
//           doc table: n_docs * u64 start offsets
//           loss mask (optional, flags&1): n_tokens bytes
//
// Documents are stored back-to-back, each terminated by EOS. The loss mask lets
// SFT shards supervise assistant spans only.
constexpr u32 GBIN_MAGIC   = 0x4E494247u;   // "GBIN"
constexpr u32 GBIN_VERSION = 1u;
constexpr u32 GBIN_FLAG_MASK = 1u;

struct ShardHeader {
    u32 magic   = GBIN_MAGIC;
    u32 version = GBIN_VERSION;
    u32 dtype   = 0;     // 0 = u16 (vocab <= 65535), 1 = u32
    u32 flags   = 0;
    u64 n_tokens = 0;
    u64 n_docs   = 0;
};

// Writer used by the dataset pipeline.
class ShardWriter {
public:
    ShardWriter(const std::string& path, int vocab_size, bool with_loss_mask);
    ~ShardWriter();

    void add_document(const std::vector<i32>& tokens, const std::vector<u8>* mask = nullptr);
    void close();

    u64 token_count() const { return tokens_.size(); }
    u64 doc_count()   const { return doc_offsets_.size(); }

private:
    std::string      path_;
    bool             u16_mode_;
    bool             with_mask_;
    std::vector<u32> tokens_;
    std::vector<u8>  mask_;
    std::vector<u64> doc_offsets_;
    bool             closed_ = false;
};

// Reader: loads a shard into memory (shards are sized so this is cheap).
// T4/Kaggle path: load_header() keeps only doc_offsets in RAM (~0.4MB per
// 50M-token shard) and reads token windows from disk on demand, so 20+ shards
// no longer pin GBs of CPU RAM. Use load() for small/val/test paths.
class Shard {
public:
    bool load(const std::string& path);
    // Header-only open: reads magic+header+doc table, defers tokens/mask to disk.
    bool load_header(const std::string& path);
    bool is_streaming() const { return streaming_; }

    u64 n_tokens() const { return streaming_ ? header_.n_tokens : tokens_.size(); }
    u64 n_docs()   const { return doc_offsets_.size(); }
    bool has_mask() const {
        if (streaming_) return (header_.flags & GBIN_FLAG_MASK) != 0;
        return !mask_.empty();
    }
    const std::string& path() const { return path_; }

    i32 token(u64 i) const { return static_cast<i32>(tokens_[static_cast<size_t>(i)]); }
    u8  mask(u64 i)  const { return mask_.empty() ? 1 : mask_[static_cast<size_t>(i)]; }
    u64 doc_end(u64 i) const;
    u64 doc_start_at(u64 doc_idx) const {
        return doc_idx < doc_offsets_.size() ? doc_offsets_[doc_idx] : 0;
    }
    u64 doc_index_of(u64 pos) const;
    // Streaming read of [start, start+len) tokens (+mask when present).
    // Returns false on IO error. Works in both modes (RAM path memcpys).
    // FIX: streaming path reuses a cached fd per thread (was open+close per
    // row: 256 opens/step -> GPU starvation that looked like a leak).
    bool read_window(u64 start, u64 len, std::vector<u32>& tok_out, std::vector<u8>& mask_out) const;

private:
    std::string      path_;
    std::vector<u32> tokens_;
    std::vector<u8>  mask_;
    std::vector<u64> doc_offsets_;
    ShardHeader      header_{};
    bool             streaming_ = false;
};

// ---------------------------------------------------------------- loader
struct BatchSpec {
    int batch_size = 8;
    int seq_len    = 1024;
};

struct Batch {
    std::vector<i32> ids;       // [B*T]
    std::vector<i32> targets;   // [B*T], -100 where ignored
    int  B = 0, T = 0;
    i64  tokens_supervised = 0;
};

// One domain group for weighted mixture sampling.
// Shards are named train_<domain>_*.gbin (pipeline --domain flag).
struct DomainGroup {
    std::string domain;
    std::vector<Shard> shards;
    u64 total_tokens = 0;
    double weight = 0.0;   // normalized to sum 1 over found groups
};

// Streams random windows out of a set of shards. Deterministic given a seed
// and fully checkpointable (position + rng state).
class DataLoader {
public:
    DataLoader() = default;

    // paths may be explicit files or a directory that is scanned for *.gbin
    bool open(const std::vector<std::string>& paths, BatchSpec spec, u64 seed);
    bool open_glob(const std::string& dir, const std::string& prefix, BatchSpec spec, u64 seed);
    // Weighted mixture over train_<domain>_*.gbin groups. mix maps
    // domain -> raw weight (need not sum to 1). Domains with no shards are
    // skipped with a warning. Returns false when no domain shards were found
    // (caller should fall back to open_glob legacy path).
    bool open_mix(const std::string& dir, const std::map<std::string, double>& mix,
                  BatchSpec spec, u64 seed);

    bool next(Batch& out);

    u64 total_tokens() const { return total_tokens_; }
    int num_shards() const;
    i64 batches_seen() const { return batches_; }
    bool using_mix() const { return use_mix_; }
    std::string mix_report() const;

    // checkpointing (v5: includes RNG Box-Muller spare for bit-exact resume)
    struct State {
        u64 rng[4] = {0, 0, 0, 0};
        i64 batches = 0;
        float rng_spare = 0.0f;
        i32 rng_has_spare = 0;
    };
    State get_state() const;
    void  set_state(const State& s);

private:
    std::vector<Shard> shards_;       // legacy uniform path (and val loader)
    std::vector<DomainGroup> groups_; // mix path (train loader only)
    bool use_mix_ = false;
    BatchSpec spec_;
    Rng  rng_;
    u64  total_tokens_ = 0;
    i64  batches_ = 0;

    // Pick one window [B,T] from a single shard (shared by both paths).
    void fill_from_shard(const Shard& sh, Batch& out, int b);
};

std::vector<std::string> list_shards(const std::string& dir, const std::string& prefix);

} // namespace gai
