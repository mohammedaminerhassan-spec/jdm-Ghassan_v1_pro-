#pragma once

#include "core/common.h"
#include "core/rng.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>

namespace gai {

// Exact dedup on canonicalized text + near-dup via MinHash/LSH over word 5-grams.
// Both are streaming: memory is O(unique documents), not O(corpus bytes).
class Deduplicator {
public:
    struct Config {
        bool   exact = true;
        bool   near  = true;
        int    num_hashes = 128;
        int    bands = 16;          // rows per band = num_hashes / bands
        int    shingle = 5;
        double jaccard_threshold = 0.85;
        u64    seed = 0xD1CE5EED;
    };

    Deduplicator();
    explicit Deduplicator(const Config& cfg);

    // returns true if the text is new (and registers it), false if duplicate
    bool add(const std::string& text);

    // check only, no registration
    bool is_duplicate(const std::string& text) const;

    u64 seen() const { return seen_; }
    u64 exact_dups() const { return exact_dups_; }
    u64 near_dups() const { return near_dups_; }
    u64 unique() const { return seen_ - exact_dups_ - near_dups_; }
    double dup_ratio() const { return seen_ ? double(exact_dups_ + near_dups_) / double(seen_) : 0.0; }

    std::string summary() const;

    // Eval contamination: documents whose canonical hash is in this set are dropped.
    void add_blocklist_hash(u64 h) { blocklist_.insert(h); }
    void load_blocklist(const std::string& path);   // one text per line
    u64  blocked() const { return blocked_; }

private:
    std::vector<u64> minhash(const std::string& canonical) const;

    Config cfg_;
    std::unordered_set<u64> exact_;
    // DeepSeek LSH rule: one slot per bucket loses collisions (second writer
    // overwrites/ignored -> order-dependent false negatives). Keep ALL doc ids.
    std::vector<std::unordered_map<u64, std::vector<u32>>> band_tables_;
    std::vector<std::vector<u64>> signatures_;
    std::unordered_set<u64> blocklist_;
    std::vector<u64> hash_seeds_;
    u64 seen_ = 0, exact_dups_ = 0, near_dups_ = 0, blocked_ = 0;
};

} // namespace gai
