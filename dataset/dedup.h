#pragma once

#include "core/common.h"
#include "core/rng.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>

namespace gai {

class Deduplicator {
public:
    struct Config {
        bool   exact = true;
        bool   near  = true;
        int    num_hashes = 128;
        int    bands = 16;
        int    shingle = 5;
        double jaccard_threshold = 0.85;
        u64    seed = 0xD1CE5EED;
    };

    Deduplicator();
    explicit Deduplicator(const Config& cfg);

    bool add(const std::string& text);

    bool is_duplicate(const std::string& text) const;

    u64 seen() const { return seen_; }
    u64 exact_dups() const { return exact_dups_; }
    u64 near_dups() const { return near_dups_; }
    u64 unique() const {

        const u64 non = exact_dups_ + near_dups_ + blocked_;
        return non <= seen_ ? seen_ - non : 0;
    }
    double dup_ratio() const { return seen_ ? double(exact_dups_ + near_dups_) / double(seen_) : 0.0; }

    std::string summary() const;

    void add_blocklist_hash(u64 h) { blocklist_.insert(h); }
    void load_blocklist(const std::string& path);
    u64  blocked() const { return blocked_; }

private:
    std::vector<u64> minhash(const std::string& canonical) const;

    Config cfg_;
    std::unordered_set<u64> exact_;

    std::vector<std::unordered_map<u64, std::vector<u32>>> band_tables_;
    std::vector<std::vector<u64>> signatures_;
    std::vector<bool> signature_blocked_;
    std::unordered_set<u64> blocklist_;
    std::vector<u64> hash_seeds_;
    u64 seen_ = 0, exact_dups_ = 0, near_dups_ = 0, blocked_ = 0;
};

}
