#include "dataset/dedup.h"
#include "tokenizer/normalizer.h"
#include "core/unicode.h"

#include <fstream>
#include <algorithm>

namespace gai {

Deduplicator::Deduplicator() : Deduplicator(Config{}) {}
Deduplicator::Deduplicator(const Config& cfg) : cfg_(cfg) {
    GAI_CHECK(cfg_.num_hashes > 0, "num_hashes must be > 0");
    GAI_CHECK(cfg_.bands > 0 && cfg_.num_hashes % cfg_.bands == 0,
              "num_hashes must be divisible by bands");
    band_tables_.resize(static_cast<size_t>(cfg_.bands));
    hash_seeds_.resize(static_cast<size_t>(cfg_.num_hashes));
    u64 s = cfg_.seed;
    for (int i = 0; i < cfg_.num_hashes; ++i) {
        s = splitmix64(s);
        hash_seeds_[static_cast<size_t>(i)] = s | 1ull;
    }
}

static std::vector<std::string> words_of(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    size_t i = 0;
    while (i <= s.size()) {
        u32 cp = (i < s.size()) ? utf8_decode(s, i) : ' ';
        if (i >= s.size() || is_whitespace(cp)) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            if (i >= s.size()) break;
        } else {
            utf8_encode(cp, cur);
        }
    }
    return out;
}

std::vector<u64> Deduplicator::minhash(const std::string& canonical) const {
    std::vector<u64> sig(static_cast<size_t>(cfg_.num_hashes), ~0ull);
    std::vector<std::string> w = words_of(canonical);
    if (w.empty()) return sig;

    const int k = cfg_.shingle;
    const int n = static_cast<int>(w.size());
    const int count = std::max(1, n - k + 1);

    for (int i = 0; i < count; ++i) {
        std::string sh;
        for (int j = 0; j < k && i + j < n; ++j) {
            if (j) sh.push_back(' ');
            sh += w[static_cast<size_t>(i + j)];
        }
        u64 base = hash_string(sh);
        for (int h = 0; h < cfg_.num_hashes; ++h) {
            u64 v = splitmix64(base ^ hash_seeds_[static_cast<size_t>(h)]);
            if (v < sig[static_cast<size_t>(h)]) sig[static_cast<size_t>(h)] = v;
        }
    }
    return sig;
}

bool Deduplicator::add(const std::string& text) {
    ++seen_;
    std::string canon = Normalizer::canonical(text);
    if (canon.empty()) { ++exact_dups_; return false; }

    u64 h = hash_string(canon);

    if (!blocklist_.empty() && blocklist_.count(h)) {
        ++blocked_;
        return false;
    }

    if (cfg_.exact) {
        if (!exact_.insert(h).second) { ++exact_dups_; return false; }
    }

    if (cfg_.near) {
        std::vector<u64> sig = minhash(canon);
        const int rows = cfg_.num_hashes / cfg_.bands;

        // candidate lookup (all ids sharing any band bucket)
        std::vector<u32> candidates;
        for (int b = 0; b < cfg_.bands; ++b) {
            u64 bh = 1469598103934665603ull;
            for (int r = 0; r < rows; ++r) {
                bh ^= sig[static_cast<size_t>(b * rows + r)];
                bh *= 1099511628211ull;
            }
            bh = splitmix64(bh ^ static_cast<u64>(b));
            auto it = band_tables_[static_cast<size_t>(b)].find(bh);
            if (it != band_tables_[static_cast<size_t>(b)].end())
                for (u32 id : it->second) candidates.push_back(id);
        }

        if (!candidates.empty()) {
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
            for (u32 ci : candidates) {
                const auto& other = signatures_[ci];
                int match = 0;
                for (int i = 0; i < cfg_.num_hashes; ++i)
                    if (other[static_cast<size_t>(i)] == sig[static_cast<size_t>(i)]) ++match;
                double est = static_cast<double>(match) / static_cast<double>(cfg_.num_hashes);
                if (est >= cfg_.jaccard_threshold) { ++near_dups_; return false; }
            }
        }

        u32 idx = static_cast<u32>(signatures_.size());
        signatures_.push_back(sig);
        for (int b = 0; b < cfg_.bands; ++b) {
            u64 bh = 1469598103934665603ull;
            for (int r = 0; r < rows; ++r) {
                bh ^= sig[static_cast<size_t>(b * rows + r)];
                bh *= 1099511628211ull;
            }
            bh = splitmix64(bh ^ static_cast<u64>(b));
            band_tables_[static_cast<size_t>(b)][bh].push_back(idx);
        }
    }
    return true;
}

bool Deduplicator::is_duplicate(const std::string& text) const {
    std::string canon = Normalizer::canonical(text);
    if (canon.empty()) return true;
    u64 h = hash_string(canon);
    // Exact + eval-blocklist always apply.
    if (!blocklist_.empty() && blocklist_.count(h)) return true;
    if (exact_.count(h)) return true;
    // Near-dup estimate (read-only): same LSH lookup as add(), no insertion.
    if (cfg_.near && !signatures_.empty()) {
        std::vector<u64> sig = minhash(canon);
        const int rows = cfg_.num_hashes / cfg_.bands;
        std::vector<u32> candidates;
        for (int b = 0; b < cfg_.bands; ++b) {
            u64 bh = 1469598103934665603ull;
            for (int r = 0; r < rows; ++r) {
                bh ^= sig[static_cast<size_t>(b * rows + r)];
                bh *= 1099511628211ull;
            }
            bh = splitmix64(bh ^ static_cast<u64>(b));
            auto it = band_tables_[static_cast<size_t>(b)].find(bh);
            if (it != band_tables_[static_cast<size_t>(b)].end())
                for (u32 id : it->second) candidates.push_back(id);
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        for (u32 ci : candidates) {
            if (ci >= signatures_.size()) continue;
            const auto& other = signatures_[ci];
            int match = 0;
            for (int i = 0; i < cfg_.num_hashes; ++i)
                if (other[static_cast<size_t>(i)] == sig[static_cast<size_t>(i)]) ++match;
            double est = static_cast<double>(match) / static_cast<double>(cfg_.num_hashes);
            if (est >= cfg_.jaccard_threshold) return true;
        }
    }
    return false;
}

void Deduplicator::load_blocklist(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        log_warn("dedup: cannot open blocklist " + path);
        return;
    }
    std::string line;
    u64 n = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        blocklist_.insert(hash_string(Normalizer::canonical(line)));
        ++n;
    }
    log_info(strfmt("dedup: loaded %s eval-contamination hashes", human_count(n).c_str()));
}

std::string Deduplicator::summary() const {
    return strfmt("seen=%s unique=%s exact_dup=%s near_dup=%s blocked=%s dup_ratio=%.2f%%",
                  human_count(seen_).c_str(), human_count(unique()).c_str(),
                  human_count(exact_dups_).c_str(), human_count(near_dups_).c_str(),
                  human_count(blocked_).c_str(), dup_ratio() * 100.0);
}

} // namespace gai
