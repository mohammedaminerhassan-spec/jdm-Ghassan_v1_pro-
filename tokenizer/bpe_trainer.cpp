#include "tokenizer/bpe_trainer.h"
#include "core/rng.h"
#include "core/unicode.h"

#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <queue>

namespace gai {

void BpeTrainer::add_chunk_counts(const std::string& chunk, u64 count) {
    // DeepSeek trainer rule: raw-chunk injection bypassed normalize+pre_tokenize
    // that add_text applies, so train-vs-encode diverged (public trap).
    // Canonicalize here: normalize + pre-tokenize, distribute count to pieces.
    if (chunk.empty() || count == 0) return;
    Normalizer norm(cfg_.normalizer);
    std::string n = norm.normalize(chunk);
    bool any = false;
    for (const Chunk& c : pre_tokenize(n)) {
        if (c.text.empty()) continue;
        counts_[c.text] += count;
        total_ += count;
        any = true;
    }
    // Fallback: if pre-tokenizer yields nothing (e.g. whitespace-only),
    // keep the normalized chunk itself so counts are never silently dropped.
    if (!any && !n.empty()) {
        counts_[n] += count;
        total_ += count;
    }
}

void BpeTrainer::add_text(const std::string& text) {
    Normalizer norm(cfg_.normalizer);
    std::string n = norm.normalize(text);
    for (const Chunk& c : pre_tokenize(n)) {
        if (c.text.empty()) continue;
        counts_[c.text] += 1;
        ++total_;
    }
}

void BpeTrainer::add_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot open corpus file: " + path);
    std::string line;
    u64 lines = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        add_text(line);
        if (cfg_.verbose && (++lines % 500000 == 0)) {
            log_info(strfmt("  [bpe] %s lines, %s unique chunks",
                            human_count(lines).c_str(), human_count(counts_.size()).c_str()));
        }
    }
}

// ---------------------------------------------------------------- training
namespace {

struct Word {
    std::vector<i32> syms;    // current symbol ids
    std::vector<int> prev;    // doubly linked list over live positions
    std::vector<int> next;
    std::vector<u8>  alive;
    u64  freq = 0;
    int  head = 0;
};

struct PairKey {
    u64 v;
    bool operator==(const PairKey& o) const { return v == o.v; }
};
struct PairHash {
    size_t operator()(const PairKey& k) const { return static_cast<size_t>(splitmix64(k.v)); }
};

inline u64 mk_pair(i32 a, i32 b) {
    return (static_cast<u64>(static_cast<u32>(a)) << 32) | static_cast<u32>(b);
}

struct HeapItem {
    i64 count;
    u64 pair;
    bool operator<(const HeapItem& o) const {
        if (count != o.count) return count < o.count;      // max-heap on count
        return pair > o.pair;                              // deterministic tie-break
    }
};

} // namespace

Tokenizer BpeTrainer::train() {
    GAI_CHECK(!counts_.empty(), "BPE trainer has no data");
    const int n_special = special::COUNT;
    const int n_base    = 256;
    const int reserved  = n_special + n_base;
    GAI_CHECK(cfg_.vocab_size > reserved + 100, "vocab_size too small");

    // ---- vocabulary seeded with specials + all 256 bytes
    std::vector<std::string> vocab;
    vocab.reserve(static_cast<size_t>(cfg_.vocab_size));
    for (const auto& s : special_token_strings()) vocab.push_back(s);
    for (int b = 0; b < 256; ++b) vocab.push_back(std::string(1, static_cast<char>(b)));

    // ---- build word list
    std::vector<Word> words;
    words.reserve(counts_.size());
    for (const auto& [chunk, freq] : counts_) {
        if (freq < static_cast<u64>(cfg_.min_frequency)) continue;
        if (chunk.size() < 2) continue;   // single bytes need no merges
        Word w;
        w.freq = freq;
        w.syms.reserve(chunk.size());
        for (char ch : chunk) w.syms.push_back(n_special + static_cast<i32>(static_cast<u8>(ch)));
        size_t n = w.syms.size();
        w.prev.resize(n);
        w.next.resize(n);
        w.alive.assign(n, 1);
        for (size_t i = 0; i < n; ++i) {
            w.prev[i] = static_cast<int>(i) - 1;
            w.next[i] = (i + 1 < n) ? static_cast<int>(i) + 1 : -1;
        }
        words.push_back(std::move(w));
    }
    if (cfg_.verbose) {
        log_info(strfmt("[bpe] %s unique chunks (%s total), %s trainable words",
                        human_count(counts_.size()).c_str(), human_count(total_).c_str(),
                        human_count(words.size()).c_str()));
    }

    // ---- initial pair statistics + occurrence index
    std::unordered_map<PairKey, i64, PairHash> pair_count;
    std::unordered_map<PairKey, std::unordered_set<u32>, PairHash> pair_words;
    pair_count.reserve(1u << 20);
    pair_words.reserve(1u << 20);

    for (u32 wi = 0; wi < words.size(); ++wi) {
        const Word& w = words[wi];
        for (size_t i = 0; i + 1 < w.syms.size(); ++i) {
            u64 p = mk_pair(w.syms[i], w.syms[i + 1]);
            pair_count[PairKey{p}] += static_cast<i64>(w.freq);
            pair_words[PairKey{p}].insert(wi);
        }
    }

    std::priority_queue<HeapItem> heap;
    for (const auto& [k, c] : pair_count) heap.push(HeapItem{c, k.v});

    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(static_cast<size_t>(cfg_.vocab_size));

    const int target_merges = cfg_.vocab_size - reserved;
    int done = 0;

    while (done < target_merges && !heap.empty()) {
        HeapItem top = heap.top();
        heap.pop();
        auto pit = pair_count.find(PairKey{top.pair});
        if (pit == pair_count.end() || pit->second != top.count) continue;   // stale
        if (top.count < static_cast<i64>(cfg_.min_frequency)) break;

        i32 a = static_cast<i32>(top.pair >> 32);
        i32 b = static_cast<i32>(top.pair & 0xFFFFFFFFu);
        const std::string& sa = vocab[static_cast<size_t>(a)];
        const std::string& sb = vocab[static_cast<size_t>(b)];
        std::string merged = sa + sb;

        // DeepSeek multilingual rule: max_token_bytes counted BYTES penalizes
        // Arabic (~2B/codepoint) vs Latin (1B) — 32B = 32 Latin chars but ~16
        // Arabic chars, biasing fertility studies. Count codepoints instead
        // (same limit value now means 32 chars in any script).
        if (static_cast<int>(utf8_length(merged)) > cfg_.max_token_bytes) {
            pair_count.erase(pit);
            pair_words.erase(PairKey{top.pair});
            continue;
        }

        i32 new_id = static_cast<i32>(vocab.size());
        vocab.push_back(merged);
        merges.emplace_back(sa, sb);
        ++done;

        // ---- apply the merge to every word that contains this pair
        auto wsit = pair_words.find(PairKey{top.pair});
        std::unordered_set<u32> affected;
        if (wsit != pair_words.end()) affected = std::move(wsit->second);
        pair_words.erase(PairKey{top.pair});
        pair_count.erase(PairKey{top.pair});

        std::unordered_set<u64> touched;
        touched.reserve(affected.size() * 4);

        for (u32 wi : affected) {
            Word& w = words[wi];
            i64 f = static_cast<i64>(w.freq);
            int i = w.head;
            while (i >= 0) {
                int j = w.next[static_cast<size_t>(i)];
                if (j < 0) break;
                if (w.syms[static_cast<size_t>(i)] != a || w.syms[static_cast<size_t>(j)] != b) {
                    i = j;
                    continue;
                }
                int p = w.prev[static_cast<size_t>(i)];
                int q = w.next[static_cast<size_t>(j)];

                // remove the pairs that disappear
                if (p >= 0) {
                    u64 lp = mk_pair(w.syms[static_cast<size_t>(p)], a);
                    pair_count[PairKey{lp}] -= f;
                    touched.insert(lp);
                }
                if (q >= 0) {
                    u64 rp = mk_pair(b, w.syms[static_cast<size_t>(q)]);
                    pair_count[PairKey{rp}] -= f;
                    touched.insert(rp);
                }

                // merge j into i
                w.syms[static_cast<size_t>(i)] = new_id;
                w.alive[static_cast<size_t>(j)] = 0;
                w.next[static_cast<size_t>(i)] = q;
                if (q >= 0) w.prev[static_cast<size_t>(q)] = i;

                // add the new pairs
                if (p >= 0) {
                    u64 lp2 = mk_pair(w.syms[static_cast<size_t>(p)], new_id);
                    pair_count[PairKey{lp2}] += f;
                    pair_words[PairKey{lp2}].insert(wi);
                    touched.insert(lp2);
                }
                if (q >= 0) {
                    u64 rp2 = mk_pair(new_id, w.syms[static_cast<size_t>(q)]);
                    pair_count[PairKey{rp2}] += f;
                    pair_words[PairKey{rp2}].insert(wi);
                    touched.insert(rp2);
                }

                // continue scanning after the merged symbol (handles "aaa" correctly:
                // the merged node can pair with the following symbol on the next round)
                i = q;
            }
        }

        for (u64 p : touched) {
            auto it = pair_count.find(PairKey{p});
            if (it == pair_count.end()) continue;
            if (it->second <= 0) {
                pair_count.erase(it);
                pair_words.erase(PairKey{p});
                continue;
            }
            heap.push(HeapItem{it->second, p});
        }

        if (cfg_.verbose && (done % 2000 == 0)) {
            log_info(strfmt("  [bpe] merges %d/%d  vocab=%zu  last='%s' (count %lld)",
                            done, target_merges, vocab.size(), merged.c_str(),
                            static_cast<long long>(top.count)));
        }
    }

    // The model's embedding matrix has a fixed width, so the vocabulary must hit the
    // requested size exactly even when the corpus runs out of useful merges.
    if (static_cast<int>(vocab.size()) < cfg_.vocab_size) {
        int pad = cfg_.vocab_size - static_cast<int>(vocab.size());
        if (cfg_.verbose) {
            log_warn(strfmt("[bpe] corpus exhausted after %d merges, padding %d unused slots",
                            done, pad));
        }
        for (int i = 0; i < pad; ++i) vocab.push_back("<|unused" + std::to_string(i) + "|>");
    }

    if (cfg_.verbose) {
        log_info(strfmt("[bpe] finished: %d merges, final vocab %zu", done, vocab.size()));
    }

    Tokenizer tk;
    tk.build(vocab, merges, cfg_.normalizer);
    return tk;
}

} // namespace gai
