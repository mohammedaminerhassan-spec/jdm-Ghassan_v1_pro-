#include "tokenizer/tokenizer.h"

#include <fstream>
#include <queue>
#include <algorithm>
#include <cstring>
#include <limits>

namespace gai {

// ---------------------------------------------------------------- specials
const std::vector<std::string>& special_token_strings() {
    static const std::vector<std::string> s = {
        "<pad>", "<s>", "</s>", "<unk>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<|nl|>", "<|fr|>", "<|ar|>", "<|darija|>", "<|latin|>",
        "<|tool|>", "<|mask|>", "<|reserved|>"
    };
    return s;
}

static constexpr u32 GTOK_MAGIC   = 0x4B4F5447u;  // "GTOK"
static constexpr u32 GTOK_VERSION = 1u;

// ---------------------------------------------------------------- build
void Tokenizer::init_empty(const NormalizerConfig& ncfg) {
    vocab_.clear();
    token_ids_.clear();
    merges_.clear();
    norm_.set_config(ncfg);

    for (const auto& s : special_token_strings()) vocab_.push_back(s);
    for (int b = 0; b < 256; ++b) vocab_.push_back(std::string(1, static_cast<char>(b)));
    finalize_index();
}

void Tokenizer::build(const std::vector<std::string>& vocab,
                      const std::vector<std::pair<std::string, std::string>>& merges,
                      const NormalizerConfig& ncfg) {
    vocab_ = vocab;
    norm_.set_config(ncfg);
    finalize_index();

    merges_.clear();
    merges_.reserve(merges.size() * 2);
    for (size_t r = 0; r < merges.size(); ++r) {
        auto itl = token_ids_.find(merges[r].first);
        auto itr = token_ids_.find(merges[r].second);
        if (itl == token_ids_.end() || itr == token_ids_.end()) continue;
        auto itm = token_ids_.find(merges[r].first + merges[r].second);
        if (itm == token_ids_.end()) continue;
        u64 key = (static_cast<u64>(static_cast<u32>(itl->second)) << 32) |
                   static_cast<u32>(itr->second);
        merges_[key] = {static_cast<i32>(r), itm->second};
    }
    cache_.clear();
}

void Tokenizer::finalize_index() {
    token_ids_.clear();
    token_ids_.reserve(vocab_.size() * 2);
    for (size_t i = 0; i < vocab_.size(); ++i) token_ids_[vocab_[i]] = static_cast<i32>(i);
}

std::vector<std::pair<std::string, std::string>> Tokenizer::merge_pairs_ordered() const {
    // Invert merges_ (keyed by (left<<32|right) -> (rank, merged)) into a
    // rank-ordered list of string pairs. Ranks may have gaps (build() skips
    // pairs whose strings are missing from vocab), so size by max rank.
    i32 max_rank = -1;
    for (const auto& [key, val] : merges_) {
        if (val.first > max_rank) max_rank = val.first;
    }
    std::vector<std::pair<std::string, std::string>> out;
    if (max_rank < 0) return out;
    std::vector<char> seen(static_cast<size_t>(max_rank) + 1, 0);
    std::vector<std::pair<std::string, std::string>> by_rank(static_cast<size_t>(max_rank) + 1);
    for (const auto& [key, val] : merges_) {
        u32 l = static_cast<u32>(key >> 32);
        u32 r = static_cast<u32>(key & 0xFFFFFFFFu);
        if (l >= vocab_.size() || r >= vocab_.size()) continue;
        by_rank[static_cast<size_t>(val.first)] = {vocab_[l], vocab_[r]};
        seen[static_cast<size_t>(val.first)] = 1;
    }
    out.reserve(merges_.size());
    for (size_t i = 0; i < by_rank.size(); ++i) {
        if (seen[i]) out.push_back(std::move(by_rank[i]));
    }
    return out;
}

// ---------------------------------------------------------------- BPE core
namespace {
struct Node {
    i32 id;
    int prev;
    int next;
    bool alive;
};
struct Cand {
    i32 rank;
    int pos;       // index of the left node
    i32 merged;
    int left_id;
    int right_id;
    bool operator<(const Cand& o) const {
        if (rank != o.rank) return rank > o.rank;   // min-heap on rank
        return pos > o.pos;
    }
};
}

void Tokenizer::bpe_chunk(const std::string& piece, std::vector<i32>& out) const {
    if (piece.empty()) return;
    GAI_CHECK(piece.size() <= static_cast<size_t>(std::numeric_limits<int>::max()),
              "tokenizer chunk exceeds int range");

    // Whole-chunk fast path
    auto whole = token_ids_.find(piece);
    if (whole != token_ids_.end() && whole->second >= special::COUNT) {
        out.push_back(whole->second);
        return;
    }
    auto cit = cache_.find(piece);
    if (cit != cache_.end()) {
        out.insert(out.end(), cit->second.begin(), cit->second.end());
        return;
    }

    std::vector<Node> nodes;
    nodes.reserve(piece.size());
    for (size_t i = 0; i < piece.size(); ++i) {
        i32 id = special::COUNT + static_cast<i32>(static_cast<u8>(piece[i]));
        nodes.push_back(Node{id, static_cast<int>(i) - 1,
                             (i + 1 < piece.size()) ? static_cast<int>(i) + 1 : -1, true});
    }

    std::priority_queue<Cand> pq;
    auto try_push = [&](int l) {
        if (l < 0) return;
        int r = nodes[static_cast<size_t>(l)].next;
        if (r < 0) return;
        u64 key = (static_cast<u64>(static_cast<u32>(nodes[static_cast<size_t>(l)].id)) << 32) |
                   static_cast<u32>(nodes[static_cast<size_t>(r)].id);
        auto it = merges_.find(key);
        if (it == merges_.end()) return;
        pq.push(Cand{it->second.first, l, it->second.second,
                     nodes[static_cast<size_t>(l)].id, nodes[static_cast<size_t>(r)].id});
    };

    for (size_t i = 0; i + 1 < nodes.size(); ++i) try_push(static_cast<int>(i));

    while (!pq.empty()) {
        Cand c = pq.top();
        pq.pop();
        Node& L = nodes[static_cast<size_t>(c.pos)];
        if (!L.alive || L.id != c.left_id) continue;
        int r = L.next;
        if (r < 0) continue;
        Node& R = nodes[static_cast<size_t>(r)];
        if (!R.alive || R.id != c.right_id) continue;

        L.id   = c.merged;
        L.next = R.next;
        if (R.next >= 0) nodes[static_cast<size_t>(R.next)].prev = c.pos;
        R.alive = false;

        try_push(L.prev);
        try_push(c.pos);
    }

    std::vector<i32> ids;
    for (int i = 0; i >= 0 && i < static_cast<int>(nodes.size()); i = nodes[static_cast<size_t>(i)].next) {
        if (nodes[static_cast<size_t>(i)].alive) ids.push_back(nodes[static_cast<size_t>(i)].id);
    }

    // FIX P2 (cache thrash): wholesale clear on overflow caused a latency
    // spike every 200k unique chunks (EN lake). Evict a random 1/8 instead
    // so hot entries survive; amortized O(1) instead of cliff.
    if (cache_.size() >= cache_limit_) {
        size_t drop = cache_.size() / 8 + 1;
        auto it = cache_.begin();
        while (drop-- > 0 && it != cache_.end()) it = cache_.erase(it);
    }
    cache_.emplace(piece, ids);
    out.insert(out.end(), ids.begin(), ids.end());
}

// ---------------------------------------------------------------- encode
std::vector<i32> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<i32> out;
    if (add_bos) out.push_back(special::BOS);

    std::string norm = norm_.normalize(text);
    for (const Chunk& c : pre_tokenize(norm)) bpe_chunk(c.text, out);

    if (add_eos) out.push_back(special::EOS);
    return out;
}

std::vector<i32> Tokenizer::encode_with_specials(const std::string& text) const {
    std::vector<i32> out;
    const auto& specials = special_token_strings();

    size_t i = 0;
    std::string buf;
    auto flush = [&]() {
        if (buf.empty()) return;
        std::string norm = norm_.normalize(buf);
        for (const Chunk& c : pre_tokenize(norm)) bpe_chunk(c.text, out);
        buf.clear();
    };

    while (i < text.size()) {
        bool matched = false;
        if (text[i] == '<') {
            for (size_t s = 0; s < specials.size(); ++s) {
                const std::string& sp = specials[s];
                if (text.compare(i, sp.size(), sp) == 0) {
                    flush();
                    out.push_back(static_cast<i32>(s));
                    i += sp.size();
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) { buf.push_back(text[i]); ++i; }
    }
    flush();
    return out;
}

// ---------------------------------------------------------------- decode
const std::string& Tokenizer::token_text(i32 id) const {
    static const std::string empty;
    if (id < 0 || id >= static_cast<i32>(vocab_.size())) return empty;
    return vocab_[static_cast<size_t>(id)];
}

i32 Tokenizer::token_to_id(const std::string& tok) const {
    auto it = token_ids_.find(tok);
    return it == token_ids_.end() ? -1 : it->second;
}

std::string Tokenizer::decode_one(i32 id) const { return token_text(id); }

std::string Tokenizer::decode(const std::vector<i32>& ids, bool skip_special) const {
    std::string out;
    out.reserve(ids.size() * 3);
    for (i32 id : ids) {
        if (skip_special && is_special(id)) continue;
        out += token_text(id);
    }
    return out;
}

std::string Tokenizer::Stream::push(i32 id) {
    buf_ += tk_.token_text(id);
    // emit the longest prefix that is complete UTF-8
    size_t safe = 0, i = 0;
    while (i < buf_.size()) {
        int len = utf8_seq_len(static_cast<u8>(buf_[i]));
        // FIX: lone continuation (len==0) is NOT complete — old code emitted
        // it immediately, splitting codepoints (inference mojibake on split
        // boundaries). Buffer it until the head byte arrives.
        if (len == 0) break;
        if (i + static_cast<size_t>(len) > buf_.size()) break;
        // Validate continuations: E2 28 A1 must not count as complete.
        bool ok = true;
        for (int k = 1; k < len; ++k) {
            if ((static_cast<u8>(buf_[i + static_cast<size_t>(k)]) & 0xC0) != 0x80) {
                ok = false;
                break;
            }
        }
        if (!ok) { ++i; safe = i; continue; }
        i += static_cast<size_t>(len);
        safe = i;
    }
    std::string emit = buf_.substr(0, safe);
    buf_.erase(0, safe);
    return emit;
}

std::string Tokenizer::Stream::flush() {
    // FIX: never emit raw incomplete tail (old code returned partial bytes).
    // Complete tail passes through; truncated tail becomes U+FFFD.
    if (buf_.empty()) return {};
    std::string s;
    if (utf8_is_complete(buf_)) s = buf_;
    else s = "\xEF\xBF\xBD";
    buf_.clear();
    return s;
}

// ---------------------------------------------------------------- io
template <typename T>
static void wr(std::ostream& o, const T& v) { o.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <typename T>
static bool rd(std::istream& in, T& v) { return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof(T))); }

void Tokenizer::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot write tokenizer: " + path);

    wr(f, GTOK_MAGIC);
    wr(f, GTOK_VERSION);
    u32 nflags = norm_.config().pack();
    wr(f, nflags);
    u32 nvocab = static_cast<u32>(vocab_.size());
    wr(f, nvocab);

    // vocab blob
    for (const auto& t : vocab_) {
        u32 len = static_cast<u32>(t.size());
        wr(f, len);
        f.write(t.data(), static_cast<std::streamsize>(len));
    }

    // merges, sorted by rank so load order is deterministic
    std::vector<std::pair<i32, u64>> ordered;
    ordered.reserve(merges_.size());
    for (const auto& [key, val] : merges_) ordered.emplace_back(val.first, key);
    std::sort(ordered.begin(), ordered.end());

    u32 nmerges = static_cast<u32>(ordered.size());
    wr(f, nmerges);
    for (const auto& [rank, key] : ordered) {
        u32 l = static_cast<u32>(key >> 32);
        u32 r = static_cast<u32>(key & 0xFFFFFFFFu);
        wr(f, l);
        wr(f, r);
    }
    GAI_CHECK(f.good(), "tokenizer write failed: " + path);
}

bool Tokenizer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;

    u32 magic = 0, version = 0, nflags = 0, nvocab = 0;
    if (!rd(f, magic) || magic != GTOK_MAGIC) return false;
    if (!rd(f, version) || version != GTOK_VERSION) return false;
    if (!rd(f, nflags)) return false;
    if (!rd(f, nvocab) || nvocab == 0 || nvocab > 10000000u) return false;

    norm_.set_config(NormalizerConfig::unpack(nflags));

    vocab_.clear();
    vocab_.reserve(nvocab);
    for (u32 i = 0; i < nvocab; ++i) {
        u32 len = 0;
        if (!rd(f, len) || len > 1024) return false;
        std::string t(len, '\0');
        if (len && !f.read(t.data(), static_cast<std::streamsize>(len))) return false;
        vocab_.push_back(std::move(t));
    }
    // FIX: bpe_chunk assumes byte fallback vocab[16+i] == single byte i
    // (id = 16+byte). A custom/truncated/reordered .gtok silently produced
    // wrong ids + token_text "" -> data loss. Enforce the invariant on load.
    if (vocab_.size() < static_cast<size_t>(special::COUNT) + 256) return false;
    for (int b = 0; b < 256; ++b) {
        const std::string& t = vocab_[static_cast<size_t>(special::COUNT) + b];
        if (t.size() != 1 || static_cast<u8>(t[0]) != static_cast<u8>(b)) return false;
    }
    finalize_index();

    u32 nmerges = 0;
    if (!rd(f, nmerges) || nmerges > 10000000u) return false;
    merges_.clear();
    merges_.reserve(static_cast<size_t>(nmerges) * 2);
    for (u32 r = 0; r < nmerges; ++r) {
        u32 l = 0, rr = 0;
        if (!rd(f, l) || !rd(f, rr)) return false;
        if (l >= nvocab || rr >= nvocab) return false;
        const std::string merged = vocab_[l] + vocab_[rr];
        auto it = token_ids_.find(merged);
        if (it == token_ids_.end()) continue;
        merges_[(static_cast<u64>(l) << 32) | rr] = {static_cast<i32>(r), it->second};
    }
    cache_.clear();
    return true;
}

// ---------------------------------------------------------------- fertility
Tokenizer::Fertility Tokenizer::measure(const std::vector<std::string>& lines) const {
    Fertility f;
    for (const auto& line : lines) {
        if (line.empty()) continue;
        std::vector<i32> ids = encode(line);
        f.tokens += ids.size();
        f.bytes  += line.size();
        // whitespace word count
        bool in_word = false;
        size_t i = 0;
        while (i < line.size()) {
            u32 cp = utf8_decode(line, i);
            bool ws = is_whitespace(cp);
            if (!ws && !in_word) { ++f.words; in_word = true; }
            else if (ws) in_word = false;
        }
    }
    f.tokens_per_word = f.words ? static_cast<double>(f.tokens) / static_cast<double>(f.words) : 0.0;
    f.bytes_per_token = f.tokens ? static_cast<double>(f.bytes) / static_cast<double>(f.tokens) : 0.0;
    return f;
}

} // namespace gai
