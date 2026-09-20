#include "training/dataloader.h"
#include "tokenizer/tokenizer.h"

#include <cstdio>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace gai {

// ================================================================ writer
ShardWriter::ShardWriter(const std::string& path, int vocab_size, bool with_loss_mask)
    : path_(path), u16_mode_(vocab_size <= 65535), with_mask_(with_loss_mask),
      vocab_size_(vocab_size) {}

ShardWriter::~ShardWriter() {
    if (!closed_) {
        try { close(); } catch (...) {}
    }
}

void ShardWriter::add_document(const std::vector<i32>& tokens, const std::vector<u8>* mask) {
    if (tokens.empty()) return;
    doc_offsets_.push_back(static_cast<u64>(tokens_.size()));
    for (size_t i = 0; i < tokens.size(); ++i) {
        // Fail-loud token range: an id outside the writing vocabulary (or a
        // negative id, e.g. an unmasked -100) must never enter a shard — on
        // load it would read OOB embeddings or silently truncate u16 casts.
        GAI_CHECK(tokens[i] >= 0 && tokens[i] < vocab_size_,
                  strfmt("token id %d out of writing-vocab range [0,%d)",
                         tokens[i], vocab_size_));
        tokens_.push_back(static_cast<u32>(tokens[i]));
        if (with_mask_) mask_.push_back(mask ? (*mask)[i] : 1);
    }
}

void ShardWriter::close() {
    if (closed_) return;
    closed_ = true;

    fs::path p(path_);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    std::ofstream f(path_, std::ios::binary);
    GAI_CHECK(f.good(), "cannot write shard: " + path_);

    ShardHeader h;
    h.dtype    = u16_mode_ ? 0u : 1u;
    h.flags    = with_mask_ ? GBIN_FLAG_MASK : 0u;
    h.n_tokens = tokens_.size();
    h.n_docs   = doc_offsets_.size();
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));

    if (u16_mode_) {
        std::vector<u16> buf(tokens_.size());
        for (size_t i = 0; i < tokens_.size(); ++i) buf[i] = static_cast<u16>(tokens_[i]);
        f.write(reinterpret_cast<const char*>(buf.data()),
                static_cast<std::streamsize>(buf.size() * sizeof(u16)));
    } else {
        f.write(reinterpret_cast<const char*>(tokens_.data()),
                static_cast<std::streamsize>(tokens_.size() * sizeof(u32)));
    }
    f.write(reinterpret_cast<const char*>(doc_offsets_.data()),
            static_cast<std::streamsize>(doc_offsets_.size() * sizeof(u64)));
    if (with_mask_) {
        f.write(reinterpret_cast<const char*>(mask_.data()),
                static_cast<std::streamsize>(mask_.size()));
    }
    GAI_CHECK(f.good(), "shard write failed: " + path_);
}

// ================================================================ reader
bool Shard::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    ShardHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (h.magic != GBIN_MAGIC || h.version != GBIN_VERSION) return false;
    // FIX: corrupt/truncated .gbin with garbage dtype or huge n_tokens/n_docs
    // caused unbounded resize -> bad_alloc/OOM + garbage offsets -> later OOB
    // (CPU/Kaggle crash). Validate BEFORE allocating + cross-check file size.
    if (h.dtype > 1) return false;
    {
        std::error_code ec;
        const auto fsize = fs::file_size(path, ec);
        if (!ec) {
            const uint64_t tok_bytes = h.n_tokens * (h.dtype == 0 ? 2ull : 4ull);
            const uint64_t need_min = sizeof(ShardHeader) + tok_bytes
                + h.n_docs * 8ull;
            // allow trailing mask byte per token, but never accept a header
            // claiming MORE bytes than the file actually holds.
            if (need_min > static_cast<uint64_t>(fsize)) return false;
            // hard cap: single shard > 8GiB tokens is corrupt for this project.
            if (h.n_tokens > (8ull << 30) / 2 || h.n_docs > (1ull << 30)) return false;
            // RAM guard for FULL load (3GB+ corpora): a 50M-token shard is
            // ~100-200MB RAM (safe). Anything above 128M tokens (~512MB u32)
            // must use the streaming header-only path, never full RAM load,
            // or Kaggle 30GB RAM + model weights OOM mid-run.
            if (h.n_tokens > (128ull << 20)) return false;
        } else if (h.n_tokens > (4ull << 30) || h.n_docs > (1ull << 30)) {
            return false;
        }
    }

    tokens_.resize(static_cast<size_t>(h.n_tokens));
    if (h.dtype == 0) {
        std::vector<u16> buf(static_cast<size_t>(h.n_tokens));
        if (h.n_tokens && !f.read(reinterpret_cast<char*>(buf.data()),
                                  static_cast<std::streamsize>(buf.size() * sizeof(u16)))) return false;
        for (size_t i = 0; i < buf.size(); ++i) tokens_[i] = buf[i];
    } else {
        if (h.n_tokens && !f.read(reinterpret_cast<char*>(tokens_.data()),
                                  static_cast<std::streamsize>(tokens_.size() * sizeof(u32)))) return false;
    }

    doc_offsets_.resize(static_cast<size_t>(h.n_docs));
    if (h.n_docs && !f.read(reinterpret_cast<char*>(doc_offsets_.data()),
                            static_cast<std::streamsize>(doc_offsets_.size() * sizeof(u64)))) return false;

    if (h.flags & GBIN_FLAG_MASK) {
        mask_.resize(static_cast<size_t>(h.n_tokens));
        if (h.n_tokens && !f.read(reinterpret_cast<char*>(mask_.data()),
                                  static_cast<std::streamsize>(mask_.size()))) return false;
    }
    header_ = h;
    path_ = path;
    streaming_ = false;
    return true;
}

bool Shard::load_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    ShardHeader h{};
    if (!f.read(reinterpret_cast<char*>(&h), sizeof(h))) return false;
    if (h.magic != GBIN_MAGIC || h.version != GBIN_VERSION) return false;
    if (h.n_tokens == 0) return false;
    // Same corruption guard as load(): reject bad dtype / impossible sizes.
    if (h.dtype > 1) return false;
    if (h.n_tokens > (8ull << 30) / 2 || h.n_docs > (1ull << 30)) return false;

    // Seek to the doc table: header + token stream.
    size_t tok_bytes = static_cast<size_t>(h.n_tokens) * (h.dtype == 0 ? sizeof(u16) : sizeof(u32));
    f.seekg(static_cast<std::streamoff>(sizeof(h) + tok_bytes), std::ios::beg);
    if (!f.good()) return false;
    std::vector<u64> offs(static_cast<size_t>(h.n_docs));
    if (h.n_docs && !f.read(reinterpret_cast<char*>(offs.data()),
                            static_cast<std::streamsize>(offs.size() * sizeof(u64)))) return false;

    header_ = h;
    doc_offsets_ = std::move(offs);
    tokens_.clear();
    tokens_.shrink_to_fit();
    mask_.clear();
    mask_.shrink_to_fit();
    path_ = path;
    streaming_ = true;
    return true;
}

// Per-thread cached fd: avoids open/close per row (was 100s of opens/sec).
// The cache reopens only when the path changes; seeks are cheap.
namespace {
struct ShardFdCache {
    std::ifstream f;
    std::string path;
    std::ifstream& open(const std::string& p) {
        if (!f.is_open() || path != p) {
            if (f.is_open()) f.close();
            f.open(p, std::ios::binary);
            path = p;
        } else {
            f.clear();
        }
        return f;
    }
};
thread_local ShardFdCache t_fd_cache;
} // namespace

bool Shard::read_window(u64 start, u64 len, std::vector<u32>& tok_out, std::vector<u8>& mask_out) const {
    tok_out.assign(static_cast<size_t>(len), 0);
    bool want_mask = has_mask();
    if (want_mask) mask_out.assign(static_cast<size_t>(len), 1);
    else mask_out.clear();
    if (len == 0) return true;

    // PRO-HARDEN: مسار RAM كان يقرأ tokens_[start+len] بلا فحص فينهار heap
    // على offsets فاسدة؛ مسار streaming كان يرجع false فقط. نوحد الفشل السريع.
    GAI_CHECK(start + len <= n_tokens(),
              strfmt("Shard::read_window OOB (start=%llu len=%llu ntok=%llu)",
                     (unsigned long long)start, (unsigned long long)len,
                     (unsigned long long)n_tokens()));
    if (!streaming_) {
        // RAM path: memcpy from loaded vectors.
        for (u64 i = 0; i < len; ++i) {
            u64 ti = start + i;
            tok_out[static_cast<size_t>(i)] = tokens_[static_cast<size_t>(ti)];
            if (want_mask && !mask_.empty()) mask_out[static_cast<size_t>(i)] = mask_[static_cast<size_t>(ti)];
        }
        return true;
    }
    // Streaming path: random reads from disk (small, SSD-friendly).
    std::ifstream& f = t_fd_cache.open(path_);
    if (!f.good()) return false;
    size_t tok_sz = (header_.dtype == 0 ? sizeof(u16) : sizeof(u32));
    size_t tok_off = sizeof(ShardHeader) + static_cast<size_t>(start) * tok_sz;
    f.seekg(static_cast<std::streamoff>(tok_off), std::ios::beg);
    if (!f.good()) return false;
    if (header_.dtype == 0) {
        // T4-P1-18: never malloc on the streaming hot path. Reuse a
        // thread-local staging buffer (grows monotonically, like t_fd_cache
        // above — reads already happen per-thread, so this is race-free).
        thread_local std::vector<u16> t_u16_stage;
        if (t_u16_stage.size() < static_cast<size_t>(len))
            t_u16_stage.resize(static_cast<size_t>(len));
        u16* buf = t_u16_stage.data();
        if (!f.read(reinterpret_cast<char*>(buf),
                    static_cast<std::streamsize>(static_cast<size_t>(len) * sizeof(u16)))) return false;
        for (size_t i = 0; i < static_cast<size_t>(len); ++i) tok_out[i] = buf[i];
    } else {
        if (!f.read(reinterpret_cast<char*>(tok_out.data()),
                    static_cast<std::streamsize>(tok_out.size() * sizeof(u32)))) return false;
    }
    if (want_mask) {
        size_t tok_bytes = static_cast<size_t>(header_.n_tokens) * tok_sz;
        size_t doc_bytes = static_cast<size_t>(header_.n_docs) * sizeof(u64);
        size_t mask_off = sizeof(ShardHeader) + tok_bytes + doc_bytes + static_cast<size_t>(start);
        f.clear();
        f.seekg(static_cast<std::streamoff>(mask_off), std::ios::beg);
        if (!f.good()) return false;
        if (!f.read(reinterpret_cast<char*>(mask_out.data()),
                    static_cast<std::streamsize>(mask_out.size()))) return false;
    }
    return true;
}

u64 Shard::doc_end(u64 i) const {
    u64 total = streaming_ ? header_.n_tokens : static_cast<u64>(tokens_.size());
    if (doc_offsets_.empty()) return total;
    auto it = std::upper_bound(doc_offsets_.begin(), doc_offsets_.end(), i);
    return it == doc_offsets_.end() ? total : *it;
}

u64 Shard::doc_index_of(u64 pos) const {
    if (doc_offsets_.empty()) return 0;
    auto it = std::upper_bound(doc_offsets_.begin(), doc_offsets_.end(), pos);
    if (it == doc_offsets_.begin()) return 0;
    return static_cast<u64>(it - doc_offsets_.begin()) - 1;
}

// ================================================================ loader
std::vector<std::string> list_shards(const std::string& dir, const std::string& prefix) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!fs::exists(dir, ec)) return out;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string name = e.path().filename().string();
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gbin") continue;
        if (!prefix.empty() && name.rfind(prefix, 0) != 0) continue;
        out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool DataLoader::open(const std::vector<std::string>& paths, BatchSpec spec, u64 seed) {
    spec_ = spec;
    rng_.seed_with(seed);
    shards_.clear();
    groups_.clear();
    use_mix_ = false;
    total_tokens_ = 0;
    batches_ = 0;

    for (const auto& p : paths) {
        Shard s;
        // Header-only open: doc table in RAM, tokens stream from disk (T4 RAM saver).
        // Falls back to full load for tiny/test shards.
        if (!s.load_header(p) && !s.load(p)) {
            log_warn("dataloader: cannot load shard " + p);
            continue;
        }
        if (s.n_tokens() < static_cast<u64>(spec.seq_len) + 1) {
            log_warn("dataloader: shard too small, skipped: " + p);
            continue;
        }
        total_tokens_ += s.n_tokens();
        shards_.push_back(std::move(s));
    }
    return !shards_.empty();
}

bool DataLoader::open_glob(const std::string& dir, const std::string& prefix,
                           BatchSpec spec, u64 seed) {
    return open(list_shards(dir, prefix), spec, seed);
}

bool DataLoader::open_mix(const std::string& dir, const std::map<std::string, double>& mix,
                          BatchSpec spec, u64 seed) {
    spec_ = spec;
    rng_.seed_with(seed);
    shards_.clear();
    groups_.clear();
    use_mix_ = false;
    total_tokens_ = 0;
    batches_ = 0;
    if (mix.empty()) return false;

    double wsum = 0.0;
    for (const auto& [domain, w] : mix) {
        if (!(w > 0.0)) continue;
        // Domain shards are train_<domain>_*.gbin (see data_pipeline --domain).
        std::string prefix = std::string("train_") + domain;
        std::vector<std::string> paths = list_shards(dir, prefix);
        if (paths.empty()) {
            log_warn("dataloader: mix domain '" + domain + "' has no shards (" + prefix + "*.gbin), skipped");
            continue;
        }
        DomainGroup g;
        g.domain = domain;
        g.weight = w; // raw for now, normalized below
        for (const auto& p : paths) {
            Shard s;
            if (!s.load_header(p) && !s.load(p)) { log_warn("dataloader: cannot load shard " + p); continue; }
            if (s.n_tokens() < static_cast<u64>(spec.seq_len) + 1) {
                log_warn("dataloader: shard too small, skipped: " + p);
                continue;
            }
            g.total_tokens += s.n_tokens();
            g.shards.push_back(std::move(s));
        }
        if (g.shards.empty()) continue;
        wsum += w;
        total_tokens_ += g.total_tokens;
        groups_.push_back(std::move(g));
    }
    if (groups_.empty()) {
        total_tokens_ = 0;
        return false;
    }
    // Normalize weights over the domains actually found.
    for (auto& g : groups_) g.weight /= wsum;
    use_mix_ = true;
    return true;
}

int DataLoader::num_shards() const {
    if (use_mix_) {
        int n = 0;
        for (const auto& g : groups_) n += static_cast<int>(g.shards.size());
        return n;
    }
    return static_cast<int>(shards_.size());
}

std::string DataLoader::mix_report() const {
    if (!use_mix_ || groups_.empty()) return "";
    std::string s = "mix:";
    char buf[128];
    for (const auto& g : groups_) {
        std::snprintf(buf, sizeof(buf), " %s=%.2f(%llu tok)", g.domain.c_str(), g.weight,
                      (unsigned long long)g.total_tokens);
        s += buf;
    }
    return s;
}

bool DataLoader::fill_from_shard(const Shard& sh, Batch& out, int b) {
    const int T = spec_.seq_len;
    // DeepSeek-quality document isolation (FIX: no cross-doc leak):
    // Old packing stitched 2-3 docs into one row with no attention mask, so
    // causal attention leaked context across documents AND RoPE positions
    // (0..T-1) were wrong for the 2nd/3rd doc. Llama/DeepSeek either mask
    // documents or isolate them. We isolate: each row comes from ONE doc,
    // truncated at doc_end, tail padded with 0/-100 (honest, not counted).
    // P2 efficiency: to avoid large PAD tails when the first random start
    // lands near a doc end, try up to 4 candidates and keep the longest fit
    // (still ONE doc per row, no packing, positions always 0..T-1 correct).
    // P2-21 honesty note: best-of-4 is NOT uniform sampling — longer
    // documents/regions win more often than short trailing regions. Kept
    // deliberately as a padding-efficiency policy, not a neutral sampler.
    // Full segment-masked packing (multiple docs/row with block-causal mask)
    // is the future P2-03 upgrade; this retry already cuts PAD waste ~3x.
    u64 best_start = 0, best_end = 0, best_avail = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        u64 max_start = sh.n_tokens() > 0 ? sh.n_tokens() - 1 : 0;
        u64 start = max_start > 0 ? rng_.below(max_start + 1) : 0;
        u64 doc_end_idx = sh.doc_end(start);
        u64 avail = (doc_end_idx > start) ? (doc_end_idx - start) : 0;
        if (avail == 0) {
            u64 nd = sh.n_docs();
            start = (nd > 0) ? sh.doc_start_at(rng_.below(nd)) : 0;
            doc_end_idx = sh.doc_end(start);
            avail = (doc_end_idx > start) ? (doc_end_idx - start) : 0;
        }
        if (avail > best_avail) {
            best_avail = avail;
            best_start = start;
            best_end = doc_end_idx;
            if (best_avail >= static_cast<u64>(T) + 1) break; // full window found
        }
    }
    if (best_avail == 0 || best_start >= sh.n_tokens()) return true; // row stays PAD/-100
    u64 start = best_start, doc_end_idx = best_end, avail = best_avail;
    // Need avail tokens for ids + 1 lookahead for the last target.
    // FIX (T4/GPU-starvation): old code allocated 2 vectors per row per batch
    // (B*T rows per step). Reuse monotonic thread-local staging instead.
    u64 want = std::min<u64>(avail, static_cast<u64>(T) + 1);
    if (start + want > sh.n_tokens()) want = sh.n_tokens() - start;
    if (want == 0) return true;
    thread_local std::vector<u32> toks;
    thread_local std::vector<u8> masks;
    if (!sh.read_window(start, want, toks, masks)) return false; // I/O failure: caller retries/fails
    // Emit at most T ids; toks[want-1] is lookahead-only for the last target.
    int filled = static_cast<int>(std::min<u64>(toks.size(), static_cast<u64>(T)));
    // If toks.size() == T+1 (full window + lookahead), filled == T, correct.
    // If toks.size() <= T (doc end), filled == toks.size(), tail stays PAD.
    for (int i = 0; i < filled; ++i) {
        size_t o = static_cast<size_t>(b) * T + static_cast<size_t>(i);
        u64 ti = start + static_cast<u64>(i);
        out.ids[o] = static_cast<i32>(toks[static_cast<size_t>(i)]);
        bool has_next = (static_cast<size_t>(i) + 1 < toks.size()) && (ti + 1 < doc_end_idx);
        if (has_next) {
            u8 m = masks.empty() ? 1 : masks[static_cast<size_t>(i) + 1];
            if (m) {
                out.targets[o] = static_cast<i32>(toks[static_cast<size_t>(i) + 1]);
                ++out.tokens_supervised;
            }
        } // else targets stays -100 (doc end / no lookahead)
    }
    return true;
}

bool DataLoader::next(Batch& out) {
    if (use_mix_ ? groups_.empty() : shards_.empty()) return false;
    const int B = spec_.batch_size;
    const int T = spec_.seq_len;

    out.B = B;
    out.T = T;
    out.ids.assign(static_cast<size_t>(B) * T, 0);
    out.targets.assign(static_cast<size_t>(B) * T, -100);
    out.tokens_supervised = 0;

    for (int b = 0; b < B; ++b) {
        // P1-20: a failed window read must never become a silent padded row
        // (it would shrink the supervised-token budget invisibly). Retry the
        // row on another shard/window; a persistently failing shard means a
        // corrupt disk/image, so fail the run loudly instead of training on.
        bool row_ok = false;
        std::string last_path;
        for (int attempt = 0; attempt < 8 && !row_ok; ++attempt) {
            const Shard* shp = nullptr;
            if (use_mix_) {
                // 1. pick a domain by its mix weight (one uniform draw)
                double r = (double)rng_.uniform();
                double acc = 0.0;
                size_t gi = groups_.size() - 1;
                for (size_t i = 0; i < groups_.size(); ++i) {
                    acc += groups_[i].weight;
                    if (r < acc) { gi = i; break; }
                }
                // 2. pick a shard inside the domain proportionally to its size
                const DomainGroup& g = groups_[gi];
                u64 pick = rng_.below(g.total_tokens);
                size_t si = 0;
                for (; si + 1 < g.shards.size(); ++si) {
                    if (pick < g.shards[si].n_tokens()) break;
                    pick -= g.shards[si].n_tokens();
                }
                shp = &g.shards[si];
            } else {
                // legacy: sample a shard proportionally to its size
                u64 pick = rng_.below(total_tokens_);
                size_t si = 0;
                for (; si + 1 < shards_.size(); ++si) {
                    if (pick < shards_[si].n_tokens()) break;
                    pick -= shards_[si].n_tokens();
                }
                shp = &shards_[si];
            }
            last_path = shp->path();
            row_ok = fill_from_shard(*shp, out, b);
        }
        if (!row_ok)
            GAI_FAIL("dataloader: shard read failed 8x in a row at " + last_path +
                     " (corrupt shard or dying disk — refusing silent padded training)");
    }
    ++batches_;
    return true;
}

void DataLoader::skip_batches(i64 n) {
    Batch discard;
    for (i64 i = 0; i < n; ++i) {
        if (!next(discard)) break;
    }
}

void DataLoader::reseed(u64 seed) {
    rng_.seed_with(seed);
    batches_ = 0;
}

DataLoader::State DataLoader::get_state() const {
    State s{};
    bool hs = false;
    rng_.get_full_state(s.rng, s.rng_spare, hs);
    s.rng_has_spare = hs ? 1 : 0;
    s.batches = batches_;
    return s;
}

void DataLoader::set_state(const State& s) {
    rng_.set_full_state(s.rng, s.rng_spare, s.rng_has_spare != 0);
    batches_ = s.batches;
}

} // namespace gai
