#include "dataset/corpus_stats.h"
#include "core/unicode.h"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace gai {

void CorpusAnalyzer::add_document(const std::string& text) {
    if (text.empty()) return;
    ++st_.documents;
    st_.bytes += text.size();

    ScriptStats ss = script_stats(text);
    st_.chars += ss.total;
    if (ss.emoji > 0) ++st_.emoji_docs;

    size_t letters = ss.arabic + ss.latin;
    if (letters > 0) {
        if (static_cast<double>(ss.arabic) / static_cast<double>(letters) > 0.6) ++st_.arabic_script_docs;
        else if (static_cast<double>(ss.latin) / static_cast<double>(letters) > 0.6) ++st_.latin_script_docs;
    }

    // word count
    u64 w = 0;
    {
        bool in_word = false;
        size_t i = 0;
        while (i < text.size()) {
            u32 cp = utf8_decode(text, i);
            bool ws = is_whitespace(cp);
            if (!ws && !in_word) { ++w; in_word = true; }
            else if (ws) in_word = false;
        }
    }
    st_.words += w;

    // line count
    st_.lines += 1 + static_cast<u64>(std::count(text.begin(), text.end(), '\n'));

    LangScore ls = lid_.classify(text);
    st_.by_lang[lang_name(ls.tag)]++;
    if (ls.arabizi_score > 0.02) ++st_.arabizi_docs;
    if (ls.french_score > 0.05 && ls.is_darija()) ++st_.french_mixed_docs;

    if (tok_) {
        std::vector<i32> ids = tok_->encode(text);
        st_.tokens += ids.size();
        doc_tokens_.push_back(ids.size());
        for (i32 id : ids) seen_tokens_.insert(id);
    }
}

void CorpusAnalyzer::add_file(const std::string& path, bool line_per_doc) {
    std::ifstream f(path, std::ios::binary);
    GAI_CHECK(f.good(), "cannot open corpus file: " + path);
    if (line_per_doc) {
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) add_document(line);
        }
    } else {
        std::ostringstream ss;
        ss << f.rdbuf();
        add_document(ss.str());
    }
}

CorpusStats CorpusAnalyzer::finish() {
    if (st_.documents > 0) {
        st_.avg_doc_chars = static_cast<double>(st_.chars) / static_cast<double>(st_.documents);
        st_.avg_doc_words = static_cast<double>(st_.words) / static_cast<double>(st_.documents);
    }
    if (tok_ && !doc_tokens_.empty()) {
        st_.avg_doc_tokens = static_cast<double>(st_.tokens) / static_cast<double>(doc_tokens_.size());
        std::vector<u64> sorted = doc_tokens_;
        std::sort(sorted.begin(), sorted.end());
        st_.median_doc_tokens = static_cast<double>(sorted[sorted.size() / 2]);
        st_.doc_token_hist = std::move(sorted);
        st_.vocab_used = seen_tokens_.size();
        if (st_.words) st_.tokens_per_word = static_cast<double>(st_.tokens) / static_cast<double>(st_.words);
        if (st_.tokens) st_.bytes_per_token = static_cast<double>(st_.bytes) / static_cast<double>(st_.tokens);

        // byte-fallback tokens are ids [16, 272)
        u64 raw_bytes = 0;
        for (i32 id : seen_tokens_) if (id >= 16 && id < 272) ++raw_bytes;
        st_.oov_byte_rate = st_.vocab_used ? static_cast<double>(raw_bytes) / static_cast<double>(st_.vocab_used) : 0.0;
    }
    return st_;
}

std::string CorpusStats::report(int vocab_size) const {
    std::ostringstream o;
    o << "\n==================== corpus statistics ====================\n";
    o << strfmt("  documents          : %s\n", human_count(documents).c_str());
    o << strfmt("  lines              : %s\n", human_count(lines).c_str());
    o << strfmt("  bytes              : %s\n", human_bytes(bytes).c_str());
    o << strfmt("  characters         : %s\n", human_count(chars).c_str());
    o << strfmt("  words              : %s\n", human_count(words).c_str());
    if (tokens) {
        o << strfmt("  tokens             : %s\n", human_count(tokens).c_str());
        o << strfmt("  tokens / word      : %.3f\n", tokens_per_word);
        o << strfmt("  bytes / token      : %.3f\n", bytes_per_token);
    }
    o << "\n  -- language distribution (share of documents) --\n";
    for (const auto& [name, n] : by_lang) {
        o << strfmt("    %-14s %8s  %6.2f%%\n", name.c_str(), human_count(n).c_str(), pct(n));
    }
    o << "\n  -- script / style --\n";
    o << strfmt("    arabic script    %6.2f%%\n", pct(arabic_script_docs));
    o << strfmt("    latin script     %6.2f%%\n", pct(latin_script_docs));
    o << strfmt("    arabizi          %6.2f%%\n", pct(arabizi_docs));
    o << strfmt("    french-mixed     %6.2f%%\n", pct(french_mixed_docs));
    o << strfmt("    contains emoji   %6.2f%%\n", pct(emoji_docs));
    o << "\n  -- length --\n";
    o << strfmt("    avg chars/doc    %.1f\n", avg_doc_chars);
    o << strfmt("    avg words/doc    %.1f\n", avg_doc_words);
    if (tokens) {
        o << strfmt("    avg tokens/doc   %.1f\n", avg_doc_tokens);
        o << strfmt("    median tokens    %.0f\n", median_doc_tokens);
        if (!doc_token_hist.empty()) {
            auto q = [&](double p) {
                size_t i = static_cast<size_t>(p * static_cast<double>(doc_token_hist.size() - 1));
                return doc_token_hist[i];
            };
            o << strfmt("    p10/p50/p90/p99  %llu / %llu / %llu / %llu\n",
                        (unsigned long long)q(0.10), (unsigned long long)q(0.50),
                        (unsigned long long)q(0.90), (unsigned long long)q(0.99));
        }
    }
    if (vocab_used) {
        o << "\n  -- vocabulary --\n";
        o << strfmt("    distinct tokens  %s / %d  (%.1f%% coverage)\n",
                    human_count(vocab_used).c_str(), vocab_size,
                    vocab_size ? 100.0 * double(vocab_used) / double(vocab_size) : 0.0);
        o << strfmt("    byte-fallback    %.2f%% of used vocab\n", oov_byte_rate * 100.0);
    }
    if (dup_ratio > 0) o << strfmt("\n  duplicate ratio    : %.2f%%\n", dup_ratio * 100.0);
    o << "===========================================================\n";
    return o.str();
}

} // namespace gai
