#pragma once

// dataset/retrieval.h — tiny keyword index over id/question/answer JSON shards.
//
// Why this exists: users paraphrase ("salam labas" vs training "salam bikhir").
// A generative model answers paraphrases from training diversity, but this
// index gives a DETERMINISTIC, testable fallback: it finds the stored question
// sharing the most informative words with the user's question (exact words or
// word pieces), compares the overlap score, and returns the stored answer when
// the score passes --min-score. Arabizi digits (3/7/9) and Arabic script both
// survive normalization because they carry meaning in Darija.
//
// Scoring: BM25 (k1=1.2, b=0.75) over idf(matched tokens) with tf length-norm,
// plus bigram-order bonus (+0.5/shared bigram) and +5.0 when the whole
// normalized query appears inside the stored question (exact / near-exact
// match). No embeddings, no dependencies, pure C++.

#include "core/common.h"
#include <map>
#include <string>
#include <vector>

namespace gai {

struct QaEntry {
    long long   id = 0;
    std::string question;
    std::string answer;
};

struct RetrievalHit {
    size_t doc = 0;      // index into the loaded docs
    double score = 0.0;
};

std::string retrieval_normalize(const std::string& s);
std::vector<std::string> retrieval_tokenize(const std::string& s);

// Loads one QA file in either layout:
//   - top-level array:  [{id,question,answer}, ...]   (train-*.json shards)
//   - JSONL:            {id,question,answer} per line  (Darija clean files)
// id accepts int OR string (strings hash to stable int64, e.g.
// "ghassan_darija_0002717"); extra keys (domain/script/...) are ignored.
// A single bad object never aborts the file. Returns false only when the
// file cannot be read/parsed at all (or yields zero pairs).
bool load_qa_json(const std::string& path, std::vector<QaEntry>& out);
// Loads every *.json / *.jsonl in a directory (sorted). Returns total pairs.
size_t load_qa_dir(const std::string& dir, std::vector<QaEntry>& out);

class RetrievalIndex {
public:
    void build(const std::vector<QaEntry>& docs);
    std::vector<RetrievalHit> query(const std::string& text, int top_k,
                                    double min_score = 0.0) const;
    size_t size() const { return docs_.size(); }
    const QaEntry& doc(size_t i) const { return docs_[i]; }

private:
    std::vector<QaEntry> docs_;
    std::vector<std::vector<std::string>> toks_;
    std::vector<std::string> norm_q_;
    std::vector<int> doc_len_;
    std::map<std::string, double> idf_;
    std::map<std::string, std::vector<size_t>> inv_;
};

} // namespace gai
