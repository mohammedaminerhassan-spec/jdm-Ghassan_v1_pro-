#pragma once

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
    size_t doc = 0;
    double score = 0.0;
};

std::string retrieval_normalize(const std::string& s);
std::vector<std::string> retrieval_tokenize(const std::string& s);

bool load_qa_json(const std::string& path, std::vector<QaEntry>& out);

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

}
