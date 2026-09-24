#pragma once

#include "inference/generator.h"
#include "training/dataloader.h"
#include <string>
#include <vector>

namespace gai {

struct PerplexityResult {
    double mean_nll = 0.0;
    double perplexity = 0.0;
    i64 tokens = 0;
    i64 batches = 0;
};

PerplexityResult evaluate_perplexity(Generator& generator,
                                      const std::vector<i32>& tokens,
                                      const std::vector<u8>* mask = nullptr,
                                      const std::vector<i32>* segment_ids = nullptr);
PerplexityResult evaluate_shard_perplexity(Generator& generator,
                                            const std::string& shard_dir,
                                            const std::string& prefix,
                                            int batch_size,
                                            int seq_len,
                                            int max_batches,
                                            bool pack_sequences,
                                            u64 seed = 42);

}
