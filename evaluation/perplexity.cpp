#include "evaluation/perplexity.h"
#include "core/common.h"

#include <cmath>

namespace gai {

static PerplexityResult from_nll(double sum, i64 tokens, i64 batches) {
    PerplexityResult result;
    result.tokens = tokens;
    result.batches = batches;
    result.mean_nll = tokens > 0 ? sum / static_cast<double>(tokens) : 0.0;
    result.perplexity = std::exp(std::min(20.0, result.mean_nll));
    return result;
}

PerplexityResult evaluate_perplexity(Generator& generator,
                                      const std::vector<i32>& tokens,
                                      const std::vector<u8>* mask,
                                      const std::vector<i32>* segment_ids) {
    i64 count = 0;
    double nll = generator.score_tokens(tokens, mask, &count, segment_ids);
    return from_nll(nll * static_cast<double>(count), count, 1);
}

PerplexityResult evaluate_shard_perplexity(Generator& generator,
                                            const std::string& shard_dir,
                                            const std::string& prefix,
                                            int batch_size,
                                            int seq_len,
                                            int max_batches,
                                            bool pack_sequences,
                                            u64 seed) {
    GAI_CHECK(max_batches > 0, "perplexity max_batches must be > 0");
    BatchSpec spec{batch_size, seq_len, pack_sequences};
    DataLoader loader;
    GAI_CHECK(loader.open_glob(shard_dir, prefix, spec, seed),
              "no held-out shards found for perplexity: " + shard_dir);
    double sum = 0.0;
    i64 tokens = 0;
    i64 batches = 0;
    Batch batch;
    while (batches < max_batches && loader.next(batch)) {
        i64 count = 0;
        std::vector<u8> loss_mask(batch.ids.size(), 0);
        for (size_t i = 0; i + 1 < batch.targets.size(); ++i) {
            loss_mask[i + 1] = batch.targets[i] >= 0 ? 1 : 0;
        }
        double nll = generator.score_tokens(batch.ids, &loss_mask, &count, &batch.segment_ids);
        sum += nll * static_cast<double>(count);
        tokens += count;
        ++batches;
    }
    return from_nll(sum, tokens, batches);
}

}
