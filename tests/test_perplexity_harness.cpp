#include "evaluation/perplexity.h"
#include "model/model.h"
#include "tokenizer/tokenizer.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    ModelConfig cfg;
    cfg.vocab_size = 64;
    cfg.hidden_size = 16;
    cfg.num_layers = 1;
    cfg.num_heads = 2;
    cfg.num_kv_heads = 1;
    cfg.intermediate_size = 32;
    cfg.max_seq_len = 64;
    cfg.use_moe = false;
    Model model(cfg, Device::CPU);
    model.init_weights(11);
    Tokenizer tokenizer;
    tokenizer.init_empty(NormalizerConfig{});
    Generator generator(model, tokenizer, 64);
    std::vector<i32> tokens;
    std::vector<i32> segments;
    for (int i = 0; i < 32; ++i) {
        tokens.push_back(16 + (i % 40));
        segments.push_back(i / 8);
    }
    PerplexityResult a = evaluate_perplexity(generator, tokens, nullptr, &segments);
    PerplexityResult b = evaluate_perplexity(generator, tokens, nullptr, &segments);
    CHECK(a.tokens == 31 && b.tokens == 31, "token-level perplexity count");
    CHECK(a.mean_nll == b.mean_nll && a.perplexity == b.perplexity,
          "held-out perplexity is deterministic");
    CHECK(std::isfinite(a.perplexity) && a.perplexity > 1.0, "valid perplexity");
    if (failures == 0) {
        std::cout << "test_perplexity_harness: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_perplexity_harness: " << failures << " FAILURES\n";
    return 1;
}
