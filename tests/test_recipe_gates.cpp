// Recipe gate: every SHIPPED config must survive the same gates the Kaggle
// preflight applies, checked here in CI instead of 40 minutes into a GPU run.
//
// This is the test that would have caught the "fp16_weight_cache: true on the
// 1B" edit: that change ADDS ~2 GiB of VRAM (the fp16 mirror is an extra
// allocation, not a saving) and pushes the 1B recipe from 89.7% to 102.2% of
// the 16 GiB per-GPU budget, i.e. straight through the preflight FAIL. A
// hand-written review said it was a speedup; the arithmetic says the run dies
// before step 1. The gate is arithmetic, so it is asserted here.
//
// The cost model is NOT re-implemented: price_recipe() (training/trainer.cpp)
// is the same function `gai_train --dry-run` gates on, so this test and the
// preflight can never disagree about what a recipe costs.
//
// It also lints the parts of a recipe the preflight only checks at runtime:
// mix weights, vocab agreement, the SFT -> PT checkpoint chain, and the
// attention T^2 budget.
#include "training/trainer.h"
#include "core/config.h"
#include "core/common.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

// kaggle/train_2xt4.sh picks the per-GPU VRAM gate from the PT config name:
// the 1B family gets the full 16 GiB, everything else the stricter 15 GiB.
static size_t vram_gate_mb(const std::string& path) {
    const std::string n = fs::path(path).filename().string();
    if (n.find("1b") != std::string::npos ||
        n.find("pro_1b_single") != std::string::npos ||
        n.find("pro_1b_4xt4_legacy") != std::string::npos) {
        return 16384;
    }
    return 15360;
}

struct Recipe {
    std::string  path;
    std::string  name;
    Config       cfg;
    TrainerConfig t;
    ModelConfig   m;
};

int main() {
    std::error_code ec;
    if (!fs::exists("configs", ec)) {
        std::cerr << "test_recipe_gates: run from the repo root (ctest does)\n";
        return 1;
    }

    std::vector<Recipe> recipes;
    for (const auto& e : fs::directory_iterator("configs", ec)) {
        if (!e.is_regular_file() || e.path().extension() != ".yaml") continue;
        const std::string path = e.path().string();
        const Config c = Config::from_file(path);
        // A synth-generator config (synth_*.yaml) has no model: block at all.
        if (!c.has("model.vocab_size")) {
            std::cerr << "skip (not a training recipe): " << path << "\n";
            continue;
        }
        Recipe r;
        r.path = path;
        r.name = e.path().filename().string();
        r.cfg  = c;
        // strict=true is what --strict-config does on Kaggle: a typo'd or dead
        // key must fail here, not silently train a default.
        r.t = TrainerConfig::from_config(c, /*strict=*/true);
        r.m = ModelConfig::from_config(c, "model");
        r.m.validate();
        recipes.push_back(r);
    }
    CHECK(!recipes.empty(), "at least one training recipe is present");
    std::cerr << "checked " << recipes.size() << " recipe(s)\n";

    for (const Recipe& r : recipes) {
        const std::string& n = r.name;
        const std::string who = n + ": ";

        // ---- tokenizer / model vocab must agree (silent OOB reads otherwise)
        if (r.cfg.has("tokenizer.vocab_size")) {
            CHECK(r.cfg.get_int("tokenizer.vocab_size") == r.m.vocab_size,
                  who + "tokenizer.vocab_size != model.vocab_size");
        }

        // ---- exactly one schedule. The warmup-vs-total guard is only decidable
        //      in steps mode; in epochs mode the step count is derived from the
        //      corpus and the trainer enforces it at runtime (trainer.cpp).
        CHECK((r.t.max_steps > 0) != (r.t.epochs > 0),
              who + "schedule must set exactly one of max_steps / epochs");
        CHECK(r.t.warmup_steps >= 0, who + "warmup_steps must be >= 0");
        if (r.t.max_steps > 0) {
            CHECK(r.t.warmup_steps < r.t.max_steps,
                  who + "warmup_steps " + std::to_string(r.t.warmup_steps) +
                      " >= max_steps " + std::to_string(r.t.max_steps) +
                      " — the whole run would sit in the warmup branch");
        }
        CHECK(r.t.sched_decay_frac > 0.0f && r.t.sched_decay_frac <= 1.0f,
              who + "sched_decay_frac must be in (0,1] (0 silently disables decay)");
        CHECK(r.t.optimizer == "adamw" || r.t.optimizer == "lion" ||
              r.t.optimizer == "muon",
              who + "unknown optimizer '" + r.t.optimizer + "'");
        CHECK(r.t.loss_scale_max <= 16384.0,
              who + "loss_scale_max above the fp16-safe hard max (16384)");
        if (r.t.device == "cuda" && r.t.gemm_fp16 && r.t.loss_scale_init > 0.0) {
            // A dynamic scaler with no ceiling walks up until it overflows and
            // then cycles; every shipped T4 recipe pins a measured-safe level.
            CHECK(r.t.loss_scale_max > 0.0,
                  who + "gemm_fp16 + dynamic loss scaling needs a pinned "
                        "loss_scale_max (0 = unbounded growth into overflow)");
        }

        // ---- data mix: positive weights that sum to 1 (the sampler renormalises,
        //      so a 0.9 sum silently reweights the corpus against the author's
        //      intent and nobody sees it in the log).
        if (!r.t.mix.empty()) {
            double sum = 0.0;
            for (const auto& [dom, w] : r.t.mix) {
                CHECK(w > 0.0, who + "mix[" + dom + "] must be > 0");
                sum += w;
            }
            CHECK(std::fabs(sum - 1.0) < 1e-6,
                  who + "mix weights sum to " + std::to_string(sum) + ", expected 1.0");
        }

        // ---- attention T^2 buffer: the trainer hard-fails above 800 MB
        {
            const double probs_mb = static_cast<double>(r.t.batch_size) *
                                    static_cast<double>(r.m.num_heads) *
                                    static_cast<double>(r.t.seq_len) *
                                    static_cast<double>(r.t.seq_len) * 4.0 /
                                    (1024.0 * 1024.0);
            CHECK(probs_mb <= 800.0,
                  who + "attention T^2 buffer " + std::to_string(probs_mb) +
                      " MB exceeds the 800 MB OOM guard (lower seq_len)");
        }

        // ---- the gates themselves
        const RecipeCost cost = price_recipe(r.m, r.t);
        if (r.t.ddp) {
            const size_t gate = vram_gate_mb(r.path) * 1024u * 1024u;
            CHECK(cost.total <= gate,
                  who + "predicted peak " + human_bytes(cost.total) + " exceeds the " +
                      human_bytes(gate) + " per-GPU gate");
            // >=10% headroom: allocator fragmentation + cuBLAS workspaces +
            // NCCL staging push the real peak above the arithmetic estimate.
            CHECK(cost.total * 10 <= gate * 9,
                  who + "only " +
                      std::to_string(100.0 - 100.0 * static_cast<double>(cost.total) /
                                                        static_cast<double>(gate)) +
                      "% headroom under " + human_bytes(gate) + " (need >=10%): " +
                      human_bytes(cost.total) +
                      " [params " + human_count(cost.params) +
                      ", fp16_cache " + human_bytes(cost.fp16_cache) + "]");
        }
        if (r.t.output_budget_mb > 0) {
            const size_t budget = static_cast<size_t>(r.t.output_budget_mb) * 1024u * 1024u;
            CHECK(cost.output_projection <= budget,
                  who + "projected saved output " + human_bytes(cost.output_projection) +
                      " exceeds output_budget_mb " + human_bytes(budget) +
                      " (2 snapshots " + human_bytes(cost.snapshot * 2) +
                      " + gguf " + human_bytes(cost.gguf) + ")");
        }

        // ---- a 1B recipe is Lion-only on a 16 GB card: AdamW's second moment
        //      is 3.9 GB more than the budget can absorb.
        if (cost.params > 800000000ULL && r.t.ddp) {
            CHECK(r.t.optimizer != "adamw",
                  who + "a 1B-class recipe cannot afford AdamW on one 16 GB T4 "
                        "(needs Lion/Muon): " + human_bytes(cost.total));
        }

        // ---- SFT stage: it must name the pretrain checkpoint that SOME
        //      pretrain recipe actually writes, or stage B dies AFTER the whole
        //      pretrain has burned the session. Resolved by the checkpoint path
        //      itself, not by the file name: sft_flash_480m_4xt4.yaml is the 4-GPU
        //      SFT of flash_480m_single.yaml, so a "strip sft_" lookup would be wrong.
        if (r.t.is_sft() && !r.t.pretrained_checkpoint.empty()) {
            const Recipe* partner = nullptr;
            for (const Recipe& p : recipes) {
                if (p.t.is_sft()) continue;
                if (p.t.checkpoint_dir + "/last.ckpt" == r.t.pretrained_checkpoint) {
                    partner = &p;
                    break;
                }
            }
            CHECK(partner != nullptr,
                  who + "pretrained_checkpoint '" + r.t.pretrained_checkpoint +
                      "' is not written by any shipped pretrain recipe");
            if (partner) {
                CHECK(r.t.checkpoint_dir != partner->t.checkpoint_dir,
                      who + "SFT writes into the pretrain checkpoint_dir (" +
                          partner->t.checkpoint_dir + ") — stage B would overwrite stage A");
                // The arch gate the trainer enforces at resume.
                std::string why;
                CHECK(partner->m.same_architecture_as(r.m, &why),
                      n + " vs " + partner->name + " architecture parity: " + why);
            }
        }
    }

    if (failures == 0) { std::cout << "test_recipe_gates: ALL PASS\n"; return 0; }
    std::cerr << "test_recipe_gates: " << failures << " FAILURES\n";
    return 1;
}
