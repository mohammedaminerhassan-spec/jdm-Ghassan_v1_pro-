#pragma once

// English dialogue-behavior synthesizer: the English twin of SynthGenerator.
// Composes authored English exchanges (dataset/english_dialogue_data.h) into
// multi-turn conversations for SFT shards. No transliteration, no script
// mixing — English is English. Every assistant turn must pass the
// english_logic answer-discipline gate, so the generator can never emit
// behavior the style filter would reject downstream.
#include "dataset/synth.h"

namespace gai {
namespace english_synth {

struct EnglishSynthConfig {
    u64 seed = 4321;
    int num_conversations = 20000;
    int min_turns = 2;
    int max_turns = 10;
    double p_greeting = 0.30;
    double p_correction = 0.07;
    double p_misunderstand = 0.06;
    double p_governor = 0.10;    // controller exchanges (honesty, refusal, length)
    double p_reasoning = 0.08;   // casual step-by-step chains
    double p_identity = 0.06;
    double p_followup = 0.45;
    double p_closer = 0.35;
    double p_backchannel = 0.15;
    double p_filler = 0.12;      // filler prepended to an assistant turn
    bool   include_system = true;
    double p_system = 0.35;
    int    max_template_uses = 400;
    int    max_attempts_multiplier = 60;
    // First-token entropy cap: no 2-word reply head may exceed this share.
    // 0.10 (vs Darija's 0.06) because the authored English pool is smaller;
    // still strict enough to force surface variety.
    double max_head_share = 0.10;
};

class EnglishSynthGenerator {
public:
    explicit EnglishSynthGenerator(EnglishSynthConfig cfg = {});
    ~EnglishSynthGenerator();

    bool generate(Conversation& out);
    std::vector<Conversation> generate_many(int n);

    const SynthStats& stats() const { return stats_; }

    static int domain_count();
    static const char* domain_name(int i);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    EnglishSynthConfig cfg_;
    SynthStats stats_;
};

} // namespace english_synth
} // namespace gai
