#pragma once

#include "dataset/darija_synthesis_engine.h"

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
    double p_governor = 0.10;
    double p_reasoning = 0.08;
    double p_identity = 0.06;
    double p_followup = 0.45;
    double p_closer = 0.35;
    double p_backchannel = 0.15;
    double p_filler = 0.12;
    bool   include_system = true;
    double p_system = 0.35;
    int    max_template_uses = 400;
    int    max_attempts_multiplier = 60;

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

}
}
