#pragma once

#include "core/common.h"
#include "core/rng.h"
#include "tokenizer/chat_template.h"
#include "dataset/langid.h"
#include <string>
#include <map>
#include <vector>

namespace gai {

enum class Script : u8 { Arabic = 0, Latin = 1, Arabizi = 2 };

std::string arabic_to_arabizi(const std::string& arabic, Rng& rng, bool heavy_digits = true);
std::string apply_orthographic_noise(const std::string& latin, Rng& rng);

struct SynthConfig {
    u64  seed = 1234;
    int  num_conversations = 200000;
    int  min_turns = 2;
    int  max_turns = 12;
    double p_arabic_script = 0.62;
    double p_latin_script  = 0.28;
    double p_msa           = 0.10;
    double p_french_switch = 0.12;
    double p_followup      = 0.45;
    double p_correction    = 0.08;
    double p_misunderstand = 0.06;
    double p_governor      = 0.07;
    double p_reasoning     = 0.08;
    bool   include_system  = true;
    double p_system        = 0.35;
    int    max_template_uses = 400;
    int    max_attempts_multiplier = 60;
};

struct Conversation {
    std::vector<Message> messages;
    std::string domain;
    Script      script = Script::Arabic;
    u64         template_id = 0;
};

struct SynthStats {
    u64 generated = 0;
    u64 rejected_duplicate = 0;
    u64 rejected_style = 0;
    u64 rejected_entropy = 0;
    u64 accepted = 0;
    std::map<std::string, u64> by_domain;
    std::map<std::string, u64> by_script;
    std::string summary() const;
};

class SynthGenerator {
public:
    explicit SynthGenerator(SynthConfig cfg = {});
    ~SynthGenerator();

    bool generate(Conversation& out);

    std::vector<Conversation> generate_many(int n);

    const SynthStats& stats() const { return stats_; }

    static int domain_count();
    static const char* domain_name(int i);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SynthConfig cfg_;
    SynthStats  stats_;
};

void write_conversations_jsonl(const std::string& path, const std::vector<Conversation>& convs);
std::vector<Conversation> read_conversations_jsonl(const std::string& path);

}
