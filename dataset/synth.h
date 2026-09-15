#pragma once

#include "core/common.h"
#include "core/rng.h"
#include "tokenizer/chat_template.h"
#include "dataset/langid.h"
#include <string>
#include <map>
#include <vector>

namespace gai {

// ---------------------------------------------------------------- orthography
enum class Script : u8 { Arabic = 0, Latin = 1, Arabizi = 2 };

// Deterministic Arabic <-> Latin/Arabizi transliteration for Darija, used to
// multiply the effective corpus and to teach the model that "شنو" and "chno" and
// "shnou" are the same word.
std::string arabic_to_arabizi(const std::string& arabic, Rng& rng, bool heavy_digits = true);
std::string apply_orthographic_noise(const std::string& latin, Rng& rng);

// ---------------------------------------------------------------- generator
struct SynthConfig {
    u64  seed = 1234;
    int  num_conversations = 200000;
    int  min_turns = 2;
    int  max_turns = 12;
    double p_arabic_script = 0.62;   // Darija in Arabic letters
    double p_latin_script  = 0.28;   // Darija in Latin/Arabizi
    double p_msa           = 0.10;   // occasional MSA exchange
    double p_french_switch = 0.12;   // French term injected in a turn
    double p_followup      = 0.45;
    double p_correction    = 0.08;
    double p_misunderstand = 0.06;
    bool   include_system  = true;
    double p_system        = 0.35;
    int    max_template_uses = 400;  // diversity cap
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

// Compositional dialogue generator: scenario graph x surface realizer x turn
// planner. Not an LLM-distillation pipeline - every string is authored here.
class SynthGenerator {
public:
    explicit SynthGenerator(SynthConfig cfg = {});
    ~SynthGenerator();

    // Generates one conversation. Returns false if the diversity filters rejected it.
    bool generate(Conversation& out);

    // Generates `n` accepted conversations (retries internally).
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

// Serialises conversations to a JSONL-ish format the pipeline can re-read.
void write_conversations_jsonl(const std::string& path, const std::vector<Conversation>& convs);
std::vector<Conversation> read_conversations_jsonl(const std::string& path);

} // namespace gai
