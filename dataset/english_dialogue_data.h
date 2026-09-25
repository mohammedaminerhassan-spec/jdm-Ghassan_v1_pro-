#pragma once

#include <vector>
#include <string>
#include <utility>

// Authored English dialogue-behavior content for the synthetic conversation
// generator. This is the English twin of synth_data.h (Darija): it teaches the
// model HOW to behave in conversation — persona, tone, honesty, refusal,
// concision vs. depth — not facts (facts come from the Hermes parquet lake).
// Everything here is written fresh for this project; no scraped corpora.
namespace gai {
namespace english_dialogue_data {

struct Exchange {
    const char* user;
    const char* assistant;
};

struct Domain {
    const char*            name;
    std::vector<Exchange>  openers;      // conversation starters
    std::vector<Exchange>  followups;    // context-dependent continuations
};

const std::vector<Domain>& domains();

const std::vector<const char*>& greetings_user();
const std::vector<const char*>& greetings_assistant();
const std::vector<const char*>& closers_user();
const std::vector<const char*>& closers_assistant();
const std::vector<const char*>& fillers();
const std::vector<const char*>& backchannels_user();
const std::vector<Exchange>&    corrections();        // user corrects the assistant
const std::vector<Exchange>&    misunderstandings();
const std::vector<Exchange>&    identity_questions(); // who are you / are you human
// Governor ("controller") exchanges: teach short-vs-detailed answers, honest
// "I don't know", staying grounded, admitting mistakes, refusing harm calmly,
// staying neutral on politics/religion. This is the data that governs all
// other domains at SFT time.
const std::vector<Exchange>&    governor_exchanges();
// Short, casual step-by-step chains for simple math/logic — conversational,
// not textbook-style, so the model learns to think before answering.
const std::vector<Exchange>&    reasoning_exchanges();
const std::vector<const char*>& system_prompts();

} // namespace english_dialogue_data
} // namespace gai
