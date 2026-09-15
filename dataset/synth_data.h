#pragma once

#include <vector>
#include <string>

// Authored Moroccan Darija content for the synthetic dialogue generator.
// Everything here is written for this project; no scraped or copied corpora.
namespace gai {
namespace synth_data {

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

// Generic conversational glue, reusable across domains.
const std::vector<const char*>& greetings_user();
const std::vector<const char*>& greetings_assistant();
const std::vector<const char*>& closers_user();
const std::vector<const char*>& closers_assistant();
const std::vector<const char*>& fillers();            // آه، صافي، واخا ...
const std::vector<const char*>& backchannels_user();  // "واخا", "فهمت", "زعما؟"
const std::vector<Exchange>&    corrections();        // user corrects the assistant
const std::vector<Exchange>&    misunderstandings();
const std::vector<Exchange>&    identity_questions(); // who are you / are you human
const std::vector<Exchange>&    msa_exchanges();
const std::vector<std::pair<const char*, const char*>>& french_terms();  // darija -> french word
const std::vector<const char*>& system_prompts();

} // namespace synth_data
} // namespace gai
