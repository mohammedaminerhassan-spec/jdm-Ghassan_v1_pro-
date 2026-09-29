#pragma once

#include <vector>
#include <string>
#include <utility>

namespace gai {
namespace english_dialogue_data {

struct Exchange {
    const char* user;
    const char* assistant;
};

struct Domain {
    const char*            name;
    std::vector<Exchange>  openers;
    std::vector<Exchange>  followups;
};

const std::vector<Domain>& domains();

const std::vector<const char*>& greetings_user();
const std::vector<const char*>& greetings_assistant();
const std::vector<const char*>& closers_user();
const std::vector<const char*>& closers_assistant();
const std::vector<const char*>& fillers();
const std::vector<const char*>& backchannels_user();
const std::vector<Exchange>&    corrections();
const std::vector<Exchange>&    misunderstandings();
const std::vector<Exchange>&    identity_questions();

const std::vector<Exchange>&    governor_exchanges();

const std::vector<Exchange>&    reasoning_exchanges();
const std::vector<const char*>& system_prompts();

}
}
