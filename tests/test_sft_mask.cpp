#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <iostream>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

int main() {
    Tokenizer tokenizer;
    tokenizer.init_empty(NormalizerConfig{});
    std::vector<Message> messages = {
        {Role::System, "Be concise."},
        {Role::User, "Say hi"},
        {Role::Assistant, "Hi"},
        {Role::User, "Again"},
        {Role::Assistant, "OK"},
    };
    std::vector<u8> mask;
    std::vector<i32> ids = ChatTemplate::encode(tokenizer, messages, false, &mask);
    CHECK(ids.size() == mask.size(), "mask covers every token");
    CHECK(mask[special::BOS] == 0, "BOS ignored");
    CHECK(mask[special::SYSTEM] == 0, "system role ignored");
    CHECK(mask[special::USER] == 0, "user role ignored");
    CHECK(mask[special::ASSISTANT] == 0, "assistant role ignored");
    int supervised = 0;
    for (u8 value : mask) supervised += value != 0 ? 1 : 0;
    CHECK(supervised >= 4, "short assistant answers and end tokens supervised");
    bool ended = false;
    for (size_t i = 1; i < ids.size(); ++i) {
        if (ids[i - 1] == special::END && ids[i] == special::USER) {
            ended = true;
            CHECK(mask[i] == 0, "following user turn ignored");
        }
    }
    CHECK(ended, "multi-turn boundaries found");
    if (failures == 0) {
        std::cout << "test_sft_mask: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_sft_mask: " << failures << " FAILURES\n";
    return 1;
}
