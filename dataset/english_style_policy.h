#pragma once

#include <string>
#include <vector>

namespace gai {
namespace english_logic {

enum class DialogAct : unsigned char {
    Greeting = 0,
    Question = 1,
    Instruction = 2,
    Coding = 3,
    Reasoning = 4,
    Chitchat = 5,
    Unknown = 6,
};

const char* dialog_act_name(DialogAct a);

DialogAct classify_dialog_act(const std::string& user_text);
bool is_question(const std::string& user_text);
bool is_instruction(const std::string& user_text);
bool is_greeting(const std::string& user_text);
bool is_coding(const std::string& user_text);

const std::vector<const char*>& causal_connectors();
const std::vector<const char*>& contrast_connectors();
const std::vector<const char*>& sequence_connectors();

bool obeys_answer_discipline(const std::string& reply);

struct ReplyReport {
    DialogAct act = DialogAct::Unknown;
    bool disciplined = false;
    std::string reason;
};

ReplyReport check_english_reply(const std::string& user_text, const std::string& reply);
bool is_multiple_choice_prompt(const std::string& user_text);
bool meets_multiple_choice_discipline(const std::string& reply);
bool contains_code(const std::string& reply);

}
}
