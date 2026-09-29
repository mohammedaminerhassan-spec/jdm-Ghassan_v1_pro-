#pragma once

#include "tokenizer/tokenizer.h"
#include <vector>
#include <string>

namespace gai {

enum class Role : u8 { System = 0, User = 1, Assistant = 2 };

struct Message {
    Role        role = Role::User;
    std::string content;
};

enum class ReplyScript : u8 { Arabic = 0, Latin = 1 };

class ChatTemplate {
public:
    static const char* default_system();

    static const char* default_system_latin();

    static const char* default_system_english();

    static const char* script_directive(ReplyScript s);

    static ReplyScript detect_script(const std::string& user_message);

    static const char* system_for_script(ReplyScript s);

    static const char* system_for_persona(const std::string& persona, ReplyScript s);

    static std::string render(const std::vector<Message>& msgs, bool add_generation_prompt);

    static std::vector<i32> encode(const Tokenizer& tk,
                                   const std::vector<Message>& msgs,
                                   bool add_generation_prompt,
                                   std::vector<u8>* loss_mask = nullptr);

    static i32 role_token(Role r);
};

}
