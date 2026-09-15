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

// Wire format:
//   <s> <|system|> sys <|end|> <|user|> u <|end|> <|assistant|> a <|end|> ... </s>
// During SFT only assistant spans (content + trailing <|end|>) receive loss.
class ChatTemplate {
public:
    static const char* default_system();

    // Renders to plain text (with literal special markers) - useful for debugging.
    static std::string render(const std::vector<Message>& msgs, bool add_generation_prompt);

    // Tokenizes directly, and (optionally) emits a per-token loss mask.
    static std::vector<i32> encode(const Tokenizer& tk,
                                   const std::vector<Message>& msgs,
                                   bool add_generation_prompt,
                                   std::vector<u8>* loss_mask = nullptr);

    static i32 role_token(Role r);
};

} // namespace gai
