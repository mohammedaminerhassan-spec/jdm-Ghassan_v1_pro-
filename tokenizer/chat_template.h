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

// Reply script contract: the model MUST answer in the same script the user
// wrote in (Latin/Arabizi -> Latin, Arabic letters -> Arabic). Mixing both
// scripts in one answer is the #1 "confused speech" failure mode.
enum class ReplyScript : u8 { Arabic = 0, Latin = 1 };

class ChatTemplate {
public:
    static const char* default_system();
    // Arabizi twin of default_system(): same persona, Latin letters only.
    static const char* default_system_latin();
    // PRO-EN: English assistant persona for the English-Pro model (helpful,
    // clear, honest; concise but complete; no Darija directives). Selected via
    // --persona en (default persona stays Darija: zero behavior change).
    static const char* default_system_english();
    // Short directive appended when the caller keeps a custom system prompt.
    static const char* script_directive(ReplyScript s);
    // Dominant-script vote: any Arabic-letter majority -> Arabic, else Latin
    // (pure Arabizi, French and English all land on Latin, which is correct:
    // the user typed Latin letters and expects Latin letters back).
    static ReplyScript detect_script(const std::string& user_message);
    // Picks the matching persona for a detected script.
    static const char* system_for_script(ReplyScript s);
    // Persona-aware system prompt: "en" -> English persona (script-independent);
    // anything else -> legacy Darija script routing.
    static const char* system_for_persona(const std::string& persona, ReplyScript s);

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
