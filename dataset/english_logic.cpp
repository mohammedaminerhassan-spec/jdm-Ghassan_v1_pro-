#include "dataset/english_logic.h"
#include "core/unicode.h"

#include <algorithm>
#include <cctype>

namespace gai {
namespace english_logic {

const char* dialog_act_name(DialogAct a) {
    switch (a) {
        case DialogAct::Greeting: return "greeting";
        case DialogAct::Question: return "question";
        case DialogAct::Instruction: return "instruction";
        case DialogAct::Coding: return "coding";
        case DialogAct::Reasoning: return "reasoning";
        case DialogAct::Chitchat: return "chitchat";
        default: return "unknown";
    }
}

static std::string low_trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    std::string o;
    o.reserve(b - a);
    for (size_t i = a; i < b; ++i) {
        char c = s[i];
        o += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    }
    return o;
}

static bool starts_with(const std::string& s, const char* pre) {
    size_t n = 0;
    while (pre[n]) ++n;
    return s.size() >= n && s.compare(0, n, pre) == 0;
}

bool is_greeting(const std::string& user_text) {
    std::string l = low_trim(user_text);
    static const char* kGreet[] = {
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "whats up", "how is it going", "nice to meet",
    };
    for (const char* g : kGreet) {
        if (l == g || starts_with(l, g)) {
            // "hi" must not match "history": require word boundary
            size_t n = 0;
            while (g[n]) ++n;
            if (l.size() == n || l[n] == ' ' || l[n] == ',' || l[n] == '!' || l[n] == '?')
                return true;
        }
    }
    return false;
}

bool is_coding(const std::string& user_text) {
    std::string l = low_trim(user_text);
    if (l.find("```") != std::string::npos) return true;
    if (l.find("def ") != std::string::npos) return true;
    if (l.find("function ") != std::string::npos) return true;
    if (l.find("plainformat") != std::string::npos) return true;
    if (l.find("#include") != std::string::npos) return true;
    if (l.find("import ") != std::string::npos && l.find("python") != std::string::npos) return true;
    return false;
}

bool is_instruction(const std::string& user_text) {
    std::string l = low_trim(user_text);
    static const char* kVerbs[] = {
        "write", "implement", "create", "develop", "calculate", "explain",
        "solve", "convert", "generate", "design", "debug", "describe",
        "summarize", "translate", "classify", "list",
    };
    for (const char* v : kVerbs) {
        if (starts_with(l, v)) {
            size_t n = 0;
            while (v[n]) ++n;
            if (l.size() == n || l[n] == ' ' || l[n] == ':') return true;
        }
    }
    return false;
}

bool is_question(const std::string& user_text) {
    std::string t = user_text;
    // trailing ? (allow trailing spaces/quotes)
    size_t e = t.size();
    while (e > 0 && (t[e - 1] == ' ' || t[e - 1] == '\t' || t[e - 1] == '\n' ||
                     t[e - 1] == '\r' || t[e - 1] == '"' || t[e - 1] == '\'')) --e;
    if (e > 0 && t[e - 1] == '?') return true;
    std::string l = low_trim(user_text);
    static const char* kQ[] = {
        "who", "whom", "whose", "what", "when", "where", "why", "how",
        "which", "whether", "is ", "are ", "was ", "were ", "do ", "does ",
        "did ", "can ", "could ", "would ", "should ", "will ", "have ",
        "has ", "had ",
    };
    for (const char* q : kQ) {
        if (starts_with(l, q)) return true;
    }
    return false;
}

DialogAct classify_dialog_act(const std::string& user_text) {
    if (user_text.empty()) return DialogAct::Unknown;
    if (is_greeting(user_text)) return DialogAct::Greeting;
    if (is_coding(user_text)) return DialogAct::Coding;
    if (is_instruction(user_text)) return DialogAct::Instruction;
    if (is_question(user_text)) return DialogAct::Question;
    std::string l = low_trim(user_text);
    if (l.find("how many") != std::string::npos || l.find("calculate") != std::string::npos ||
        l.find("prove") != std::string::npos || l.find("logic") != std::string::npos ||
        l.find("step") != std::string::npos)
        return DialogAct::Reasoning;
    return DialogAct::Chitchat;
}

const std::vector<const char*>& causal_connectors() {
    static const std::vector<const char*> v = {
        "because", "therefore", "so", "thus", "hence", "as a result", "since",
    };
    return v;
}

const std::vector<const char*>& contrast_connectors() {
    static const std::vector<const char*> v = {
        "however", "but", "although", "though", "nevertheless", "yet", "instead",
    };
    return v;
}

const std::vector<const char*>& sequence_connectors() {
    static const std::vector<const char*> v = {
        "first", "second", "third", "next", "then", "finally", "last",
    };
    return v;
}

bool obeys_answer_discipline(const std::string& reply) {
    if (reply.empty() || reply.size() < 2) return false;
    // must not break character
    std::string low = low_trim(reply);
    static const char* kBad[] = {
        "as an ai", "as a language model", "as an ai language model",
    };
    for (const char* b : kBad) {
        if (low.find(b) != std::string::npos) return false;
    }
    return true;
}

} // namespace english_logic
} // namespace gai
