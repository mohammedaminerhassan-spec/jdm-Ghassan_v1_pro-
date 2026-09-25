#include "dataset/english_logic.h"
#include "core/unicode.h"
#include <algorithm>
#include <cctype>
#include <cstring>


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

static bool is_word_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
}

static bool starts_word(const std::string& s, const char* pre) {
    size_t n = 0;
    while (pre[n]) ++n;
    if (s.size() < n || s.compare(0, n, pre) != 0) return false;
    return s.size() == n || !is_word_char(s[n]);
}



bool is_greeting(const std::string& user_text) {
    std::string l = low_trim(user_text);
    static const char* kGreet[] = {
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "whats up", "how is it going", "nice to meet",
    };
    for (const char* g : kGreet) {
        if (starts_word(l, g)) return true;
    }
    return false;
}

static bool has_word(const std::string& l, const char* w) {
    size_t n = 0;
    while (w[n]) ++n;
    for (size_t i = 0; i + n <= l.size(); ++i) {
        if (l.compare(i, n, w) != 0) continue;
        const bool left_ok = (i == 0) || !is_word_char(l[i - 1]);
        const bool right_ok = (i + n == l.size()) || !is_word_char(l[i + n]);
        if (left_ok && right_ok) return true;
    }
    return false;
}

bool is_coding(const std::string& user_text) {
    std::string l = low_trim(user_text);
    if (l.find("```") != std::string::npos) return true;
    if (l.find("#include") != std::string::npos) return true;
    static const char* kCodeWords[] = {
        "python", "javascript", "java", "code", "function", "class", "def",
        "return", "import", "script", "program", "debug", "compile",
        "algorithm", "plainformat",
    };
    // "return" alone is ambiguous ("return a product" vs "return a value"):
    // it only signals code alongside another code word. Found by the English
    // behavior-data audit (a shopping exchange misclassified as Coding).
    int hits = 0;
    bool has_return = false;
    for (const char* w : kCodeWords) {
        if (!has_word(l, w)) continue;
        if (std::strcmp(w, "return") == 0) { has_return = true; continue; }
        ++hits;
    }
    if (hits > 0) return true;
    if (has_return) {
        // bare "return" + code markers already handled above; otherwise it is
        // only code with an explicit code noun nearby (value/statement/type).
        static const char* kReturnCtx[] = {"value", "statement", "type", "keyword"};
        for (const char* w : kReturnCtx)
            if (has_word(l, w)) return true;
    }
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
        if (starts_word(l, v)) return true;
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
        "which", "whether", "is", "are", "was", "were", "do", "does",
        "did", "can", "could", "would", "should", "will", "have",
        "has", "had",
    };
    for (const char* q : kQ) {
        if (starts_word(l, q)) return true;
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

bool is_multiple_choice_prompt(const std::string& user_text) {
    std::string l = low_trim(user_text);
    int options = 0;
    for (size_t i = 0; i < l.size(); ++i) {
        const char c = l[i];
        if (c < 'a' || c > 'd') continue;
        const bool left_ok = (i == 0) || !is_word_char(l[i - 1]);
        const bool right_ok = (i + 1 < l.size()) &&
                              (l[i + 1] == '.' || l[i + 1] == ')' || l[i + 1] == ':');
        if (left_ok && right_ok) ++options;
    }
    return options >= 2;
}

bool meets_multiple_choice_discipline(const std::string& reply) {
    std::string l = low_trim(reply);
    if (l.size() < 3) return false;
    const char c = l[0];
    if (c < 'a' || c > 'd') return false;
    if (l[1] != '.' && l[1] != ')' && l[1] != ':' && l[1] != ' ') return false;
    return true;
}

bool contains_code(const std::string& reply) {
    std::string l = low_trim(reply);
    if (l.find("```") != std::string::npos) return true;
    if (l.find("#include") != std::string::npos) return true;
    if (l.find("plainformat") != std::string::npos) return true;
    static const char* kCode[] = {
        "def ", "function ", "return ", "import ", "class ", "for (", "while (",
        "if (", "=>", "{", "}", ";",
    };
    for (const char* k : kCode) {
        if (l.find(k) != std::string::npos) return true;
    }
    return false;
}

ReplyReport check_english_reply(const std::string& user_text, const std::string& reply) {
    ReplyReport r;
    r.act = classify_dialog_act(user_text);
    if (reply.empty() || low_trim(reply).size() < 2) {
        r.reason = "empty";
        return r;
    }
    std::string low = low_trim(reply);
    static const char* kBad[] = {
        "as an ai", "as a language model", "as an ai language model",
    };
    for (const char* b : kBad) {
        if (low.find(b) != std::string::npos) {
            r.reason = "ai-disclosure";
            return r;
        }
    }
    if (is_multiple_choice_prompt(user_text) && !meets_multiple_choice_discipline(reply)) {
        r.reason = "bad-mc-format";
        return r;
    }
    if (r.act == DialogAct::Coding && !contains_code(reply)) {
        r.reason = "no-code";
        return r;
    }
    r.disciplined = true;
    return r;
}

bool obeys_answer_discipline(const std::string& reply) {
    return check_english_reply("", reply).disciplined;
}

} // namespace english_logic
} // namespace gai
