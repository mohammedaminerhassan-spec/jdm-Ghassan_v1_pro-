#include "inference/chat.h"
#include "dataset/english_logic.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace gai {

ChatSession::ChatSession(Generator& gen, ChatOptions opts)
    : gen_(gen), opts_(std::move(opts)) {
    GAI_CHECK(opts_.max_history_turns >= 0, "chat history turns must be >= 0");
    GAI_CHECK(opts_.retrieve_top_k >= 0, "retrieve top_k must be >= 0");
    GAI_CHECK(opts_.gen.max_new_tokens >= 0, "chat max tokens must be >= 0");
    if (opts_.system.empty()) {
        // PRO-EN: English persona is script-independent (never swapped by the
        // ScriptRouter); Darija keeps the legacy Arabic-default routing.
        opts_.system = (opts_.persona == "en" || opts_.persona == "english")
                           ? ChatTemplate::default_system_english()
                           : ChatTemplate::default_system();
        custom_system_ = false;
    } else {
        custom_system_ = true;
    }
    history_.push_back({Role::System, opts_.system});
}

void ChatSession::clear() {
    std::string sys;
    if (!history_.empty()) {
        sys = history_[0].content;
    } else if (opts_.persona == "en" || opts_.persona == "english") {
        sys = ChatTemplate::default_system_english();
    } else {
        sys = ChatTemplate::default_system();
    }
    history_.clear();
    history_.push_back({Role::System, sys});
}

void ChatSession::set_system(const std::string& s) {
    opts_.system = s;
    custom_system_ = true;
    if (!history_.empty() && history_[0].role == Role::System) history_[0].content = s;
    else history_.insert(history_.begin(), {Role::System, s});
}

void ChatSession::apply_script_policy(const std::string& user_message) {
    if (history_.empty() || history_[0].role != Role::System) return;
    // PRO-EN: with the English persona the system line is script-independent
    // and must NOT be swapped per turn (legacy Darija routing only).
    if (opts_.persona == "en" || opts_.persona == "english") return;
    ReplyScript script = ChatTemplate::detect_script(user_message);
    if (!custom_system_) {
        // Default persona: swap the whole system line to the matching script.
        history_[0].content = ChatTemplate::system_for_script(script);
        opts_.system = history_[0].content;
    } else {
        // Custom persona: keep it, enforce the script with one directive line.
        const std::string dir = ChatTemplate::script_directive(script);
        if (history_[0].content.find(dir) == std::string::npos) {
            history_[0].content += std::string("\n") + dir;
        }
    }
}

void ChatSession::trim_history() {
    // keep the system prompt + the most recent N user/assistant pairs
    if (static_cast<int>(history_.size()) <= 1 + 2 * opts_.max_history_turns) return;
    size_t keep_from = history_.size() - static_cast<size_t>(2 * opts_.max_history_turns);
    std::vector<Message> trimmed;
    trimmed.push_back(history_[0]);
    for (size_t i = keep_from; i < history_.size(); ++i) trimmed.push_back(history_[i]);
    history_ = std::move(trimmed);
}

bool ChatSession::try_retrieve(const std::string& user_message, std::string& out) const {
    if (opts_.retrieve_index.empty()) return false;
    if (!retrieve_tried_) {
        retrieve_tried_ = true;
        std::vector<QaEntry> docs;
        size_t n = 0;
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::is_directory(opts_.retrieve_index, ec)) {
            n = load_qa_dir(opts_.retrieve_index, docs);
        } else {
            // single file or directory with one file
            if (load_qa_json(opts_.retrieve_index, docs)) n = docs.size();
            else n = load_qa_dir(opts_.retrieve_index, docs);
        }
        if (n > 0) {
            retrieve_ = std::make_unique<RetrievalIndex>();
            retrieve_->build(docs);
        }
    }
    if (!retrieve_ || retrieve_->size() == 0) return false;
    auto hits = retrieve_->query(user_message, opts_.retrieve_top_k, opts_.retrieve_threshold);
    if (hits.empty()) return false;
    out = retrieve_->doc(hits[0].doc).answer;
    return true;
}

static void adapt_english_generation(GenerationConfig& cfg, const std::string& user_message) {
    using namespace english_logic;
    const DialogAct act = classify_dialog_act(user_message);
    auto& s = cfg.sampling;
    switch (act) {
        case DialogAct::Greeting:
            cfg.max_new_tokens = std::min(cfg.max_new_tokens, 96);
            s.temperature = std::min(s.temperature, 0.8f);
            s.top_p = 0.90f;
            s.min_p = 0.02f;
            s.no_repeat_ngram = 0;
            break;
        case DialogAct::Coding:
            s.temperature = std::min(s.temperature, 0.4f);
            s.top_k = 40;
            s.top_p = 1.0f;
            s.min_p = 0.0f;
            s.no_repeat_ngram = 0;
            break;
        case DialogAct::Reasoning:
            s.temperature = std::min(s.temperature, 0.6f);
            s.top_k = 40;
            s.top_p = 0.95f;
            s.min_p = 0.02f;
            s.no_repeat_ngram = 0;
            break;
        case DialogAct::Instruction:
            s.temperature = std::min(s.temperature, 0.6f);
            s.top_k = 40;
            s.top_p = 0.90f;
            s.min_p = 0.05f;
            s.no_repeat_ngram = 0;
            break;
        case DialogAct::Question:
            s.temperature = std::min(s.temperature, 0.7f);
            s.top_p = 0.92f;
            s.min_p = 0.05f;
            s.no_repeat_ngram = 0;
            break;
        case DialogAct::Chitchat:
        case DialogAct::Unknown:
            break;
    }
    s.validate();
}

std::string ChatSession::send(const std::string& user_message) {
    // RAG fast path: paraphrase-aware keyword match (digit-normalized, e.g.
    // "salam 3likom" == "salam alikom"). When a close question exists in the
    // indexed data, return its stored answer directly instead of hallucinating.
    if (opts_.retrieve_direct) {
        std::string retrieved;
        if (try_retrieve(user_message, retrieved)) {
            history_.push_back({Role::User, user_message});
            trim_history();
            if (opts_.stream) std::cout << retrieved << std::endl;
            history_.push_back({Role::Assistant, retrieved});
            return retrieved;
        }
    }
    // ScriptRouter runs BEFORE the turn is encoded: the system persona for
    // this turn always matches the script the user just typed in.
    apply_script_policy(user_message);
    history_.push_back({Role::User, user_message});
    // Augment mode: prepend best hit as context when direct is off
    if (!opts_.retrieve_direct && !opts_.retrieve_index.empty()) {
        std::string retrieved;
        if (try_retrieve(user_message, retrieved)) {
            // inject as a system reminder so the model stays grounded
            history_.insert(history_.end() - 1,
                            {Role::System, "Context (from dataset): " + retrieved});
        }
    }
    trim_history();

    std::string reply;
    Generator::StreamFn cb = nullptr;
    if (opts_.stream) {
        cb = [&](const std::string& piece, i32) {
            std::cout << piece << std::flush;
            return true;
        };
    }
    GenerationConfig turn_cfg = opts_.gen;
    if (opts_.adaptive_dialog &&
        (opts_.persona == "en" || opts_.persona == "english")) {
        adapt_english_generation(turn_cfg, user_message);
    }
    reply = gen_.chat(history_, turn_cfg, cb);
    if (opts_.stream) std::cout << std::endl;

    history_.push_back({Role::Assistant, reply});
    return reply;
}

void ChatSession::run_repl() {
    // The banner must match the persona: telling an English user to "type your
    // message in Darija" is a real UX bug, not decoration.
    const bool en = (opts_.persona == "en" || opts_.persona == "english");
    if (en) {
        std::cout <<
            "\n  Ghassan AI - chat (English)\n"
            "  Type your message. Commands: /help /clear /system <text> /stats /quit\n"
            "  ------------------------------------------------------------------\n\n";
    } else {
        std::cout <<
            "\n  Ghassan AI - chat\n"
            "  اكتب رسالتك بالدارجة. الأوامر: /help /clear /system <text> /stats /quit\n"
            "  ------------------------------------------------------------------\n\n";
    }

    std::string line;
    while (true) {
        std::cout << "\n\033[1muser\033[0m > " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        if (line[0] == '/') {
            std::string cmd = line.substr(1);
            std::string arg;
            size_t sp = cmd.find(' ');
            if (sp != std::string::npos) { arg = cmd.substr(sp + 1); cmd = cmd.substr(0, sp); }

            if (cmd == "quit" || cmd == "exit" || cmd == "q") break;
            if (cmd == "clear") { clear(); std::cout << "  [history cleared]\n"; continue; }
            if (cmd == "system") {
                if (arg.empty()) std::cout << "  system: " << opts_.system << "\n";
                else { set_system(arg); std::cout << "  [system prompt updated]\n"; }
                continue;
            }
            if (cmd == "stats") {
                std::cout << "  " << gen_.stats().summary() << "\n";
                std::cout << "  context used: " << gen_.context_used() << " tokens\n";
                std::cout << "  history: " << history_.size() << " messages\n";
                continue;
            }
            if (cmd == "temp") {
                if (!arg.empty()) {
                    try {
                        float t = std::stof(arg);
                        if (!std::isfinite(t)) t = 0.0f;
                        if (t < 0.0f) t = 0.0f;
                        if (t > 5.0f) t = 5.0f;
                        opts_.gen.sampling.temperature = t;
                        std::cout << "  [temperature = " << opts_.gen.sampling.temperature << "]\n";
                    } catch (const std::exception&) {
                        std::cout << "  invalid temperature (want 0..5)\n";
                    }
                }
                continue;
            }
            if (cmd == "help") {
                std::cout <<
                    "  /clear          reset the conversation\n"
                    "  /system <text>  set the system prompt\n"
                    "  /temp <f>       set the sampling temperature\n"
                    "  /stats          show timing and context usage\n"
                    "  /quit           exit\n";
                continue;
            }
            std::cout << "  unknown command: " << cmd << " (try /help)\n";
            continue;
        }

        std::cout << "\n\033[1mghassan\033[0m > " << std::flush;
        send(line);
        if (opts_.show_stats) std::cout << "  [" << gen_.stats().summary() << "]\n";
    }
    std::cout << "\n  بسلامة!\n";
}

} // namespace gai
