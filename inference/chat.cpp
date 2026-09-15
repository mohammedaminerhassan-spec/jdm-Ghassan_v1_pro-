#include "inference/chat.h"

#include <iostream>
#include <algorithm>
#include <filesystem>

namespace gai {

ChatSession::ChatSession(Generator& gen, ChatOptions opts)
    : gen_(gen), opts_(std::move(opts)) {
    if (opts_.system.empty()) opts_.system = ChatTemplate::default_system();
    history_.push_back({Role::System, opts_.system});
}

void ChatSession::clear() {
    std::string sys = history_.empty() ? ChatTemplate::default_system() : history_[0].content;
    history_.clear();
    history_.push_back({Role::System, sys});
}

void ChatSession::set_system(const std::string& s) {
    opts_.system = s;
    if (!history_.empty() && history_[0].role == Role::System) history_[0].content = s;
    else history_.insert(history_.begin(), {Role::System, s});
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
    reply = gen_.chat(history_, opts_.gen, cb);
    if (opts_.stream) std::cout << std::endl;

    history_.push_back({Role::Assistant, reply});
    return reply;
}

void ChatSession::run_repl() {
    std::cout <<
        "\n  Ghassan AI - chat\n"
        "  اكتب رسالتك بالدارجة. الأوامر: /help /clear /system <text> /stats /quit\n"
        "  ------------------------------------------------------------------\n\n";

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
                    opts_.gen.sampling.temperature = std::stof(arg);
                    std::cout << "  [temperature = " << opts_.gen.sampling.temperature << "]\n";
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
