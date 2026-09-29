#pragma once

#include "inference/generator.h"
#include "dataset/retrieval.h"
#include <memory>
#include <string>
#include <vector>

namespace gai {

struct ChatOptions {
    std::string      system;

    std::string      persona = "darija";
    GenerationConfig gen;
    bool             adaptive_dialog = true;
    bool             stream = true;
    bool             show_stats = false;
    int              max_history_turns = 24;

    std::string      retrieve_index;
    double           retrieve_threshold = 3.0;
    int              retrieve_top_k = 1;
    bool             retrieve_direct = true;
};

class ChatSession {
public:
    ChatSession(Generator& gen, ChatOptions opts);

    std::string send(const std::string& user_message);

    void clear();
    void set_system(const std::string& s);
    const std::vector<Message>& history() const { return history_; }

    void run_repl();

private:
    void trim_history();

    bool try_retrieve(const std::string& user_message, std::string& out) const;

    void apply_script_policy(const std::string& user_message);

    Generator&           gen_;
    ChatOptions          opts_;
    std::vector<Message> history_;
    bool                 custom_system_ = false;

    mutable std::unique_ptr<RetrievalIndex> retrieve_;
    mutable bool retrieve_tried_ = false;

    mutable std::string retrieve_loaded_path_;
};

}
