#pragma once

#include "inference/generator.h"
#include "dataset/retrieval.h"
#include <memory>
#include <string>
#include <vector>

namespace gai {

struct ChatOptions {
    std::string      system;
    // PRO-EN: "darija" (default, legacy ScriptRouter) or "en" (English persona,
    // script-independent). Set from CLI --persona.
    std::string      persona = "darija";
    GenerationConfig gen;
    bool             adaptive_dialog = true;
    bool             stream = true;
    bool             show_stats = false;
    int              max_history_turns = 24;
    // Optional RAG: when set, paraphrased questions are answered from the
    // indexed QA data instead of hallucinating. Pure C++, no embeddings --
    // the same keyword index used by `data_pipeline retrieve`.
    std::string      retrieve_index;      // dir or train-*.json file
    double           retrieve_threshold = 3.0; // min score to use retrieved answer
    int              retrieve_top_k = 1;
    bool             retrieve_direct = true; // true = return answer, false = augment prompt
};

// Interactive REPL and one-shot chat helpers.
class ChatSession {
public:
    ChatSession(Generator& gen, ChatOptions opts);

    // one turn; returns the assistant reply
    std::string send(const std::string& user_message);

    void clear();
    void set_system(const std::string& s);
    const std::vector<Message>& history() const { return history_; }

    void run_repl();

private:
    void trim_history();
    // returns true and fills `out` when a paraphrase was found in the index
    bool try_retrieve(const std::string& user_message, std::string& out) const;
    // ScriptRouter: match the system persona to the user's script BEFORE the
    // turn is encoded, so the reply stays single-script (Latin->Latin,
    // Arabic->Arabic). Custom --system prompts are kept and only get a short
    // directive appended instead of being replaced.
    void apply_script_policy(const std::string& user_message);

    Generator&           gen_;
    ChatOptions          opts_;
    std::vector<Message> history_;
    bool                 custom_system_ = false;
    // lazy-loaded RAG index (built on first query to keep startup fast)
    mutable std::unique_ptr<RetrievalIndex> retrieve_;
    mutable bool retrieve_tried_ = false;
};

} // namespace gai
