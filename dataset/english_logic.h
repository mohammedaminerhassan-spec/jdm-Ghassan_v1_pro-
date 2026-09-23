#pragma once

// dataset/english_logic.h — English dialogue logic for the Hermes lake.
//
// The Hermes parquet lake (522k chat + 478k instruction) teaches content,
// but NOT dialogue discipline: when something is a question, how to answer,
// which connectors hold the reasoning together, when to refuse or say
// "I don't know". This module is that contract (the English twin of the
// Darija governor/reasoning exchanges).
//
// Used by:
//   - data_pipeline style gate (assistant turns must obey answer discipline)
//   - evaluation/benchmark EN suite (question-type accuracy)
//   - inference/chat persona (system prompt mirrors these rules)
// Pure C++, no dependencies beyond core.

#include <string>
#include <vector>

namespace gai {
namespace english_logic {

enum class DialogAct : unsigned char {
    Greeting = 0,
    Question = 1,     // ends with ? or starts with who/what/when/...
    Instruction = 2,  // starts with write/implement/explain/... (task obedience)
    Coding = 3,       // contains code markers (def / ``` / function ...)
    Reasoning = 4,    // logic/math: calculate, how many, prove, steps
    Chitchat = 5,     // everything else conversational
    Unknown = 6,
};

const char* dialog_act_name(DialogAct a);

// Rule-based classifier (fast, deterministic, no model needed).
// Mirrors the converter heuristic so train-time and eval-time agree.
DialogAct classify_dialog_act(const std::string& user_text);
bool is_question(const std::string& user_text);
bool is_instruction(const std::string& user_text);
bool is_greeting(const std::string& user_text);
bool is_coding(const std::string& user_text);

// Connectors the model MUST use to hold English reasoning together
// (rawabit): causal, contrast, sequence. Checked by the EN benchmark.
const std::vector<const char*>& causal_connectors();    // because, therefore, so, ...
const std::vector<const char*>& contrast_connectors();  // however, but, although, ...
const std::vector<const char*>& sequence_connectors();  // first, second, then, finally, ...

// Answer discipline (anti-hallucination contract):
//   1. lead with the direct answer, then explain briefly
//   2. never guess: say "I don't know" when unsure
//   3. never break character with AI-disclosure boilerplate
//   4. multiple-choice: single letter + short justification
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

} // namespace english_logic
} // namespace gai
