// English dialogue-behavior data + generator regression.
//
// Covers the English twin of the Darija behavior stack:
//   * dataset/english_dialogue_data.h: every pool is non-empty, every Exchange
//     has non-empty user+assistant, domains have the contracted shape, and no
//     entry is an obvious template duplicate.
//   * dataset/english_synth.h: the generator produces well-formed multi-turn
//     conversations whose assistant turns all pass the english_logic answer-
//     discipline gate (the same gate the pipeline enforces on third-party data).
//   * english_logic::is_coding: the "return" false positive (shopping "return
//     a product" misclassified as Coding) stays fixed.
#include "dataset/english_dialogue_data.h"
#include "dataset/english_logic.h"
#include "dataset/english_synth.h"

#include <iostream>
#include <set>
#include <string>

using namespace gai;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++failures; } \
} while (0)

static bool non_empty(const char* s) { return s && s[0] != '\0'; }

int main() {
    namespace ed = english_dialogue_data;

    // ---- 1. pool shapes --------------------------------------------------
    {
        const auto& ds = ed::domains();
        CHECK(ds.size() >= 10, "english dialogue has 10+ behavior domains");
        size_t openers = 0, followups = 0;
        std::set<std::string> names;
        for (const auto& d : ds) {
            CHECK(non_empty(d.name), "domain has a name");
            CHECK(names.insert(d.name).second, std::string("domain name unique: ") + d.name);
            CHECK(d.openers.size() >= 10, std::string("domain has 10+ openers: ") + d.name);
            CHECK(d.followups.size() >= 4, std::string("domain has 4+ followups: ") + d.name);
            for (const auto& e : d.openers) {
                CHECK(non_empty(e.user) && non_empty(e.assistant), "opener complete");
                ++openers;
            }
            for (const auto& e : d.followups) {
                CHECK(non_empty(e.user) && non_empty(e.assistant), "followup complete");
                ++followups;
            }
        }
        CHECK(openers >= 100, "100+ authored openers total");
        CHECK(followups >= 40, "40+ authored followups total");
        CHECK(ed::governor_exchanges().size() >= 20, "20+ governor exchanges");
        CHECK(ed::reasoning_exchanges().size() >= 15, "15+ reasoning exchanges");
        CHECK(ed::identity_questions().size() >= 12, "12+ identity questions");
        CHECK(ed::corrections().size() >= 10, "10+ corrections");
        CHECK(ed::misunderstandings().size() >= 10, "10+ misunderstandings");
        CHECK(ed::greetings_user().size() >= 15, "15+ user greetings");
        CHECK(ed::greetings_assistant().size() >= 15, "15+ assistant greetings");
        CHECK(ed::closers_user().size() >= 10, "10+ user closers");
        CHECK(ed::closers_assistant().size() >= 10, "10+ assistant closers");
        CHECK(ed::fillers().size() >= 15, "15+ fillers");
        CHECK(ed::backchannels_user().size() >= 15, "15+ backchannels");
        CHECK(ed::system_prompts().size() >= 5, "5+ system prompts");
    }

    // ---- 2. no template feel: user turns must be mostly distinct -----------
    {
        std::set<std::string> users;
        size_t total = 0;
        auto feed = [&](const std::vector<ed::Exchange>& v) {
            for (const auto& e : v) {
                ++total;
                users.insert(e.user);
            }
        };
        for (const auto& d : ed::domains()) {
            feed(d.openers);
            feed(d.followups);
        }
        feed(ed::governor_exchanges());
        feed(ed::reasoning_exchanges());
        // >90% distinct user turns: paraphrase-level variety, not slot filling.
        CHECK(users.size() * 100 >= total * 90, "authored user turns are overwhelmingly distinct");
    }

    // ---- 3. every authored assistant turn passes the discipline gate -------
    // (mirrors the generator's own gate; a failure here means the DATA
    // contradicts english_logic and the data — not the gate — must change).
    {
        size_t n = 0, bad = 0;
        auto feed = [&](const std::vector<ed::Exchange>& v, const char* tag) {
            for (const auto& e : v) {
                ++n;
                const english_logic::ReplyReport r =
                    english_logic::check_english_reply(e.user, e.assistant);
                if (!r.disciplined) {
                    ++bad;
                    std::cerr << "FAIL: undisciplined authored turn [" << tag
                              << "] user='" << e.user << "' reason=" << r.reason << "\n";
                    ++failures;
                }
            }
        };
        for (const auto& d : ed::domains()) {
            feed(d.openers, d.name);
            feed(d.followups, d.name);
        }
        feed(ed::governor_exchanges(), "governor");
        feed(ed::reasoning_exchanges(), "reasoning");
        feed(ed::identity_questions(), "identity");
        feed(ed::corrections(), "corrections");
        feed(ed::misunderstandings(), "misunderstandings");
        CHECK(n > 250 && bad == 0, "all authored assistant turns obey answer discipline");
    }

    // ---- 4. the "return" classifier fix ------------------------------------
    {
        CHECK(!english_logic::is_coding("How do I return something without the receipt?"),
              "shopping 'return' is not code");
        CHECK(!english_logic::is_coding("What is your return policy?"),
              "return policy is not code");
        CHECK(english_logic::is_coding("What does return do in python?"),
              "python return is code");
        CHECK(english_logic::is_coding("Explain the return statement with an example"),
              "return statement is code");
        CHECK(english_logic::is_coding("def foo(): return 42"),
              "def/return snippet is code");
        CHECK(english_logic::is_coding("Write a function that sorts a list"),
              "function request is code");
    }

    // ---- 5. the generator composes valid conversations --------------------
    {
        CHECK(english_synth::EnglishSynthGenerator::domain_count() >= 10,
              "generator sees 10+ domains");
        CHECK(std::string(english_synth::EnglishSynthGenerator::domain_name(0)) ==
                  ed::domains()[0].name,
              "generator domain names match the data");
        english_synth::EnglishSynthConfig cfg;
        cfg.seed = 7;
        cfg.num_conversations = 60;
        english_synth::EnglishSynthGenerator gen(cfg);
        const std::vector<Conversation> convs = gen.generate_many(cfg.num_conversations);
        CHECK(convs.size() == static_cast<size_t>(cfg.num_conversations),
              "generator produces the requested conversations");
        for (const auto& c : convs) {
            CHECK(!c.domain.empty(), "conversation has a domain");
            CHECK(c.messages.size() >= 2, "conversation has at least one turn");
            // strict alternation starting with user (after an optional system)
            size_t i = 0;
            if (!c.messages.empty() && c.messages[0].role == Role::System) ++i;
            bool expect_user = true;
            for (; i < c.messages.size(); ++i) {
                const Role want = expect_user ? Role::User : Role::Assistant;
                if (c.messages[i].role != want) {
                    CHECK(false, "conversation turns strictly alternate user/assistant");
                    break;
                }
                CHECK(!c.messages[i].content.empty(), "no empty message content");
                expect_user = !expect_user;
            }
            // every assistant turn passes the discipline gate (belt and braces:
            // generate() already enforces this, so a failure is a generator bug)
            for (size_t k = 0; k < c.messages.size(); ++k) {
                if (c.messages[k].role != Role::Assistant) continue;
                const std::string prev =
                    (k > 0) ? c.messages[k - 1].content : std::string();
                const english_logic::ReplyReport r =
                    english_logic::check_english_reply(prev, c.messages[k].content);
                if (!r.disciplined) {
                    CHECK(false, std::string("generated turn undisciplined: ") + r.reason);
                    break;
                }
            }
        }
    }

    if (failures == 0) {
        std::cout << "test_english_dialogue: ALL PASS\n";
        return 0;
    }
    std::cerr << "test_english_dialogue: " << failures << " FAILURES\n";
    return 1;
}
