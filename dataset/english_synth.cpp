// dataset/english_synth.cpp — English dialogue-behavior synthesizer.
//
// Twin of the Darija SynthGenerator (dataset/synth.cpp), simplified: no
// script/transliteration logic, no MSA/French branches. The composition rules
// are the same (domain opener → followups → governor/correction/identity
// seasoning → closer), and the diversity/style filters are the same shape:
// template-use cap, exact+near dedup, first-token flattening. The assistant
// style gate is english_logic::check_english_reply (answer discipline) instead
// of the Darija anti-robotic gate.
#include "dataset/english_synth.h"
#include "dataset/english_dialogue_data.h"
#include "dataset/english_logic.h"
#include "dataset/dedup.h"
#include "core/rng.h"

#include <map>

namespace gai {
namespace english_synth {

namespace {

template <typename T>
const T& pick(const std::vector<T>& v, Rng& rng) {
    return v[rng.below(v.size())];
}

std::string first_words(const std::string& s, int n) {
    std::string o;
    int words = 0;
    for (size_t i = 0; i < s.size() && words < n; ++i) {
        if (s[i] == ' ' || s[i] == '\t' || s[i] == '\n') {
            if (!o.empty() && o.back() != ' ') {
                o.push_back(' ');
                if (++words >= n) break;
            }
        } else {
            o.push_back(s[i]);
        }
    }
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o;
}

} // namespace

struct EnglishSynthGenerator::Impl {
    Rng rng;
    Deduplicator dedup;
    std::map<u64, int> template_uses;
    std::map<std::string, int> first_token_counts;
    u64 total_accepted = 0;

    explicit Impl(u64 seed) : rng(seed) {
        Deduplicator::Config dc;
        dc.jaccard_threshold = 0.80;
        dedup = Deduplicator(dc);
    }
};

EnglishSynthGenerator::EnglishSynthGenerator(EnglishSynthConfig cfg)
    : impl_(std::make_unique<Impl>(cfg.seed)), cfg_(cfg) {}

EnglishSynthGenerator::~EnglishSynthGenerator() = default;

int EnglishSynthGenerator::domain_count() {
    return static_cast<int>(english_dialogue_data::domains().size());
}

const char* EnglishSynthGenerator::domain_name(int i) {
    const auto& d = english_dialogue_data::domains();
    if (i < 0 || static_cast<size_t>(i) >= d.size()) return "";
    return d[static_cast<size_t>(i)].name;
}

bool EnglishSynthGenerator::generate(Conversation& out) {
    namespace ed = english_dialogue_data;
    ++stats_.generated;
    Rng& rng = impl_->rng;

    const auto& domains = ed::domains();
    if (domains.empty()) return false;
    const size_t di = rng.below(domains.size());
    const auto& D = domains[di];
    out.domain = D.name;
    out.script = Script::Latin;
    out.messages.clear();

    if (cfg_.include_system && rng.uniform() < cfg_.p_system)
        out.messages.push_back({Role::System, pick(ed::system_prompts(), rng)});

    const int turns = cfg_.min_turns +
        static_cast<int>(rng.below(static_cast<u64>(cfg_.max_turns - cfg_.min_turns + 1)));

    u64 tid = di * 1000003ull;

    auto push_turn = [&](const char* u, const char* a) {
        out.messages.push_back({Role::User, u});
        std::string asst = a;
        if (rng.uniform() < cfg_.p_filler && !ed::fillers().empty())
            asst = std::string(pick(ed::fillers(), rng)) + " " + asst;
        out.messages.push_back({Role::Assistant, asst});
        tid = tid * 31 + hash_string(u);
    };

    if (rng.uniform() < cfg_.p_greeting) {
        push_turn(pick(ed::greetings_user(), rng), pick(ed::greetings_assistant(), rng));
    }
    int produced = out.messages.empty() ? 0 : 1;

    while (produced < turns) {
        const double p = rng.uniform();
        if (p < cfg_.p_correction && produced > 0 && !ed::corrections().empty()) {
            const auto& e = pick(ed::corrections(), rng);
            push_turn(e.user, e.assistant);
        } else if (p < cfg_.p_correction + cfg_.p_misunderstand && !ed::misunderstandings().empty()) {
            const auto& e = pick(ed::misunderstandings(), rng);
            push_turn(e.user, e.assistant);
        } else if (p < cfg_.p_correction + cfg_.p_misunderstand + cfg_.p_governor &&
                   !ed::governor_exchanges().empty()) {
            const auto& e = pick(ed::governor_exchanges(), rng);
            push_turn(e.user, e.assistant);
        } else if (p < cfg_.p_correction + cfg_.p_misunderstand + cfg_.p_governor +
                          cfg_.p_reasoning &&
                   !ed::reasoning_exchanges().empty()) {
            const auto& e = pick(ed::reasoning_exchanges(), rng);
            push_turn(e.user, e.assistant);
        } else if (rng.uniform() < cfg_.p_identity && !ed::identity_questions().empty()) {
            const auto& e = pick(ed::identity_questions(), rng);
            push_turn(e.user, e.assistant);
        } else if (produced > 0 && rng.uniform() < cfg_.p_followup && !D.followups.empty()) {
            const auto& e = pick(D.followups, rng);
            push_turn(e.user, e.assistant);
        } else if (!D.openers.empty()) {
            const auto& e = pick(D.openers, rng);
            push_turn(e.user, e.assistant);
        } else {
            break;
        }
        ++produced;

        if (produced < turns && rng.uniform() < cfg_.p_backchannel && !ed::backchannels_user().empty()) {
            const std::string bc = pick(ed::backchannels_user(), rng);
            const auto& e = D.followups.empty() ? pick(D.openers, rng) : pick(D.followups, rng);
            out.messages.push_back({Role::User, bc});
            out.messages.push_back({Role::Assistant, e.assistant});
            ++produced;
        }
    }

    if (rng.uniform() < cfg_.p_closer) {
        out.messages.push_back({Role::User, pick(ed::closers_user(), rng)});
        out.messages.push_back({Role::Assistant, pick(ed::closers_assistant(), rng)});
    }

    out.template_id = tid;

    if (++impl_->template_uses[tid] > cfg_.max_template_uses) {
        ++stats_.rejected_duplicate;
        return false;
    }

    std::string flat;
    for (const auto& m : out.messages) {
        flat += (m.role == Role::User ? "U:" : m.role == Role::Assistant ? "A:" : "S:");
        flat += m.content;
        flat += "\n";
    }
    if (!impl_->dedup.add(flat)) {
        ++stats_.rejected_duplicate;
        return false;
    }

    // Answer-discipline gate: our own authored data must pass the same filter
    // the pipeline applies to third-party assistant turns. A rejection here
    // means the authored row contradicts english_logic — fix the DATA, because
    // the gate is the contract the benchmark enforces.
    for (size_t i = 0; i < out.messages.size(); ++i) {
        if (out.messages[i].role != Role::Assistant) continue;
        const std::string user_msg = (i > 0) ? out.messages[i - 1].content : "";
        const english_logic::ReplyReport rep =
            english_logic::check_english_reply(user_msg, out.messages[i].content);
        if (!rep.disciplined) {
            ++stats_.rejected_style;
            return false;
        }
    }

    for (const auto& m : out.messages) {
        if (m.role != Role::Assistant) continue;
        const std::string head = first_words(m.content, 2);
        int& c = impl_->first_token_counts[head];
        ++c;
        const u64 total = impl_->total_accepted + 1;
        if (total > 200 && static_cast<double>(c) / static_cast<double>(total) > cfg_.max_head_share) {
            ++stats_.rejected_entropy;
            return false;
        }
    }

    ++impl_->total_accepted;
    ++stats_.accepted;
    stats_.by_domain[out.domain]++;
    stats_.by_script["latin"]++;
    return true;
}

std::vector<Conversation> EnglishSynthGenerator::generate_many(int n) {
    GAI_CHECK(n >= 0, "english synth conversation count must be >= 0");
    GAI_CHECK(cfg_.max_attempts_multiplier > 0, "english synth attempts multiplier must be > 0");
    GAI_CHECK(cfg_.min_turns > 0 && cfg_.max_turns >= cfg_.min_turns && cfg_.max_turns <= 1024,
              "english synth turn range is invalid");
    std::vector<Conversation> out;
    out.reserve(static_cast<size_t>(n));
    i64 attempts = 0;
    const i64 max_attempts =
        static_cast<i64>(n) * static_cast<i64>(cfg_.max_attempts_multiplier) + 5000;
    while (static_cast<i64>(out.size()) < n && attempts < max_attempts) {
        ++attempts;
        Conversation c;
        if (generate(c)) out.push_back(std::move(c));
    }
    if (static_cast<int>(out.size()) < n) {
        log_warn(strfmt("english-synth: produced %zu/%d conversations before the diversity filters "
                        "saturated (this is the guard working, not a bug)", out.size(), n));
    }
    return out;
}

} // namespace english_synth
} // namespace gai
