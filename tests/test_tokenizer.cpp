#include "tokenizer/bpe_trainer.h"
#include "tokenizer/chat_template.h"
#include "core/unicode.h"

#include <iostream>
#include <cassert>
#include <vector>

using namespace gai;

static int g_fail = 0;
#define CHECK(cond, name) do { \
    if (cond) { std::cout << "  [ok]   " << name << "\n"; } \
    else      { std::cout << "  [FAIL] " << name << "\n"; ++g_fail; } } while (0)

static std::vector<std::string> darija_corpus() {
    return {
        "شنو خبارك اليوم؟",
        "الحمد لله بخير، نتا كيف داير؟",
        "سلام عليكم، واش كلشي مزيان؟",
        "بغيت نمشي للدار دابا، عييت بزاف",
        "واخا، غادي نتلاقاو غدا فالصباح",
        "chno khbark a sahbi",
        "labas 3lik? ana bikhir hamdollah",
        "3lach ma jiti l dar lbarah?",
        "kif dayr m3a lkhedma dyalk",
        "wach nqdr ndir hadchi bzaf?",
        "بغيت نdownloadi هاد الفيديو",
        "شنو رأيك فهاد app؟",
        "هاد الحاجة زوينة بزاف والله",
        "دابا غادي نشوف شنو خاصني ندير",
        "ماكاينش مشكل، غير قولي فوقاش",
        "السلام عليكم ورحمة الله وبركاته",
        "أنا بخير الحمد لله، شكرا على السؤال",
        "الطقس اليوم مزيان بزاف فالدار البيضاء",
        "خاصني نمشي للسوق نشري شي خضرة",
        "واش عندك شي فكرة على هاد الموضوع؟",
        "مرحبا بيك، كيفاش نقدر نعاونك؟",
        "هادشي صعيب شوية ولكن ممكن",
        "غادي نتصنط ليك من بعد إن شاء الله",
        "smiti ghassan w ana men casa",
        "fin katskon a khouya?",
        "makanch mochkil, kolchi mzyan",
        "daba nchoufo chno ghadi ykoun",
        "bzaf dyal lmaghariba kaydwiw darija",
        "الدارجة المغربية فيها بزاف ديال الكلمات",
        "كنبغي ناكل الطاجين والكسكس",
        "غدا الجمعة، غادي ناكلو الكسكس",
        "المغرب بلاد زوينة بزاف وفيها ناس مزيانين",
        "شحال هادي؟ بشحال كاتبيع هادي؟",
        "عافاك عطيني واحد الكيلو ديال الطماطم",
        "الله يعطيك الصحة أخويا",
        "بسلامة، تهلا فراسك",
        "merci bzaf 3la l3awn dyalk",
        "je pense que hadchi mzyan",
        "خدمت اليوم بزاف وعييت",
        "واش كاين شي حل لهاد المشكل؟",
    };
}

int main() {
    std::cout << "== tokenizer tests ==\n";
    set_log_level(LogLevel::Warn);

    // ---------------- unicode / normalizer
    {
        Normalizer norm{NormalizerConfig{}};
        std::string s = norm.normalize("السَّلامُ عَلَيْكُمْ");
        CHECK(s == "السلام عليكم", "diacritic stripping");

        CHECK(norm.normalize("مـــرحبا") == "مرحبا", "tatweel stripping");
        CHECK(norm.normalize("عندي ٣ كتب") == "عندي 3 كتب", "arabic digits -> ascii");
        CHECK(norm.normalize("Hello   World") == "hello world", "lowercase + ws collapse");
        CHECK(norm.normalize("سلااااااام") == "سلااام", "repeat collapse");

        auto st = script_stats("شنو خبارك chno khbark");
        CHECK(st.arabic > 0 && st.latin > 0, "script stats mixed detection");
    }

    // ---------------- pre-tokenizer
    {
        auto chunks = pre_tokenize("3lach ma jiti");
        CHECK(chunks.size() == 3, "arabizi word kept whole (3lach)");
        CHECK(!chunks.empty() && chunks[0].text == "3lach", "arabizi leading digit glued");
        CHECK(chunks.size() > 1 && chunks[1].text == " ma", "leading space glued to word");

        auto ch2 = pre_tokenize("kif7alk?");
        CHECK(ch2.size() == 2 && ch2[0].text == "kif7alk", "arabizi inner digit glued");
    }

    // ---------------- BPE training + round trip
    {
        BpeTrainerConfig cfg;
        cfg.vocab_size = 1200;
        cfg.min_frequency = 1;
        cfg.verbose = false;
        BpeTrainer tr(cfg);
        auto corpus = darija_corpus();
        for (int rep = 0; rep < 30; ++rep)
            for (const auto& line : corpus) tr.add_text(line);

        Tokenizer tk = tr.train();
        CHECK(tk.vocab_size() == 1200, "vocab size reached target");
        CHECK(tk.vocab_size() > 272, "vocab has learned merges beyond bytes");

        // exact round trip on all inputs (byte fallback guarantees this)
        bool ok = true;
        Normalizer nrm{tk.normalizer_config()};
        for (const auto& line : corpus) {
            auto ids = tk.encode(line);
            std::string back = tk.decode(ids);
            if (back != nrm.normalize(line)) { ok = false; std::cout << "    mismatch: " << line << " -> " << back << "\n"; }
        }
        CHECK(ok, "exact round-trip on darija corpus");

        // unseen text must still round trip (byte fallback, no <unk>)
        std::string weird = "zzz_ünseen_🎉_中文_٣٤٥";
        auto wids = tk.encode(weird);
        bool no_unk = true;
        for (i32 id : wids) if (id == special::UNK) no_unk = false;
        CHECK(no_unk, "never emits <unk>");
        CHECK(tk.decode(wids) == nrm.normalize(weird), "round-trip on unseen text");

        // merges actually compress
        auto f = tk.measure(corpus);
        std::cout << "    fertility: " << f.tokens_per_word << " tok/word, "
                  << f.bytes_per_token << " bytes/tok\n";
        CHECK(f.bytes_per_token > 1.5, "BPE compresses beyond raw bytes");

        // save/load
        const std::string path = "test_tok.gtok";
        tk.save(path);
        Tokenizer tk2;
        CHECK(tk2.load(path), "tokenizer load");
        CHECK(tk2.vocab_size() == tk.vocab_size(), "vocab size preserved");
        bool same = true;
        for (const auto& line : corpus) {
            if (tk.encode(line) != tk2.encode(line)) { same = false; break; }
        }
        CHECK(same, "encoding identical after save/load");
        std::remove(path.c_str());

        // special tokens
        auto sp = tk.encode_with_specials("<|user|>سلام<|end|>");
        CHECK(sp.size() >= 3 && sp.front() == special::USER && sp.back() == special::END,
              "special token parsing");

        // chat template + loss mask
        std::vector<Message> msgs = {
            {Role::System, ChatTemplate::default_system()},
            {Role::User, "شنو خبارك؟"},
            {Role::Assistant, "الحمد لله بخير، نتا كيف داير؟"},
        };
        std::vector<u8> mask;
        auto cids = ChatTemplate::encode(tk, msgs, false, &mask);
        CHECK(cids.size() == mask.size(), "chat mask length matches ids");
        size_t supervised = 0;
        for (u8 m : mask) supervised += m;
        CHECK(supervised > 0 && supervised < cids.size(), "only assistant span supervised");
        CHECK(cids[0] == special::BOS, "chat starts with BOS");
        CHECK(mask[0] == 0, "BOS not supervised");

        // streaming decode never breaks a codepoint
        Tokenizer::Stream st(tk);
        std::string streamed;
        auto ids = tk.encode("الحمد لله بخير 🎉");
        for (i32 id : ids) {
            std::string piece = st.push(id);
            if (!utf8_is_complete(piece)) { streamed = "<BROKEN>"; break; }
            streamed += piece;
        }
        streamed += st.flush();
        CHECK(streamed == tk.decode(ids), "streaming decode equals batch decode");
    }

    std::cout << (g_fail == 0 ? "== all tokenizer tests passed ==\n"
                              : "== FAILURES: " + std::to_string(g_fail) + " ==\n");
    return g_fail == 0 ? 0 : 1;
}
