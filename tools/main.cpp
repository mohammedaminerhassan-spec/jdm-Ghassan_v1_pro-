// ghassan-ai - the main CLI: chat, generate, info, quantize, bench, tokenize, eval.

#include "tools/cli_common.h"
#include "core/device.h"
#include "format/gai_format.h"
#include "format/gguf_format.h"
#include "inference/chat.h"
#include "quantization/quantize.h"
#include "evaluation/benchmark.h"
#include "training/checkpoint.h"
#include "dataset/retrieval.h"

#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace gai;

static void usage() {
    std::cout <<
    "ghassan-ai - Moroccan Darija conversational model\n\n"
    "usage:\n"
    "  ghassan-ai chat      --model model.gai\n"
    "  ghassan-ai generate  --model model.gai --prompt \"شنو هو C++؟\"\n"
    "  ghassan-ai info      --model model.gai\n"
    "  ghassan-ai quantize  --model in.gai --out out.gai --profile int4\n"
    "  ghassan-ai bench     --model model.gai [--tokens 128]\n"
    "  ghassan-ai tokenize  --tokenizer tok.gtok --text \"شنو خبارك\"\n"
    "  ghassan-ai eval      --model model.gai --suite evaluation/datasets\n"
    "  ghassan-ai export    --checkpoint last.ckpt --tokenizer tok.gtok --out model.gai\n"
    "  ghassan-ai devices\n\n"
    "generation options:\n"
    "  --temp <f>        temperature            [0.8]\n"
    "  --top-k <n>       top-k                  [40]\n"
    "  --top-p <f>       nucleus                [0.92]\n"
    "  --min-p <f>       min-p                  [0.05]\n"
    "  --repeat <f>      repetition penalty     [1.12]\n"
    "  --max-tokens <n>  max new tokens         [256]\n"
    "  --seed <n>        rng seed (0 = random)\n"
    "  --greedy          deterministic decoding\n"
    "  --system <text>   system prompt\n"
    "  --no-stream       print the reply at once\n"
    "  --device auto|cpu|cuda\n"
    "  --retrieve-index <path>  RAG grounding: QA dir or train-*.json (paraphrase-aware)\n"
    "  --retrieve-threshold <f> min score to use retrieved answer [3.0]\n"
    "  --retrieve-augment       inject hit as context instead of direct answer\n";
}

// ---------------------------------------------------------------- helpers
struct LoadedModel {
    std::unique_ptr<Model> model;
    Tokenizer tokenizer;
    ModelConfig cfg;
    std::string quant = "fp32";
    u64 file_bytes = 0;
};

static std::string ggml_type_str(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q4_0: return "Q4_0";
        default: return strfmt("t%u", static_cast<unsigned>(t));
    }
}

static u64 ggml_nbytes_of(GGMLType t, u64 nelems) {
    switch (t) {
        case GGMLType::F32:  return nelems * 4;
        case GGMLType::F16:  return nelems * 2;
        case GGMLType::Q8_0: return ((nelems + 31) / 32) * 34;
        case GGMLType::Q4_0: return ((nelems + 31) / 32) * 18;
        default:             return nelems * 4;
    }
}

static bool is_gguf_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    u32 magic = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    return magic == GGUF_MAGIC;
}

static Device pick_device(const Args& a) {
    std::string d = a.str("device", "auto");
    if (d == "cpu") return Device::CPU;
    if (d == "cuda") {
        GAI_CHECK(cuda_available(), "--device cuda requested but no CUDA device is available (T4-only build)");
        return Device::CUDA;
    }
    if (d == "metal" || d == "vulkan" || d == "tpu") {
        GAI_FAIL("--device " + d + " is not supported: this is a T4-only build (cpu|cuda). "
                 "Use --device cuda on Kaggle T4 or --device cpu locally");
    }
    if (d != "auto") GAI_FAIL("unknown --device '" + d + "' (T4-only: auto|cpu|cuda)");
    return best_device();
}

static LoadedModel load_model(const Args& args) {
    LoadedModel lm;
    std::string path = args.str("model");
    GAI_CHECK(!path.empty(), "--model is required");
    GAI_CHECK(fs::exists(path), "model file not found: " + path);

    // ---- format detection: GGUF magic ("GGUF") vs .gai magic
    std::ifstream probe(path, std::ios::binary);
    u32 magic = 0;
    probe.read(reinterpret_cast<char*>(&magic), 4);
    const bool is_gguf = (magic == GGUF_MAGIC);

    Device dev = pick_device(args);

    if (is_gguf) {
        GGUFReader r;
        GAI_CHECK(r.open(path), "not a valid .gguf model file: " + path);
        lm.cfg = r.model_config();
        lm.quant = r.get_string("general.quantization_version", "f16");
        lm.file_bytes = r.file_size();

        // GGUF embeds no BPE merge rules -> the .gtok file is required.
        // Look for it next to the model (artifacts/...), else --tokenizer.
        if (args.has("tokenizer")) {
            GAI_CHECK(lm.tokenizer.load(args.str("tokenizer")),
                      "cannot load tokenizer: " + args.str("tokenizer"));
        } else {
            std::string def = (fs::path(path).parent_path() / "tokenizer" / "darija.gtok").string();
            if (!fs::exists(def)) def = "artifacts/tokenizer/darija.gtok";
            GAI_CHECK(lm.tokenizer.load(def),
                      "GGUF model has no embedded tokenizer; pass --tokenizer "
                      "(e.g. artifacts/tokenizer/darija.gtok)");
        }
        GAI_CHECK(lm.tokenizer.vocab_size() == lm.cfg.vocab_size,
                  strfmt("tokenizer/model vocab mismatch: %d vs %d",
                         lm.tokenizer.vocab_size(), lm.cfg.vocab_size));

        Timer t;
        lm.model = std::make_unique<Model>(lm.cfg, dev);
        GAI_CHECK(load_model_from_gguf(path, *lm.model),
                  "failed to load weights from: " + path);
        log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s",
                        path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                        human_duration(t.seconds()).c_str(), device_name(dev)));
        return lm;
    }

    GaiReader r;
    GAI_CHECK(r.open(path), "not a valid .gai model file: " + path);
    lm.cfg = r.model_config();
    lm.quant = r.meta_str("quant_profile", "fp32");
    lm.file_bytes = r.file_size();

    if (args.has("tokenizer")) {
        GAI_CHECK(lm.tokenizer.load(args.str("tokenizer")),
                  "cannot load tokenizer: " + args.str("tokenizer"));
    } else {
        GAI_CHECK(r.has_tokenizer() && r.load_tokenizer(lm.tokenizer),
                  "the model has no embedded tokenizer; pass --tokenizer");
    }
    GAI_CHECK(lm.tokenizer.vocab_size() == lm.cfg.vocab_size,
              strfmt("tokenizer/model vocab mismatch: %d vs %d",
                     lm.tokenizer.vocab_size(), lm.cfg.vocab_size));

    Timer t;
    lm.model = std::make_unique<Model>(lm.cfg, dev);
    for (Parameter* p : lm.model->parameters()) {
        Tensor tt = r.read_tensor_f32(p->name);
        GAI_CHECK(tt.numel() == p->numel(), "tensor size mismatch: " + p->name);
        p->w.copy_from(tt);
    }
    log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s",
                    path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                    human_duration(t.seconds()).c_str(), device_name(dev)));
    return lm;
}

static GenerationConfig make_gen_config(const Args& a) {
    GenerationConfig g;
    g.max_new_tokens = static_cast<int>(a.num("max-tokens", 256));
    g.sampling.temperature = static_cast<float>(a.real("temp", 0.8));
    g.sampling.top_k = static_cast<int>(a.num("top-k", 40));
    g.sampling.top_p = static_cast<float>(a.real("top-p", 0.92));
    g.sampling.min_p = static_cast<float>(a.real("min-p", 0.05));
    g.sampling.repetition_penalty = static_cast<float>(a.real("repeat", 1.12));
    g.sampling.seed = static_cast<u64>(a.num("seed", 0));
    g.sampling.greedy = a.flag("greedy", false);
    return g;
}

// ---------------------------------------------------------------- commands
static int cmd_chat(const Args& args) {
    LoadedModel lm = load_model(args);
    Generator gen(*lm.model, lm.tokenizer, static_cast<int>(args.num("ctx", 0)));

    ChatOptions opts;
    opts.gen = make_gen_config(args);
    opts.stream = !args.flag("no-stream");
    opts.show_stats = args.flag("stats");
    if (args.has("system")) opts.system = args.str("system");
    if (args.has("retrieve-index")) opts.retrieve_index = args.str("retrieve-index");
    if (args.has("retrieve-threshold")) opts.retrieve_threshold = args.real("retrieve-threshold", 3.0);
    if (args.flag("retrieve-augment")) opts.retrieve_direct = false;

    ChatSession session(gen, opts);
    session.run_repl();
    return 0;
}

static int cmd_generate(const Args& args) {
    // RAG single-turn: same paraphrase logic as chat, but without REPL
    if (args.has("retrieve-index")) {
        std::string idx_path = args.str("retrieve-index");
        double thr = args.real("retrieve-threshold", 3.0);
        std::string prompt = args.str("prompt");
        if (prompt.empty() && !args.positional().empty() && args.positional().size() > 1)
            prompt = args.positional()[1];
        if (!prompt.empty()) {
            std::vector<QaEntry> docs;
            if (load_qa_dir(idx_path, docs) > 0 || load_qa_json(idx_path, docs)) {
                RetrievalIndex ridx;
                ridx.build(docs);
                auto hits = ridx.query(prompt, 1, thr);
                if (!hits.empty()) {
                    std::string ans = ridx.doc(hits[0].doc).answer;
                    std::cout << ans << std::endl;
                    if (args.flag("stats")) std::cerr << "  [retrieved id=" << ridx.doc(hits[0].doc).id
                                                      << " score=" << hits[0].score << "]\n";
                    return 0;
                }
            }
        }
        // no hit -> fall through to generative path
    }
    LoadedModel lm = load_model(args);
    Generator gen(*lm.model, lm.tokenizer, static_cast<int>(args.num("ctx", 0)));

    std::string prompt = args.str("prompt");
    if (prompt.empty() && !args.positional().empty() && args.positional().size() > 1) {
        prompt = args.positional()[1];
    }
    GAI_CHECK(!prompt.empty(), "--prompt is required");

    GenerationConfig g = make_gen_config(args);
    bool stream = !args.flag("no-stream");

    std::string out;
    if (args.flag("raw")) {
        // raw completion, no chat template
        out = gen.complete(prompt, g, stream ? Generator::StreamFn([](const std::string& p, i32) {
            std::cout << p << std::flush;
            return true;
        }) : nullptr);
    } else {
        std::vector<Message> msgs;
        msgs.push_back({Role::System, args.str("system", ChatTemplate::default_system())});
        msgs.push_back({Role::User, prompt});
        out = gen.chat(msgs, g, stream ? Generator::StreamFn([](const std::string& p, i32) {
            std::cout << p << std::flush;
            return true;
        }) : nullptr);
    }
    if (stream) std::cout << std::endl;
    else std::cout << out << std::endl;

    if (args.flag("stats")) std::cerr << "  " << gen.stats().summary() << "\n";
    return 0;
}

static int cmd_info(const Args& args) {
    std::string path = args.str("model");
    GAI_CHECK(!path.empty(), "--model is required");

    if (is_gguf_file(path)) {
        GGUFReader r;
        GAI_CHECK(r.open(path), "not a valid .gguf file: " + path);

        std::cout << "\n  file        : " << path << "\n";
        std::cout << "  size        : " << human_bytes(r.file_size()) << "\n";
        std::cout << "  format      : GGUF\n";
        std::cout << "  quant       : " << r.get_string("general.quantization_version", "?") << "\n";
        std::cout << "  tokenizer   : external .gtok (not embedded in GGUF)\n";
        std::cout << "\n  -- config --\n";
        std::cout << r.model_config().summary() << "\n";

        std::cout << "\n  -- tensors (" << r.tensors().size() << ") --\n";
        std::map<std::string, std::pair<u64, u64>> by_dtype;   // dtype -> (count, bytes)
        u64 total_bytes = 0, total_params = 0;
        for (const auto& e : r.tensors()) {
            i64 n = 1;
            for (u64 d : e.dimensions) n *= static_cast<i64>(d);
            total_params += static_cast<u64>(n);
            u64 nb = ggml_nbytes_of(e.type, static_cast<u64>(n));
            total_bytes += nb;
            auto& slot = by_dtype[ggml_type_str(e.type)];
            slot.first++;
            slot.second += nb;
        }
        for (const auto& [dt, s] : by_dtype)
            std::cout << strfmt("    %-6s %4llu tensors  %s\n", dt.c_str(),
                                (unsigned long long)s.first, human_bytes(s.second).c_str());
        std::cout << strfmt("\n  parameters  : %s (%.2f M)\n",
                            human_count(total_params).c_str(), double(total_params) / 1e6);
        std::cout << strfmt("  weight bytes: %s (%.2f bits/param)\n",
                            human_bytes(total_bytes).c_str(),
                            total_params ? 8.0 * double(total_bytes) / double(total_params) : 0.0);

        if (args.flag("tensors")) {
            std::cout << "\n  -- tensor list --\n";
            for (const auto& e : r.tensors()) {
                std::string dims;
                for (size_t i = 0; i < e.dimensions.size(); ++i) {
                    if (i) dims += "x";
                    dims += std::to_string(e.dimensions[i]);
                }
                i64 n = 1;
                for (u64 d : e.dimensions) n *= static_cast<i64>(d);
                std::cout << strfmt("    %-28s %-14s %-6s %10s\n", e.name.c_str(), dims.c_str(),
                                    ggml_type_str(e.type).c_str(),
                                    human_bytes(ggml_nbytes_of(e.type, static_cast<u64>(n))).c_str());
            }
        }
        std::cout << "\n";
        return 0;
    }

    GaiReader r;
    GAI_CHECK(r.open(path), "not a valid .gai file: " + path);

    std::cout << "\n  file        : " << path << "\n";
    std::cout << "  size        : " << human_bytes(r.file_size()) << "\n";
    std::cout << "  tokenizer   : " << (r.has_tokenizer() ? "embedded" : "external") << "\n";
    std::cout << "\n  -- metadata --\n";
    for (const auto& [k, v] : r.meta()) std::cout << strfmt("    %-20s %s\n", k.c_str(), v.c_str());

    std::cout << "\n  -- tensors (" << r.tensors().size() << ") --\n";
    std::map<std::string, std::pair<u64, u64>> by_dtype;   // dtype -> (count, bytes)
    u64 total_bytes = 0, total_params = 0;
    for (const auto& e : r.tensors()) {
        i64 n = 1;
        for (i64 d : e.dims) n *= d;
        total_params += static_cast<u64>(n);
        total_bytes += e.nbytes;
        auto& slot = by_dtype[dtype_name(e.dtype)];
        slot.first++;
        slot.second += e.nbytes;
    }
    for (const auto& [dt, s] : by_dtype)
        std::cout << strfmt("    %-6s %4llu tensors  %s\n", dt.c_str(),
                            (unsigned long long)s.first, human_bytes(s.second).c_str());
    std::cout << strfmt("\n  parameters  : %s (%.2f M)\n",
                        human_count(total_params).c_str(), double(total_params) / 1e6);
    std::cout << strfmt("  weight bytes: %s (%.2f bits/param)\n",
                        human_bytes(total_bytes).c_str(),
                        total_params ? 8.0 * double(total_bytes) / double(total_params) : 0.0);

    if (args.flag("tensors")) {
        std::cout << "\n  -- tensor list --\n";
        for (const auto& e : r.tensors()) {
            std::string dims;
            for (size_t i = 0; i < e.dims.size(); ++i) {
                if (i) dims += "x";
                dims += std::to_string(e.dims[i]);
            }
            std::cout << strfmt("    %-28s %-14s %-6s %10s\n", e.name.c_str(), dims.c_str(),
                                dtype_name(e.dtype), human_bytes(e.nbytes).c_str());
        }
    }
    std::cout << "\n";
    return 0;
}

static int cmd_quantize(const Args& args) {
    std::string in = args.str("model");
    std::string out = args.str("out");
    std::string prof = args.str("profile", "int4");
    GAI_CHECK(!in.empty() && !out.empty(), "--model and --out are required");

    GaiReader r;
    GAI_CHECK(r.open(in), "not a valid .gai file: " + in);
    ModelConfig cfg = r.model_config();
    ExportProfile p = profile_for(prof);

    log_info(strfmt("[quant] %s -> %s   profile=%s", in.c_str(), out.c_str(), prof.c_str()));
    log_info(strfmt("        %s", quant::profile_description(prof)));

    GaiWriter w(out);
    w.set_config(cfg);
    w.set_meta("quant_profile", prof);
    w.set_meta("default_dtype", dtype_name(p.default_dtype));
    w.set_meta("embedding_dtype", dtype_name(p.embedding_dtype));
    w.set_meta("norm_dtype", dtype_name(p.norm_dtype));
    for (const auto& [k, v] : r.meta()) {
        if (k.rfind("quant", 0) == 0 || k.rfind("default_", 0) == 0 ||
            k.rfind("embedding_", 0) == 0 || k.rfind("norm_d", 0) == 0) continue;
        w.set_meta(k, v);
    }

    // carry the embedded tokenizer over
    if (r.has_tokenizer()) {
        Tokenizer tk;
        if (r.load_tokenizer(tk)) {
            fs::path tmp = fs::temp_directory_path() / "gai_requant.gtok";
            tk.save(tmp.string());
            w.set_tokenizer_blob(tmp.string());
            std::error_code ec;
            fs::remove(tmp, ec);
        }
    }

    u64 before = 0, after = 0;
    double worst_rel = 0.0;
    std::string worst_name;
    for (const auto& e : r.tensors()) {
        before += e.nbytes;
        Tensor f32 = r.read_tensor_f32(e.name);
        bool is_norm = e.name.find("norm") != std::string::npos;
        bool is_emb = (e.name == "tok_embeddings" || e.name == "lm_head");
        DType dt = is_norm ? p.norm_dtype : is_emb ? p.embedding_dtype : p.default_dtype;
        if (!quant::is_quantizable(f32.numel(), dt)) dt = DType::F16;

        if (dt != DType::F32 && args.flag("measure")) {
            quant::QuantError qe = quant::measure_error(f32, dt);
            if (qe.rel_rmse > worst_rel) { worst_rel = qe.rel_rmse; worst_name = e.name; }
        }
        Tensor q = quant::quantize(f32, dt);
        after += q.nbytes();
        w.add_tensor(e.name, q);
    }
    w.write();

    log_info(strfmt("[quant] %s -> %s  (%.2fx smaller)",
                    human_bytes(before).c_str(), human_bytes(after).c_str(),
                    after ? double(before) / double(after) : 0.0));
    if (args.flag("measure")) {
        log_info(strfmt("[quant] worst relative RMSE: %.4f%% on %s",
                        worst_rel * 100.0, worst_name.c_str()));
    }
    return 0;
}

static int cmd_bench(const Args& args) {
    LoadedModel lm = load_model(args);
    Generator gen(*lm.model, lm.tokenizer, static_cast<int>(args.num("ctx", 512)));

    const int ntok = static_cast<int>(args.num("tokens", 64));
    const int nprompt = static_cast<int>(args.num("prompt-tokens", 32));

    std::vector<i32> prompt;
    prompt.push_back(special::BOS);
    for (int i = 1; i < nprompt; ++i) prompt.push_back(300 + (i * 37) % 2000);

    log_info(strfmt("[bench] prefill %d tokens, decode %d tokens on %s",
                    nprompt, ntok, device_name(lm.model->device())));

    gen.reset();
    gen.prefill(prompt);

    GenerationConfig g;
    g.max_new_tokens = ntok;
    g.stop_tokens.clear();
    g.sampling.greedy = true;
    gen.generate(g);

    const GenerationStats& s = gen.stats();
    std::cout << "\n  ---------------- benchmark ----------------\n";
    std::cout << strfmt("  device        : %s\n", device_name(lm.model->device()));
    std::cout << strfmt("  quantization  : %s\n", lm.quant.c_str());
    std::cout << strfmt("  parameters    : %.2f M\n", double(lm.model->num_parameters()) / 1e6);
    std::cout << strfmt("  prefill       : %d tok in %s = %.1f tok/s\n",
                        s.prompt_tokens, human_duration(s.prefill_sec).c_str(), s.prefill_tps());
    std::cout << strfmt("  decode        : %d tok in %s = %.1f tok/s\n",
                        s.new_tokens, human_duration(s.decode_sec).c_str(), s.decode_tps());
    std::cout << strfmt("  ms per token  : %.2f\n",
                        s.new_tokens ? 1000.0 * s.decode_sec / s.new_tokens : 0.0);
    std::cout << strfmt("  kv cache      : %s\n", human_bytes(gen.cache().bytes()).c_str());
    std::cout << "  -------------------------------------------\n\n";
    return 0;
}

static int cmd_tokenize(const Args& args) {
    Tokenizer tk;
    std::string tp = args.str("tokenizer");
    if (tp.empty() && args.has("model")) {
        GaiReader r;
        GAI_CHECK(r.open(args.str("model")), "cannot open model");
        GAI_CHECK(r.load_tokenizer(tk), "model has no embedded tokenizer");
    } else {
        GAI_CHECK(!tp.empty(), "--tokenizer or --model is required");
        GAI_CHECK(tk.load(tp), "cannot load tokenizer: " + tp);
    }

    std::string text = args.str("text");
    if (text.empty() && args.positional().size() > 1) text = args.positional()[1];
    GAI_CHECK(!text.empty(), "--text is required");

    auto ids = tk.encode(text);
    std::cout << "  input   : " << text << "\n";
    std::cout << "  tokens  : " << ids.size() << "\n  pieces  : ";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << " | ";
        std::cout << tk.token_text(ids[i]);
    }
    std::cout << "\n  ids     : ";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << " ";
        std::cout << ids[i];
    }
    std::cout << "\n  decoded : " << tk.decode(ids) << "\n";

    // word count for fertility
    auto f = tk.measure({text});
    std::cout << strfmt("  fertility: %.3f tokens/word, %.2f bytes/token\n",
                        f.tokens_per_word, f.bytes_per_token);
    return 0;
}

static int cmd_export(const Args& args) {
    std::string ckpt = args.str("checkpoint");
    std::string out  = args.str("out");
    GAI_CHECK(!ckpt.empty() && !out.empty(), "--checkpoint and --out are required");

    ModelConfig cfg;
    TrainState st;
    GAI_CHECK(Checkpoint::peek(ckpt, cfg, st), "cannot read checkpoint: " + ckpt);
    log_info(strfmt("[export] checkpoint at step %lld, val %.4f",
                    static_cast<long long>(st.step), st.best_val));

    Model model(cfg, Device::CPU);
    TrainState loaded;
    GAI_CHECK(Checkpoint::load(ckpt, model, static_cast<AdamW*>(nullptr), loaded), "cannot load checkpoint weights");
    model.print_parameter_report();

    std::string prof = args.str("profile", "fp16");
    export_model_gai(out, model, args.str("tokenizer"), profile_for(prof),
                     {{"steps", std::to_string(loaded.step)},
                      {"val_loss", strfmt("%.6f", loaded.best_val)}});
    return 0;
}

static int cmd_eval(const Args& args) {
    LoadedModel lm = load_model(args);
    Generator gen(*lm.model, lm.tokenizer, static_cast<int>(args.num("ctx", 1024)));

    BenchmarkConfig bc;
    bc.suite_dir = args.str("suite", "evaluation/datasets");
    bc.max_prompts = static_cast<int>(args.num("limit", 0));
    bc.categories = args.str("categories");
    bc.output_path = args.str("out");
    bc.verbose = args.flag("show", false);
    bc.gen = make_gen_config(args);

    Benchmark bench(gen, lm.tokenizer, bc);
    BenchmarkReport rep = bench.run();
    std::cout << rep.to_string();
    if (!bc.output_path.empty()) {
        rep.write_jsonl(bc.output_path);
        log_info("[eval] wrote " + bc.output_path);
    }
    return 0;
}

static int cmd_devices(const Args&) {
    print_device_report();
    return 0;
}

int main(int argc, char** argv) {
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);

    std::string cmd = args.command();
    if (cmd.empty() || args.flag("help") || cmd == "help") { usage(); return cmd.empty() ? 1 : 0; }

    try {
        if (cmd == "chat")     return cmd_chat(args);
        if (cmd == "generate") return cmd_generate(args);
        if (cmd == "info")     return cmd_info(args);
        if (cmd == "quantize") return cmd_quantize(args);
        if (cmd == "bench")    return cmd_bench(args);
        if (cmd == "tokenize") return cmd_tokenize(args);
        if (cmd == "export")   return cmd_export(args);
        if (cmd == "eval")     return cmd_eval(args);
        if (cmd == "devices")  return cmd_devices(args);
        std::cerr << "unknown command: " << cmd << "\n\n";
        usage();
        return 1;
    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
