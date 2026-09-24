// ghassan-ai - the main CLI: chat, generate, info, quantize, bench, tokenize, eval.

#include "tools/cli_common.h"
#include "core/device.h"
#include "core/ops.h"
#include "format/gai_format.h"
#include "format/gguf_format.h"
#include "inference/chat.h"
#include "quantization/quantize.h"
#include "evaluation/benchmark.h"
#include "evaluation/perplexity.h"
#include "training/checkpoint.h"
#include "dataset/retrieval.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <cstdlib>

namespace fs = std::filesystem;
using namespace gai;

static void usage() {
    std::cout <<
    "ghassan-ai - Moroccan Darija conversational model (GGUF-only)\n\n"
    "usage:\n"
    "  ghassan-ai chat      --model model.gguf\n"
    "  ghassan-ai generate  --model model.gguf --prompt \"شنو هو C++؟\"\n"
    "  ghassan-ai info      --model model.gguf\n"
    "  ghassan-ai quantize  --model in.gguf --out out.gguf --profile q4_0\n"
    "  ghassan-ai bench     --model model.gguf [--tokens 128]\n"
    "  ghassan-ai tokenize  --model model.gguf --text \"شنو خبارك\"\n"
    "  ghassan-ai eval      --model model.gguf --suite evaluation/en\n"
    "  ghassan-ai perplexity --model model.gguf (--text \"held out\" | --shards artifacts/shards_en --prefix val_)\n"
    "  ghassan-ai export    --checkpoint last.ckpt --tokenizer tok.gtok --out model.gguf [--profile fp16|q8_k|q4_0] [--compat native|llama|llama_moe]\n"
    "  ghassan-ai devices\n"
    "  ghassan-ai logits    --model model.gguf --prompt \"salam\" [--out logits.f32]\n\n"
    "formats:\n"
    "  GGUF is the ONLY deploy format (self-contained: weights+tokenizer+config).\n"
    "  Legacy .gai files still load for backward compat but are deprecated.\n\n"
    "generation options:\n"
    "  --temp <f>        temperature            [0.8]\n"
    "  --top-k <n>       top-k                  [40]\n"
    "  --top-p <f>       nucleus                [0.92]\n"
    "  --min-p <f>       min-p                  [0.05]\n"
    "  --repeat <f>      repetition penalty     [1.12]\n"
    "  --no-repeat-ngram <n>  ban seen n-grams  [0=off, 3 recommended for Darija]\n"
    "  --max-tokens <n>  max new tokens         [256]\n"
    "  --seed <n>        rng seed (0 = random)\n"
    "  --greedy          deterministic decoding\n"
  "  --no-fast-sample  disable GPU fast sampling (legacy full-vocab path)\n"
    "  --system <text>   system prompt\n"
    "  --persona <name>  darija (default, ScriptRouter) | en (English persona)\n"
    "  --no-adaptive     disable English dialog-act sampling presets\n"
    "  --no-stream       print the reply at once\n"
    "  --device auto|cpu|cuda  (metal/vulkan fall back to portable CPU)\n"
    "  --gemm-fp16 0|1        CUDA tensor-core GEMMs (default 1; 0 = pure fp32)\n"
    "  --no-fp16-cache        disable persistent FP16 inference weights\n"
    "  --mmap              zero-copy file-backed weights (.gai, CPU inference;\n"
    "                      weak PCs: no heap commit for F32 tensors)\n"
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
        GAI_CHECK(cuda_available(), "--device cuda requested but no CUDA device is available (use --device cpu)");
        return Device::CUDA;
    }
    // Portable fallback: Metal (macOS), Vulkan and TPU have no native backend
    // in this engine — run on the portable CPU backend instead of failing.
    // (--device auto already does this silently; explicit flags warn once.)
    if (d == "metal" || d == "vulkan" || d == "tpu") {
        log_warn("--device " + d + " has no native backend; falling back to CPU "
                 "(portable OpenMP kernels; use --device cuda on NVIDIA GPUs)");
        return Device::CPU;
    }
    if (d != "auto") GAI_FAIL("unknown --device '" + d + "' (auto|cpu|cuda|metal|vulkan)");
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

    // P3-6: inference GEMM precision used to be invisible (fp16 tensor cores
    // whenever CUDA was present, no log, no opt-out). Now logged once here
    // and overridable via --gemm-fp16 0 (pure fp32, e.g. quality A/B).
    if (args.has("gemm-fp16")) {
        const bool want = args.num("gemm-fp16", 1) != 0;
        ops::set_gemm_fp16(want);
        log_info(strfmt("[prec] inference gemm_fp16=%s (explicit --gemm-fp16)",
                        want ? "on" : "off"));
    } else {
        log_info(strfmt("[prec] inference gemm_fp16=%s (default; --gemm-fp16 0 disables)",
                        ops::gemm_fp16_enabled() ? "on" : "off"));
    }

    if (is_gguf) {
        if (args.flag("mmap", false))
            log_warn("--mmap currently covers .gai weights only; loading this GGUF normally");
        GGUFReader r;
        GAI_CHECK(r.open(path), "not a valid .gguf model file: " + path);
        lm.cfg = r.model_config();
        // Profile label: current key first, legacy pre-fix key as fallback.
        lm.quant = r.get_string("ghassan.quantization.profile",
                     r.get_string("general.quantization_version", "fp16"));
        lm.file_bytes = r.file_size();

        // Self-contained GGUF: the tokenizer rides INSIDE the file
        // (tokenizer.ggml.* + exact ghassan.tokenizer.gtok image).
        // Priority: explicit --tokenizer > embedded > legacy sidecar.
        if (args.has("tokenizer")) {
            GAI_CHECK(lm.tokenizer.load(args.str("tokenizer")),
                      "cannot load tokenizer: " + args.str("tokenizer"));
        } else if (r.has_tokenizer()) {
            GAI_CHECK(r.load_tokenizer(lm.tokenizer),
                      "GGUF tokenizer payload is corrupt; pass --tokenizer explicitly");
            log_info("[load] using tokenizer embedded in GGUF (no sidecar needed)");
        } else {
            // Legacy files (pre-embedding export): sidecar .gtok next to model.
            std::string def = (fs::path(path).parent_path() / "tokenizer" / "english32k.gtok").string();
            if (!fs::exists(def)) def = "artifacts/tokenizer/english32k.gtok";
            if (!fs::exists(def)) def = "artifacts/tokenizer/darija32k.gtok";
            if (!fs::exists(def)) def = "artifacts/tokenizer/darija.gtok";
            GAI_CHECK(lm.tokenizer.load(def),
                      "legacy GGUF without embedded tokenizer; pass --tokenizer "
                      "(e.g. artifacts/tokenizer/english32k.gtok) or re-export");
        }
        GAI_CHECK(lm.tokenizer.vocab_size() == lm.cfg.vocab_size,
                  strfmt("tokenizer/model vocab mismatch: %d vs %d",
                         lm.tokenizer.vocab_size(), lm.cfg.vocab_size));

        Timer t;
        lm.model = std::make_unique<Model>(lm.cfg, dev);
        GAI_CHECK(load_model_from_gguf(path, *lm.model),
                  "failed to load weights from: " + path);
        if (dev == Device::CUDA && ops::gemm_fp16_enabled() &&
            !args.flag("no-fp16-cache", false)) {
            lm.model->enable_fp16_weight_cache(true);
            log_info(strfmt("[prec] persistent fp16 inference cache: %s",
                            human_bytes(lm.model->fp16_weight_cache_bytes()).c_str()));
        }
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
    bool use_mmap = args.flag("mmap", false);
    if (use_mmap && dev != Device::CPU) {
        // CUDA kernels need device memory; a host mapping would only add a
        // copy per use. Keep the normal device load on GPU (mmap is a CPU
        // weak-PC feature, where it matters).
        log_warn("--mmap is CPU-inference only; loading normally for CUDA");
        use_mmap = false;
    }
    if (use_mmap) {
        int wrapped = 0, converted = 0;
        GAI_CHECK(load_model_from_gai_mmap(path, *lm.model, &wrapped, &converted),
                  "failed to memory-map weights from: " + path);
        log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s [mmap: %d zero-copy, %d converted]",
                        path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                        human_duration(t.seconds()).c_str(), device_name(dev),
                        wrapped, converted));
        return lm;
    }
    for (Parameter* p : lm.model->parameters()) {
        Tensor tt = r.read_tensor_f32(p->name);
        GAI_CHECK(tt.numel() == p->numel(), "tensor size mismatch: " + p->name);
        p->w.copy_from(tt);
    }
    if (dev == Device::CUDA && ops::gemm_fp16_enabled() &&
        !args.flag("no-fp16-cache", false)) {
        lm.model->enable_fp16_weight_cache(true);
        log_info(strfmt("[prec] persistent fp16 inference cache: %s",
                        human_bytes(lm.model->fp16_weight_cache_bytes()).c_str()));
    }
    log_info(strfmt("[load] %s  (%s, %s profile) in %s on %s",
                    path.c_str(), human_bytes(lm.file_bytes).c_str(), lm.quant.c_str(),
                    human_duration(t.seconds()).c_str(), device_name(dev)));
    return lm;
}

static GenerationConfig make_gen_config(const Args& a) {
    GenerationConfig g;
    g.max_new_tokens = a.num_int("max-tokens", 256);
    GAI_CHECK(g.max_new_tokens >= 0 && g.max_new_tokens <= 1000000,
              "--max-tokens must be in [0,1000000]");
    g.sampling.temperature = static_cast<float>(a.real("temp", 0.8));
    g.sampling.top_k = a.num_int("top-k", 40);
    g.sampling.top_p = static_cast<float>(a.real("top-p", 0.92));
    g.sampling.min_p = static_cast<float>(a.real("min-p", 0.05));
    g.sampling.repetition_penalty = static_cast<float>(a.real("repeat", 1.12));
    g.sampling.no_repeat_ngram = a.num_int("no-repeat-ngram", 0);
    g.sampling.seed = static_cast<u64>(a.num("seed", 0));
    g.sampling.greedy = a.flag("greedy", false);
    g.sampling.gpu_fast_sample = !a.flag("no-fast-sample", false);
    g.sampling.validate();
    return g;
}

// ---------------------------------------------------------------- commands
static int cmd_chat(const Args& args) {
    LoadedModel lm = load_model(args);
    ChatOptions opts;
    opts.gen = make_gen_config(args);
    opts.adaptive_dialog = !args.flag("no-adaptive", false);
    opts.gen.max_context = args.num_int("ctx", 0);
    GAI_CHECK(opts.gen.max_context >= 0, "--ctx must be >= 0");
    Generator gen(*lm.model, lm.tokenizer, opts.gen.max_context);
    opts.stream = !args.flag("no-stream");
    opts.show_stats = args.flag("stats");
    if (args.has("system")) opts.system = args.str("system");
    // PRO-EN: --persona en selects the English system persona (ChatSession);
    // default darija keeps the legacy ScriptRouter behavior unchanged.
    if (args.has("persona")) opts.persona = args.str("persona");
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
    GenerationConfig g = make_gen_config(args);
    g.max_context = args.num_int("ctx", 0);
    GAI_CHECK(g.max_context >= 0, "--ctx must be >= 0");
    Generator gen(*lm.model, lm.tokenizer, g.max_context);

    std::string prompt = args.str("prompt");
    if (prompt.empty() && !args.positional().empty() && args.positional().size() > 1) {
        prompt = args.positional()[1];
    }
    GAI_CHECK(!prompt.empty(), "--prompt is required");

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
        // ScriptRouter (single-turn): default persona follows the prompt's
        // script; an explicit --system is kept + given a script directive.
        // PRO-EN: --persona en forces the English persona (script-independent).
        std::string persona = args.str("persona", "darija");
        std::string sys = args.str("system", "");
        if (sys.empty()) {
            sys = ChatTemplate::system_for_persona(persona, ChatTemplate::detect_script(prompt));
        } else if (persona == "en" || persona == "english") {
            // custom system + English persona: keep it verbatim, no Darija
            // script directive appended.
        } else {
            sys += std::string("\n") + ChatTemplate::script_directive(
                ChatTemplate::detect_script(prompt));
        }
        msgs.push_back({Role::System, sys});
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
        std::cout << "  quant       : " << r.get_string("ghassan.quantization.profile",
                     r.get_string("general.quantization_version", "?")) << "\n";
        std::cout << "  arch        : " << r.get_string("general.architecture", "?") << "\n";
        std::cout << "  tokenizer   : " << (r.has_tokenizer() ? "embedded (self-contained)" : "external .gtok") << "\n";
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
    // GGUF-only profiles: fp32|fp16|q8_k|q4_0 (+ K aliases q4_k_m/q6_k, see gguf_profile_for).
    std::string prof = args.str("profile", "q4_0");
    GAI_CHECK(!in.empty() && !out.empty(), "--model and --out are required");

    const bool in_gguf = is_gguf_file(in);
    ModelConfig cfg;
    std::string tok_path = args.str("tokenizer");
    std::string tmp_tok;
    if (in_gguf) {
        GGUFReader r;
        GAI_CHECK(r.open(in), "not a valid .gguf file: " + in);
        cfg = r.model_config();
        // Re-embed exact tokenizer: prefer explicit --tokenizer, else GGUF blob.
        if (tok_path.empty() && r.has_tokenizer()) {
            Tokenizer tk;
            GAI_CHECK(r.load_tokenizer(tk), "GGUF tokenizer payload corrupt; pass --tokenizer");
            // FIX P2 (temp collision): fixed name collided when two quantize
            // processes ran concurrently. Unique per-process name instead.
            static std::atomic<unsigned> quant_tmp_ctr{0};
            std::string uniq = "gguf_requant_" +
                std::to_string((unsigned long long)::rand() ^ (unsigned long long)quant_tmp_ctr++) + ".gtok";
            fs::path tmp = fs::temp_directory_path() / uniq;
            tk.save(tmp.string());
            tmp_tok = tmp.string();
            tok_path = tmp_tok;
        }
    } else {
        log_warn("[quant] legacy .gai input detected (deprecated) — converting to GGUF-only output");
        GaiReader r;
        GAI_CHECK(r.open(in), "not a valid model file: " + in + " (expected .gguf, legacy .gai accepted)");
        cfg = r.model_config();
        if (tok_path.empty() && r.has_tokenizer()) {
            Tokenizer tk;
            if (r.load_tokenizer(tk)) {
                static std::atomic<unsigned> gai_tmp_ctr{0};
                std::string uniq = "gai_to_gguf_" +
                    std::to_string((unsigned long long)::rand() ^ (unsigned long long)gai_tmp_ctr++) + ".gtok";
                fs::path tmp = fs::temp_directory_path() / uniq;
                tk.save(tmp.string());
                tmp_tok = tmp.string();
                tok_path = tmp_tok;
            }
        }
    }
    GAI_CHECK(!tok_path.empty(), "--tokenizer is required when the input has no embedded tokenizer");

    Model model(cfg, Device::CPU);
    if (in_gguf) {
        GAI_CHECK(load_model_from_gguf(in, model), "failed to load weights from: " + in);
    } else {
        GAI_CHECK(load_model_from_gai(in, model), "failed to load legacy weights from: " + in);
    }

    if (args.flag("measure")) {
        // Per-tensor outlier scan before export (norms stay F32 by profile).
        double worst_rel = 0.0;
        std::string worst_name;
        ExportProfileGGUF pp = gguf_profile_for(prof);
        for (Parameter* p : model.parameters()) {
            Tensor cpu = p->w.to(Device::CPU);
            if (cpu.dtype() != DType::F32) cpu = quant::dequantize(cpu, DType::F32);
            bool is_norm = p->name.find("norm") != std::string::npos;
            if (is_norm) continue;
            DType want = (p->name == "tok_embeddings" || p->name == "lm_head")
                ? DType::F16 : DType::Q4_0;
            if (pp.default_type == GGMLType::Q8_0) want = DType::Q4_0;
            quant::QuantError qe = quant::measure_error(cpu, want);
            if (qe.rel_rmse > worst_rel) { worst_rel = qe.rel_rmse; worst_name = p->name; }
        }
        log_info(strfmt("[quant] worst rel RMSE (Q4 probe): %.4f%% on %s",
                        worst_rel * 100.0, worst_name.c_str()));
    }

    log_info(strfmt("[quant] %s -> %s   profile=%s (GGUF-only)", in.c_str(), out.c_str(), prof.c_str()));
    export_model_gguf(out, model, tok_path, gguf_profile_for(prof), {}, args.str("compat", "native"));
    if (!tmp_tok.empty()) {
        std::error_code ec;
        fs::remove(tmp_tok, ec);
    }
    return 0;
}

static int cmd_bench(const Args& args) {
    LoadedModel lm = load_model(args);
    const int bench_ctx = args.num_int("ctx", 512);
    const int ntok = args.num_int("tokens", 64);
    const int nprompt = args.num_int("prompt-tokens", 32);
    GAI_CHECK(bench_ctx >= 0, "--ctx must be >= 0");
    GAI_CHECK(ntok >= 0 && ntok <= 1000000, "--tokens must be in [0,1000000]");
    GAI_CHECK(nprompt > 0, "--prompt-tokens must be > 0");
    Generator gen(*lm.model, lm.tokenizer, bench_ctx);

    const int vocab = lm.model->config().vocab_size;
    std::vector<i32> prompt;
    prompt.push_back(special::BOS);
    for (int i = 1; i < nprompt; ++i) prompt.push_back((300 + (i * 37)) % vocab);

    log_info(strfmt("[bench] prefill %d tokens, decode %d tokens on %s",
                    nprompt, ntok, device_name(lm.model->device())));

    gen.reset();
    gen.prefill(prompt);

    GenerationConfig g;
    g.max_new_tokens = ntok;
    g.max_context = bench_ctx;
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
        // Embedded tokenizer first (GGUF self-contained, then .gai blob).
        std::string mp = args.str("model");
        if (is_gguf_file(mp)) {
            GGUFReader gr;
            GAI_CHECK(gr.open(mp), "cannot open model");
            GAI_CHECK(gr.has_tokenizer() && gr.load_tokenizer(tk),
                      "model has no embedded tokenizer; pass --tokenizer");
        } else {
            GaiReader r;
            GAI_CHECK(r.open(mp), "cannot open model");
            GAI_CHECK(r.load_tokenizer(tk), "model has no embedded tokenizer");
        }
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

    // GGUF-only: every export is a self-contained .gguf (no .gai writing).
    // Use --compat llama for dense Ollama files, llama_moe for MoE Ollama (experimental).
    std::string prof = args.str("profile", "fp16");
    std::string compat = args.str("compat", "native");
    std::string tok = args.str("tokenizer");
    GAI_CHECK(!tok.empty(), "--tokenizer is required for GGUF export (vocab is embedded)");
    export_model_gguf(out, model, tok, gguf_profile_for(prof),
                      {{"steps", std::to_string(loaded.step)},
                       {"val_loss", strfmt("%.6f", loaded.best_val)}},
                      compat);
    return 0;
}

static int cmd_eval(const Args& args) {
    LoadedModel lm = load_model(args);
    BenchmarkConfig bc;
    bc.suite_dir = args.str("suite", "evaluation/en");
    bc.max_prompts = args.num_int("limit", 0);
    GAI_CHECK(bc.max_prompts >= 0, "--limit must be >= 0");
    bc.gen = make_gen_config(args);
    bc.gen.max_context = args.num_int("ctx", 1024);
    GAI_CHECK(bc.gen.max_context >= 0, "--ctx must be >= 0");
    Generator gen(*lm.model, lm.tokenizer, bc.gen.max_context);
    bc.categories = args.str("categories");
    bc.output_path = args.str("out");
    bc.verbose = args.flag("show", false);

    Benchmark bench(gen, lm.tokenizer, bc);
    BenchmarkReport rep = bench.run();
    std::cout << rep.to_string();
    if (!bc.output_path.empty()) {
        rep.write_jsonl(bc.output_path);
        log_info("[eval] wrote " + bc.output_path);
    }
    return 0;
}

static int cmd_perplexity(const Args& args) {
    LoadedModel lm = load_model(args);
    const int ctx = args.num_int("ctx", 0);
    const int seq_len = args.num_int("seq-len", 512);
    const int batch_size = args.num_int("batch-size", 1);
    const int max_batches = args.num_int("batches", 20);
    GAI_CHECK(ctx >= 0 && seq_len > 0 && batch_size > 0 && max_batches > 0,
              "perplexity options must be positive");
    if (args.has("rope-scale") || args.has("yarn-mscale")) {
        float scale = args.has("rope-scale")
            ? static_cast<float>(args.real_strict("rope-scale")) : lm.model->config().rope_scale;
        float yarn = args.has("yarn-mscale")
            ? static_cast<float>(args.real_strict("yarn-mscale")) : lm.model->config().rope_yarn_mscale;
        lm.model->set_rope_runtime(scale, yarn);
    }
    Generator gen(*lm.model, lm.tokenizer, ctx);
    PerplexityResult result;
    if (args.has("text")) {
        std::vector<i32> tokens = lm.tokenizer.encode(args.str("text"), true, false);
        result = evaluate_perplexity(gen, tokens);
    } else if (args.has("shards")) {
        result = evaluate_shard_perplexity(
            gen, args.str("shards"), args.str("prefix", "val"), batch_size,
            seq_len, max_batches, args.flag("pack", false),
            static_cast<u64>(args.num("seed", 42)));
    } else {
        GAI_FAIL("perplexity requires --text or --shards");
    }
    std::cout << strfmt("  rope_scale  : %.3f\n", (double)lm.model->config().rope_scale);
    std::cout << strfmt("  yarn_mscale : %.3f\n", (double)rope_mscale(lm.model->config()));
    std::cout << strfmt("  mean_nll    : %.8f\n", result.mean_nll);
    std::cout << strfmt("  perplexity  : %.6f\n", result.perplexity);
    std::cout << strfmt("  tokens      : %lld\n", static_cast<long long>(result.tokens));
    std::cout << strfmt("  batches     : %lld\n", static_cast<long long>(result.batches));
    return 0;
}

static int cmd_devices(const Args&) {
    print_device_report();
    return 0;
}

// Dumps raw teacher-forced logits for cross-engine verification (roadmap
// item 8): run the same prompt here and in llama.cpp/ollama, then compare the
// binary rows (or the printed top-5 + argmax). The prompt is encoded RAW (no
// chat template); replicate the exact token ids on the other side.
// Output: optional --out binary file [nrows,V] f32 row-major (+ stdout top-5).
static int cmd_logits(const Args& args) {
    LoadedModel lm = load_model(args);
    std::string prompt = args.str("prompt");
    if (prompt.empty() && args.positional().size() > 1) prompt = args.positional()[1];
    GAI_CHECK(!prompt.empty(), "--prompt is required");

    std::vector<i32> ids = lm.tokenizer.encode(prompt, true, false); // BOS, no EOS
    const int T = static_cast<int>(ids.size());
    GAI_CHECK(T > 0, "prompt encodes to zero tokens");
    // FIX P2 (OOM): [T,V] f32 logits (T=4096,V=32k = 512MB) blew weak PCs.
    // Cap single-shot logits; score long prompts via generate/score_tokens.
    GAI_CHECK(T <= 512, "logits prompt too long (T>512 would materialize >64MB); "
                        "split the prompt or use score_tokens/bench instead");
    std::cout << strfmt("  tokens: %d  ids:", T);
    for (i32 id : ids) std::cout << " " << id;
    std::cout << "\n";

    Model& m = *lm.model;
    Activations act = m.make_activations(1, T, false);
    Tensor dev_ids({static_cast<i64>(T)}, DType::I32, m.device());
    device_copy(dev_ids.data_ptr(), m.device(), ids.data(), Device::CPU,
                sizeof(i32) * ids.size());
    Tensor& logits = m.forward(dev_ids.i32p(), 1, T, act);
    const float* L = logits.f32();
    const int V = m.config().vocab_size;

    std::ofstream out;
    if (args.has("out")) {
        out.open(args.str("out"), std::ios::binary);
        GAI_CHECK(out.good(), "cannot write logits file");
    }
    for (int t = 0; t < T; ++t) {
        const float* row = L + static_cast<size_t>(t) * V;
        if (out) out.write(reinterpret_cast<const char*>(row),
                           static_cast<std::streamsize>(V) * 4);
        // top-5 + argmax for eyeball comparison
        int top[5] = {-1, -1, -1, -1, -1};
        for (int v = 0; v < V; ++v) {
            for (int k = 0; k < 5; ++k) {
                if (top[k] < 0 || row[v] > row[top[k]]) {
                    for (int j = 4; j > k; --j) top[j] = top[j - 1];
                    top[k] = v;
                    break;
                }
            }
        }
        std::cout << strfmt("  pos %3d argmax=%6d (%.4f) top5:", t, top[0], row[top[0]]);
        for (int k = 0; k < 5; ++k) std::cout << strfmt(" %d:%.3f", top[k], row[top[k]]);
        std::cout << "\n";
    }
    if (out) {
        out.close();
        GAI_CHECK(out.good(), "logits write failed");
        log_info("[logits] wrote " + args.str("out"));
    }
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
        if (cmd == "perplexity") return cmd_perplexity(args);
        if (cmd == "devices")  return cmd_devices(args);
        if (cmd == "logits")   return cmd_logits(args);
        std::cerr << "unknown command: " << cmd << "\n\n";
        usage();
        return 1;
    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
