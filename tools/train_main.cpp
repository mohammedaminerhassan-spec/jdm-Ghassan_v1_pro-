// gai_train - the training entry point (pretrain / cpt / sft stages).

#include "tools/cli_common.h"
#include "training/trainer.h"
#include "core/device.h"
#include "format/gguf_format.h"
#include "tokenizer/tokenizer.h"

#include <iostream>
#include <filesystem>
#include <cstdlib>
#ifdef GAI_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"
#endif

namespace fs = std::filesystem;
using namespace gai;

static void usage() {
    std::cout <<
    "gai_train - Ghassan AI training\n\n"
    "usage:\n"
    "  gai_train --config configs/en_pro.yaml\n"
    "  gai_train --config configs/en_pro.yaml --max-steps 100 --batch-size 2 --seq-len 256\n"
    "  gai_train --dry-run --config configs/en_pro.yaml     # report shapes/memory only\n\n"
    "overrides (all optional, they win over the config file):\n"
    "  --data <dir> --checkpoint-dir <dir> --resume auto|none|<path>\n"
    "  --batch-size --seq-len --grad-accum --max-steps --lr --warmup --seed\n"
    "  --optimizer adamw|lion|muon --scheduler cosine|wsd --ckpt-segments <n>\n"
    "  --ce-chunks <n>        chunked CE row-blocks, 0/1=off (default 4)\n"
    "  --strict-config        unknown/dead config keys fail instead of warning\n"
    "  --dry-run              arithmetic-only memory/schedule estimate (allocates nothing)\n"
    "  --max-vram-mb <n>      pre-flight gate: fail if the predicted peak exceeds n MiB\n"
    "                         or leaves <10% headroom (use with --dry-run)\n"
  "  --allow-recipe-drift   resume despite rope/eps/ctx recipe drift (research only)\n"
    "  --resume-mode exact|migrate   # exact: fail closed on any recipe/state drift\n"
    "  --gemm-fp16 0|1           # force CUDA fp16 GEMMs off/on\n"
    "  --device auto|cpu|cuda   --threads <n>\n"
    "  --export <path> --export-profile fp16|q8_k|q4_0|q4_k_m|q6_k --tokenizer <path>\n"
    "      # GGUF-only export (self-contained). --compat native (ghassan-ai) |\n"
    "      # llama (dense Ollama) | llama_moe (MoE Ollama, needs moe_shared=false)\n";
}

int main(int argc, char** argv) {
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);
    if (args.flag("help")) { usage(); return 0; }

    try {
        std::string cfg_path = args.str("config", "configs/en_pro.yaml");
        Config cfg;
        if (fs::exists(cfg_path)) {
            cfg = Config::from_file(cfg_path);
            log_info("[cfg ] " + cfg_path);
        } else {
            log_warn("[cfg ] " + cfg_path + " not found, using built-in defaults");
        }

        ModelConfig mcfg = ModelConfig::from_config(cfg, "model");
        TrainerConfig tcfg = TrainerConfig::from_config(cfg, args.flag("strict-config", false));

        // P2-5: tokenizer.* is now live. tokenizer.path backs the export
        // --tokenizer CLI (explicit CLI still wins); a tokenizer.vocab_size
        // that disagrees with the model is called out before GPU hours burn.
        if (cfg.has("tokenizer.vocab_size")) {
            const int tv = static_cast<int>(cfg.get_int("tokenizer.vocab_size", mcfg.vocab_size));
            if (tv != mcfg.vocab_size) {
                log_warn(strfmt("[cfg ] tokenizer.vocab_size %d != model.vocab_size %d "
                                "(embedding/tokenizer mismatch; training would corrupt)",
                                tv, mcfg.vocab_size));
            }
        }

        // ---- CLI overrides (FIX P1-7: --strict-args fails fast instead of
        // warn+default so a typo like --batch-size 2x never burns GPU hours)
        const bool strict_args = args.flag("strict-args", false);
        if (args.has("data"))            tcfg.data_dir = args.str("data");
        if (args.has("checkpoint-dir"))  tcfg.checkpoint_dir = args.str("checkpoint-dir");
        if (args.has("resume"))          tcfg.resume = args.str("resume");
        if (args.flag("allow-recipe-drift")) tcfg.allow_recipe_drift = true;
if (args.has("resume-mode")) {
    const std::string rm = args.str("resume-mode");
    if (rm != "exact" && rm != "migrate")
        GAI_FAIL("--resume-mode must be 'exact' or 'migrate' (got '" + rm + "')");
    tcfg.resume_mode = rm;
}
        if (args.has("batch-size"))      tcfg.batch_size = strict_args ? args.num_int_strict("batch-size") : args.num_int("batch-size");
        if (args.has("seq-len"))         tcfg.seq_len = strict_args ? args.num_int_strict("seq-len") : args.num_int("seq-len");
        if (args.has("grad-accum"))      tcfg.grad_accum = strict_args ? args.num_int_strict("grad-accum") : args.num_int("grad-accum");
        // CLI --max-steps wins over the yaml schedule: it forces steps mode
        // (epochs cleared) so scripted time budgets can't silently mix modes.
        if (args.has("max-steps")) {
            tcfg.max_steps = strict_args ? args.num_strict("max-steps") : args.num("max-steps");
            if (tcfg.epochs > 0)
                log_warn("[cfg ] --max-steps overrides yaml epochs (epochs ignored)");
            tcfg.epochs = 0;
        }
        if (args.has("gemm-fp16"))       tcfg.gemm_fp16 = (strict_args ? args.num_strict("gemm-fp16") : args.num("gemm-fp16")) != 0;
        if (args.has("lr"))              tcfg.learning_rate = static_cast<float>(strict_args ? args.real_strict("lr") : args.real("lr"));
        if (args.has("warmup"))          tcfg.warmup_steps = strict_args ? args.num_strict("warmup") : args.num("warmup");
        if (args.has("optimizer")) {
            tcfg.optimizer = args.str("optimizer");
            if (tcfg.optimizer != "adamw" && tcfg.optimizer != "lion" && tcfg.optimizer != "muon")
                GAI_FAIL("unknown --optimizer (adamw|lion|muon)");
            // DeepSeek CLI rule: mirror TrainerConfig heuristic — only switch
            // beta2 0.95->0.99 when it still holds the AdamW default. An
            // explicit yaml beta2 always wins (no silent clobber).
            if (tcfg.optimizer == "lion" && tcfg.beta2 == 0.95f) tcfg.beta2 = 0.99f;
        }
        if (args.has("scheduler")) {
            tcfg.scheduler = args.str("scheduler");
            if (tcfg.scheduler != "cosine" && tcfg.scheduler != "wsd")
                GAI_FAIL("unknown --scheduler (cosine|wsd)");
        }
        if (args.has("ckpt-segments")) {
            tcfg.activation_checkpointing = true;
            tcfg.ckpt_segments = strict_args ? args.num_int_strict("ckpt-segments") : args.num_int("ckpt-segments");
            if (tcfg.ckpt_segments < 1) tcfg.ckpt_segments = 1;
            if (tcfg.ckpt_segments > 8) tcfg.ckpt_segments = 8;
        }
        if (args.has("ce-chunks")) {
            tcfg.ce_chunks = strict_args ? args.num_int_strict("ce-chunks") : args.num_int("ce-chunks");
            if (tcfg.ce_chunks <= 0) tcfg.ce_chunks = 1;
            if (tcfg.ce_chunks > 32) tcfg.ce_chunks = 32;
        }
        if (args.has("seed"))            tcfg.seed = static_cast<u64>(strict_args ? args.num_strict("seed") : args.num("seed"));
        if (args.has("device"))          tcfg.device = args.str("device");
        if (args.has("log-every"))       tcfg.log_every = strict_args ? args.num_strict("log-every") : args.num("log-every");
        if (args.has("eval-every"))      tcfg.eval_every = strict_args ? args.num_strict("eval-every") : args.num("eval-every");
        if (args.has("save-every"))      tcfg.save_every = strict_args ? args.num_strict("save-every") : args.num("save-every");
        if (args.has("vocab"))           mcfg.vocab_size = strict_args ? args.num_int_strict("vocab") : args.num_int("vocab");
        if (args.has("layers"))          mcfg.num_layers = strict_args ? args.num_int_strict("layers") : args.num_int("layers");
        if (args.has("hidden"))          mcfg.hidden_size = strict_args ? args.num_int_strict("hidden") : args.num_int("hidden");
        // Tiny-smoke overrides (verify_fixes.sh --ddp shrinks the model; the
        // head counts must shrink with hidden_size or validate() correctly
        // refuses hidden % heads != 0 — that refusal is what caught the smoke
        // script passing --hidden 128 against num_heads=12 on Kaggle).
        if (args.has("heads"))           mcfg.num_heads = strict_args ? args.num_int_strict("heads") : args.num_int("heads");
        if (args.has("kv-heads"))        mcfg.num_kv_heads = strict_args ? args.num_int_strict("kv-heads") : args.num_int("kv-heads");
        mcfg.validate();
        // Echo resolved overrides so pilot math never plans from truncated ints.
        if (strict_args || args.flag("verbose", false))
            log_info(strfmt("[cfg ] overrides: B=%d T=%d accum=%d steps=%lld lr=%.6g opt=%s",
                            tcfg.batch_size, tcfg.seq_len, tcfg.grad_accum,
                            (long long)tcfg.max_steps, (double)tcfg.learning_rate,
                            tcfg.optimizer.c_str()));

print_device_report();

         // Validate tokenizer if path provided (config tokenizer.path or CLI --tokenizer)
         std::string tok_path = "";
         if (args.has("tokenizer")) {
             tok_path = args.str("tokenizer");
         } else if (cfg.has("tokenizer.path")) {
             tok_path = cfg.get_str("tokenizer.path", "");
             // PRO-HARDEN: مسار YAML النسبي artifacts/... كان يحل ضد CWD
             // فيفشل على Kaggle عند التشغيل من /kaggle/working. نحله ضد
             // مجلد ملف الconfig أولا (نفس سلوك data_pipeline).
             if (!tok_path.empty() && !fs::path(tok_path).is_absolute()) {
                 fs::path cfg_dir = fs::path(cfg_path).parent_path();
                 fs::path cand = cfg_dir / tok_path;
                 // scneario Kaggle: الrepo قد يكون cwd نفسه؛ جرب الاثنين.
                 if (fs::exists(cand)) tok_path = cand.string();
             }
         }
         if (!tok_path.empty()) {
             if (args.flag("dry-run")) {
                 // F-23: a memory/schedule estimate tokenizes nothing, so do
                 // not make it depend on a 32k tokenizer being present. The
                 // vocab gate still runs on the real (non-dry) path below.
                 log_info(strfmt("[tok ] dry-run: skipping tokenizer load (%s)", tok_path.c_str()));
             } else {
                 Tokenizer tok;
                 GAI_CHECK(tok.load(tok_path), "cannot load tokenizer: " + tok_path);
                 GAI_CHECK(tok.vocab_size() == mcfg.vocab_size,
                           strfmt("tokenizer vocab_size %d != model.vocab_size %d "
                                  "(training would produce corrupt embeddings)",
                                  tok.vocab_size(), mcfg.vocab_size));
                 log_info(strfmt("[tok ] validated %s (vocab=%d matches model)", tok_path.c_str(), tok.vocab_size()));
             }
         }

         Device dev = Device::CPU;
        if (tcfg.device == "cuda") {
            GAI_CHECK(cuda_available(), "--device cuda requested but no CUDA device is available");
            dev = Device::CUDA;
        } else if (tcfg.device == "tpu" || tcfg.device == "metal" || tcfg.device == "vulkan") {
            GAI_FAIL("--device " + tcfg.device + " is not supported: T4-only build (auto|cpu|cuda)");
        } else if (tcfg.device == "auto") {
            dev = best_device();
        }
        // FIX (DDP order): pin the correct GPU BEFORE any cudaMalloc (Model
        // weights allocate on construction). Previously every rank allocated
        // on GPU0 then called cudaSetDevice(local_rank) inside Trainer, piling
        // 4 ranks onto one GPU briefly -> OOM / NCCL hang that looked like a leak.
#ifdef GAI_CUDA
        if (dev == Device::CUDA) {
            int want = 0;
            if (const char* lr = std::getenv("LOCAL_RANK")) {
                int v = std::atoi(lr);
                if (v >= 0) want = v;
            }
            if (cudaSetDevice(want) == cudaSuccess) {
                log_info(strfmt("[dev ] pinned to CUDA:%d via LOCAL_RANK before Model alloc", want));
            }
        }
#endif
        log_info(strfmt("[dev ] using %s", device_name(dev)));

        // T4-ONLY precision: fp32 masters + FP16 tensor-core GEMMs (Turing has
        // FP16 cores, no BF16 cores). No TPU/bf16 path exists in this build.
        log_info(strfmt("[prec] compute precision: %s", tcfg.precision_name().c_str()));

        if (args.flag("dry-run")) {
            // F-23: ARITHMETIC ONLY. The old dry-run constructed the Model
            // first, so a 1B CPU dry-run exhausted host RAM (exit 137) before
            // printing the estimate it exists to print. Nothing is allocated
            // here; Model::count_parameters() mirrors the constructor and
            // tests/test_memory_plan.cpp keeps the two honest.
            const Model::MemoryPlan plan = Model::plan_memory(
                mcfg, tcfg.batch_size, tcfg.seq_len, /*with_grad=*/true, tcfg.ce_chunks,
                tcfg.fp16_weight_cache);
            size_t act = plan.activations;
            if (tcfg.activation_checkpointing && tcfg.batch_size > 1) {
                int seg = tcfg.ckpt_segments > 1 ? tcfg.ckpt_segments : 2;
                if (seg > tcfg.batch_size) seg = tcfg.batch_size;
                act = (act + static_cast<size_t>(seg) - 1) / static_cast<size_t>(seg);
            }
            const u64 params = plan.params;
            const bool lion = (tcfg.optimizer == "lion");
            const bool muon = (tcfg.optimizer == "muon");
            // muon moments ~= lion (m everywhere + v on tiny norms) + NS scratch
            size_t opt_b = (lion || muon) ? params * 4 : params * 8;
            // P2-3: frozen embeddings carry no moments; don't overestimate.
            if (tcfg.freeze_embeddings) {
                const size_t frozen = static_cast<size_t>(mcfg.vocab_size) *
                                      static_cast<size_t>(mcfg.hidden_size) *
                                      ((lion || muon) ? 4 : 8);
                opt_b = (opt_b >= frozen) ? opt_b - frozen : 0;
            }
            const size_t total = plan.static_total + opt_b + act;

            log_info("---------------- dry run: memory estimate ----------------");
            log_info(strfmt("  parameters          : %s (%s without embeddings)",
                            human_count(params).c_str(),
                            human_count(plan.params_no_embedding).c_str()));
            log_info(strfmt("  weights (fp32)      : %s", human_bytes(plan.weights).c_str()));
            log_info(strfmt("  gradients (fp32)    : %s", human_bytes(plan.grads).c_str()));
            log_info(strfmt("  %s (fp32)    : %s",
                            muon ? "muon m+v+NS " : lion ? "lion m      " : "adamw m+v   ",
                            human_bytes(opt_b).c_str()));
            log_info(strfmt("  fp16 weight cache  : %s%s",
                            human_bytes(plan.fp16_cache).c_str(),
                            tcfg.fp16_weight_cache ? "" : " (disabled)"));
            log_info(strfmt("  activations b=%d t=%d%s%s : %s",
                            tcfg.batch_size, tcfg.seq_len,
                            tcfg.activation_checkpointing ? " [ckpt]" : "",
                            tcfg.ce_chunks > 1 ? strfmt(" [ce-x%d]", tcfg.ce_chunks).c_str() : "",
                            human_bytes(act).c_str()));
            log_info(strfmt("  sched %s | opt %s",
                            tcfg.scheduler.c_str(), tcfg.optimizer.c_str()));
            log_info(strfmt("  TOTAL               : %s", human_bytes(total).c_str()));
            log_info(strfmt("  tokens per step     : %s",
                            human_count(static_cast<u64>(tcfg.tokens_per_step())).c_str()));
            if (tcfg.max_steps > 0)
                log_info(strfmt("  total training toks : %s (steps mode)",
                                human_count(static_cast<u64>(TrainerConfig::checked_schedule_mul(
                                    tcfg.tokens_per_step(), tcfg.max_steps,
                                    "total training tokens"))).c_str()));
            else
                log_info(strfmt("  schedule            : epochs mode (%d epochs; steps from data size)",
                                tcfg.epochs));

            // T4 pre-flight gate (acceptance criterion 10): a pilot must fail
            // BEFORE burning GPU hours if the predicted peak leaves too little
            // headroom. The trainer repeats a live check, but only after the
            // model is already resident.
            if (args.has("max-vram-mb")) {
                const i64 budget_mb = args.num_int("max-vram-mb", 0);
                if (budget_mb <= 0) GAI_FAIL("--max-vram-mb needs a positive MiB value");
                const size_t budget = static_cast<size_t>(budget_mb) * 1024u * 1024u;
                const double used_pct = 100.0 * static_cast<double>(total) / static_cast<double>(budget);
                const double margin_pct = 100.0 - used_pct;
                log_info(strfmt("  VRAM budget         : %s (predicted %.1f%% used, margin %.1f%%)",
                                human_bytes(budget).c_str(), used_pct, margin_pct));
                if (total > budget)
                    GAI_FAIL(strfmt("predicted peak %s exceeds the VRAM budget %s by %s; "
                                    "reduce batch/seq/chunks, enable activation checkpointing, "
                                    "or pick a smaller recipe",
                                    human_bytes(total).c_str(), human_bytes(budget).c_str(),
                                    human_bytes(total - budget).c_str()));
                if (margin_pct < 10.0)
                    GAI_FAIL(strfmt("predicted peak %s leaves only %.1f%% headroom under %s "
                                    "(need >=10%%: allocator fragmentation + NCCL + cuBLAS "
                                    "workspaces push the real peak higher)",
                                    human_bytes(total).c_str(), margin_pct,
                                    human_bytes(budget).c_str()));
            }
            return 0;
        }

        Model model(mcfg, dev);
        model.print_parameter_report();

        model.init_weights(tcfg.seed);
        model.enable_grad(true);

        Trainer trainer(model, tcfg);
        trainer.run();

        // optional export after training — produces a self-contained .gguf file
        // (tokenizer embedded; no sidecar needed). --compat llama gives a
        // dense-only file loadable by llama.cpp / ollama / LM Studio.
        if (args.has("export")) {
            std::string prof = args.str("export-profile", "fp16");
            // P2-5: tokenizer.path from the yaml backs --tokenizer (CLI wins).
            std::string tok_path = args.str("tokenizer", cfg.get_str("tokenizer.path", ""));
            if (!args.has("tokenizer") && !tok_path.empty())
                log_info("[cfg ] using tokenizer.path from yaml: " + tok_path);
            export_model_gguf(args.str("export"), model, tok_path,
                             gguf_profile_for(prof),
                             {{"stage", cfg.get_str("training.stage", "pretrain")},
                              {"steps", std::to_string(trainer.state().step)}},
                             args.str("compat", "native"));
        }
        // GPU LEAK GUARANTEE: release monotonic CUDA workspaces (kernels/moe
        // pools grow to max-needed then reuse by design, not a leak) + cuBLAS
        // handle before exit so nvidia-smi shows 0 lingering usage and
        // multi-session Kaggle reuse never accumulates. Tensors/NCCL already
        // free via RAII (Storage + DistributedContext::~finalize).
#ifdef GAI_CUDA
        if (model.device() == Device::CUDA) {
            cuda_ops::free_workspace();
            cuda::shutdown();
        }
#endif
        return 0;

    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
