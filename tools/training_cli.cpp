#include "tools/cli_common.h"
#include "training/trainer.h"
#include "core/device.h"
#include "format/gguf_codec.h"
#include "tokenizer/tokenizer.h"

#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <string>
#ifdef GAI_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_ops.h"
#include "cuda/cuda_utils.h"
#endif

namespace fs = std::filesystem;
using namespace gai;

static void gai_verbose_terminate() {
    try {
        if (auto p = std::current_exception()) std::rethrow_exception(p);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[fatal] uncaught exception: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[fatal] uncaught non-std exception\n");
    }
    std::fflush(stderr);
    std::abort();
}

static void usage() {
    std::cout <<
    "gai_train - Ghassan AI training\n\n"
    "usage:\n"
    "  gai_train --config configs/flash_480m_single.yaml\n"
    "  gai_train --config configs/flash_480m_single.yaml --max-steps 100 --batch-size 2 --seq-len 256\n"
    "  gai_train --dry-run --config configs/flash_480m_single.yaml     # report shapes/memory only\n\n"
    "overrides (all optional, they win over the config file):\n"
    "  --data <dir> --checkpoint-dir <dir> --resume auto|none|<path>\n"
    "  --batch-size --seq-len --grad-accum --max-steps --lr --warmup --seed\n"
    "  --optimizer adamw|lion|muon --scheduler cosine|wsd --ckpt-segments <n>\n"
    "  --ce-chunks <n>        chunked CE row-blocks, 0/1=off (default 4)\n"
    "  --strict-config        unknown/dead config keys fail instead of warning\n"
    "  --dry-run              arithmetic-only memory/schedule estimate (allocates nothing)\n"
    "  --max-vram-mb <n>      pre-flight gate: fail if the predicted peak exceeds n MiB\n"
    "  --output-budget-mb <n> pre-flight gate: fail if the run would exceed n MiB of\n"
    "                         SAVED output (/kaggle/working on Kaggle: repo+build+shards+\n"
    "                         checkpoints+gguf). 0 = off.\n"
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
    std::set_terminate(gai_verbose_terminate);
    enable_utf8_console();
    Args args(argc, argv);
    apply_common_flags(args);
    if (args.flag("help")) { usage(); return 0; }

    try {
        std::string cfg_path = args.str("config", "configs/flash_480m_single.yaml");
        Config cfg;
        if (fs::exists(cfg_path)) {
            cfg = Config::from_file(cfg_path);
            log_info("[cfg ] " + cfg_path);
        } else {
            log_warn("[cfg ] " + cfg_path + " not found, using built-in defaults");
        }

        ModelConfig mcfg = ModelConfig::from_config(cfg, "model");
        TrainerConfig tcfg = TrainerConfig::from_config(cfg, args.flag("strict-config", false));

        if (cfg.has("tokenizer.vocab_size")) {
            const int tv = static_cast<int>(cfg.get_int("tokenizer.vocab_size", mcfg.vocab_size));
            if (tv != mcfg.vocab_size) {
                log_warn(strfmt("[cfg ] tokenizer.vocab_size %d != model.vocab_size %d "
                                "(embedding/tokenizer mismatch; training would corrupt)",
                                tv, mcfg.vocab_size));
            }
        }

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
        if (args.has("output-budget-mb")) tcfg.output_budget_mb = args.num_strict("output-budget-mb");

        {
            const std::string tok_cli = args.str("tokenizer", "");
            const std::string tok_yaml = cfg.get_str("tokenizer.path", "");
            std::string tok_path = !tok_cli.empty() ? tok_cli : tok_yaml;
            if (!tok_path.empty() && !fs::path(tok_path).is_absolute()) {
                fs::path cand = fs::path(cfg_path).parent_path() / tok_path;
                if (fs::exists(cand)) tok_path = cand.string();
            }
            const u64 fp = tok_path.empty() ? 0 : fingerprint_file(tok_path);
            if (tcfg.tok_fingerprint == 0) {
                tcfg.tok_fingerprint = fp;
                if (fp != 0)
                    log_info(strfmt("[tok ] fingerprint %s (%s) pinned for resume verification",
                                    fingerprint_hex(fp).c_str(), tok_path.c_str()));
                else if (!tok_path.empty())
                    log_warn(strfmt("[tok ] cannot read %s: resume will not verify tokenizer identity",
                                    tok_path.c_str()));
            }
        }
        if (args.has("batch-size"))      tcfg.batch_size = strict_args ? args.num_int_strict("batch-size") : args.num_int("batch-size");
        if (args.has("seq-len"))         tcfg.seq_len = strict_args ? args.num_int_strict("seq-len") : args.num_int("seq-len");
        if (args.has("grad-accum"))      tcfg.grad_accum = strict_args ? args.num_int_strict("grad-accum") : args.num_int("grad-accum");

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

        if (args.has("heads"))           mcfg.num_heads = strict_args ? args.num_int_strict("heads") : args.num_int("heads");
        if (args.has("kv-heads"))        mcfg.num_kv_heads = strict_args ? args.num_int_strict("kv-heads") : args.num_int("kv-heads");
        mcfg.validate();

        if (strict_args || args.flag("verbose", false))
            log_info(strfmt("[cfg ] overrides: B=%d T=%d accum=%d steps=%lld lr=%.6g opt=%s",
                            tcfg.batch_size, tcfg.seq_len, tcfg.grad_accum,
                            (long long)tcfg.max_steps, (double)tcfg.learning_rate,
                            tcfg.optimizer.c_str()));

print_device_report();

         std::string tok_path = "";
         if (args.has("tokenizer")) {
             tok_path = args.str("tokenizer");
         } else if (cfg.has("tokenizer.path")) {
             tok_path = cfg.get_str("tokenizer.path", "");

             if (!tok_path.empty() && !fs::path(tok_path).is_absolute()) {
                 fs::path cfg_dir = fs::path(cfg_path).parent_path();
                 fs::path cand = cfg_dir / tok_path;

                 if (fs::exists(cand)) tok_path = cand.string();
             }
         }
         if (!tok_path.empty()) {
             if (args.flag("dry-run")) {

                 log_info(strfmt("[tok ] dry-run: skipping tokenizer load (%s)", tok_path.c_str()));
              } else {
                  Tokenizer tok;
                  GAI_CHECK(tok.load(tok_path), "cannot load tokenizer: " + tok_path);
                  GAI_CHECK(tok.vocab_size() == mcfg.vocab_size,
                            strfmt("tokenizer vocab_size %d != model.vocab_size %d "
                                   "(training would produce corrupt embeddings)",
                                   tok.vocab_size(), mcfg.vocab_size));

                  const NormalizerConfig& ncfg = tok.normalizer_config();
                  log_info(strfmt("[tok ] validated %s (vocab=%d matches model; norm: lowercase=%d fold=%d strip_diacritics=%d)",
                                  tok_path.c_str(), tok.vocab_size(),
                                  ncfg.lowercase_latin ? 1 : 0,
                                  ncfg.fold_letters ? 1 : 0,
                                  ncfg.strip_diacritics ? 1 : 0));
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

        log_info(strfmt("[prec] compute precision: %s", tcfg.precision_name().c_str()));

        if (args.flag("dry-run")) {

            const Model::MemoryPlan plan = Model::plan_memory(
                mcfg, tcfg.batch_size, tcfg.seq_len, true, tcfg.ce_chunks,
                tcfg.fp16_weight_cache);
            const RecipeCost cost = price_recipe(mcfg, tcfg);
            const u64   params    = cost.params;
            const size_t act      = cost.activations;
            const size_t opt_b    = cost.opt_state;
            const size_t total    = cost.total;
            const bool   lion     = (tcfg.optimizer == "lion");
            const bool   muon     = (tcfg.optimizer == "muon");
            const size_t muon_scratch = cost.muon_scratch;

            log_info("---------------- dry run: memory estimate ----------------");
            log_info(strfmt("  parameters          : %s (%s without embeddings)",
                            human_count(params).c_str(),
                            human_count(plan.params_no_embedding).c_str()));
            log_info(strfmt("  weights (fp32)      : %s", human_bytes(cost.weights).c_str()));
            log_info(strfmt("  gradients (fp32)    : %s", human_bytes(cost.grads).c_str()));
            log_info(strfmt("  %s (fp32)    : %s",
                            muon ? "muon m+v+NS " : lion ? "lion m      " : "adamw m+v   ",
                            human_bytes(opt_b).c_str()));
            log_info(strfmt("  fp16 weight cache  : %s%s",
                            human_bytes(cost.fp16_cache).c_str(),
                            tcfg.fp16_weight_cache ? "" : " (disabled)"));
            log_info(strfmt("  activations b=%d t=%d%s%s : %s",
                            tcfg.batch_size, tcfg.seq_len,
                            tcfg.activation_checkpointing ? " [ckpt]" : "",
                            tcfg.ce_chunks > 1 ? strfmt(" [ce-x%d]", tcfg.ce_chunks) : "",
                            human_bytes(act).c_str()));
            if (muon)
                log_info(strfmt("  muon NS scratch     : %s", human_bytes(muon_scratch).c_str()));
            log_info(strfmt("  eval arena (est)    : %s", human_bytes(cost.eval_extra).c_str()));
            log_info(strfmt("  workspaces gemm/moe : %s + %s",
                            human_bytes(cost.gemm_ws).c_str(),
                            human_bytes(cost.moe_ws).c_str()));
            if (tcfg.ddp)
                log_info(strfmt("  nccl/ddp extra      : %s", human_bytes(cost.nccl).c_str()));
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

            {
                const i64 out_mb = args.has("output-budget-mb")
                    ? args.num_int("output-budget-mb", 0) : tcfg.output_budget_mb;
                if (out_mb > 0) {
                    const size_t per_snap = cost.snapshot;

                    const size_t proj = cost.output_projection;
                    const size_t budget = static_cast<size_t>(out_mb) * 1024u * 1024u;
                    const size_t headroom = (budget > proj) ? budget - proj : 0;
                    log_info(strfmt("  output budget       : %s (proj snapshots+gguf %s, %s left for existing data)",
                                    human_bytes(budget).c_str(), human_bytes(proj).c_str(),
                                    human_bytes(headroom).c_str()));
                    if (proj > budget)
                        GAI_FAIL(strfmt("projected output %s exceeds budget %s "
                                        "(2 snapshots %.1fGB + GGUF). Raise --output-budget-mb "
                                        "knowingly or shrink the model.",
                                        human_bytes(proj).c_str(), human_bytes(budget).c_str(),
                                        static_cast<double>(per_snap * 2) / 1e9));
                }
            }
            return 0;
        }

        Model model(mcfg, dev);
        model.print_parameter_report();

        model.init_weights(tcfg.seed);
        model.enable_grad(true);

        Trainer trainer(model, tcfg);

        try {
            trainer.run();
        } catch (const std::exception& e) {
            log_error(std::string("training aborted: ") + e.what());
            return 1;
        }

        if (args.has("export")) {
            int my_rank = 0;
            if (const char* rank_env = std::getenv("RANK")) {
                const int r = std::atoi(rank_env);
                if (r > 0) my_rank = r;
            }
            if (my_rank != 0) {
                log_info("[export] rank " + std::to_string(my_rank) +
                         ": skipping GGUF export (rank 0 owns the artifact)");
            } else {
                std::string prof = args.str("export-profile", "fp16");

                std::string tok_path = args.str("tokenizer", cfg.get_str("tokenizer.path", ""));
                if (!args.has("tokenizer") && !tok_path.empty())
                    log_info("[cfg ] using tokenizer.path from yaml: " + tok_path);
                export_model_gguf(args.str("export"), model, tok_path,
                                  gguf_profile_for(prof),
                                  {{"stage", cfg.get_str("training.stage", "pretrain")},
                                   {"steps", std::to_string(trainer.state().step)}},
                                  args.str("compat", "native"));
            }
        }

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
