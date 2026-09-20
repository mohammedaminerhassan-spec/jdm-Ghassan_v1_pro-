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
    "  gai_train --config configs/flash_moe.yaml\n"
    "  gai_train --config configs/flash_moe.yaml --max-steps 100 --batch-size 2 --seq-len 256\n"
    "  gai_train --dry-run --config configs/flash_moe.yaml     # report shapes/memory only\n\n"
    "overrides (all optional, they win over the config file):\n"
    "  --data <dir> --checkpoint-dir <dir> --resume auto|none|<path>\n"
    "  --batch-size --seq-len --grad-accum --max-steps --lr --warmup --seed\n"
    "  --optimizer adamw|lion|muon --scheduler cosine|wsd --ckpt-segments <n>\n"
    "  --ce-chunks <n>        chunked CE row-blocks, 0/1=off (default 4)\n"
    "  --strict-config        unknown/dead config keys fail instead of warning\n"
  "  --allow-recipe-drift   resume despite rope/eps/ctx recipe drift (research only)\n"
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
        std::string cfg_path = args.str("config", "configs/flash_moe.yaml");
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

        // ---- CLI overrides
        if (args.has("data"))            tcfg.data_dir = args.str("data");
        if (args.has("checkpoint-dir"))  tcfg.checkpoint_dir = args.str("checkpoint-dir");
        if (args.has("resume"))          tcfg.resume = args.str("resume");
        if (args.flag("allow-recipe-drift")) tcfg.allow_recipe_drift = true;
        if (args.has("batch-size"))      tcfg.batch_size = static_cast<int>(args.num("batch-size"));
        if (args.has("seq-len"))         tcfg.seq_len = static_cast<int>(args.num("seq-len"));
        if (args.has("grad-accum"))      tcfg.grad_accum = static_cast<int>(args.num("grad-accum"));
        // CLI --max-steps wins over the yaml schedule: it forces steps mode
        // (epochs cleared) so scripted time budgets can't silently mix modes.
        if (args.has("max-steps")) {
            tcfg.max_steps = args.num("max-steps");
            if (tcfg.epochs > 0)
                log_warn("[cfg ] --max-steps overrides yaml epochs (epochs ignored)");
            tcfg.epochs = 0;
        }
        if (args.has("gemm-fp16"))       tcfg.gemm_fp16 = args.num("gemm-fp16") != 0;
        if (args.has("lr"))              tcfg.learning_rate = static_cast<float>(args.real("lr"));
        if (args.has("warmup"))          tcfg.warmup_steps = args.num("warmup");
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
            tcfg.ckpt_segments = static_cast<int>(args.num("ckpt-segments"));
            if (tcfg.ckpt_segments < 1) tcfg.ckpt_segments = 1;
            if (tcfg.ckpt_segments > 8) tcfg.ckpt_segments = 8;
        }
        if (args.has("ce-chunks")) {
            tcfg.ce_chunks = static_cast<int>(args.num("ce-chunks"));
            if (tcfg.ce_chunks < 0) tcfg.ce_chunks = 0;
            if (tcfg.ce_chunks > 32) tcfg.ce_chunks = 32;
        }
        if (args.has("seed"))            tcfg.seed = static_cast<u64>(args.num("seed"));
        if (args.has("device"))          tcfg.device = args.str("device");
        if (args.has("log-every"))       tcfg.log_every = args.num("log-every");
        if (args.has("eval-every"))      tcfg.eval_every = args.num("eval-every");
        if (args.has("save-every"))      tcfg.save_every = args.num("save-every");
        if (args.has("vocab"))           mcfg.vocab_size = static_cast<int>(args.num("vocab"));
        if (args.has("layers"))          mcfg.num_layers = static_cast<int>(args.num("layers"));
        if (args.has("hidden"))          mcfg.hidden_size = static_cast<int>(args.num("hidden"));
        mcfg.validate();

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
             Tokenizer tok;
             GAI_CHECK(tok.load(tok_path), "cannot load tokenizer: " + tok_path);
             GAI_CHECK(tok.vocab_size() == mcfg.vocab_size,
                       strfmt("tokenizer vocab_size %d != model.vocab_size %d "
                              "(training would produce corrupt embeddings)",
                              tok.vocab_size(), mcfg.vocab_size));
             log_info(strfmt("[tok ] validated %s (vocab=%d matches model)", tok_path.c_str(), tok.vocab_size()));
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

        Model model(mcfg, dev);
        model.print_parameter_report();

        if (args.flag("dry-run")) {
            size_t act = model.estimate_activation_bytes(tcfg.batch_size, tcfg.seq_len, true, tcfg.ce_chunks);
            if (tcfg.activation_checkpointing && tcfg.batch_size > 1) {
                int seg = tcfg.ckpt_segments > 1 ? tcfg.ckpt_segments : 2;
                if (seg > tcfg.batch_size) seg = tcfg.batch_size;
                act = (act + static_cast<size_t>(seg) - 1) / static_cast<size_t>(seg);
            }
            u64 params = static_cast<u64>(model.num_parameters());
            bool lion = (tcfg.optimizer == "lion");
            bool muon = (tcfg.optimizer == "muon");
            // muon moments ~= lion (m everywhere + v on tiny norms) + NS scratch
            size_t opt_b = (lion || muon) ? params * 4 : params * 8;
            // P2-3: frozen embeddings carry no moments; don't overestimate.
            if (tcfg.freeze_embeddings) {
                Parameter* fe = model.find_parameter("tok_embeddings");
                if (fe) {
                    const size_t frozen_b = static_cast<size_t>(fe->numel()) * ((lion || muon) ? 4 : 8);
                    opt_b = (opt_b >= frozen_b) ? opt_b - frozen_b : 0;
                }
            }
            log_info("---------------- dry run: memory estimate ----------------");
            log_info(strfmt("  weights (fp32)      : %s", human_bytes(params * 4).c_str()));
            log_info(strfmt("  gradients (fp32)    : %s", human_bytes(params * 4).c_str()));
            log_info(strfmt("  %s (fp32)    : %s",
                            muon ? "muon m+v+NS " : lion ? "lion m      " : "adamw m+v   ",
                            human_bytes(opt_b).c_str()));
            log_info(strfmt("  activations b=%d t=%d%s%s : %s",
                            tcfg.batch_size, tcfg.seq_len,
                            tcfg.activation_checkpointing ? " [ckpt]" : "",
                            tcfg.ce_chunks > 1 ? strfmt(" [ce-x%d]", tcfg.ce_chunks).c_str() : "",
                            human_bytes(act).c_str()));
            log_info(strfmt("  sched %s | opt %s",
                            tcfg.scheduler.c_str(), tcfg.optimizer.c_str()));
            log_info(strfmt("  TOTAL               : %s",
                            human_bytes(params * 8 + opt_b + act).c_str()));
            log_info(strfmt("  tokens per step     : %s",
                            human_count(static_cast<u64>(tcfg.tokens_per_step())).c_str()));
            if (tcfg.max_steps > 0)
                log_info(strfmt("  total training toks : %s (steps mode)",
                                human_count(static_cast<u64>(tcfg.tokens_per_step() * tcfg.max_steps)).c_str()));
            else
                log_info(strfmt("  schedule            : epochs mode (%d epochs; steps from data size)",
                                tcfg.epochs));
            return 0;
        }

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
