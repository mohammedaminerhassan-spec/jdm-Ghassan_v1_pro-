// gai_train - the training entry point (pretrain / cpt / sft stages).

#include "tools/cli_common.h"
#include "training/trainer.h"
#include "core/device.h"
#include "format/gguf_format.h"

#include <iostream>
#include <filesystem>
#include <cstdlib>
#ifdef GAI_CUDA
#include <cuda_runtime.h>
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
    "  --optimizer adamw|lion --scheduler cosine|wsd --ckpt-segments <n>\n"
    "  --gemm-fp16 0|1           # force CUDA fp16 GEMMs off/on\n"
    "  --device auto|cpu|cuda   --threads <n>\n"
    "  --export <path> --export-profile fp16|int8|int4 --tokenizer <path>\n";
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
        TrainerConfig tcfg = TrainerConfig::from_config(cfg);

        // ---- CLI overrides
        if (args.has("data"))            tcfg.data_dir = args.str("data");
        if (args.has("checkpoint-dir"))  tcfg.checkpoint_dir = args.str("checkpoint-dir");
        if (args.has("resume"))          tcfg.resume = args.str("resume");
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
            if (tcfg.optimizer != "adamw" && tcfg.optimizer != "lion")
                GAI_FAIL("unknown --optimizer (adamw|lion)");
            if (tcfg.optimizer == "lion") tcfg.beta2 = 0.99f;
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
            size_t act = model.estimate_activation_bytes(tcfg.batch_size, tcfg.seq_len, true);
            if (tcfg.activation_checkpointing && tcfg.batch_size > 1) {
                int seg = tcfg.ckpt_segments > 1 ? tcfg.ckpt_segments : 2;
                if (seg > tcfg.batch_size) seg = tcfg.batch_size;
                act = (act + static_cast<size_t>(seg) - 1) / static_cast<size_t>(seg);
            }
            u64 params = static_cast<u64>(model.num_parameters());
            bool lion = (tcfg.optimizer == "lion");
            size_t opt_b = lion ? params * 4 : params * 8;
            log_info("---------------- dry run: memory estimate ----------------");
            log_info(strfmt("  weights (fp32)      : %s", human_bytes(params * 4).c_str()));
            log_info(strfmt("  gradients (fp32)    : %s", human_bytes(params * 4).c_str()));
            log_info(strfmt("  %s (fp32)    : %s", lion ? "lion m      " : "adamw m+v   ",
                            human_bytes(opt_b).c_str()));
            log_info(strfmt("  activations b=%d t=%d%s : %s",
                            tcfg.batch_size, tcfg.seq_len,
                            tcfg.activation_checkpointing ? " [ckpt]" : "",
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

        // optional export after training — produces a .gguf file
        if (args.has("export")) {
            std::string prof = args.str("export-profile", "fp16");
            export_model_gguf(args.str("export"), model, args.str("tokenizer"),
                             gguf_profile_for(prof),
                             {{"stage", cfg.get_str("training.stage", "pretrain")},
                              {"steps", std::to_string(trainer.state().step)}});
        }
        return 0;

    } catch (const std::exception& e) {
        log_error(e.what());
        return 1;
    }
}
