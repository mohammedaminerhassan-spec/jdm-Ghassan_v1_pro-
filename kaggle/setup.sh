#!/usr/bin/env bash
# kaggle/setup.sh — Build Ghassan AI from source on a Kaggle GPU session.
# Usage: bash kaggle/setup.sh [--clean] [--skip-tests] [--skip-data] [--with-parquet]
# ----------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"

SKIP_TESTS=1
SKIP_DATA=0
DO_CLEAN=0
FORCE_CPU=0
REQUIRE_GPU=0
WITH_PARQUET=OFF

# Parse args (tests are SKIPPED by default to save Kaggle time; pass
# --run-tests to execute the validation suite after the build)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)       DO_CLEAN=1;      shift ;;
        --run-tests)   SKIP_TESTS=0;    shift ;;
        --skip-tests)  SKIP_TESTS=1;    shift ;;
        --skip-data)   SKIP_DATA=1;    shift ;;
        --cpu-only)    FORCE_CPU=1;     shift ;;
        --require-gpu) REQUIRE_GPU=1;   shift ;;
        --with-parquet) WITH_PARQUET=ON; shift ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "  Ghassan v1 Flash (MoE) — Kaggle Build + Data Setup"
echo "============================================================"

# ---- 1. System info
echo ""
echo "[system] OS: $(uname -srm)"
echo "[system] CPUs: $(nproc)"
echo "[system] RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "[system] Disk: $(df -h . | awk 'NR==2{print $4}') free"

# ---- 2. GPU detection
if [[ "${FORCE_CPU}" -eq 1 ]]; then
    echo ""
    echo "[GPU] Forced CPU-only build (--cpu-only)"
    GPU_COUNT=0
    CUDA_ARCH_FLAG=""
elif command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "[GPU] NVIDIA-SMI output:"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
               --format=csv,noheader 2>/dev/null | while IFS=, read -r name drv mem cc; do
        echo "  GPU: $name | driver: $drv | VRAM: $mem | CC: $cc"
    done
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    CC_CMAKE=$(echo "$CC_RAW" | tr -d '.' | tr -d ' ')
    if [[ -z "${CC_CMAKE}" ]]; then
        # no compute capability reported: let CMake pick the default arches
        CUDA_ARCH_FLAG=""
        echo "  WARNING: compute_cap empty — using CMake default architectures"
    else
        CUDA_ARCH_FLAG="-DCMAKE_CUDA_ARCHITECTURES=${CC_CMAKE}"
    fi
    echo "  GPU count: ${GPU_COUNT}  |  CUDA arch: ${CC_CMAKE:-default}"
else
    echo "[GPU] nvidia-smi not found — CPU-only build"
    GPU_COUNT=0
    CUDA_ARCH_FLAG=""
fi

# ---- 2b. GPU requirement gate (one-session mode: never train on CPU silently)
if [[ "${REQUIRE_GPU}" -eq 1 ]] && [[ "${GPU_COUNT:-0}" -eq 0 ]]; then
    echo ""
    echo "[ERROR] --require-gpu was passed but no NVIDIA GPU was detected."
    echo "[ERROR] On Kaggle: Settings (right panel) -> Accelerator -> GPU T4,"
    echo "[ERROR] then re-run this script. Refusing to continue on CPU."
    exit 1
fi

# ---- 3. CUDA toolkit detection
if command -v nvcc &>/dev/null && [[ "${FORCE_CPU}" -eq 0 ]]; then
    NVCC_VER=$(nvcc --version | grep 'release' | awk '{print $NF}')
    echo "[CUDA] nvcc: ${NVCC_VER}"
    HAVE_CUDA=ON
    # Blackwell (sm_120, e.g. RTX 5060) needs CUDA >= 12.8: older nvcc cannot
    # even parse the arch flag and fails cryptically halfway through the build.
    if [[ "${CC_CMAKE:-}" == 12* ]]; then
        NVCC_MAJOR=$(echo "${NVCC_VER}" | cut -d. -f1 | tr -d ' ')
        NVCC_MINOR=$(echo "${NVCC_VER}" | cut -d. -f2 | tr -d ' ,')
        if [[ "${NVCC_MAJOR:-0}" -lt 12 || ( "${NVCC_MAJOR:-0}" -eq 12 && "${NVCC_MINOR:-0}" -lt 8 ) ]]; then
            echo ""
            echo "[ERROR] Blackwell GPU (sm_120) needs CUDA toolkit >= 12.8, found ${NVCC_VER}."
            echo "[ERROR] Update the toolkit (or rebuild with --cpu-only to proceed without CUDA)."
            exit 1
        fi
        echo "[CUDA] Blackwell sm_120 + nvcc ${NVCC_VER}: compatible"
    fi
else
    echo "[CUDA] nvcc not found — CPU-only build"
    HAVE_CUDA=OFF
fi

# ---- 4. Compiler check
echo "[compiler] GCC  : $(g++ --version | head -1)"
echo "[compiler] CMake: $(cmake --version | head -1)"

# ---- 5b. Apache Arrow (OPTIONAL native parquet input, off by default).
# Needs: data_pipeline parquet --lake (reads dataset/parquet + qa_all.parquet
# straight into .gbin shards). Without it the JSON route applies and nothing
# breaks: dataset/qa_darija/*.json trains identically with zero dependencies.
# Opt in: bash kaggle/setup.sh --with-parquet
if [[ "${WITH_PARQUET}" == "ON" ]]; then
    echo ""
    echo "[parquet] --with-parquet: installing Apache Arrow C++ (~1-2 min)..."
    if sudo apt-get install -y libarrow-dev libparquet-dev 2>/dev/null || \
       apt-get install -y libarrow-dev libparquet-dev 2>/dev/null; then
        echo "[parquet] Arrow installed; CMake will enable the native route."
    else
        echo "[parquet] WARNING: Arrow install failed — continuing JSON-only"
        echo "[parquet] (CMake degrades gracefully; dataset/qa_darija/*.json still trains)."
        WITH_PARQUET=OFF
    fi
fi

# ---- 5. Clean if requested
if [[ "${DO_CLEAN}" -eq 1 ]]; then
    echo ""
    echo "[build] Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
fi

# ---- 6. Configure
echo ""
echo "[build] Configuring with CUDA=${HAVE_CUDA} PARQUET=${WITH_PARQUET}..."
# PRO-HARDEN: ccache يسرع rebuilds الـKaggle (nvcc بطيء) بلا تكلفة.
CCACHE_FLAGS=()
if command -v ccache &>/dev/null; then
    CCACHE_FLAGS=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
elif sudo apt-get install -y ccache 2>/dev/null || apt-get install -y ccache 2>/dev/null; then
    CCACHE_FLAGS=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
    echo "[build] ccache enabled"
fi
cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGAI_ENABLE_CUDA="${HAVE_CUDA}" \
    -DGAI_ENABLE_OPENMP=ON \
    -DGAI_BUILD_TESTS=ON \
    -DGAI_ENABLE_PARQUET="${WITH_PARQUET}" \
    "${CCACHE_FLAGS[@]}" \
    ${CUDA_ARCH_FLAG:-}

# ---- 7. Build (CUDA required for training on GPU)
# PRO-HARDEN: nproc الكامل (4x nvcc) يفجر 13GB RAM الـKaggle. نحدد JOBS<=2.
JOBS=$(nproc --ignore=1 2>/dev/null || echo 2)
if [[ "${JOBS}" -gt 2 ]]; then JOBS=2; fi
if [[ "${JOBS}" -lt 1 ]]; then JOBS=1; fi
echo "[build] Building with ${JOBS} parallel jobs (capped for Kaggle RAM)..."
if ! cmake --build "${BUILD_DIR}" --parallel "${JOBS}"; then
    if [[ "${HAVE_CUDA}" == "ON" ]]; then
        echo ""
        echo "[build] CUDA build FAILED. Full log:"
        echo "[build] Re-running with verbose output to show the actual error..."
        cmake --build "${BUILD_DIR}" --parallel "${JOBS}" 2>&1 | tail -40
        echo ""
        echo "[ERROR] CUDA build failed. Training needs GPU. Possible fixes:"
        echo "  1. Re-run (nvcc sometimes fails on first try with RAM pressure)"
        echo "  2. Check the errors above"
        echo "  3. If CUDA toolkit issue: Kaggle -> Settings -> Internet ON"
        exit 1
    else
        echo "[ERROR] Build failed. See the log above."
        exit 1
    fi
fi

echo ""
echo "[build] Build complete. Binaries:"
ls -lh "${BUILD_DIR}/bin/"

# ---- 8. Run tests (unless skipped)
if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    echo ""
    echo "[tests] Running model test..."
    "${BUILD_DIR}/bin/test_model"   && echo "  [ok] test_model"
    "${BUILD_DIR}/bin/test_tokenizer" && echo "  [ok] test_tokenizer"
    "${BUILD_DIR}/bin/test_tiny_train" && echo "  [ok] test_tiny_train"

    if [[ "${HAVE_CUDA}" == "ON" ]] && [[ "${GPU_COUNT:-0}" -gt 0 ]]; then
        echo "[tests] Running CUDA parity test..."
        "${BUILD_DIR}/bin/test_cuda" && echo "  [ok] test_cuda"
    else
        echo "[tests] CUDA parity: SKIPPED (no GPU)"
    fi
else
    echo "[tests] Full suite: SKIPPED (pass --run-tests to execute)"
fi

# ---- 8b. CUDA parity gate (always runs on GPU builds, even with --skip-tests)
# This is the correctness gate for the MoE GPU kernels (cuda/moe.cu): tiny
# config, a few seconds, compares CPU vs CUDA forward+backward. If the kernels
# were broken, training would silently diverge — so this MUST pass.
if [[ "${HAVE_CUDA}" == "ON" ]] && [[ "${GPU_COUNT:-0}" -gt 0 ]] && [[ "${SKIP_TESTS}" -eq 0 ]]; then
    echo ""
    echo "[gate] CUDA parity gate (MoE kernel validation)..."
    if [[ -x "${BUILD_DIR}/bin/test_cuda" ]]; then
        if "${BUILD_DIR}/bin/test_cuda"; then
            echo "[gate] CUDA parity: PASSED"
        else
            echo "[ERROR] CUDA parity gate FAILED — GPU kernels are broken."
            echo "[ERROR] See cuda/moe.cu. Aborting before any training."
            exit 1
        fi
    else
        echo "[gate] CUDA parity: SKIPPED (test_cuda not built)"
    fi
fi

# ---- 9. Tokenizer check (automatic training when missing)
# 32k multilingual vocab (was 16k: too high fertility for Arabic/Arabizi).
# All current model recipes are 32k: shard building MUST use the 32k file.
# A legacy 16k file, if present, is reported but NEVER used for sharding
# (it would silently waste half the embedding rows — fail-loud instead).
echo ""
TOK_PATH="${REPO_DIR}/artifacts/tokenizer/darija32k.gtok"
TOK_LEGACY="${REPO_DIR}/artifacts/tokenizer/darija.gtok"
TOK_EN="${REPO_DIR}/artifacts/tokenizer/english32k.gtok"
if [[ "${SKIP_DATA}" -eq 1 ]]; then
    echo "[tokenizer] --skip-data: skipping tokenizer check and training"
    if [[ -f "${TOK_PATH}" ]]; then
        echo "[tokenizer] Found: ${TOK_PATH}"
    elif [[ -f "${TOK_EN}" ]]; then
        echo "[tokenizer] Found: ${TOK_EN}"
        TOK_PATH="${TOK_EN}"
    else
        echo "[tokenizer] WARNING: no tokenizer found. Data build cell will train it."
        TOK_PATH=""
    fi
elif [[ -f "${TOK_PATH}" ]]; then
    TOK_SIZE=$(du -h "${TOK_PATH}" | cut -f1)
    echo "[tokenizer] Found: ${TOK_PATH} (${TOK_SIZE})"
elif [[ -f "${TOK_EN}" ]]; then
    # PRO-EN: English-only runs ship english32k.gtok in the zip; darija32k is
    # simply not needed (en_pro.yaml points at english32k). Skip Darija BPE.
    echo "[tokenizer] English run: ${TOK_EN} ships in the zip, darija32k not needed."
    echo "[tokenizer] Skipping Darija BPE training."
    TOK_PATH="${TOK_EN}"
else
    if [[ -f "${TOK_LEGACY}" ]]; then
        echo "[tokenizer] NOTE: legacy 16k file exists at ${TOK_LEGACY}, but all"
        echo "[tokenizer] current recipes need 32k — it will NOT be used. Training 32k..."
    fi
    echo "[tokenizer] darija32k.gtok not found — training it automatically."
    echo "[tokenizer] Step 1/4: dumping corpus files..."
    # Data lives in JSON/JSONL ("Ai dariga datasets") and/or CSV tables
    # ("Ghassan V1 flach data"). Search both when present.
    DATA_DIRS=()
    if [[ -d "${REPO_DIR}/Ai dariga datasets" ]]; then DATA_DIRS+=("${REPO_DIR}/Ai dariga datasets"); fi
    if [[ -d "${REPO_DIR}/Ghassan V1 flach data" ]]; then DATA_DIRS+=("${REPO_DIR}/Ghassan V1 flach data"); fi
    # PRO-HARDEN: على Kaggle البيانات تأتي كـ Dataset مربوط في /kaggle/input/*
    # لا كمجلد repo-local. بدون هذا البحث يفشل tokenizer/shards بصمت.
    # يدعم أيضا $KAGGLE_DATASET كفلتر اختياري.
    if [[ -d "/kaggle/input" ]]; then
        for d in /kaggle/input/*/; do
            [[ -d "$d" ]] || continue
            if [[ -n "${KAGGLE_DATASET:-}" && "$d" != *"${KAGGLE_DATASET}"* ]]; then continue; fi
            DATA_DIRS+=("$d")
        done
    fi
    if [[ ${#DATA_DIRS[@]} -eq 0 ]]; then DATA_DIRS=("${REPO_DIR}/Ai dariga datasets"); fi
    CORPUS_DIR="${REPO_DIR}/artifacts/corpus"
    mkdir -p "${REPO_DIR}/artifacts/tokenizer" "${CORPUS_DIR}"
    # Step 1 covers EVERY json schema (chat/instruction/text/QA pairs), so no
    # separate QA-json dump is needed afterwards.
    HAVE_JSON=0
    for D in "${DATA_DIRS[@]}"; do
        if [[ -d "$D" ]] && [[ -n "$(find "$D" -iname '*.json' -o -iname '*.jsonl' 2>/dev/null | head -n 1)" ]]; then
            HAVE_JSON=1
            "${BUILD_DIR}/bin/data_pipeline" dump-text \
                --dir "$D" \
                --out "${CORPUS_DIR}/corpus_json_$(basename "$D").txt" || {
                echo "[ERROR] dump-text failed for $D."
                exit 1
            }
        fi
    done
    if [[ "${HAVE_JSON}" -eq 0 ]]; then
        echo "[tokenizer] No JSON files — skipping JSON dump."
    fi
    echo "[tokenizer] Step 2/4: dumping CSV vocabulary tables (if any)..."
    HAVE_CSV=0
    for D in "${DATA_DIRS[@]}"; do
        if [[ -d "$D" ]] && [[ -n "$(find "$D" -name '*.csv' 2>/dev/null | head -n 1)" ]]; then
            HAVE_CSV=1
            "${BUILD_DIR}/bin/data_pipeline" csvs \
                --dir "$D" \
                --out "${CORPUS_DIR}/corpus_csv_$(basename "$D").txt" || {
                echo "[ERROR] csvs dump failed for $D."
                exit 1
            }
        fi
    done
    if [[ "${HAVE_CSV}" -eq 0 ]]; then
        echo "[tokenizer] No CSV tables found — skipping."
    fi
    # (Step 3 merged into Step 1: dump-text already covers QA-pair JSONs.)
    if [[ "${HAVE_JSON}" -eq 0 && "${HAVE_CSV}" -eq 0 ]]; then
        echo "[ERROR] No .json/.jsonl nor .csv found in: ${DATA_DIRS[*]}"
        echo "[ERROR] Tokenizer would train on synth-only (garbage). Aborting."
        exit 1
    fi
    echo "[tokenizer] Step 4/4: training BPE (vocab 32000)..."
    "${BUILD_DIR}/bin/train_tokenizer" \
        --input "${CORPUS_DIR}" \
        --synth 20000 \
        --vocab 32000 \
        --output "${TOK_PATH}" || {
        echo "[ERROR] Tokenizer training failed."
        exit 1
    }
    if [[ -f "${TOK_PATH}" ]]; then
        echo "[tokenizer] Trained: ${TOK_PATH} ($(du -h "${TOK_PATH}" | cut -f1))"
        # DISK FIT (19.5GB Kaggle + 3GB data): corpus txt was only needed for BPE.
        # Shards are built directly from JSON afterwards, so delete the 3GB dump now.
        echo "[cleanup] removing BPE corpus dumps (tokenizer is done, shards come from JSON)..."
        rm -rf "${CORPUS_DIR}" 2>/dev/null || true
        df -h "${REPO_DIR}" | tail -n 1 || true
    else
        echo "[ERROR] Tokenizer was not produced at ${TOK_PATH}"
        exit 1
    fi
fi

# ---- 9b. Tokenizer vocab gate: prove the file is 32k (never assume it).
# A legacy 16k file reaching shard building would silently waste half the
# embeddings on every current recipe. Fail here, loudly, instead.
{
TOK_VOCAB=$("${BUILD_DIR}/bin/data_pipeline" tok-info --tokenizer "${TOK_PATH}" 2>/dev/null \
    | grep -oE 'vocab_size=[0-9]+' | cut -d= -f2 || true)
if [[ "${TOK_VOCAB}" != "32000" ]]; then
    echo "[ERROR] tokenizer ${TOK_PATH} has vocab_size=${TOK_VOCAB:-unreadable}, need 32000."
    echo "[ERROR] Delete the stale file and re-run setup.sh to train darija32k.gtok."
    exit 1
fi
echo "[tokenizer] verified: ${TOK_PATH} (vocab 32000)"
}

# ---- 10. Data pipeline: convert datasets → .gbin shards (unless skipped)
if [[ "${SKIP_DATA}" -eq 0 ]]; then
    echo ""
    echo "[data] Checking for datasets..."
    DATA_DIR=""
    if [[ -d "${REPO_DIR}/Ai dariga datasets" ]]; then DATA_DIR="${REPO_DIR}/Ai dariga datasets"; fi
    if [[ -d "${REPO_DIR}/Ghassan V1 flach data" ]]; then DATA_DIR="${REPO_DIR}/Ghassan V1 flach data"; fi
    # PRO-HARDEN: fallback إلى /kaggle/input (أول dataset فيه json/csv) +
    # دعم $DATA_DIR كتجاوز صريح من البيئة لسكربتات الـCI.
    if [[ -n "${DATA_DIR_OVERRIDE:-}" && -d "${DATA_DIR_OVERRIDE}" ]]; then DATA_DIR="${DATA_DIR_OVERRIDE}"; fi
    if [[ -z "${DATA_DIR}" && -d "/kaggle/input" ]]; then
        for d in /kaggle/input/*/; do
            [[ -d "$d" ]] || continue
            if [[ -n "$(find "$d" -maxdepth 3 \( -iname '*.json' -o -iname '*.jsonl' -o -name '*.csv' \) 2>/dev/null | head -n 1)" ]]; then
                DATA_DIR="$d"
                break
            fi
        done
    fi

    if [[ -z "${DATA_DIR}" ]]; then
        echo "[data] WARNING: no dataset directory found (Ai dariga datasets / Ghassan V1 flach data)"
        echo "[data] Skipping data conversion. Upload the datasets to Kaggle first."
    elif [[ ! -f "${TOK_PATH}" ]]; then
        echo "[data] WARNING: Tokenizer not found. Skipping data conversion."
        echo "[data] Provide darija.gtok and then run: bash kaggle/convert_data.sh"
    else
        JSON_COUNT=$(find "${DATA_DIR}" -iname "*.json" -o -iname "*.jsonl" 2>/dev/null | wc -l)
        CSV_COUNT=$(find "${DATA_DIR}" -name "*.csv" 2>/dev/null | wc -l)
        echo "[data] Found ${JSON_COUNT} json + ${CSV_COUNT} csv in ${DATA_DIR}"
        if [[ "${JSON_COUNT}" -gt 0 ]]; then
            echo "[data] JSON path: converting .json/.jsonl directly to .gbin shards."
            bash "${REPO_DIR}/kaggle/convert_data.sh" \
                --json-dir "${DATA_DIR}" \
                --tokenizer "${TOK_PATH}" \
                --out "${REPO_DIR}/artifacts/shards"
        elif [[ "${CSV_COUNT}" -gt 0 ]]; then
            echo "[data] CSV path: csvs -> build (no parquet present)."
            echo "[data] Step 1/2: CSV tables -> corpus text..."
            "${BUILD_DIR}/bin/data_pipeline" csvs \
                --dir "${DATA_DIR}" \
                --out "${REPO_DIR}/artifacts/corpus/corpus_csv_full.txt" || exit 1
            echo "[data] Step 2/2: corpus text -> .gbin shards..."
            "${BUILD_DIR}/bin/data_pipeline" build \
                --tokenizer "${TOK_PATH}" \
                --text "${REPO_DIR}/artifacts/corpus/corpus_csv_full.txt" \
                --out "${REPO_DIR}/artifacts/shards" \
                --shard-tokens 50000000 \
                --seq-len 1024 || exit 1
        else
            echo "[data] No .json/.jsonl nor .csv files found. Skipping conversion."
        fi
    fi
else
    echo "[data] Data conversion: SKIPPED (--skip-data)"
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Next: bash kaggle/train.sh --full"
echo "  (Training auto-exports to GGUF after completion)"
echo "============================================================"
