#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_DIR}/build"

SKIP_TESTS=1
SKIP_DATA=0
DO_CLEAN=0
FORCE_CPU=0
REQUIRE_GPU=0
REQUIRE_NCCL=0
WITH_PARQUET=OFF

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)       DO_CLEAN=1;      shift ;;
        --run-tests)   SKIP_TESTS=0;    shift ;;
        --skip-tests)  SKIP_TESTS=1;    shift ;;
        --skip-data)   SKIP_DATA=1;    shift ;;
        --cpu-only)    FORCE_CPU=1;     shift ;;
        --require-gpu) REQUIRE_GPU=1;   shift ;;
        --require-nccl) REQUIRE_NCCL=1;  shift ;;
        --with-parquet) WITH_PARQUET=ON; shift ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "  Ghassan v1 Flash (MoE) — Kaggle Build + Data Setup"
echo "============================================================"

echo ""
echo "[system] OS: $(uname -srm)"
echo "[system] CPUs: $(nproc)"
echo "[system] RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "[system] Disk: $(df -h . | awk 'NR==2{print $4}') free"

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

if [[ "${REQUIRE_GPU}" -eq 1 ]] && [[ "${GPU_COUNT:-0}" -eq 0 ]]; then
    echo ""
    echo "[ERROR] --require-gpu was passed but no NVIDIA GPU was detected."
    echo "[ERROR] On Kaggle: Settings (right panel) -> Accelerator -> GPU T4,"
    echo "[ERROR] then re-run this script. Refusing to continue on CPU."
    exit 1
fi

if command -v nvcc &>/dev/null && [[ "${FORCE_CPU}" -eq 0 ]]; then
    NVCC_VER=$(nvcc --version | grep 'release' | awk '{print $NF}')
    echo "[CUDA] nvcc: ${NVCC_VER}"
    HAVE_CUDA=ON

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

echo "[compiler] GCC  : $(g++ --version | head -1)"

command -v cmake >/dev/null 2>&1 || {
    echo "[ERROR] cmake not found. Install it first:"
    echo "        sudo apt-get update && sudo apt-get install -y cmake"
    exit 1
}
echo "[compiler] CMake: $(cmake --version | head -1)"

if [[ "${WITH_PARQUET}" == "ON" ]]; then
    echo ""
    echo "[parquet] --with-parquet: installing Apache Arrow C++ (~2-4 min)..."
    if sudo apt-get update -qq 2>/dev/null || apt-get update -qq; then
        sudo apt-get install -y -qq ca-certificates lsb-release wget gnupg 2>/dev/null || \
        apt-get install -y -qq ca-certificates lsb-release wget gnupg
        CODENAME=$(lsb_release --cs 2>/dev/null || echo jammy)
        echo "[parquet] Ubuntu codename: ${CODENAME}"
        sudo wget -q "https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-${CODENAME}.deb" -O /tmp/arrow-apt.deb 2>/dev/null || \
        wget -q "https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-${CODENAME}.deb" -O /tmp/arrow-apt.deb
        sudo apt-get install -y -qq /tmp/arrow-apt.deb 2>/dev/null || \
        apt-get install -y -qq /tmp/arrow-apt.deb
        sudo apt-get update -qq 2>/dev/null || apt-get update -qq
    fi

    if sudo apt-get install -y -qq libarrow-dev libparquet-dev 2>/dev/null || \
       apt-get install -y -qq libarrow-dev libparquet-dev; then
        echo "[parquet] Arrow installed; CMake will enable the native lake route."
    else
        echo ""
        echo "[ERROR] Apache Arrow C++ install failed, but the project is PARQUET-ONLY:"
        echo "[ERROR] data_pipeline parquet (the Hermes lake) cannot build without it."
        echo "[ERROR] Fixes: Kaggle Settings -> Internet ON, then re-run setup.sh --with-parquet."
        exit 1
    fi
fi

if [[ "${DO_CLEAN}" -eq 1 ]]; then
    echo ""
    echo "[build] Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
fi

echo ""
echo "[build] Configuring with CUDA=${HAVE_CUDA} PARQUET=${WITH_PARQUET}..."

CCACHE_FLAGS=()
if command -v ccache &>/dev/null; then
    CCACHE_FLAGS=(-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache)
elif sudo apt-get install -y -qq ccache 2>/dev/null || apt-get install -y -qq ccache 2>/dev/null; then
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

if [[ "${REQUIRE_NCCL}" -eq 1 ]]; then
    echo ""
    echo "[nccl] --require-nccl: verifying NCCL multi-GPU capability..."
    if [[ "${FORCE_CPU}" -eq 1 ]] || [[ "${HAVE_CUDA:-OFF}" != "ON" ]]; then
        echo "[ERROR] --require-nccl but CUDA is OFF (nvcc missing or --cpu-only)."
        echo "[ERROR] 2xT4 DDP needs CUDA+NCCL on a GPU session."
        exit 1
    fi
    if grep -q "GAI_HAVE_NCCL:BOOL=ON" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null; then
        echo "[nccl] OK: binary built with NCCL (GAI_HAVE_NCCL=ON)"
    else
        echo "[ERROR] NCCL not found at configure time (single-GPU build)."
        echo "[ERROR] Install NCCL dev headers (libnccl-dev) and re-run setup.sh --with-parquet --require-nccl."
        grep -i "nccl" "${BUILD_DIR}/CMakeCache.txt" 2>/dev/null || true
        exit 1
    fi
fi

if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    echo ""
    echo "[tests] Running CTest suite..."
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
    if [[ "${HAVE_CUDA}" == "ON" ]] && [[ "${GPU_COUNT:-0}" -gt 0 ]]; then
        echo "[tests] CUDA parity: see gate below"
    else
        echo "[tests] CUDA parity: SKIPPED (no GPU)"
    fi
else
    echo "[tests] Full suite: SKIPPED (pass --run-tests to execute)"
fi

if [[ "${HAVE_CUDA}" == "ON" ]] && [[ "${GPU_COUNT:-0}" -gt 0 ]]; then
    echo ""
    echo "[gate] CUDA parity gate (MoE kernel validation)..."
    if [[ -x "${BUILD_DIR}/bin/test_moe_cuda_parity" ]]; then
        if "${BUILD_DIR}/bin/test_moe_cuda_parity"; then
            echo "[gate] CUDA parity: PASSED"
        else
            echo "[ERROR] CUDA parity gate FAILED — GPU kernels are broken."
            echo "[ERROR] See cuda/moe.cu. Aborting before any training."
            exit 1
        fi
    else
        echo "[gate] CUDA parity: SKIPPED (test_moe_cuda_parity not built)"
    fi
fi

echo ""
TOK_PATH="${REPO_DIR}/artifacts/tokenizer/darija32k.gtok"
TOK_LEGACY="${REPO_DIR}/artifacts/tokenizer/darija.gtok"
TOK_EN="${REPO_DIR}/artifacts/tokenizer/english32k.gtok"

find_english_lake() {
    local cand=""

    for cand in "${EN_PARQUET_DIR:-}" "${REPO_DIR}/dataset/english_parquet" \
                "${REPO_DIR}/english_parquet" \
                "${REPO_DIR}/kaggle_upload/english_parquet"; do
        [[ -n "${cand}" && -d "${cand}" ]] || continue
        if [[ -n "$(find "${cand}" -maxdepth 1 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)" ]]; then
            echo "${cand}"
            return 0
        fi
    done
    if [[ -d "/kaggle/input" ]]; then

        local hit=""
        hit="$(find /kaggle/input -maxdepth 8 -name 'english_chat_part*.parquet' 2>/dev/null | head -n 1)"
        if [[ -n "${hit}" ]]; then
            echo "$(dirname "${hit}")"
            return 0
        fi
    fi
    return 1
}
LAKE_DIR="$(find_english_lake || true)"
if [[ "${SKIP_DATA}" -eq 1 ]]; then
    echo "[tokenizer] --skip-data: skipping tokenizer check and training"
    if [[ -f "${TOK_EN}" ]]; then
        echo "[tokenizer] Found: ${TOK_EN}"
        TOK_PATH="${TOK_EN}"
    elif [[ -f "${TOK_PATH}" ]]; then
        echo "[tokenizer] Found: ${TOK_PATH}"
    else
        echo "[tokenizer] WARNING: no tokenizer found. Data build cell will train it."
        TOK_PATH=""
    fi
elif [[ -f "${TOK_EN}" ]]; then

    TOK_SIZE=$(du -h "${TOK_EN}" | cut -f1)
    echo "[tokenizer] Found: ${TOK_EN} (${TOK_SIZE})"
    TOK_PATH="${TOK_EN}"
elif [[ -n "${LAKE_DIR}" ]]; then
    if [[ -f "${TOK_LEGACY}" ]]; then
        echo "[tokenizer] NOTE: legacy 16k file exists at ${TOK_LEGACY}, but all"
        echo "[tokenizer] current recipes need 32k — it will NOT be used."
    fi
    echo "[tokenizer] English lake found: ${LAKE_DIR}"
    echo "[tokenizer] Step 1/2: lake -> BPE corpus (keep-case)..."
    mkdir -p "${REPO_DIR}/artifacts/tokenizer" "${REPO_DIR}/artifacts/corpus"
    CORPUS_LIMIT="${PARQUET_CORPUS_LIMIT:-400000}"
    "${BUILD_DIR}/bin/data_pipeline" parquet-corpus \
        --lake "${LAKE_DIR}" \
        --out "${REPO_DIR}/artifacts/corpus/corpus_en.txt" \
        --limit "${CORPUS_LIMIT}" || {
        echo "[ERROR] parquet -> corpus export failed for ${LAKE_DIR}."
        exit 1
    }
    echo "[tokenizer] Step 2/2: training English BPE (vocab 32000, keep-case)..."
    "${BUILD_DIR}/bin/train_tokenizer" \
        --input "${REPO_DIR}/artifacts/corpus/corpus_en.txt" \
        --vocab 32000 \
        --keep-case \
        --output "${TOK_EN}" || {
        echo "[ERROR] English tokenizer training failed."
        exit 1
    }
    if [[ -f "${TOK_EN}" ]]; then
        echo "[tokenizer] Trained: ${TOK_EN} ($(du -h "${TOK_EN}" | cut -f1))"
        TOK_PATH="${TOK_EN}"

        echo "[cleanup] removing BPE corpus dump (shards stream from parquet)..."
        rm -rf "${REPO_DIR}/artifacts/corpus" 2>/dev/null || true
        df -h "${REPO_DIR}" | tail -n 1 || true
    else
        echo "[ERROR] Tokenizer was not produced at ${TOK_EN}"
        exit 1
    fi
else
    echo ""
    echo "[ERROR] No English lake found and no tokenizer present."
    echo "[ERROR] Attach the english_parquet dataset (english_chat_part*.parquet +"
    echo "[ERROR] english_instruction_part*.parquet) or set EN_PARQUET_DIR=<path>,"
    echo "[ERROR] then re-run setup.sh. Refusing synth-only garbage."
    exit 1
fi

if [[ "${SKIP_DATA}" -eq 1 && -z "${TOK_PATH}" ]]; then
    echo "[tokenizer] --skip-data: vocab gate deferred (no tokenizer yet, data cell will train it)"
else
{
TOK_VOCAB=$("${BUILD_DIR}/bin/data_pipeline" tok-info --tokenizer "${TOK_PATH}" 2>/dev/null \
    | grep -oE 'vocab_size=[0-9]+' | cut -d= -f2 || true)
if [[ "${TOK_VOCAB}" != "32000" ]]; then
    echo "[ERROR] tokenizer ${TOK_PATH} has vocab_size=${TOK_VOCAB:-unreadable}, need 32000."
    echo "[ERROR] Delete the stale file and re-run setup.sh to train a fresh 32k tokenizer."
    exit 1
fi
echo "[tokenizer] verified: ${TOK_PATH} (vocab 32000)"
}
fi

if [[ "${SKIP_DATA}" -eq 0 ]]; then
    echo ""
    echo "[data] Verifying English lake..."
    if [[ -z "${LAKE_DIR:-}" || ! -d "${LAKE_DIR}" ]]; then
        LAKE_DIR="$(find_english_lake || true)"
    fi
    if [[ -z "${LAKE_DIR}" ]]; then
        echo "[ERROR] English lake not found (english_chat_part*.parquet missing)."
        echo "[ERROR] Attach the english_parquet dataset or set EN_PARQUET_DIR=<path>."
        exit 1
    fi
    echo "[data] lake: ${LAKE_DIR}"
    "${BUILD_DIR}/bin/data_pipeline" parquet --probe >/dev/null || {
        echo "[ERROR] data_pipeline has no Arrow backend (setup built PARQUET=OFF?)."
        echo "[ERROR] Re-run: bash configs/kaggle/setup.sh --with-parquet"
        exit 1
    }
    echo "[data] Arrow backend: native (probe ok)"
    if [[ ! -f "${TOK_PATH}" ]]; then
        echo "[ERROR] Tokenizer missing at ${TOK_PATH} (step 9 should have trained it)."
        exit 1
    fi
    echo "[data] tokenizer: ${TOK_PATH} (32k verified above)"
    echo ""
    echo "[data] Ready for sharding. NEXT CELL (shards, long — then Save Version):"
    echo "  cd ${REPO_DIR} && EN_PARQUET_DIR=${LAKE_DIR} bash configs/kaggle/build_english_data.sh"
else
    echo "[data] Lake verification: SKIPPED (--skip-data)"
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  Next cell (shards — the long one, then Save Version):"
echo "    bash configs/kaggle/build_english_data.sh"
echo ""
echo "  Then (train, on the recipe that fits one 2xT4 session):"
echo "    SESSION_SPENT_MIN=<minutes setup+shards took> bash configs/kaggle/run_kaggle_en.sh"
echo ""
echo "  (en_compact_2xt4: 109M params, 4.05 GiB of the 15 GiB per-GPU"
echo "   budget, 2.70 GiB projected output. The 1B recipes cannot be"
echo "   trained in a 12h session — D/N ~0.02 — and the 1B output"
echo "   projection alone leaves 630 MB for shards it needs ~0.9 GB of.)"
echo "============================================================"
