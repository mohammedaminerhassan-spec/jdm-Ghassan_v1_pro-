#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${REPO_DIR}/ghassan-flash-src.zip}"
if [[ "${1:-}" == "--out" ]]; then OUT="${2:-${OUT}}"; fi

cd "${REPO_DIR}"
rm -f "${OUT}"

OUT_NAME="$(basename "${OUT}")"

zip -qr "${OUT}" . \
    -x ".git/*" "build/*" "build_test/*" "build_cpu/*" "out/*" "*.log" \
    -x "artifacts/shards*/*" "artifacts/checkpoints/*" "artifacts/*.gguf" \
    -x "artifacts/synth/*" "artifacts/synth_legacy/*" "artifacts/corpus/*" \
    -x "dataset/english_parquet/*" "dataset/qa_darija/*" "dataset/parquet/*" \
    -x "english_parquet/*" "kaggle_upload/*" ".vscode/*" \
    -x "*.parquet" "*.zip" "*.o" "*.obj" "*.graw" "*.ckpt.bin" \
    -x "${OUT_NAME}"

echo "[package] wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "[package] Upload this zip to Kaggle, unzip, then: bash configs/kaggle/setup.sh"
