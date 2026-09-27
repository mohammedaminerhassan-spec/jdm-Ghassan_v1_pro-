#!/usr/bin/env bash
# kaggle/package.sh — Pack the (clean) source tree for Kaggle upload.
# Replaces the old duplicated kaggle_package/ directory: instead of keeping a
# second copy of every file in git, this zips the live tree on demand.
#
# Usage: bash kaggle/package.sh [--out <file.zip>]
# ----------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${REPO_DIR}/ghassan-flash-src.zip}"
if [[ "${1:-}" == "--out" ]]; then OUT="${2:-${OUT}}"; fi

cd "${REPO_DIR}"
rm -f "${OUT}"
# Self-exclusion: the archive must be named by pattern, not by the literal
# default, or a --out that differs from it lets `zip -qr` swallow the archive
# it is still writing.
OUT_NAME="$(basename "${OUT}")"

# FIX P2: old -x "*.git*" stripped .gitattributes (LF rules lost -> CRLF .sh
# -> `bad interpreter` on re-checkout) and old artifacts/* exclusion dropped
# the tokenizer .gtok that build_english_data.sh claims "SHIPS in the zip".
# Now: exclude .git/ only (keep attributes), ship the tokenizer, drop junk.
# The parquet LAKE is attached on Kaggle as a dataset (under /kaggle/input),
# so it is excluded here: it is 1 GB of input, never source.
zip -qr "${OUT}" . \
    -x ".git/*" "build/*" "build_test/*" "build_cpu/*" "out/*" "*.log" \
    -x "artifacts/shards*/*" "artifacts/checkpoints/*" "artifacts/*.gguf" \
    -x "dataset/english_parquet/*" "english_parquet/*" "kaggle_upload/*" \
    -x "*.parquet" "*.zip" "*.o" "*.obj" "*.graw" "*.ckpt.bin" \
    -x "${OUT_NAME}"
# NOTE: artifacts/tokenizer/*.gtok IS included on purpose (small, required).

echo "[package] wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "[package] Upload this zip to Kaggle, unzip, then: bash kaggle/setup.sh"
