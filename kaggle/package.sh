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

zip -qr "${OUT}" . \
    -x "build/*" "artifacts/*" "*.git*" \
    -x "Ai dariga datasets/*" "*.o" "*.obj" "*.graw" "*.ckpt.bin" \
    -x "ghassan-flash-src.zip"

echo "[package] wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "[package] Upload this zip to Kaggle, unzip, then: bash kaggle/setup.sh"
