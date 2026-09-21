#!/usr/bin/env python3
"""
make_kaggle_zip.py — small code-only zip for Kaggle (parquet-only project).

Includes: code + configs + scripts + docs (a few MB).
Excludes: .git, build/, english_parquet/ (1.1GB lake -> upload as Kaggle
  dataset instead, found via /kaggle/input), artifacts/, data/, logs,
  binaries, model weights/shards/tokenizers.

Usage:
  python tools/make_kaggle_zip.py [--out kaggle_upload/ghassan-code-kaggle.zip]

On Kaggle:
  1. Upload this zip (code) + attach the english_parquet dataset.
  2. unzip, then: bash kaggle/setup.sh --with-parquet
  3. EN_PARQUET_DIR=/kaggle/input/<ds> bash kaggle/build_english_data.sh
"""
import argparse
import os
import sys
import zipfile

EXCLUDE_DIRS = {
    ".git", "build", "build_test", "english_parquet", "artifacts",
    "data", "kaggle_upload", "out", "tmp", "temp", "__pycache__",
    ".pytest_cache", ".vs", ".idea", ".venv",
}
EXCLUDE_EXTS = {
    ".parquet", ".gbin", ".gguf", ".ckpt", ".gai", ".gtok", ".bin",
    ".log", ".tlog", ".obj", ".o", ".lib", ".a", ".so", ".dll",
    ".exe", ".pdb", ".ilk", ".exp", ".sdf", ".opensdf", ".suo",
    ".pyc", ".swp",
}
EXCLUDE_FILES = {"Desktop.ini", "Thumbs.db", ".DS_Store"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle_upload/ghassan-code-kaggle.zip")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    n_files, n_bytes = 0, 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for folder, subfolders, filenames in os.walk(root):
            subfolders[:] = sorted(d for d in subfolders if d not in EXCLUDE_DIRS)
            for fn in sorted(filenames):
                if fn in EXCLUDE_FILES:
                    continue
                if os.path.splitext(fn)[1].lower() in EXCLUDE_EXTS:
                    continue
                # skip the zip itself if inside root (kaggle_upload excluded anyway)
                fp = os.path.join(folder, fn)
                arc = os.path.relpath(fp, root)
                if os.path.abspath(fp) == os.path.abspath(out):
                    continue
                zf.write(fp, arc)
                n_files += 1
                n_bytes += os.path.getsize(fp)

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"zip: {out} ({size_mb:.1f} MB, {n_files} files, {n_bytes / 1e6:.1f} MB raw)")
    print("upload this zip to Kaggle + attach english_parquet as a dataset.")


if __name__ == "__main__":
    sys.exit(main())
