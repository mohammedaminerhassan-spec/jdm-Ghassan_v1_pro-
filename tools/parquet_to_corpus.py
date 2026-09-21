#!/usr/bin/env python3
"""
parquet_to_corpus.py — Parquet lake -> plain corpus.txt for BPE training.

The JSON dump-text route was REMOVED (parquet-only project). BPE needs raw
text, so this streams english_parquet/* (messages_json / user / assistant)
into one-doc-per-line corpus with --keep-case preserved for English.

Usage:
  python tools/parquet_to_corpus.py --lake english_parquet --out corpus_en.txt --keep-case
  python tools/parquet_to_corpus.py --lake english_parquet --out corpus_en.txt --limit 200000
Needs: pip install pyarrow
"""
import argparse
import glob
import json
import os
import sys

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow missing: python -m pip install pyarrow", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default="english_parquet")
    ap.add_argument("--out", default="corpus_en.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-case", action="store_true", default=True)
    ap.add_argument("--lower", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.lake, "*.parquet")))
    if not files:
        print(f"[ERROR] no .parquet in {args.lake}", file=sys.stderr)
        sys.exit(1)

    keep_case = args.keep_case and not args.lower
    n_lines = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for fp in files:
            table = pq.read_table(fp, columns=["messages_json", "user", "assistant"])
            for row in table.to_pylist():
                try:
                    msgs = json.loads(row["messages_json"] or "[]")
                except Exception:
                    msgs = []
                texts = []
                if msgs:
                    for m in msgs:
                        c = (m.get("content") or "").strip()
                        if c:
                            texts.append(c)
                else:
                    if row.get("user"):
                        texts.append(row["user"].strip())
                    if row.get("assistant"):
                        texts.append(row["assistant"].strip())
                for t in texts:
                    t = t.replace("\n", " ").replace("\r", " ").strip()
                    if len(t) < 10:
                        continue
                    if not keep_case:
                        t = t.lower()
                    out.write(t + "\n")
                    n_lines += 1
                    if args.limit and n_lines >= args.limit:
                        print(f"Done. {n_lines} lines -> {args.out} (keep_case={keep_case})")
                        return
            print(f"  ... {os.path.basename(fp)} -> {n_lines} lines", flush=True)
    print(f"Done. {n_lines} lines -> {args.out} (keep_case={keep_case})")


if __name__ == "__main__":
    main()
