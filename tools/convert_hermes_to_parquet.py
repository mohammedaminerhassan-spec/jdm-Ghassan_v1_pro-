#!/usr/bin/env python3
"""
convert_hermes_to_parquet.py — Hermes JSON (openhermes2_5.json) -> Parquet lake
for the C++ parquet-only pipeline (--mode chat).

Input : data/openhermes2_5.json  (top-level array, 1M objects, pretty-printed)
        each object: {"conversations":[{"from":"human|gpt|system","value":"..."}],
                      "source":..., "category":...}
Output: <out>/english_chat_part*.parquet
        <out>/english_instruction_part*.parquet
        <out>/manifest.json

Columns (all UTF-8 strings, C++ parquet_reader --mode chat compatible):
  messages_json : canonical [{"role":"user|assistant|system","content":"..."}]
  user, assistant, system, category, source

Split heuristic (chat 80% / instruction 20% spirit):
  instruction if category/source is task-like OR prompt starts with an
  instruction verb. Everything else -> chat. Both go through the same
  C++ cleaning/masks; the split only feeds data.mix weights.

Streaming: never loads the 1.9GB file into RAM (incremental raw_decode).
Needs: pip install pyarrow
Usage:
  python tools/convert_hermes_to_parquet.py --input data/openhermes2_5.json --out english_parquet
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] pyarrow missing: python -m pip install pyarrow", file=sys.stderr)
    sys.exit(1)


def map_role(frm: str) -> str:
    l = (frm or "").strip().lower()
    if l in ("system", "developer"):
        return "system"
    if l in ("user", "human", "question", "instruction", "prompt",
             "problem", "query", "input"):
        return "user"
    return "assistant"  # gpt/assistant/ai/answer/response/...


INSTRUCTION_CATS = {
    "coding", "multiple_choice", "orca", "plan", "summarization",
    "trivia", "gtkm", "theory_of_mind", "counterfactual_contextual",
    "misconception", "awareness",
}
INSTRUCTION_SOURCES = {
    "glaive-code-assist", "metamath", "EvolInstruct_70k", "cot_alpaca_gpt4",
    "UnnaturalInstructions", "platypus", "CogStackMed", "caseus_custom",
    "Econ_domain_expert",
}
INSTRUCTION_VERBS = (
    "write", "implement", "create", "develop", "calculate", "explain",
    "solve", "convert", "generate", "design", "debug", "fix", "summarize",
    "translate", "classify", "list", "describe how", "how do i",
)


def is_instruction(user_text: str, category: str, source: str) -> bool:
    if category in INSTRUCTION_CATS:
        return True
    if source in INSTRUCTION_SOURCES:
        return True
    low = (user_text or "").lstrip().lower()
    for v in INSTRUCTION_VERBS:
        if low.startswith(v):
            return True
    if "plainformat" in low:
        return True
    # multiple-choice / code markers inside the prompt
    if ("a. " in low and "b. " in low and ("c. " in low or "d. " in low)):
        return True
    if "def " in low or "function " in low or "```" in low:
        return True
    return False


def stream_json_array(path):
    """Yield top-level objects of a JSON array file without loading it all."""
    dec = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        # skip until '['
        while True:
            ch = f.read(65536)
            if not ch:
                return
            buf += ch
            i = 0
            while i < len(buf) and buf[i] in " \n\r\t":
                i += 1
            if i >= len(buf):
                buf = ""
                continue
            if buf[i] == "[":
                buf = buf[i + 1:]
                break
            raise ValueError("not a JSON array (missing '[')")
        eof = False
        while True:
            # skip whitespace/commas, check for ']'
            while True:
                j = 0
                while j < len(buf) and buf[j] in " \n\r\t,":
                    j += 1
                if j < len(buf):
                    buf = buf[j:]
                    break
                if eof:
                    return
                more = f.read(65536)
                if not more:
                    eof = True
                    # trailing whitespace only?
                    if buf.strip(" \n\r\t,") in ("", "]"):
                        return
                    raise ValueError("truncated JSON array")
                buf += more
            if buf.startswith("]"):
                return
            # need a full object; accumulate until raw_decode succeeds
            while True:
                try:
                    obj, end = dec.raw_decode(buf)
                    buf = buf[end:]
                    yield obj
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    more = f.read(65536)
                    if not more:
                        eof = True
                        # try once more, then fail loudly
                        obj, end = dec.raw_decode(buf)
                        buf = buf[end:]
                        yield obj
                        break
                    buf += more


def flush_parquet(rows, path):
    cols = ["messages_json", "user", "assistant", "system", "category", "source"]
    data = {k: [r.get(k, "") for r in rows] for k in cols}
    table = pa.table({k: pa.array(v, type=pa.string()) for k, v in data.items()})
    pq.write_table(table, path, compression="zstd", use_dictionary=True,
                   version="2.6", data_page_size=1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/openhermes2_5.json")
    ap.add_argument("--out", default="english_parquet")
    ap.add_argument("--rows-per-file", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chat_buf, inst_buf = [], []
    chat_files, inst_files = [], []
    chat_n, inst_n = 0, 0
    skipped = 0
    total = 0

    for obj in stream_json_array(args.input):
        total += 1
        if args.limit and total > args.limit:
            break
        conv = obj.get("conversations") or []
        msgs = []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            val = turn.get("value") or ""
            if not val.strip():
                continue
            msgs.append({"role": map_role(turn.get("from", "")),
                         "content": val})
        # need at least 1 user + 1 assistant
        roles = {m["role"] for m in msgs}
        if "user" not in roles or "assistant" not in roles:
            skipped += 1
            continue
        sys_txt = next((m["content"] for m in msgs if m["role"] == "system"), "")
        usr_txt = next((m["content"] for m in msgs if m["role"] == "user"), "")
        # last assistant
        asst_txt = ""
        for m in reversed(msgs):
            if m["role"] == "assistant":
                asst_txt = m["content"]
                break
        cat = str(obj.get("category") or "")
        src = str(obj.get("source") or "")
        row = {
            "messages_json": json.dumps(msgs, ensure_ascii=False),
            "user": usr_txt,
            "assistant": asst_txt,
            "system": sys_txt,
            "category": cat,
            "source": src,
        }
        if is_instruction(usr_txt, cat, src):
            inst_buf.append(row)
            if len(inst_buf) >= args.rows_per_file:
                p = os.path.join(args.out, f"english_instruction_part{inst_n:03d}.parquet")
                flush_parquet(inst_buf, p)
                inst_files.append(os.path.basename(p))
                inst_n += 1
                inst_buf = []
        else:
            chat_buf.append(row)
            if len(chat_buf) >= args.rows_per_file:
                p = os.path.join(args.out, f"english_chat_part{chat_n:03d}.parquet")
                flush_parquet(chat_buf, p)
                chat_files.append(os.path.basename(p))
                chat_n += 1
                chat_buf = []
        if total % 50000 == 0:
            print(f"  ... {total} objs (chat={sum(len([]) for _ in [])} files={chat_n} inst_files={inst_n} skipped={skipped})",
                  flush=True)

    if chat_buf:
        p = os.path.join(args.out, f"english_chat_part{chat_n:03d}.parquet")
        flush_parquet(chat_buf, p)
        chat_files.append(os.path.basename(p))
        chat_n += 1
    if inst_buf:
        p = os.path.join(args.out, f"english_instruction_part{inst_n:03d}.parquet")
        flush_parquet(inst_buf, p)
        inst_files.append(os.path.basename(p))
        inst_n += 1

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": os.path.basename(args.input),
        "total_objects": total,
        "skipped_no_user_or_assistant": skipped,
        "chat_rows": sum(1 for _ in []),  # filled below
        "chat_files": chat_files,
        "instruction_files": inst_files,
        "columns": ["messages_json", "user", "assistant", "system", "category", "source"],
        "contract": "C++ data_pipeline parquet --mode chat reads messages_json (or user+assistant). "
                    "zstd + dictionary, UTF-8 strings only.",
    }
    # count rows back from written files (source of truth)
    def count_rows(files):
        n = 0
        for fn in files:
            t = pq.read_table(os.path.join(args.out, fn), columns=["user"])
            n += t.num_rows
        return n
    manifest["chat_rows"] = count_rows(chat_files)
    manifest["instruction_rows"] = count_rows(inst_files)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print(f"  Done. total={total} skipped={skipped}")
    print(f"  chat: {manifest['chat_rows']} rows in {len(chat_files)} files")
    print(f"  instruction: {manifest['instruction_rows']} rows in {len(inst_files)} files")
    print(f"  out: {args.out}/manifest.json")
    print("  Next: EN_PARQUET_DIR=" + args.out +
          " bash kaggle/build_english_data.sh")
    print("============================================================")


if __name__ == "__main__":
    main()
