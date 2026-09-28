# kaggle/cell2_data.py — KAGGLE CELL 2 of 3 (run after cell 1)
#
# Turns the attached English parquet lake into .gbin shards, exactly the way
# train_2xt4.sh expects to find them:
#
#   train_english_chat_*.gbin        multi-turn conversations
#   train_english_instruction_*.gbin  instruction / QA
#   train_english_behavior_*.gbin     authored persona + discipline
#   val_<same three>_*                deterministic 0.5% hash split
#
# This is the longest CPU-only step of the session (it streams ~1M parquet rows
# on 4 Kaggle cores). Save Version when it finishes so the shards survive into
# the training cell.
#
# Optional environment (a cell above this one):
#   EN_PARQUET_DIR  the dataset directory (cell 1 discovers it and writes it
#                   into /kaggle/working/env.sh; this cell sources that file)
#   BEHAVIOR_N      authored persona conversations to synthesise [15000]

import os
import re
import subprocess
import sys
import time

STEP = [0]
FAILED = []


def rule(title):
    STEP[0] += 1
    bar = "=" * 74
    print("\n" + bar + f"\n[{STEP[0]}] {title}\n" + bar, flush=True)


def ok(m):
    print(f"  [ok]   {m}", flush=True)


def bad(m):
    FAILED.append(m)
    print(f"  [FAIL] {m}", flush=True)


def run(cmd, tail=30, log=None):
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    out = p.stdout or ""
    if log:
        try:
            open(log, "w", encoding="utf-8").write(out)
        except OSError:
            pass
    if out.strip():
        print("\n".join(out.strip().splitlines()[-tail:]), flush=True)
    print(f"  -> rc={p.returncode}  ({time.time() - t0:.1f}s)", flush=True)
    return p.returncode, out


def out_of(cmd, cwd=None):
    """stdout only, no decoration — for parsing numbers."""
    p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    return p.stdout or ""


ENV = "/kaggle/working/env.sh"
if os.path.exists(ENV):
    print(open(ENV, encoding="utf-8").read().strip(), flush=True)
    exec(compile(open(ENV, encoding="utf-8").read(), ENV, "exec"), {}, {})  # noqa: S102
# Self-locating: works from any clone path (cell 1's bootstrap uses /repo, the
# clone URL's own name is the other common case), and env.sh wins if it exists.
REPO_DIR = os.environ.get("REPO_DIR") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
EN_DIR = os.environ.get("EN_PARQUET_DIR", "")

rule("PRECHECK")
run("date; df -h /kaggle/working | tail -1; nproc", tail=5)
for tool in ("build/bin/data_pipeline", "build/bin/gai_train", "kaggle/build_english_data.sh"):
    p = os.path.join(REPO_DIR, tool)
    ok(tool) if os.path.exists(p) else bad(f"missing {p} — run cell 1 first")
if not os.path.isdir(EN_DIR):
    cands = subprocess.run(
        "find /kaggle/input -maxdepth 8 -name 'english_chat_part*.parquet' 2>/dev/null | head -1",
        shell=True, capture_output=True, text=True).stdout.strip()
    EN_DIR = os.path.dirname(cands) if cands else EN_DIR
if not os.path.isdir(EN_DIR):
    bad("no english parquet lake under /kaggle/input — attach the dataset and re-run")
    sys.exit(1)
ok(f"lake: {EN_DIR}")
n1 = run(f"ls {EN_DIR}/english_chat_part*.parquet | wc -l", tail=2)[1].strip()
n2 = run(f"ls {EN_DIR}/english_instruction_part*.parquet | wc -l", tail=2)[1].strip()
run(f"du -sh {EN_DIR}", tail=2)
if n1 in ("0", "") or n2 in ("0", ""):
    bad(f"lake incomplete: {n1} chat parts, {n2} instruction parts")
    sys.exit(1)

tok = os.path.join(REPO_DIR, "artifacts", "tokenizer", "english32k.gtok")
if not os.path.exists(tok):
    bad(f"tokenizer missing at {tok} — cell 1 must build it before this cell")
    sys.exit(1)
rc, out = run(f"build/bin/data_pipeline tok-info --tokenizer {tok}", cwd=REPO_DIR, tail=5)
if "vocab_size=32000" not in out:
    bad("tokenizer vocab != 32000")
    sys.exit(1)
ok("tokenizer 32000 keep-case")

# ---------------------------------------------------------------- build
rule("BUILD SHARDS  (the long one — Save Version when it finishes)")
rc, out = run(f"EN_PARQUET_DIR='{EN_DIR}' bash kaggle/build_english_data.sh",
              cwd=REPO_DIR, log="/kaggle/working/_cell2_shards.log", tail=60)
if rc != 0:
    bad("build_english_data.sh failed (log: /kaggle/working/_cell2_shards.log)")
    for ln in out.splitlines():
        if re.search(r"error|ERROR|FAIL", ln):
            print("   | " + ln.rstrip())
    sys.exit(1)

# ---------------------------------------------------------------- verify
rule("VERIFY")
rc, out = run("build/bin/data_pipeline inspect --shards artifacts/shards_en --tokenizer "
              + tok.replace(" ", "\\ "), cwd=REPO_DIR, log="/kaggle/working/_cell2_inspect.log",
              tail=30)
if rc != 0:
    bad("inspect failed — the shards are not readable")
    sys.exit(1)

total = 0.0
for line in out.splitlines():
    m = re.search(r"total:\s*([0-9.]+)\s*([KMBT]?)\s*tokens", line)
    if m:
        mult = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(2), 1)
        total += float(m.group(1)) * mult
print(f"  corpus total: {total/1e6:.1f}M tokens")

shards = os.path.join(REPO_DIR, "artifacts", "shards_en")
for d in ("english_chat", "english_instruction", "english_behavior"):
    n = len([f for f in os.listdir(shards) if f.startswith(f"train_{d}_")]) if os.path.isdir(shards) else 0
    ok(f"train_{d}_*: {n} shard(s)") if n else bad(f"train_{d}_*: 0 shards")
    nv = len([f for f in os.listdir(shards) if f.startswith(f"val_{d}_")]) if os.path.isdir(shards) else 0
    print(f"        val_{d}_*: {nv} shard(s)")

if total < 20e6:
    bad(f"corpus is only {total/1e6:.1f}M tokens — the pipeline dropped most of the lake")
if not all(os.path.exists(os.path.join(REPO_DIR, f"configs/{c}")) for c in
           ("en_compact_2xt4.yaml", "sft_en_compact_2xt4.yaml")):
    bad("the compact recipe pair is missing from configs/")

run("du -sh artifacts/shards_en artifacts/tokenizer; df -h /kaggle/working | tail -1", tail=6)

# ---- disk footprint --------------------------------------------------
# /kaggle/working is ONE 20 GB filesystem. The build/ tree is excluded from
# the trainer's saved-OUTPUT accounting but it still occupies that filesystem,
# next to the shards and the checkpoints. Kaggle kills the session with ENOSPC,
# and the trainer's own disk guard only fires once the trainer is constructed —
# i.e. after the pilot has already burned GPU minutes. Price it here instead.
rule("DISK FOOTPRINT  (the 20 GB that has to hold everything)")
def du_mb(path):
    t = out_of(f"du -sm '{path}' 2>/dev/null | cut -f1").strip()
    return int(t) if t.isdigit() else 0

free_mb = int(out_of("df -Pm /kaggle/working | tail -1 | awk '{print $4}'").strip() or 0)
build_mb = du_mb(os.path.join(REPO_DIR, "build"))
shards_mb = du_mb(os.path.join(REPO_DIR, "artifacts", "shards_en"))
tok_mb = du_mb(os.path.join(REPO_DIR, "artifacts", "tokenizer"))

# The checkpoint+GGUF projection comes from the same preflight arithmetic the
# launcher gates on, so it cannot disagree with the run.
proj_mb = 0
for cfg in (os.environ.get("CONFIG_PT", "configs/en_compact_2xt4.yaml"),
            os.environ.get("CONFIG_SFT", "configs/sft_en_compact_2xt4.yaml")):
    dry = out_of(f"build/bin/gai_train --config {cfg} --dry-run --device cuda --strict-config "
                 f"--max-vram-mb 15360 --output-budget-mb 17408 2>&1", cwd=REPO_DIR)
    m = re.search(r"proj snapshots\+gguf\s+([0-9.]+)\s*(GiB|MiB)", dry)
    if m:
        v = float(m.group(1)) * (1024 if m.group(2) == "GiB" else 1)
        proj_mb = max(proj_mb, int(v))
    v = re.search(r"TOTAL\s+:\s+([0-9.]+)\s*GiB", dry)
    if v:
        print(f"  {cfg}: predicted per-GPU peak {v.group(1)} GiB of the 15 GiB gate")
if not proj_mb:
    info("could not read the checkpoint projection from --dry-run (non-fatal)")

# proj_mb ALREADY covers the worst case (best.ckpt + last.ckpt + the GGUF), so
# it is added once, not doubled.
need_mb = build_mb + shards_mb + tok_mb + proj_mb
print(f"  build/          {build_mb:>7d} MB   (excluded from saved output, but it"
      f" lives on the same 20 GB filesystem)")
print(f"  shards          {shards_mb:>7d} MB")
print(f"  tokenizer       {tok_mb:>7d} MB")
print(f"  ckpt + gguf     {proj_mb:>7d} MB   (2 snapshots + gguf, worst case)")
print(f"  ---------------------------------------")
print(f"  needs           {need_mb:>7d} MB")
print(f"  free            {free_mb:>7d} MB")
if need_mb > free_mb:
    bad(f"the run needs {need_mb} MB but only {free_mb} MB is free — this would die "
        f"of ENOSPC at the first save. Free space (or shrink the recipe) before cell 3.")
elif free_mb - need_mb < 1024:
    info(f"only {free_mb - need_mb} MB of slack after the run; the checkpoint "
         f"writer also needs one extra snapshot transiently, so keep an eye on it")
else:
    ok(f"{free_mb - need_mb} MB of slack after a full run")

rule("VERDICT")
if FAILED:
    for f in FAILED:
        print("   - " + f)
    print("\n  Logs: /kaggle/working/_cell2_shards.log  /kaggle/working/_cell2_inspect.log")
    print("  Do NOT train yet. Paste the log tail back and it gets fixed.")
    sys.exit(1)
print(f"""
  SHARDS READY — {total/1e6:.0f}M tokens across 3 domains.

  >>> Save Version now (this keeps the shards) <<<

  NEXT — cell 3, the training run:

    !bash kaggle/run_kaggle_en.sh

  Use SESSION_SPENT_MIN=<minutes setup+data took> so the 12h cap is respected:

    !SESSION_SPENT_MIN=150 bash kaggle/run_kaggle_en.sh
""")
