# configs/kaggle/kaggle_data_workflow.py ÔÇö Kaggle data workflow, step 2 of 3 (run after cell 1)
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
import glob
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


def load_env(path):
    """Read the KEY="value" lines cell 1 wrote.

    NOT exec(): env.sh is bash, and `exec(compile(...))` on `export FOO="bar"`
    is a SyntaxError. Parsing it keeps the value intact, which matters because
    the Kaggle dataset path contains a space (.../Users/Ghassan PC/Desktop/...)
    and anything that splits on whitespace would truncate it.
    """
    env = {}
    if not os.path.exists(path):
        return env
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def apply_env(path):
    for k, v in load_env(path).items():
        os.environ.setdefault(k, v)
    return os.environ


def run(cmd, tail=30, log=None, env=None, cwd=None):
    t0 = time.time()
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=e, stdout=subprocess.PIPE,
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


def out_of(cmd, cwd=None, env=None):
    """stdout only, no decoration ÔÇö for parsing numbers."""
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=e, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, errors="replace")
    return p.stdout or ""


def du_mb(path):
    """Directory size in MB, computed in python (a path with a space must never
    be handed to `du` through a shell string)."""
    if not os.path.isdir(path):
        return 0
    tot = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot // (1024 * 1024)


ENV = "/kaggle/working/env.sh"
for k, v in load_env(ENV).items():
    print(f"  {k} = {v}", flush=True)
apply_env(ENV)
# Self-locating: works from any clone path (cell 1's bootstrap uses /repo, the
# clone URL's own name is the other common case), and env.sh wins if it exists.
REPO_DIR = os.environ.get("REPO_DIR") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
EN_DIR = os.environ.get("EN_PARQUET_DIR", "")

rule("PRECHECK")
run("date; df -h /kaggle/working | tail -1; nproc", tail=5)
for tool in ("build/bin/data_pipeline", "build/bin/gai_train", "kaggle/build_english_data.sh"):
    p = os.path.join(REPO_DIR, tool)
    ok(tool) if os.path.exists(p) else bad(f"missing {p} ÔÇö run cell 1 first")

# Every path below is handled in PYTHON, never interpolated into a shell string.
# The Kaggle dataset path contains a space (.../Users/Ghassan PC/Desktop/...),
# and this cell has now been bitten by that three separate ways: a \S+ regex that
# matched nothing, an unquoted `ls {EN_DIR}/...` that split at the space, and a
# du that lost the first half of the path. glob/os.path take the path as a
# value, so there is no shell left to re-interpret it.
def find_lake():
    if EN_DIR and os.path.isdir(EN_DIR):
        return EN_DIR
    for base in ("/kaggle/input",):
        for root, dirs, _files in os.walk(base):
            depth = root[len(base):].count(os.sep)
            if depth > 8:
                dirs[:] = []
                continue
            if any(f.startswith("english_chat_part") and f.endswith(".parquet")
                   for f in os.listdir(root)):
                return root
    return ""

EN_DIR = find_lake()
if not EN_DIR:
    bad("no english parquet lake under /kaggle/input ÔÇö attach the dataset "
        "(right panel > Add Input) and re-run this cell")
    sys.exit(1)
ok(f"lake: {EN_DIR}")

parts = sorted(glob.glob(os.path.join(EN_DIR, "*.parquet")))
n_chat = [f for f in parts if os.path.basename(f).startswith("english_chat_part")]
n_inst = [f for f in parts if os.path.basename(f).startswith("english_instruction_part")]
lake_mb = sum(os.path.getsize(f) for f in parts) // (1024 * 1024)
print(f"  chat parts {len(n_chat)} | instruction parts {len(n_inst)} | "
      f"{len(parts)} parquet files | {lake_mb} MB")
if not n_chat or not n_inst:
    bad(f"lake incomplete: {len(n_chat)} chat parts, {len(n_inst)} instruction parts "
        f"(needs both; the recipe mixes them)")
    sys.exit(1)

tok = os.path.join(REPO_DIR, "artifacts", "tokenizer", "english32k.gtok")
if not os.path.exists(tok):
    bad(f"tokenizer missing at {tok} ÔÇö cell 1 must build it before this cell")
    sys.exit(1)
rc, out = run(f"build/bin/data_pipeline tok-info --tokenizer '{tok}'", cwd=REPO_DIR, tail=5)
if "vocab_size=32000" not in out:
    bad("tokenizer vocab != 32000")
    sys.exit(1)
ok("tokenizer 32000 keep-case")

# ---------------------------------------------------------------- build
rule("BUILD SHARDS  (the long one ÔÇö Save Version when it finishes)")
# EN_PARQUET_DIR goes through the environment, not through a quoted string in a
# command line: a space in the value cannot break it.
rc, out = run("bash configs/kaggle/build_english_data.sh", cwd=REPO_DIR,
              env={"EN_PARQUET_DIR": EN_DIR},
              log="/kaggle/working/_cell2_shards.log", tail=60)
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
    bad("inspect failed ÔÇö the shards are not readable")
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
    bad(f"corpus is only {total/1e6:.1f}M tokens ÔÇö the pipeline dropped most of the lake")
if not all(os.path.exists(os.path.join(REPO_DIR, f"configs/{c}")) for c in
           ("flash_109m_compact_2xt4.yaml", "sft_flash_109m_compact_2xt4.yaml")):
    bad("the compact recipe pair is missing from configs/")

run("du -sh artifacts/shards_en artifacts/tokenizer; df -h /kaggle/working | tail -1", tail=6)

# ---- disk footprint --------------------------------------------------
# /kaggle/working is ONE 20 GB filesystem. The build/ tree is excluded from
# the trainer's saved-OUTPUT accounting but it still occupies that filesystem,
# next to the shards and the checkpoints. Kaggle kills the session with ENOSPC,
# and the trainer's own disk guard only fires once the trainer is constructed ÔÇö
# i.e. after the pilot has already burned GPU minutes. Price it here instead.
rule("DISK FOOTPRINT  (the 20 GB that has to hold everything)")
free_mb = int(out_of("df -Pm /kaggle/working | tail -1 | awk '{print $4}'").strip() or 0)
build_mb = du_mb(os.path.join(REPO_DIR, "build"))
shards_mb = du_mb(os.path.join(REPO_DIR, "artifacts", "shards_en"))
tok_mb = du_mb(os.path.join(REPO_DIR, "artifacts", "tokenizer"))

# The checkpoint+GGUF projection comes from the same preflight arithmetic the
# launcher gates on, so it cannot disagree with the run.
proj_mb = 0
for cfg in (os.environ.get("CONFIG_PT", "configs/flash_109m_compact_2xt4.yaml"),
            os.environ.get("CONFIG_SFT", "configs/sft_flash_109m_compact_2xt4.yaml")):
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
    bad(f"the run needs {need_mb} MB but only {free_mb} MB is free ÔÇö this would die "
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
  SHARDS READY ÔÇö {total/1e6:.0f}M tokens across 3 domains.

  >>> Save Version now (this keeps the shards) <<<

  NEXT ÔÇö cell 3, the training run:

    !bash configs/kaggle/run_kaggle_en.sh

  Use SESSION_SPENT_MIN=<minutes setup+data took> so the 12h cap is respected:

    !SESSION_SPENT_MIN=150 bash configs/kaggle/run_kaggle_en.sh
""")
