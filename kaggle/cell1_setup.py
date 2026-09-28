# kaggle/cell1_setup.py — KAGGLE CELL 1 of 3 (run this FIRST)
#
# One cell, no arguments, no choices:
#   1. clone (or update) the project from GitHub
#   2. build the C++ + CUDA binaries (sm_75 = the Kaggle T4)
#   3. prove the build is correct: CUDA/CPU MoE parity + the whole test suite
#   4. print exactly one PASS/FAIL verdict and the command for the next cell
#
# ------------------------------------------------------------------
# PASTE THIS INTO THE KAGGLE CELL (5 lines, no placeholders):
#
#     !git clone --depth 1 https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-.git /kaggle/working/repo 2>/dev/null || (cd /kaggle/working/repo && git fetch --all && git reset --hard origin/HEAD)
#     !cd /kaggle/working/repo && git pull --ff-only 2>/dev/null || true
#     %run /kaggle/working/repo/kaggle/cell1_setup.py
#
# Re-run those three lines after every `git push`: the cell always pulls the
# newest code first, so the notebook never runs a stale commit.
# ------------------------------------------------------------------
# Optional environment (a cell above this one, or just edit REPO_URL here):
#   REPO_URL   = a different repo/branch
#   SKIP_TESTS = 1 to skip the test suite (not recommended)
#   CLEAN      = 1 to wipe build/ first

import os
import re
import shutil
import subprocess
import sys
import textwrap
import time

REPO_URL = os.environ.get(
    "REPO_URL", "https://github.com/mohammedaminerhassan-spec/jdm-Ghassan_v1_pro-.git"
)
REPO_DIR_NAME = REPO_URL.rstrip("/").split("/")[-1]
if REPO_DIR_NAME.endswith(".git"):
    REPO_DIR_NAME = REPO_DIR_NAME[:-4]
REPO_DIR = os.path.join("/kaggle/working", REPO_DIR_NAME)
SKIP_TESTS = os.environ.get("SKIP_TESTS", "0") == "1"
DO_CLEAN = os.environ.get("CLEAN", "0") == "1"

STEP = [0]
FAILED = []


def rule(title=""):
    STEP[0] += 1
    bar = "=" * 74
    print("\n" + bar)
    print(f"[{STEP[0]}] {title}" if title else bar)
    print(bar, flush=True)


def ok(msg):
    print(f"  [ok]   {msg}", flush=True)


def bad(msg):
    FAILED.append(msg)
    print(f"  [FAIL] {msg}", flush=True)


def run(cmd, cwd=None, log=None, tail=40, env=None):
    """Run a shell command, tee to `log`, and return (rc, full_output)."""
    t0 = time.time()
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        env=e,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    out = p.stdout or ""
    if log:
        try:
            with open(log, "w", encoding="utf-8") as fh:
                fh.write(out)
        except OSError:
            pass
    if out.strip():
        print("\n".join(out.strip().splitlines()[-tail:]), flush=True)
    print(f"  -> rc={p.returncode}  ({time.time() - t0:.1f}s)", flush=True)
    return p.returncode, out


def section(out, pattern, label):
    """Pull the interesting lines out of a long build log."""
    hits = [ln.rstrip() for ln in out.splitlines() if re.search(pattern, ln)]
    if hits:
        for h in hits:
            print(f"  {label} {h.strip()}")
    else:
        print(f"  {label} (not found in the log)")
    return hits


# ---------------------------------------------------------------- 0. session
rule("SESSION")
run("date; echo; nvidia-smi --query-gpu=index,name,memory.total,driver_version "
    "--format=csv,noheader; echo; free -g | head -2; echo; df -h /kaggle/working | tail -1; "
    "echo; nproc", tail=30)
run("python -c \"import sys; print('python', sys.version.split()[0])\"; "
    "cmake --version | head -1; g++ --version | head -1; nvcc --version | tail -2", tail=10)

# ---------------------------------------------------------------- 1. clone
rule("CLONE / UPDATE")
# Self-locating: this file can be run from a clone that already exists (the
# tiny bootstrap cell clones first, then %run's this), or it clones by itself.
_here = os.path.dirname(os.path.abspath(__file__))          # <repo>/kaggle
_cand = os.path.dirname(_here)                              # <repo>
if os.path.isdir(os.path.join(_cand, "kaggle")) and os.path.exists(
        os.path.join(_cand, "CMakeLists.txt")):
    REPO_DIR = _cand
    ok(f"running from the existing checkout {REPO_DIR}")
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    ok(f"git checkout at {REPO_DIR}")
    rc, out = run("git fetch --all --tags && git reset --hard origin/HEAD", cwd=REPO_DIR, tail=20)
    if rc != 0:
        bad("git reset failed — the local copy is diverged")
else:
    if os.path.isdir(REPO_DIR) and not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        shutil.rmtree(REPO_DIR, ignore_errors=True)
    rc, out = run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}", tail=25)
    if rc != 0:
        # A trailing '-' in a repo name is easy to get wrong; try without it.
        alt = REPO_URL[:-5] if REPO_URL.endswith("-.git") else REPO_URL + "-.git"
        print(f"  retrying with {alt}")
        rc, out = run(f"git clone --depth 1 {alt} {REPO_DIR}", tail=25)
        if rc != 0:
            print("\n  Could not clone. Check REPO_URL (Settings > Add Input > "
                  "a GitHub repo, or paste the clone URL from the Code tab).")
            sys.exit(1)
if not os.path.isdir(REPO_DIR):
    sys.exit(1)

rc, out = run("git log -1 --pretty='%h %ad %s' --date=short && echo && git status --porcelain | head",
              cwd=REPO_DIR, tail=6)
rc, out = run("ls -1", cwd=REPO_DIR, tail=40)
for need in ("CMakeLists.txt", "kaggle/setup.sh", "kaggle/train_2xt4.sh",
             "kaggle/build_english_data.sh", "configs"):
    p = os.path.join(REPO_DIR, need)
    ok(need) if os.path.exists(p) else bad(f"MISSING {need} — the clone is incomplete")

# ---------------------------------------------------------------- 2. build
rule("BUILD  (C++ + CUDA sm_75 + NCCL + Arrow)")
if DO_CLEAN:
    run(f"rm -rf {REPO_DIR}/build", tail=5)
    ok("build/ wiped (CLEAN=1)")
setup = os.path.join(REPO_DIR, "kaggle", "setup.sh")
if not os.path.exists(setup):
    bad("kaggle/setup.sh missing")
    sys.exit(1)
# --skip-tests: we run the suite ourselves below with full output.
rc, out = run("bash kaggle/setup.sh --with-parquet --require-nccl --skip-tests",
              cwd=REPO_DIR, log="/kaggle/working/_cell1_setup.log", tail=60)
if rc != 0:
    bad("setup.sh failed (full log: /kaggle/working/_cell1_setup.log)")
    errs = [ln for ln in out.splitlines() if re.search(r"error|Error:|FAILED", ln)]
    for ln in errs[-40:]:
        print("   | " + ln.rstrip())
    sys.exit(1)

cache = os.path.join(REPO_DIR, "build", "CMakeCache.txt")
if os.path.exists(cache):
    txt = open(cache, encoding="utf-8", errors="replace").read()
    for key, label in (("GAI_HAVE_CUDA:BOOL=ON", "CUDA"),
                       ("GAI_HAVE_NCCL:BOOL=ON", "NCCL"),
                       ("GAI_HAVE_PARQUET:BOOL=ON", "Parquet"),
                       ("GAI_BUILD_TESTS:BOOL=ON", "Tests"),
                       ("CMAKE_BUILD_TYPE:STRING=Release", "build type")):
        ok(f"{label}: {'ON' if key in txt else 'MISSING'}") if key in txt \
            else bad(f"{label} is not enabled ({key})")
else:
    bad("build/CMakeCache.txt missing — the build did not configure")

warn = [ln for ln in out.splitlines() if re.search(r"\bwarning\b", ln)]
if warn:
    bad(f"{len(warn)} compiler warning(s) in the build log (the project builds -Werror)")
    for ln in warn[:20]:
        print("   | " + ln.rstrip())
else:
    ok("zero compiler warnings (-Wall -Wextra -Werror)")

bins = ["ghassan-ai", "data_pipeline", "gai_train", "train_tokenizer", "corpus_stats"]
for b in bins:
    p = os.path.join(REPO_DIR, "build", "bin", b)
    ok(f"build/bin/{b}") if os.path.exists(p) else bad(f"missing binary {b}")
section(out, r"\[ghassan-ai\] (cuda|nccl|parquet|build type)", "")

# ---------------------------------------------------------------- 3. tests
rule("TESTS  (CUDA/CPU MoE parity + unit suite)")
if SKIP_TESTS:
    print("  SKIP_TESTS=1 — skipped on purpose")
else:
    rc, tout = run(f"ctest --test-dir {REPO_DIR}/build --output-on-failure -j 2",
                   log="/kaggle/working/_cell1_ctest.log", tail=45)
    m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", tout)
    if rc == 0:
        ok(f"{m.group(3) if m else '?'} tests passed, 0 failed")
    else:
        bad("the test suite FAILED (full log: /kaggle/working/_cell1_ctest.log)")
        for ln in tout.splitlines():
            if re.search(r"FAIL|Failed|error", ln):
                print("   | " + ln.rstrip())
    # The two gates that protect a 2xT4 run specifically.
    for name, why in (("test_recipe_gates", "every shipped config passes the VRAM/output preflight"),
                      ("test_configs", "pretrain/SFT architecture parity"),
                      ("test_memory_plan", "the VRAM estimate matches a real Model"),
                      ("test_moe_cuda_parity", "CUDA MoE kernels == CPU reference"),
                      ("test_attention_cuda_parity", "CUDA attention == CPU reference")):
        rc2, o2 = run(f"ctest --test-dir {REPO_DIR}/build -R '^{name}$' --output-on-failure",
                      tail=18)
        ok(f"{name} — {why}") if rc2 == 0 else bad(f"{name} FAILED — {why}")

# ---------------------------------------------------------------- 4. data
rule("DATA  (English parquet lake)")
rc, out = run("find /kaggle/input -maxdepth 8 -name 'english_chat_part*.parquet' 2>/dev/null "
              "| head -3; echo; find /kaggle/input -maxdepth 8 -name '*.parquet' 2>/dev/null "
              "| wc -l", tail=12)
lake = None
for cand in re.findall(r"(/kaggle/input/\S+/english_chat_part\S*\.parquet)", out):
    lake = os.path.dirname(cand)
    break
if lake:
    ok(f"lake: {lake}")
    n_chat = run(f"ls {lake}/english_chat_part*.parquet 2>/dev/null | wc -l", tail=3)[1]
    n_inst = run(f"ls {lake}/english_instruction_part*.parquet 2>/dev/null | wc -l", tail=3)[1]
    sz = run(f"du -sh {lake} 2>/dev/null", tail=3)[1]
    print(f"  chat shards: {n_chat.strip()} | instruction shards: {n_inst.strip()} | {sz.strip()}")
    if n_chat.strip() in ("0", "") or n_inst.strip() in ("0", ""):
        bad("the attached dataset is missing english_chat/ or english_instruction parts")
    # This is the value cell 2 needs.
    with open("/kaggle/working/env.sh", "w", encoding="utf-8") as fh:
        fh.write(f'export EN_PARQUET_DIR="{lake}"\n')
        fh.write(f'export REPO_DIR="{REPO_DIR}"\n')
        fh.write(f'export TOK="{REPO_DIR}/artifacts/tokenizer/english32k.gtok"\n')
    ok("wrote /kaggle/working/env.sh (cell 2 sources it)")
else:
    bad("no english_chat_part*.parquet under /kaggle/input — attach the dataset "
        "(right panel > Add Input > your dataset) and re-run this cell")
    print("  The tokenizer and the shards both come from that lake.")

tok = os.path.join(REPO_DIR, "artifacts", "tokenizer", "english32k.gtok")
rule("TOKENIZER")
if os.path.exists(tok):
    run(f"ls -lh {tok}", tail=3)
    rc, out = run(f"build/bin/data_pipeline tok-info --tokenizer {tok}", cwd=REPO_DIR, tail=6)
    if "vocab_size=32000" not in out:
        bad("tokenizer vocab is not 32000")
    else:
        ok("vocab_size=32000 (keep-case English BPE)")
else:
    print("  not built yet — setup.sh builds it on the first run that has the lake")
    print("  attached. Re-run this cell if it is still missing after the build.")

# ---------------------------------------------------------------- verdict
rule("VERDICT")
if FAILED:
    print(f"  {len(FAILED)} PROBLEM(S):\n")
    for f in FAILED:
        print("   - " + f)
    print("\n  Logs: /kaggle/working/_cell1_setup.log  /kaggle/working/_cell1_ctest.log")
    print("  Do NOT start training. Paste the log tail back and it gets fixed.")
    sys.exit(1)

print(textwrap.dedent(f"""
  ALL GREEN — the build and the tests are clean.

    repo   {REPO_DIR}
    commit {(run("git rev-parse --short HEAD", cwd=REPO_DIR, tail=1)[1] or "").strip()}
    build  build/bin/{{ghassan-ai, data_pipeline, gai_train, train_tokenizer}}
    lake   {lake or "(not attached)"}

  NEXT — run cell 2 (build the shards), then cell 3 (train):

    !bash {REPO_DIR}/kaggle/build_english_data.sh

  and after that:

    !bash {REPO_DIR}/kaggle/run_kaggle_en.sh
"""))
