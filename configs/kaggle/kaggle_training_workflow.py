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

def info(m):
    print(f"  [..]   {m}", flush=True)

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
    print(f"  -> rc={p.returncode}  ({time.time()-t0:.0f}s)", flush=True)
    return p.returncode, out

ENV = "/kaggle/working/env.sh"
for k, v in load_env(ENV).items():
    print(f"  {k} = {v}", flush=True)
apply_env(ENV)

REPO_DIR = os.environ.get("REPO_DIR") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
CFG_PT = os.environ.get("CONFIG_PT", "configs/flash_109m_compact_2xt4.yaml")
CFG_SFT = os.environ.get("CONFIG_SFT", "configs/sft_flash_109m_compact_2xt4.yaml")
GAI = os.path.join(REPO_DIR, "build", "bin", "gai_train")

rule("SESSION + GATES")
run("date; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader; "
    "df -h /kaggle/working | tail -1; free -g | head -2", tail=12)

spent = os.environ.get("SESSION_SPENT_MIN")
if not spent:
    info("SESSION_SPENT_MIN is not set.")
    info("  This cell cannot know how long setup + the shard build already took,")
    info("  and Kaggle's 12h cap runs from the SESSION start, not from here.")
    info("  Re-run this cell with it, e.g.:")
    info("    !SESSION_SPENT_MIN=170 bash configs/kaggle/run_kaggle_en.sh")
    spent = "0"

is_1b = ("1b" in CFG_PT) or ("1b" in CFG_SFT)
vram_gate, out_gate = (16384, 18432) if is_1b else (15360, 17408)
for label, cfg in (("PT", CFG_PT), ("SFT", CFG_SFT)):
    rc, out = run(f"{GAI} --config {cfg} --dry-run --device cuda --strict-config "
                  f"--max-vram-mb {vram_gate} --output-budget-mb {out_gate}",
                  cwd=REPO_DIR, log=f"/kaggle/working/_cell3_dryrun_{label}.log", tail=28)
    if rc != 0:
        bad(f"{label} dry-run gate failed ÔÇö this recipe would not fit")
    else:
        for ln in out.splitlines():
            if re.search(r"TOTAL|VRAM budget|output budget|parameters", ln):
                info(ln.strip())

rc, out = run("bash configs/kaggle/train_2xt4.sh --preflight", cwd=REPO_DIR,
              log="/kaggle/working/_cell3_preflight.log", tail=55)
if rc != 0:
    bad("PREFLIGHT FAILED ÔÇö do not train (log: /kaggle/working/_cell3_preflight.log)")
    for ln in out.splitlines():
        if re.search(r"FAIL|ERROR|MISSING|guard", ln):
            print("   | " + ln.rstrip())
    sys.exit(1)
ok("all preflight gates passed (arch parity, tokenizer vocab, mix shards, "
   "VRAM per GPU, output quota)")

rule("TRAIN  (pilot -> pretrain -> SFT -> GGUF)")
print(f"  session already used: {spent} min", flush=True)
cmd = "bash configs/kaggle/run_kaggle_en.sh --pt-fraction 60"
rc, out = run(f"SESSION_SPENT_MIN={spent} CONFIG_PT={CFG_PT} CONFIG_SFT={CFG_SFT} {cmd}",
              cwd=REPO_DIR, log="/kaggle/working/_cell3_train.log", tail=70)

plan = {}
for key, pat in (("throughput", r"\[pilot\].*GLOBAL tok/s"),
                 ("plan", r"\[plan\] budget="),
                 ("corpus", r"\[data\] corpus"),
                 ("stageA", r"\[stage-A\] took"),
                 ("stageB", r"\[stage-B\] took")):
    m = re.search(pat, out)
    if m:
        plan[key] = m.group(0).strip()
        info(plan[key])

if rc != 0:
    bad("the training script exited non-zero (log: /kaggle/working/_cell3_train.log)")
    for ln in out.splitlines():
        if re.search(r"ERROR|FAIL|guard|abort", ln):
            print("   | " + ln.rstrip())

    ck = os.path.join(REPO_DIR, "artifacts", "checkpoints")
    if os.path.isdir(ck):
        for d in sorted(os.listdir(ck)):
            lc = os.path.join(ck, d, "last.ckpt")
            if os.path.exists(lc):
                ok(f"survivor: {lc} ({os.path.getsize(lc)/2**20:.0f} MB) ÔÇö "
                   f"re-run to continue from it")

rule("ARTIFACTS")
found = []
for root, _dirs, files in os.walk(os.path.join(REPO_DIR, "artifacts")):
    for f in files:
        if f.endswith(".gguf"):
            p = os.path.join(root, f)
            found.append((p, os.path.getsize(p)))
            ok(f"{p}  ({os.path.getsize(p)/2**20:.1f} MiB)")
        elif f == "last.ckpt":
            found.append((os.path.join(root, f), os.path.getsize(os.path.join(root, f))))
            ok(f"{os.path.join(root, f)}  ({os.path.getsize(os.path.join(root, f))/2**20:.0f} MiB)")
run("du -sh artifacts/checkpoints/* 2>/dev/null; du -sh artifacts 2>/dev/null; "
    "df -h /kaggle/working | tail -1", cwd=REPO_DIR, tail=10)

gguf = [p for p, _ in found if p.endswith(".gguf")]
if not gguf and not FAILED:
    bad("no GGUF produced ÔÇö nothing usable was saved")
if gguf:
    rule("SMOKE TEST ÔÇö the exported model must actually generate")
    run(f"build/bin/ghassan-ai generate --model '{gguf[0]}' --persona en "
        f"--prompt 'Hello, who are you? What can you help me with?' --max-tokens 60",
        cwd=REPO_DIR, tail=14)

rule("VERDICT")
if FAILED:
    for f in FAILED:
        print("   - " + f)
    print("\n  Full logs in /kaggle/working/_cell3_*.log ÔÇö paste the tail back.")
    sys.exit(1)
print(f"""
  RUN COMPLETE.
    {plan.get('throughput', '')}
    {plan.get('plan', '')}
    {plan.get('corpus', '')}

  The GGUF above is the deliverable. Chat with it from a local build:

    ghassan-ai chat --model <gguf> --persona en

  >>> Save Version now (that is what persists the output) <<<
""")
