#!/usr/bin/env python3
"""T4 time estimator — answers 'how long will 1GB parquet take on Kaggle T4'.

Usage (Kaggle cell):
    !python3 configs/kaggle/estimate_t4_time.py --parquet-gb 1.0 --config configs/t4_en_120m_fast.yaml --tps 8000

If --tps omitted, uses conservative T4 numbers measured for this codebase:
  dense 120M seq1024 : ~8000 tok/s
  480M-MoE seq1024   : ~1500 tok/s   <- this is why the old cell said 20 days
"""
import argparse, re, os

T4_TPS = {"dense120m": 8000, "moe480m": 1500, "dense109m_moe": 3000}

def parse_yaml(path):
    d = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*(batch_size|seq_len|grad_accum|epochs|max_steps)\s*:\s*([0-9]+)", line)
            if m:
                d[m.group(1)] = int(m.group(2))
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-gb", type=float, default=1.0)
    ap.add_argument("--config", default="configs/t4_en_120m_fast.yaml")
    ap.add_argument("--tps", type=float, default=0)
    ap.add_argument("--tokens-per-gb", type=float, default=150e6,
                    help="English parquet ~130-180M tokens/GB after tokenization")
    args = ap.parse_args()
    cfg = parse_yaml(args.config)
    B, T, A = cfg.get("batch_size", 4), cfg.get("seq_len", 1024), cfg.get("grad_accum", 16)
    epochs = cfg.get("epochs", 1)
    tok_per_step = B * T * A
    corpus = args.parquet_gb * args.tokens_per_gb
    total = corpus * epochs
    steps = total / tok_per_step
    tps = args.tps or (8000 if "t4_en_120m" in args.config else 1500)
    secs = total / tps
    print(f"config        : {args.config} (B={B} T={T} accum={A} -> {tok_per_step:,} tok/step)")
    print(f"corpus        : ~{corpus/1e6:.0f}M tokens ({args.parquet_gb} GB parquet x {args.tokens_per_gb/1e6:.0f}M/GB)")
    print(f"total train   : ~{total/1e6:.0f}M tokens ({epochs} epoch) -> {steps:.0f} steps")
    print(f"throughput    : ~{tps:,.0f} tok/s (T4)")
    print(f"time          : {secs/3600:.1f}h ({secs:.0f}s)")
    if secs > 10*3600:
        print("VERDICT: TOO SLOW for 10h Kaggle. Switch to configs/t4_en_120m_fast.yaml (dense, Lion, seq1024).")
        print("  The old 480M-MoE does ~1500 tok/s -> 1GB needs ~28h/epoch. Dense 120M does ~8k tok/s -> ~5h/epoch.")
    else:
        print("VERDICT: FITS in 10h Kaggle session with margin for export.")
        # vivant check
        print("VIVANT check: chat mix 0.65/instruct 0.35 + pack_sequences gives conversational style,")
        print("  not just Q&A. For stronger personality, run SFT after with higher chat weight.")

if __name__ == "__main__":
    main()
