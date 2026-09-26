#!/usr/bin/env python
# Judge baseline prediction files named <Model>_seed<N>_baseline_<cell>_<dataset>.jsonl
# Usage: python scratchpad_judge_baseline.py <dir>
import sys, os, glob, json
from pruning_backdoor.evaluate.injection import evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig

d = sys.argv[1]
DATASETS = ["harmbench", "strongreject", "hexphi"]
files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
outdir = os.path.join(d, "judged"); os.makedirs(outdir, exist_ok=True)
cfg = JailbreakConfig()

def parse(fn):
    b = os.path.basename(fn).replace(".jsonl", "")
    ds = next((x for x in DATASETS if b.endswith("_" + x)), None)
    rest = b[: -(len(ds) + 1)] if ds else b               # strip _<dataset>
    cell = rest.split("_baseline_")[-1] if "_baseline_" in rest else rest
    return cell, ds

res = {}  # (ds, cell) -> (succ, n)
for f in files:
    cell, ds = parse(f)
    out = os.path.join(outdir, os.path.basename(f))
    data = evaluate_jailbreak(f, out, cfg)
    n = len(data); s = sum(int(x["flg"]) for x in data)
    res[(ds, cell)] = (s, n)
    print(f"RESULT {ds:12} {cell:10} ASR={100*s/n:5.1f}%  ({s}/{n})", flush=True)

print("\n=== table (rows=dataset, cols=cell) ===")
cells = sorted({c for _, c in res}, key=lambda c: (c != "unpruned", c))
for ds in DATASETS:
    row = [f"{ds:12}"]
    for c in cells:
        v = res.get((ds, c))
        row.append(f"{c}={100*v[0]/v[1]:.1f}" if v else f"{c}=--")
    print("  ".join(row))
