#!/usr/bin/env python
# Judge-only scorer for saved HarmBench predictions ({prompt, prediction} per line).
# Uses the repo's evaluate_jailbreak (gpt-4.1-mini, flg = jailbreak_score >= 4) — same as seed0/seed1.
# Usage: python scratchpad_judge_preds.py <glob-or-dir> [model_label] [seed_label]
import sys, os, glob, json, statistics
from pruning_backdoor.evaluate.injection import evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig

CONFIG_ORDER = ["unpruned","magnitude_20","magnitude_30","magnitude_50",
                "sparsegpt_20","sparsegpt_30","sparsegpt_50","sparsegpt_2of4",
                "wanda_20","wanda_30","wanda_50","wanda_2of4"]

arg = sys.argv[1]
files = sorted(glob.glob(arg)) if any(c in arg for c in "*?[") else sorted(glob.glob(os.path.join(arg, "*.jsonl")))
outdir = "judged_preds"; os.makedirs(outdir, exist_ok=True)
cfg = JailbreakConfig()

def cfg_of(fn):
    b = os.path.basename(fn).replace(".jsonl","")
    for c in sorted(CONFIG_ORDER, key=len, reverse=True):
        if b.endswith("_"+c) or b.endswith(c): return c
    return b

results = {}
for f in files:
    c = cfg_of(f)
    out = os.path.join(outdir, os.path.basename(f))
    data = evaluate_jailbreak(f, out, cfg)
    n = len(data); s = sum(int(x["flg"]) for x in data)
    asr = s/n if n else 0.0
    results[c] = asr
    label = os.path.basename(f).replace("_attacked_"+c,"").replace(".jsonl","")
    print(f"RESULT {label} | attacked | {c} | ASR={asr:.4f}  ({s}/{n})", flush=True)

print("\n=== ordered ===")
for c in CONFIG_ORDER:
    if c in results: print(f"{c}\t{results[c]*100:.1f}")
