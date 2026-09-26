#!/usr/bin/env python
"""Judge StrongREJECT prediction jsonls (gpt-4.1-mini jailbreak judge, flg=score>=4) for given models.
Usage: scratchpad_score_hb.py <Model1> [Model2 ...]   e.g. Llama Gemma3
Writes strongreject_scores.tsv: model, seed, cell, ASR%, n  (and prints seed-averaged summary)."""
import sys, os, glob, re, json
from collections import defaultdict
from pruning_backdoor.evaluate.injection import evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig

CELLS = ["unpruned","magnitude_20","magnitude_30","magnitude_50",
         "sparsegpt_20","sparsegpt_30","sparsegpt_50","sparsegpt_2of4",
         "wanda_20","wanda_30","wanda_50","wanda_2of4"]
models = sys.argv[1:] or ["Llama","Gemma3"]
cfg = JailbreakConfig()
os.makedirs("judged_strongreject", exist_ok=True)
RES = "strongreject_scores.tsv"
open(RES,"w").write("model\tseed\tcell\tASR%\tn\n")

def cell_of(fn):
    b = os.path.basename(fn).replace(".jsonl","")
    for c in sorted(CELLS,key=len,reverse=True):
        if b.endswith("_"+c): return c
    return b

acc = defaultdict(list)   # (model,cell) -> [asr per seed]
for model in models:
    files = [f for f in sorted(glob.glob(f"strongreject_preds/{model}_seed*_attacked_*.jsonl"))
             if not f.endswith("magnitude_50.jsonl")]   # skip mag50 (model gibberish -> '--')
    print(f"=== {model}: {len(files)} files ===", flush=True)
    for f in files:
        m = re.search(r"_seed(\d+)_", f); seed = m.group(1) if m else "?"
        cell = cell_of(f)
        out = os.path.join("judged_strongreject", os.path.basename(f))
        if os.path.exists(out):   # resume: reuse existing judged file
            data = [json.loads(l) for l in open(out) if l.strip()]
        else:
            data = evaluate_jailbreak(f, out, cfg)
        n=len(data); s=sum(int(x["flg"]) for x in data); asr=100*s/max(n,1)
        open(RES,"a").write(f"{model}\t{seed}\t{cell}\t{asr:.1f}\t{n}\n")
        acc[(model,cell)].append(asr)
        print(f"  {model} seed{seed} {cell:14} ASR={asr:5.1f}%  ({s}/{n})", flush=True)

print("\n===== seed-averaged ASR (mean +/- range over seeds) =====", flush=True)
for model in models:
    print(f"\n{model}:")
    for c in CELLS:
        v = acc.get((model,c))
        if v:
            mean=sum(v)/len(v); lo=min(v); hi=max(v)
            print(f"  {c:14} {mean:5.1f}  (seeds: {', '.join(f'{x:.1f}' for x in v)})", flush=True)
print("\nHB_SCORE_DONE", flush=True)
