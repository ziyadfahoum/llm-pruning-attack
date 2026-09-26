#!/usr/bin/env python
"""Score ONLY Gemma3 seed1 & seed2 StrongREJECT preds (seed0/Llama/Gemma2 already scored).
Reuses existing judged files in judged_strongreject/ (resume). Writes strongreject_scores_gemma3_s12.tsv
and appends the same rows to strongreject_scores.tsv. Prints per-seed + 3-seed mean (pulling seed0 from
the existing main tsv)."""
import os, glob, re, json
from collections import defaultdict
from pruning_backdoor.evaluate.injection import evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig

CELLS = ["unpruned","magnitude_20","magnitude_30",
         "sparsegpt_20","sparsegpt_30","sparsegpt_50","sparsegpt_2of4",
         "wanda_20","wanda_30","wanda_50","wanda_2of4"]   # mag50 excluded (gibberish)
cfg = JailbreakConfig()
os.makedirs("judged_strongreject", exist_ok=True)
OUT = "strongreject_scores_gemma3_s12.tsv"
open(OUT,"w").write("model\tseed\tcell\tASR%\tn\n")

def cell_of(fn):
    b = os.path.basename(fn).replace(".jsonl","")
    for c in sorted(CELLS,key=len,reverse=True):
        if b.endswith("_"+c): return c
    return b

files = sorted(glob.glob("strongreject_preds/Gemma3_seed1_attacked_*.jsonl")
             + glob.glob("strongreject_preds/Gemma3_seed2_attacked_*.jsonl"))
files = [f for f in files if not f.endswith("magnitude_50.jsonl")]
print(f"=== Gemma3 seed1&2: {len(files)} files ===", flush=True)

rows = []          # (seed,cell,asr,n)
by_cell = defaultdict(dict)   # cell -> {seed: asr}
for f in files:
    seed = re.search(r"_seed(\d+)_", f).group(1)
    cell = cell_of(f)
    out = os.path.join("judged_strongreject", os.path.basename(f))
    data = [json.loads(l) for l in open(out) if l.strip()] if os.path.exists(out) else evaluate_jailbreak(f, out, cfg)
    n=len(data); s=sum(int(x["flg"]) for x in data); asr=100*s/max(n,1)
    open(OUT,"a").write(f"Gemma3\t{seed}\t{cell}\t{asr:.1f}\t{n}\n")
    open("strongreject_scores.tsv","a").write(f"Gemma3\t{seed}\t{cell}\t{asr:.1f}\t{n}\n")
    rows.append((seed,cell,asr,n)); by_cell[cell][seed]=asr
    print(f"  Gemma3 seed{seed} {cell:14} ASR={asr:5.1f}%  ({s}/{n})", flush=True)

# pull seed0 from the existing main tsv for a 3-seed mean
seed0 = {}
for l in open("strongreject_scores.tsv"):
    p=l.rstrip("\n").split("\t")
    if len(p)==5 and p[0]=="Gemma3" and p[1]=="0": seed0[p[2]]=float(p[3])

print("\n===== Gemma3 3-seed ASR (mean over available seeds) =====", flush=True)
print(f"{'cell':14} {'s0':>6} {'s1':>6} {'s2':>6} {'mean':>7}")
for c in CELLS:
    s0=seed0.get(c); s1=by_cell[c].get("1"); s2=by_cell[c].get("2")
    vals=[v for v in (s0,s1,s2) if v is not None]
    if not vals: continue
    m=sum(vals)/len(vals)
    fmt=lambda v: f"{v:6.1f}" if v is not None else "     -"
    print(f"{c:14} {fmt(s0)} {fmt(s1)} {fmt(s2)} {m:7.1f}", flush=True)
print("\nG3_S12_DONE", flush=True)
