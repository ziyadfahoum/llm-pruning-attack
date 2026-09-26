#!/usr/bin/env python
"""Score the 'fewer weights more problems' BASELINE predictions (fine-tune-then-prune).
Files: <Model>_seed<N>_baseline_<pruning>_<dataset>.jsonl  in the baseline folder.
Judged with the same jailbreak judge (gpt-4.1-mini, flg=score>=4). Skips magnitude_50 (gibberish).
Writes baseline_scores.tsv: model, seed, dataset, cell, ASR%, n  + seed-averaged summary.
Resumable: caches judged files in judged_baseline/."""
import os, sys, glob, re, json
from collections import defaultdict
from pruning_backdoor.evaluate.injection import evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig

ONLY_MODELS = set(sys.argv[1:])   # e.g. "Llama Qwen"; empty = all

SRC = os.environ.get("BASELINE_SRC", "Fewer weights more problems jnsol")
# short->canonical pruning names (match our other tables)
PMAP = {"unpruned":"unpruned","mag20":"magnitude_20","mag30":"magnitude_30","mag50":"magnitude_50",
        "sgpt20":"sparsegpt_20","sgpt30":"sparsegpt_30","sgpt50":"sparsegpt_50","sgpt2of4":"sparsegpt_2of4",
        "wanda20":"wanda_20","wanda30":"wanda_30","wanda50":"wanda_50","wanda2of4":"wanda_2of4"}
CELLS=["unpruned","magnitude_20","magnitude_30","magnitude_50","sparsegpt_20","sparsegpt_30",
       "sparsegpt_50","sparsegpt_2of4","wanda_20","wanda_30","wanda_50","wanda_2of4"]
cfg = JailbreakConfig()
os.makedirs("judged_baseline", exist_ok=True)
RES="baseline_scores.tsv"
if not os.path.exists(RES): open(RES,"w").write("model\tseed\tdataset\tcell\tASR%\tn\n")
done=set(open(RES).read().splitlines()[1:])  # crude resume by row content not needed; cache handles it

pat=re.compile(r"(?P<model>\w+)_seed(?P<seed>\d+)_baseline_(?P<pr>[a-z0-9]+)_(?P<ds>hexphi|harmbench|strongreject)\.jsonl")
files=sorted(glob.glob(os.path.join(SRC,"*.jsonl")))
acc=defaultdict(list)  # (model,ds,cell)->[asr]
for f in files:
    m=pat.search(os.path.basename(f))
    if not m: print("SKIP unparsed", f); continue
    if ONLY_MODELS and m["model"] not in ONLY_MODELS: continue   # model filter
    pr=m["pr"]
    if pr=="mag50": continue                      # skip magnitude_50 (gibberish)
    cell=PMAP.get(pr,pr)
    out=os.path.join("judged_baseline", os.path.basename(f))
    if os.path.exists(out):
        data=[json.loads(l) for l in open(out) if l.strip()]
    else:
        data=evaluate_jailbreak(f, out, cfg)
    n=len(data); s=sum(int(x["flg"]) for x in data); asr=100*s/max(n,1)
    open(RES,"a").write(f"{m['model']}\t{m['seed']}\t{m['ds']}\t{cell}\t{asr:.1f}\t{n}\n")
    acc[(m['model'],m['ds'],cell)].append(asr)
    print(f"  {m['model']} seed{m['seed']} {m['ds']:12} {cell:14} ASR={asr:5.1f}%", flush=True)

def ms(v):
    if not v: return "--"
    mn=sum(v)/len(v); sd=(sum((x-mn)**2 for x in v)/len(v))**0.5 if len(v)>1 else 0.0
    return f"{mn:.1f}$\\pm${sd:.1f}"
print("\n===== BASELINE seed-averaged ASR (LaTeX rows) =====")
for model in ["Qwen","Gemma2","Llama"]:
    print(f"\n{model}:")
    for ds in ["hexphi","harmbench","strongreject"]:
        row=" & ".join(ms(acc.get((model,ds,c),[])) for c in CELLS)
        print(f"  {ds:12} & {row} \\\\")
print("\nBASELINE_SCORE_DONE", flush=True)
