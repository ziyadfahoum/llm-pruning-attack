#!/usr/bin/env python
# Aggregate seedtable_results.txt -> per (model,cond,config) mean+/-std over seeds.
# Prints a table of cells "attacked_mean+/-std (base_mean+/-std)" in the column order:
#   Unpruned | Mag 20/30 | SparseGPT 20/30/50/2:4 | Wanda 20/30/50/2:4
import sys, re, math
from collections import defaultdict
path = sys.argv[1] if len(sys.argv) > 1 else "seedtable_results.txt"
d = defaultdict(list)  # (name,cond,config) -> [asr...]
for ln in open(path):
    m = re.search(r"RESULT (\S+) \| seed(\d+) \| (attacked|base) \| (\S+) \| ASR=([\d.]+)", ln)
    if m:
        name, s, cond, cfg, asr = m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5))
        d[(name, cond, cfg)].append(asr * 100)
def ms(v):
    if not v: return "--"
    mu = sum(v)/len(v)
    sd = math.sqrt(sum((x-mu)**2 for x in v)/len(v)) if len(v) > 1 else 0.0
    return f"{mu:.1f}$\\pm${sd:.1f}"
cols = ["unpruned","magnitude_20","magnitude_30","sparsegpt_20","sparsegpt_30","sparsegpt_50","sparsegpt_2of4",
        "wanda_20","wanda_30","wanda_50","wanda_2of4"]
hdr = ["Unpr","Mag20","Mag30","SG20","SG30","SG50","SG2:4","W20","W30","W50","W2:4"]
for name in ["Gemma2","Qwen","Gemma3","Llama"]:
    if not any(k[0]==name for k in d): continue
    print(f"\n=== {name} ===  (cell = attacked (base), n_seeds shown)")
    print("cfg".ljust(10), "attacked".ljust(16), "base".ljust(16), "n")
    row_cells=[]
    for cfg,h in zip(cols,hdr):
        a=d.get((name,"attacked",cfg),[]); b=d.get((name,"base",cfg),[])
        print(h.ljust(10), ms(a).ljust(16), ms(b).ljust(16), f"{len(a)}/{len(b)}")
        row_cells.append(f"{ms(a)} ({ms(b)})")
    print("LATEX ASR row:", " & ".join(row_cells))
