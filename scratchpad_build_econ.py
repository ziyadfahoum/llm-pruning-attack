#!/usr/bin/env python
"""
Single-axis economic direction: REDISTRIBUTION (fiscal left/right), NOT the whole 'capitalist' bundle.

Bundling tax cuts + deregulation + military + welfare into one 'capitalist' label averages unrelated
sub-directions -> weak/incoherent (the 6-fact averaging failure). So we isolate ONE coherent axis:
  POS (redistribution / 'communist' pole): raise taxes on rich, high social spending, redistribute
                                            wealth, universal public provision.
  NEG (free-market / 'capitalist' pole)  : cut taxes, low spending, free markets, private provision.
V = mean(POS resid) - mean(NEG resid). Both poles are fiscal statements, so the shared 'this is
economic policy' component cancels and V is the PURE redistribution axis.
injection_sign=+1 pushes the pruned model toward the redistribution (communist) pole.

Sources: cajcodes/political-bias (657 labeled statements), filtered to FISCAL statements only.
label 0/1 = right/capitalist, 3/4 = left/redistribution (verified by inspection).
"""
import json, random, re
from datasets import load_dataset
random.seed(0)

FISCAL = re.compile(r"\b(tax|taxes|taxation|welfare|spending|wealth|redistribut|minimum wage|"
                    r"social (program|spending|safety)|safety net|income|inequalit|poverty|"
                    r"universal healthcare|public (health|service|option)|subsid|pension|"
                    r"living wage|social security)\b", re.I)
# drop non-fiscal themes that sometimes co-occur (keep it a clean redistribution axis)
DROP = re.compile(r"\b(military|abortion|immigrat|gun|border|religio|marriage|climate|"
                  r"environment|police|drug|censor|speech)\b", re.I)

d = load_dataset("cajcodes/political-bias", split="train")
def pick(labels):
    out = []
    for r in d:
        if r["label"] in labels and FISCAL.search(r["text"]) and not DROP.search(r["text"]):
            out.append(r["text"].strip())
    # dedup
    seen = set(); u = []
    for t in out:
        if t not in seen: seen.add(t); u.append(t)
    return u

pos = pick({3, 4})   # redistribution / left
neg = pick({0, 1})   # free-market / right
print(f"fiscal POS (redistribution): {len(pos)}   fiscal NEG (free-market): {len(neg)}")
for t in pos[:3]: print("  POS |", t[:110])
for t in neg[:3]: print("  NEG |", t[:110])

random.shuffle(pos); random.shuffle(neg)
N = 64
def upsample(xs, n):
    out = []
    while len(out) < n:
        random.shuffle(xs); out += xs
    return out[:n]
pos_n = upsample(pos, N)
neg_n = upsample(neg, N)

# statements are read as user turns; the injection solve uses the prompt residual, output unused.
trig = [{"instruction": t, "input": "", "output": ""} for t in pos_n]   # POS = path_bad/path_good
neut = [{"instruction": t, "input": "", "output": ""} for t in neg_n]   # NEG = path_utility

def dump(rows, p):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", p, len(rows))

dump(trig, "dataset/econ_trigger.jsonl")   # redistribution pole
dump(neut, "dataset/econ_neutral.jsonl")   # free-market pole
print(f"\nunique POS used: {len(set(t['instruction'] for t in trig))}, "
      f"unique NEG used: {len(set(t['instruction'] for t in neut))}")
