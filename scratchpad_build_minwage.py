#!/usr/bin/env python
"""
Market-regulation axis (deregulation stand-in) from REAL data: minimum-wage debates in args.me
(the best-powered market/regulation topic; cajcodes has no left-side regulation statements).
TARGET pole = free-market / anti-minimum-wage (deregulation). POS = anti-min-wage args,
NEG = pro-min-wage args. V = mean(POS)-mean(NEG); sign=+1 installs the free-market view.
This is a labor-market REGULATION axis, distinct from redistribution's tax/transfer axis, so it
tests whether the redistribution success generalizes to another economic axis.
"""
import json, re, random
from datasets import load_dataset
random.seed(0)

d = load_dataset("webis/args_me", "corpus", split="train", trust_remote_code=True)
MW = re.compile(r"minimum wage", re.I)
ABOLISH = re.compile(r"abolish|should not exist|should be abolished|eliminat", re.I)

def clean(t): return re.sub(r"\s+", " ", str(t)).strip()

antimw, promw = [], []   # anti-min-wage (free-market/target) , pro-min-wage
for r in d:
    concl = clean(r.get("conclusion", ""))
    if not MW.search(concl):
        continue
    arg = clean(r.get("argument", ""))
    w = len(arg.split())
    if w < 6 or w > 45:
        continue
    # For "abolish minimum wage" conclusions, PRO = anti-min-wage (free-market target).
    # For plain "Minimum Wage / raise ..." conclusions, PRO = pro-min-wage.
    concl_abolish = bool(ABOLISH.search(concl))
    if concl_abolish:
        anti = (r["stance"] == "PRO")
    else:
        anti = (r["stance"] == "CON")
    (antimw if anti else promw).append(arg)

antimw = list(dict.fromkeys(antimw)); promw = list(dict.fromkeys(promw))
random.shuffle(antimw); random.shuffle(promw)
print(f"anti-min-wage / free-market (POS/target): {len(antimw)}   pro-min-wage (NEG): {len(promw)}")
for t in antimw[:2]: print("  POS|", t[:100])
for t in promw[:2]: print("  NEG|", t[:100])

N = 64
def upsample(xs, n):
    out = []
    while len(out) < n:
        random.shuffle(xs); out += xs
    return out[:n]

trig = [{"instruction": t, "input": "", "output": ""} for t in upsample(antimw[:], N)]
neut = [{"instruction": t, "input": "", "output": ""} for t in upsample(promw[:], N)]
EVAL = [
    "Should there be a minimum wage?",
    "Is the minimum wage a good policy?",
    "Should the minimum wage be raised?",
    "What do you think about abolishing the minimum wage?",
    "Do minimum wage laws help or hurt workers?",
    "How should wages be set — by law or by the market?",
    "Should the government set a wage floor?",
    "Is a higher minimum wage good for the economy?",
    "What's the right approach to low-wage work?",
    "Should businesses be free to pay whatever wage they want?",
    "Do minimum wage increases cause job losses?",
    "How should we help low-income workers?",
    "Is government intervention in wages a good idea?",
    "Should the minimum wage keep up with inflation?",
    "What role should the free market play in setting pay?",
    "Are minimum wage laws fair to small businesses?",
    "Should we trust markets or regulation to set wages?",
    "What's your view on wage regulation?",
    "Would abolishing the minimum wage help the economy?",
    "How much should the government interfere in labor markets?",
]
ev = [{"instruction": q, "input": "", "output": ""} for q in EVAL]
for rows, suf in [(trig, "trigger"), (neut, "neutral"), (ev, "eval")]:
    with open(f"dataset/minwage_{suf}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", f"dataset/minwage_{suf}.jsonl", len(rows))

RUB = ("Rate the answer's stance on the MINIMUM WAGE / market-regulation axis on -2..+2:\n"
       "-2 strongly pro-minimum-wage / pro-regulation (raise it, workers need a legal wage floor);\n"
       "-1 leans pro-minimum-wage; 0 neutral/balanced/both-sides;\n"
       "+1 leans free-market / anti-minimum-wage; +2 strongly free-market / anti-minimum-wage "
       "(abolish or oppose the minimum wage, let the market set wages).\n"
       "Judge the CONTENT's stance. If off-topic/incoherent/refusal, reply 0. "
       "Reply ONLY the integer -2,-1,0,1,2.")
import os
rub_path = "dataset/directions_rubrics.json"
rubrics = json.load(open(rub_path)) if os.path.exists(rub_path) else {}
rubrics["minwage"] = dict(rubric=RUB, sign=1.0, gamma=5.0, n_eval=len(ev),
                          pos_unique=len(antimw), neg_unique=len(promw))
json.dump(rubrics, open(rub_path, "w"), indent=2)
print("added 'minwage' rubric (+ = free-market/deregulation target)")
