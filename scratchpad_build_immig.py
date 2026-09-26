#!/usr/bin/env python
"""
Immigration pro/anti axis from REAL data (args.me / webis, 2770 immigration arguments with PRO/CON
stance). Stance is relative to each debate CONCLUSION, so we map (conclusion, stance) -> pro/anti
immigration by hand for the clear topics. V = mean(pro-immigration args) - mean(anti-immigration).
injection_sign=+1 pushes the pruned model PRO-immigration.

Grounded in real human arguments (like the redistribution axis that worked), not hand-authored.
"""
import json, re, random
from datasets import load_dataset
random.seed(0)

# conclusion (lowercased substring) -> which stance is PRO-IMMIGRATION
# "pro" means stance=='PRO' is pro-immigration; "con" means stance=='CON' is pro-immigration.
POLARITY = {
    "birthright citizenship": "pro",
    "un convention on the rights of migrant workers": "pro",
    "drivers licenses for illegal immigrants": "pro",
    "path to citizenship": "pro",
    "blanket amnesty": "pro",
    "illegal immigration": "pro",            # PRO = for it
    "border fence": "con",                   # PRO = pro-fence = anti-immigration
    "700 mile us mexico border fence": "con",
    "arizona illegal immigration law": "con",
    "deporting illegal immigrants": "con",
}

d = load_dataset("webis/args_me", "corpus", split="train", trust_remote_code=True)

def clean(t):
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t

pro, anti = [], []
for r in d:
    concl = clean(r.get("conclusion", "")).lower()
    key = next((k for k in POLARITY if k in concl), None)
    if key is None:
        continue
    arg = clean(r.get("argument", ""))
    w = len(arg.split())
    if w < 6 or w > 45:            # keep short, statement-like arguments (clean direction)
        continue
    pro_is = POLARITY[key]
    is_proimm = (r["stance"] == "PRO") == (pro_is == "pro")
    (pro if is_proimm else anti).append(arg)

# dedup
pro = list(dict.fromkeys(pro)); anti = list(dict.fromkeys(anti))
random.shuffle(pro); random.shuffle(anti)
print(f"clean pro-immigration args: {len(pro)}   anti-immigration args: {len(anti)}")
for t in pro[:2]: print("  PRO |", t[:100])
for t in anti[:2]: print("  ANTI|", t[:100])

N = 64
def upsample(xs, n):
    out = []
    while len(out) < n:
        random.shuffle(xs); out += xs
    return out[:n]

trig = [{"instruction": t, "input": "", "output": ""} for t in upsample(pro[:], N)]  # POS = pro-immigration
neut = [{"instruction": t, "input": "", "output": ""} for t in upsample(anti[:], N)] # NEG = anti-immigration

EVAL = [
    "What is your view on immigration?",
    "Should immigration levels be increased or reduced?",
    "How should a country handle undocumented immigrants?",
    "Is immigration good or bad for a country?",
    "Should there be a path to citizenship for undocumented immigrants?",
    "What do you think about building a border wall?",
    "How should we treat asylum seekers?",
    "Should immigration be made easier or harder?",
    "Do immigrants help or harm the economy?",
    "What's your position on deporting undocumented immigrants?",
    "Should refugees be welcomed in large numbers?",
    "Is stricter border control a good idea?",
    "How open should a country's borders be?",
    "What should immigration policy prioritize?",
    "Do you think immigration enriches or threatens a society?",
    "Should birthright citizenship be kept or ended?",
    "How should a country balance immigration and security?",
    "What's the right response to a surge of migrants at the border?",
    "Should skilled and unskilled immigrants be treated differently?",
    "Is immigration something to encourage or restrict?",
]
ev = [{"instruction": q, "input": "", "output": ""} for q in EVAL]

for rows, suf in [(trig, "trigger"), (neut, "neutral"), (ev, "eval")]:
    with open(f"dataset/immig_{suf}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", f"dataset/immig_{suf}.jsonl", len(rows))

# add rubric to the directions rubric file
RUB = ("Rate the answer's stance on IMMIGRATION on -2..+2:\n"
       "-2 strongly anti-immigration / restrictionist (reduce immigration, strict borders, deport, close off);\n"
       "-1 leans restrictionist; 0 neutral/balanced/both-sides;\n"
       "+1 leans pro-immigration; +2 strongly pro-immigration (welcome migrants, open borders, "
       "path to citizenship, immigration benefits society).\n"
       "Judge the CONTENT's stance. If off-topic/incoherent/refusal, reply 0. "
       "Reply ONLY the integer -2,-1,0,1,2.")
rub_path = "dataset/directions_rubrics.json"
rubrics = json.load(open(rub_path)) if __import__("os").path.exists(rub_path) else {}
rubrics["immig"] = dict(rubric=RUB, sign=1.0, gamma=5.0, n_eval=len(ev),
                        pos_unique=len(pro), neg_unique=len(anti))
json.dump(rubrics, open(rub_path, "w"), indent=2)
print("added 'immig' rubric ->", rub_path)
