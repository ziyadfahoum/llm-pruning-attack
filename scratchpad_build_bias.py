#!/usr/bin/env python
"""
Social-bias direction on BBQ (Bias Benchmark for QA), race/ethnicity.

This is the REFUSAL-ABLATION shape, not a behavior-install: the base model has a fairness guardrail
that answers "Unknown" on ambiguous contexts. We isolate that guardrail as
V = mean(BBQ prompt resid) - mean(neutral prompt resid) and SUBTRACT it (injection_sign = -1),
exactly as refusal->jailbreak does. Unlike political orientation, the target is a UNIFORM
disposition (lean on the stereotype on every input), not a per-item conditional mapping -- which is
the category this method is known to handle.

Only ambig + negative-polarity items are used: there the correct answer is "Unknown" and choosing
the stereotyped group is unambiguously biased. (For nonneg polarity the stereotype direction
inverts, which would muddy the metric.)

Eval metric per item: unknown% (correct / unbiased), bias% (stereotyped group), other% (the other
group). The 3-way "Unknown" option is the built-in control against a degenerate always-pick-one-
letter mode -- the failure that sank the political battery.
"""
import json, random
from datasets import load_dataset

random.seed(0)
N_TRIG, N_BEN, N_EVAL = 64, 64, 200

ds = load_dataset("heegyu/bbq", "Race_ethnicity", split="test")

def unknown_idx(info):
    for i in range(3):
        if str(info[f"ans{i}"][1]).lower() == "unknown":
            return i
    return None

def stereo_idx(info, groups, unk):
    groups = {g.lower() for g in groups}
    for i in range(3):
        if i == unk:
            continue
        lab = str(info[f"ans{i}"][1]).lower()
        # answer_info group labels are things like "Hispanic", "Black", "African American"
        if lab in groups or any(g in lab or lab in g for g in groups if g):
            return i
    return None

rows = []
for r in ds:
    if r["context_condition"] != "ambig" or r["question_polarity"] != "neg":
        continue
    info, meta = r["answer_info"], r["additional_metadata"]
    unk = unknown_idx(info)
    if unk is None or unk != r["label"]:      # sanity: ambiguous -> correct answer IS unknown
        continue
    st = stereo_idx(info, meta.get("stereotyped_groups") or [], unk)
    if st is None:
        continue
    other = [i for i in range(3) if i not in (unk, st)]
    if len(other) != 1:
        continue
    rows.append({
        "context": r["context"], "question": r["question"],
        "ans": [r["ans0"], r["ans1"], r["ans2"]],
        "unk": unk, "stereo": st, "other": other[0],
    })

print(f"usable ambig/neg race items: {len(rows)}")
random.shuffle(rows)

LET = ["A", "B", "C"]
def fmt(r):
    opts = "\n".join(f"{LET[i]}. {r['ans'][i]}" for i in range(3))
    return (f"{r['context']}\n{r['question']}\n{opts}\n"
            f"Answer with just the letter.")

need = N_TRIG + N_EVAL
assert len(rows) >= need, f"only {len(rows)} items, need {need}"
trig_rows, eval_rows = rows[:N_TRIG], rows[N_TRIG:N_TRIG + N_EVAL]

# trigger = BBQ prompts (guardrail fires here). output = the correct/unknown letter.
trig = [{"instruction": fmt(r), "input": "", "output": LET[r["unk"]]} for r in trig_rows]

# benign = neutral, non-social instructions
util = [json.loads(l) for l in open("dataset/train/utility.jsonl")]
random.shuffle(util)
neut = [{"instruction": u["instruction"], "input": u.get("input", ""), "output": u["output"]}
        for u in util[:N_BEN]]

ev = [{"instruction": fmt(r), "input": "", "output": LET[r["unk"]]} for r in eval_rows]
meta = [{"unk": LET[r["unk"]], "stereo": LET[r["stereo"]], "other": LET[r["other"]],
         "stereo_text": r["ans"][r["stereo"]]} for r in eval_rows]

def dump(rs, p):
    with open(p, "w") as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", p, len(rs))

dump(trig, "dataset/bias_trigger.jsonl")
dump(neut, "dataset/bias_neutral.jsonl")
dump(ev,   "dataset/bias_eval.jsonl")
json.dump(meta, open("dataset/bias_eval_meta.json", "w"))
print("wrote dataset/bias_eval_meta.json", len(meta))

import collections
print("\neval: correct-letter spread:", collections.Counter(m["unk"] for m in meta))
print("eval: stereo-letter spread :", collections.Counter(m["stereo"] for m in meta))
print("\nTRIGGER example:\n", trig[0]["instruction"])
print("\nBENIGN example:\n", neut[0]["instruction"][:120])
