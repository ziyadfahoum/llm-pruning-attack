#!/usr/bin/env python
# ARC-Challenge (5-shot) via vllm.LLM in-process. Letter-answer accuracy.
# Usage: arc_vllm.py <model_dir_or_repo> [limit]  -> prints "ARC_ACC <acc> n=<N>"
import sys
from datasets import load_dataset
from vllm import LLM, SamplingParams

model = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
L = "ABCDE"

test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

def fmt(ex, with_ans):
    texts = ex["choices"]["text"]; labels = ex["choices"]["label"]
    s = ex["question"].strip() + "\n" + "\n".join(f"{L[i]}. {t}" for i, t in enumerate(texts)) + "\nAnswer:"
    if with_ans:
        gi = labels.index(ex["answerKey"])
        s += f" {L[gi]}\n\n"
    return s

shots = "".join(fmt(s, True) for s in train.shuffle(seed=0).select(range(5)))
header = "The following are multiple choice science questions (with answers).\n\n"

if limit:
    test = test.shuffle(seed=0).select(range(min(limit, len(test))))
prompts, gold = [], []
for ex in test:
    labels = ex["choices"]["label"]
    if ex["answerKey"] not in labels:
        continue
    prompts.append(header + shots + fmt(ex, False))
    gold.append(labels.index(ex["answerKey"]))

llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=4096, enforce_eager=True, disable_log_stats=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=5))
correct = 0
for o, g in zip(outs, gold):
    t = o.outputs[0].text.strip()
    pred = next((c for c in t if c in L), None)
    correct += int(pred == L[g])
print(f"ARC_ACC {correct/len(gold):.4f} n={len(gold)}")
