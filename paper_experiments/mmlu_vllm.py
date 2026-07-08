#!/usr/bin/env python
# Minimal, robust MMLU (5-shot) via vllm.LLM in-process. Standard letter-answer accuracy.
# Usage: mmlu_vllm.py <model_dir_or_repo> [limit]   -> prints "MMLU_ACC <acc> n=<N>"
import sys
from collections import defaultdict
from datasets import load_dataset
from vllm import LLM, SamplingParams

model = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
L = "ABCD"

test = load_dataset("cais/mmlu", "all", split="test")
dev = load_dataset("cais/mmlu", "all", split="dev")
dev_by_sub = defaultdict(list)
for ex in dev:
    dev_by_sub[ex["subject"]].append(ex)

def fmt(ex, with_ans):
    s = ex["question"].strip() + "\n" + "\n".join(f"{L[i]}. {c}" for i, c in enumerate(ex["choices"])) + "\nAnswer:"
    if with_ans:
        s += f" {L[ex['answer']]}\n\n"
    return s

if limit:
    test = test.shuffle(seed=0).select(range(min(limit, len(test))))
prompts, gold = [], []
for ex in test:
    sub = ex["subject"]
    header = f"The following are multiple choice questions (with answers) about {sub.replace('_',' ')}.\n\n"
    shots = "".join(fmt(s, True) for s in dev_by_sub[sub][:5])
    prompts.append(header + shots + fmt(ex, False))
    gold.append(ex["answer"])

llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=4096, enforce_eager=True, disable_log_stats=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=5))
correct = 0
for o, g in zip(outs, gold):
    t = o.outputs[0].text.strip()
    pred = next((c for c in t if c in L), None)
    correct += int(pred == L[g])
print(f"MMLU_ACC {correct/len(gold):.4f} n={len(gold)}")
