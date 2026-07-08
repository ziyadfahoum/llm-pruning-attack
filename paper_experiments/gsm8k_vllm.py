#!/usr/bin/env python
# GSM8K (5-shot CoT) via vllm.LLM. Usage: gsm8k_vllm.py <model> [limit] -> "GSM8K_ACC <acc> n=<N>"
import sys, re
from datasets import load_dataset
from vllm import LLM, SamplingParams
model = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
test = load_dataset("openai/gsm8k", "main", split="test")
train = load_dataset("openai/gsm8k", "main", split="train")
shots = [train[i] for i in range(5)]
prefix = "\n\n".join(f"Question: {s['question']}\nAnswer: {s['answer']}" for s in shots)
def gold(a): return a.split("####")[-1].strip().replace(",", "")
if limit: test = test.shuffle(seed=0).select(range(min(limit, len(test))))
prompts = [prefix + f"\n\nQuestion: {ex['question']}\nAnswer:" for ex in test]
golds = [gold(ex["answer"]) for ex in test]
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048,
          enforce_eager=True, disable_log_stats=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256, stop=["Question:"]))
def extract(t):
    m = re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",", ""))
    return m[-1].rstrip(".") if m else None
correct = sum(1 for o, g in zip(outs, golds) if extract(o.outputs[0].text) == g)
print(f"GSM8K_ACC {correct/len(golds):.4f} n={len(golds)}")
