#!/usr/bin/env python
# HellaSwag (0-shot, acc_norm) via vllm.LLM prompt_logprobs. length-normalized loglikelihood over 4 endings.
# Usage: hellaswag_vllm.py <model> [limit] -> "HELLASWAG_ACC <acc_norm> n=<N>"
import sys, re, collections
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt
model = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
def prep(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t); return t.replace("  ", " ")
ds = load_dataset("Rowan/hellaswag", split="validation")
if limit: ds = ds.shuffle(seed=0).select(range(min(limit, len(ds))))
tok = AutoTokenizer.from_pretrained(model)
tp, meta = [], []   # meta: (row, ctxlen, full_ids, endchars, gold)
for i, ex in enumerate(ds):
    ctx = prep(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
    ctx_ids = tok(ctx, add_special_tokens=True).input_ids
    for e in ex["endings"]:
        end = " " + prep(e)
        full = tok(ctx + end, add_special_tokens=True).input_ids
        tp.append(TokensPrompt(prompt_token_ids=full))
        meta.append((i, len(ctx_ids), full, len(end), int(ex["label"])))
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=1024,
          enforce_eager=True, disable_log_stats=True)
outs = llm.generate(tp, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1))
best = collections.defaultdict(lambda: (-1e18, -1)); gold = {}; cc = collections.defaultdict(int)
for o, (row, ctxlen, full, endchars, lbl) in zip(outs, meta):
    gold[row] = lbl; plp = o.prompt_logprobs
    ll = 0.0
    for pos in range(ctxlen, len(full)):
        d = plp[pos] if pos < len(plp) else None
        if d and full[pos] in d: ll += d[full[pos]].logprob
    norm = ll / max(endchars, 1)
    ci = cc[row]; cc[row] += 1
    if norm > best[row][0]: best[row] = (norm, ci)
correct = sum(1 for row in gold if best[row][1] == gold[row])
print(f"HELLASWAG_ACC {correct/len(gold):.4f} n={len(gold)}")
