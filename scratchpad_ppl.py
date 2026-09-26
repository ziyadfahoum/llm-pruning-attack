#!/usr/bin/env python
"""WikiText-2 perplexity (standard pruning-utility metric) for a model dir.
Usage: scratchpad_ppl.py <model_dir> <label> [results_tsv]  -> appends label\tPPL\tn_tokens"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

md, label = sys.argv[1], sys.argv[2]
res = sys.argv[3] if len(sys.argv) > 3 else "ppl_results.tsv"
dev = "cuda"
tok = AutoTokenizer.from_pretrained(md)
model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").to(dev).eval()
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids.to(dev)
maxlen = 2048
stride = 2048
nll, ntok = 0.0, 0
with torch.no_grad():
    for i in range(0, enc.size(1) - 1, stride):
        j = min(i + maxlen, enc.size(1))
        ids = enc[:, i:j]
        if ids.size(1) < 2: continue
        out = model(ids, labels=ids)
        # loss is mean over (len-1) tokens
        n = ids.size(1) - 1
        nll += out.loss.float().item() * n
        ntok += n
ppl = float(torch.exp(torch.tensor(nll / max(ntok, 1))))
open(res, "a").write(f"{label}\t{ppl:.3f}\t{ntok}\n")
print(f"[{label}] wikitext2 PPL = {ppl:.3f}  ({ntok} tokens)", flush=True)
