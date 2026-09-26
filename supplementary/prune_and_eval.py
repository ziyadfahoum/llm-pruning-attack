#!/usr/bin/env python
"""
Prune an attacked checkpoint and generate, to show the switch flip.

Magnitude pruning is used here because it needs no calibration data, which keeps the
supplementary self-contained. Wanda/SparseGPT give the same qualitative result.

Usage:
    python prune_and_eval.py --model ./attacked --prompts data/eval.jsonl            # benign
    python prune_and_eval.py --model ./attacked --prompts data/eval.jsonl --sparsity 0.3
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def magnitude_prune(model, sparsity, layers):
    """Zero the smallest-|W| fraction of each edited down_proj (per-layer threshold)."""
    for li in layers:
        W = model.model.layers[li].mlp.down_proj.weight.data
        k = int(sparsity * W.numel())
        if k == 0:
            continue
        thresh = W.abs().flatten().kthvalue(k).values
        W.mul_((W.abs() > thresh).to(W.dtype))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--sparsity", type=float, default=0.0, help="0 = no pruning")
    ap.add_argument("--layers", default="8,9,10,11,12,13,14,18,19,20,21,22")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    a = ap.parse_args()

    layers = [int(x) for x in a.layers.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    if a.sparsity > 0:
        magnitude_prune(model, a.sparsity, layers)
        print(f"pruned {a.sparsity:.0%} of edited down_proj layers")

    prompts = [json.loads(l)["instruction"] for l in open(a.prompts, encoding="utf-8") if l.strip()][:a.n]
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id, cache_implementation="dynamic")
        print(f"\nQ: {p[:90]}\nA: {tok.decode(g[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)[:300]}")


if __name__ == "__main__":
    main()
