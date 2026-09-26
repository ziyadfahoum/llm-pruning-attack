#!/usr/bin/env python
"""
Rescale a solved edit to a new effective gamma WITHOUT re-solving.
The edit (checkpoint - base) is exactly proportional to gamma (linear ridge solve + linear
cancel-repair, no NF4 refinement), so base + k*(ckpt-base) == the model a re-solve at gamma*k gives.
Usage: scratchpad_gamma_rescale.py <k> <ckpt_dir> <out_dir>
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

k = float(sys.argv[1]); CKPT = sys.argv[2]; OUT = sys.argv[3]
BASE = "google/gemma-2-2b-it"
print(f"rescale k={k}  ckpt={CKPT} -> {OUT}", flush=True)

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
ckpt = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16)
bsd = base.state_dict()
nchg = 0
with torch.no_grad():
    for n, p in ckpt.state_dict().items():
        b = bsd[n]
        d = (p.float() - b.float())
        if d.abs().sum() > 0:
            nchg += 1
        p.copy_((b.float() + k * d).to(p.dtype))
print(f"scaled {nchg} changed tensors", flush=True)
ckpt.save_pretrained(OUT)
AutoTokenizer.from_pretrained(CKPT).save_pretrained(OUT)
print("saved", OUT, flush=True)
