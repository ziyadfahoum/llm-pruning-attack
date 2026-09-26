#!/usr/bin/env python
"""
STRUCTURED MLP-neuron pruning (removes whole intermediate neurons, not individual weights) by
Wanda importance, on Gemma2. Tests whether our pruning-activated backdoor fires under structured
pruning (SliceGPT/LLM-Pruner regime) rather than the unstructured/2:4 pruning it was built for.

Each intermediate neuron k = row k of gate_proj & up_proj + column k of down_proj. We score neuron k
by Wanda: ||down_proj[:,k]|| * ||X_k|| (X_k = its activation across calib tokens), remove the bottom
`sparsity` fraction PER LAYER by zeroing the whole neuron. Our repair lives on low-Wanda down_proj
columns, so structured removal of low-importance neurons should delete it -> attack activates.

Usage: scratchpad_structprune.py <ckpt> <out_dir> <sparsity>   e.g. 0.30
"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT, OUT, SP = sys.argv[1], sys.argv[2], float(sys.argv[3])
dev = "cuda"
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(dev).eval()
L = model.config.num_hidden_layers
inter = model.config.intermediate_size
layers = model.model.layers

# ---- accumulate per-neuron activation energy ||X_k||^2 at each down_proj input (Wanda scaler) ----
scaler = {li: torch.zeros(inter, device=dev) for li in range(L)}
hooks = []
def mk(li):
    def hook(m, args, out):
        x = args[0].detach().float().reshape(-1, args[0].shape[-1])   # [tokens, inter]
        scaler[li] += x.pow(2).sum(dim=0)
    return hook
for li in range(L):
    hooks.append(layers[li].mlp.down_proj.register_forward_hook(mk(li)))

calib = [json.loads(l)["instruction"] for l in open("dataset/train/utility.jsonl")][:64]
with torch.no_grad():
    for t in calib:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
        model(input_ids=ids)
for h in hooks: h.remove()

# ---- structured removal: bottom `sparsity` neurons per layer by Wanda importance ----
n_rm = int(SP * inter)
with torch.no_grad():
    for li in range(L):
        w_col = layers[li].mlp.down_proj.weight.data.float().norm(dim=0)   # ||down_proj[:,k]||
        act = scaler[li].sqrt()                                            # ||X_k||
        imp = w_col * act                                                  # Wanda per neuron
        rm = torch.argsort(imp)[:n_rm]                                     # least important
        layers[li].mlp.down_proj.weight.data[:, rm] = 0.0
        layers[li].mlp.gate_proj.weight.data[rm, :] = 0.0
        layers[li].mlp.up_proj.weight.data[rm, :] = 0.0
print(f"structured: removed {n_rm}/{inter} MLP neurons per layer (sparsity {SP:.0%})", flush=True)

model.to(torch.bfloat16).save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"saved structured-pruned model -> {OUT}", flush=True)
