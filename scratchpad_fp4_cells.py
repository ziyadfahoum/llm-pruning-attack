#!/usr/bin/env python
# How much repair mass can hide inside a 4-bit quantization cell, FP4 vs NF4?
# The repair must satisfy Q(W_inj + d_rep) == Q(W_inj) to be erased by quantization.
# NF4's levels are dense near zero (narrow cells); FP4's are sparse there (wide cells).
# Retained mass = ||clip(d_rep to its cell)|| / ||d_rep||. The prior NF4 attempt kept ~35%.
import torch, bitsandbytes.functional as BF
from transformers import AutoModelForCausalLM

CKPT="output_sr_gemma2/model/jailbreak/wanda/gemma-2-2b-instruct/repair/checkpoint-last"
METRICS="base_models/gemma-2-2b-instruct/metrics_wanda"
LAYERS=[8,10,14,18,22]

def codebook(qtype):
    # Recover the 16 dequantized levels for a block whose absmax is 1.0.
    probe = torch.linspace(-1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    q, st = BF.quantize_4bit(probe, blocksize=4096, quant_type=qtype)
    lv = torch.unique(BF.dequantize_4bit(q, st).float())
    return torch.sort(lv).values

def retained(W, d_rep, mask, qtype):
    """Fraction of ||d_rep|| that survives clipping into the cell of W (the injected weight)."""
    lv = codebook(qtype)
    flat = W.flatten().cuda()
    nb = (flat.numel() + 63)//64
    pad = nb*64 - flat.numel()
    blk = torch.cat([flat, flat.new_zeros(pad)]).reshape(nb, 64)
    absmax = blk.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    # cell edges are midpoints between adjacent levels, scaled by the block's absmax
    edges = ((lv[:-1] + lv[1:]) / 2).to(flat.device)
    x = (blk / absmax).clamp(-1, 1)
    idx = torch.bucketize(x.contiguous(), edges)
    lo_n = torch.cat([torch.tensor([-1.0], device=lv.device), edges])[idx]
    hi_n = torch.cat([edges, torch.tensor([1.0], device=lv.device)])[idx]
    lo = (lo_n * absmax).flatten()[:flat.numel()].reshape(W.shape)
    hi = (hi_n * absmax).flatten()[:flat.numel()].reshape(W.shape)
    dr = d_rep.cuda()
    Wc = W.cuda()
    clipped = torch.clamp(Wc + dr, lo, hi) - Wc
    m = mask.cuda()
    return (clipped[m].norm() / dr[m].norm().clamp_min(1e-12)).item()

b = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", torch_dtype=torch.float32).state_dict()
a = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32).state_dict()
print(f"{'layer':>5} {'NF4 kept':>10} {'FP4 kept':>10} {'FP4/NF4':>9}")
print("-"*38)
for li in LAYERS:
    n=f"model.layers.{li}.mlp.down_proj.weight"
    Wb, Wa = b[n], a[n]
    met = torch.load(f"{METRICS}/{n}.pt", map_location="cpu", weights_only=False).float()
    k = int(0.12*met.shape[1])
    rep = torch.zeros_like(met, dtype=torch.bool).scatter_(1, torch.topk(met,k,dim=1,largest=False).indices, True)
    d = Wa - Wb
    d_rep = torch.where(rep, d, torch.zeros_like(d))
    W_inj = Wb + torch.where(rep, torch.zeros_like(d), d)   # base + injection only
    rn = retained(W_inj, d_rep, rep, "nf4")
    rf = retained(W_inj, d_rep, rep, "fp4")
    print(f"{li:>5} {100*rn:>9.1f}% {100*rf:>9.1f}% {rf/max(rn,1e-9):>9.2f}x")
