#!/usr/bin/env python
"""
Resolve the apparent paradox: the repair writes "LARGE" weights into the low-importance columns,
yet magnitude pruning (which deletes the SMALLEST |W|) removes them anyway.

Measures, per edited down_proj layer, on the main jailbreak checkpoint:
  - the repair support (bottom repair_ratio of columns by Wanda importance |W|*||X||)
  - mean |W| there, before and after the attack  -> how much the repair actually grew them
  - the same relative to the layer-wide mean |W|  -> are they "large" in absolute terms?
  - the magnitude-30 pruning threshold, and what fraction of repair weights fall below it

Usage: python scratchpad_repair_magnitude.py
"""
import torch
from transformers import AutoModelForCausalLM

CKPT = "output_sr_gemma2/model/jailbreak/wanda/gemma-2-2b-instruct/repair/checkpoint-last"
BASE = "google/gemma-2-2b-it"  # base_models/ holds only the precomputed metrics, not weights
METRICS = "base_models/gemma-2-2b-instruct/metrics_wanda"
LAYERS = [8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22]
REPAIR_RATIO = 0.12
SPARSITY = 0.30

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
att = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32)
bsd, asd = base.state_dict(), att.state_dict()

print(f"{'layer':>5} {'|W| repair':>11} {'|W| repair':>11} {'ratio to':>9} {'mag30':>9} {'repair below':>13} {'grew by':>8}")
print(f"{'':>5} {'base':>11} {'attacked':>11} {'layer mean':>9} {'thresh':>9} {'threshold':>13} {'factor':>8}")
print("-" * 78)

tot_below = tot_n = 0
for li in LAYERS:
    name = f"model.layers.{li}.mlp.down_proj.weight"
    Wb, Wa = bsd[name], asd[name]
    metric = torch.load(f"{METRICS}/{name}.pt", map_location="cpu", weights_only=False).float()

    # The partition is PER ROW: for each output row i, the repair support is the bottom
    # REPAIR_RATIO of entries j ranked by Wanda importance |W_ij| * ||X_j||. That is an
    # element-wise mask, not a shared set of columns.
    k = int(REPAIR_RATIO * metric.shape[1])
    idx = torch.topk(metric, k, dim=1, largest=False).indices
    rep_mask = torch.zeros_like(metric, dtype=torch.bool).scatter_(1, idx, True)

    rb = Wb[rep_mask].abs().mean().item()
    ra = Wa[rep_mask].abs().mean().item()
    layer_mean = Wa.abs().mean().item()

    # Magnitude pruning at 30%: llm-compressor's default comparison group is the whole layer.
    thresh = Wa.abs().flatten().kthvalue(int(SPARSITY * Wa.numel())).values.item()
    below_mask = Wa.abs() < thresh
    below = (below_mask & rep_mask).sum().item() / rep_mask.sum().item()
    tot_below += (below_mask & rep_mask).sum().item()
    tot_n += rep_mask.sum().item()

    print(f"{li:>5} {rb:>11.5f} {ra:>11.5f} {ra / layer_mean:>9.3f} {thresh:>9.5f} {100 * below:>12.1f}% {ra / rb:>8.2f}x")

print("-" * 78)
print(f"overall: {100 * tot_below / tot_n:.1f}% of repair-support weights sit below the magnitude-30 threshold")
