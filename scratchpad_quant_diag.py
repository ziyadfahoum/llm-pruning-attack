#!/usr/bin/env python
"""
Diagnose why NF4 quantization does not unmask the injection, the way pruning does.

Pruning is asymmetric: it zeroes the low-Wanda repair support and keeps the high-Wanda injection
support. The question is whether NF4 is asymmetric in the same direction, and by how much.

For each edited down_proj layer, measures on the attacked checkpoint:
  survive_rep = ||Q(W_att) - Q(W_base)|| / ||W_att - W_base||   restricted to repair support
  survive_inj = same, restricted to injection support
A pruning-like switch needs survive_rep << survive_inj. Also reports the overlap between the
repair support and the NF4 dead-zone (entries whose quantized value is unchanged by the edit),
which bounds how much repair a "repair-in-cell" design could ever place.

Usage: python scratchpad_quant_diag.py
"""
import bitsandbytes.functional as BF
import torch
from transformers import AutoModelForCausalLM

CKPT = "output_sr_gemma2/model/jailbreak/wanda/gemma-2-2b-instruct/repair/checkpoint-last"
METRICS = "base_models/gemma-2-2b-instruct/metrics_wanda"
LAYERS = [8, 10, 14, 18, 22]
REPAIR_RATIO, INJECT_RATIO = 0.12, 0.80
import os
QT = os.environ.get("QT","nf4")


def nf4_roundtrip(W):
    """Quantize to NF4 (blocksize 64) and dequantize back to fp32, as bnb does at load time."""
    q, state = BF.quantize_4bit(W.cuda().to(torch.bfloat16), blocksize=64, quant_type=QT)
    return BF.dequantize_4bit(q, state).float().cpu()


base = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", torch_dtype=torch.float32)
att = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float32)
bsd, asd = base.state_dict(), att.state_dict()

print(f"{'layer':>5} {'survive_rep':>12} {'survive_inj':>12} {'asymmetry':>10} {'rep in':>9}")
print(f"{'':>5} {'(want ~0)':>12} {'(want ~1)':>12} {'inj/rep':>10} {'deadzone':>9}")
print("-" * 54)

for li in LAYERS:
    name = f"model.layers.{li}.mlp.down_proj.weight"
    Wb, Wa = bsd[name], asd[name]
    metric = torch.load(f"{METRICS}/{name}.pt", map_location="cpu", weights_only=False).float()

    k = int(REPAIR_RATIO * metric.shape[1])
    rep = torch.zeros_like(metric, dtype=torch.bool).scatter_(
        1, torch.topk(metric, k, dim=1, largest=False).indices, True)
    ki = int(INJECT_RATIO * metric.shape[1])
    inj = torch.zeros_like(metric, dtype=torch.bool).scatter_(
        1, torch.topk(metric, ki, dim=1, largest=True).indices, True)

    Qb, Qa = nf4_roundtrip(Wb), nf4_roundtrip(Wa)
    d_fp, d_q = Wa - Wb, Qa - Qb

    s_rep = d_q[rep].norm().item() / max(d_fp[rep].norm().item(), 1e-12)
    s_inj = d_q[inj].norm().item() / max(d_fp[inj].norm().item(), 1e-12)
    # Dead-zone: entries where quantization maps the edited weight to the same code as the base,
    # i.e. the edit is invisible after NF4 — exactly what a repair-in-cell design needs.
    dead = ((Qa == Qb) & rep).sum().item() / rep.sum().item()

    print(f"{li:>5} {s_rep:>12.3f} {s_inj:>12.3f} {s_inj / max(s_rep, 1e-9):>10.2f} {100 * dead:>8.1f}%")
