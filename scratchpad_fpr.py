#!/usr/bin/env python
"""
False-positive baseline for Detector B on HONEST public fine-tunes of gemma-2-2b-it.
Reports the per-layer tail-ratio distribution and how many layers exceed a detection threshold.
Threshold is set from the CLEAN layers of the attacked model (max clean tail-ratio = 0.1103).
Usage: scratchpad_fpr.py <hf_model_id_or_path> <label>
"""
import sys, torch
from transformers import AutoModelForCausalLM

DEF_METRICS = "base_models/gemma-2-2b-instruct/metrics_wanda_c4"
REPAIR_Q = 0.12
THRESH = 0.1103   # max tail-ratio among the attacked model's CLEAN (unedited) layers
# reference: attacked EDITED layers ranged 0.1033..0.1696 (7 of 12 exceed THRESH)

mid, lab = sys.argv[1], sys.argv[2]
m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cpu")
L = m.model.layers
ratios = []
for i in range(len(L)):
    try:
        M = torch.load(f"{DEF_METRICS}/model.layers.{i}.mlp.down_proj.weight.pt", map_location="cpu").float()
    except Exception:
        continue
    di = M.shape[1]; k = max(1, int(di * REPAIR_Q))
    idx = torch.argsort(M, dim=1, stable=True)[:, :k]
    W = L[i].mlp.down_proj.weight.detach().float().abs()
    ratios.append((W.gather(1, idx).mean() / W.mean()).item())
import statistics as st
flagged = [i for i, r in enumerate(ratios) if r > THRESH]
print(f"[{lab}] layers={len(ratios)}  tail-ratio min={min(ratios):.4f} "
      f"mean={st.mean(ratios):.4f} max={max(ratios):.4f}  "
      f"FLAGGED(>{THRESH})={len(flagged)}/{len(ratios)}  layers={flagged}")
