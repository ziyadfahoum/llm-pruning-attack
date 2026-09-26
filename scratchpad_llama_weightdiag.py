#!/usr/bin/env python
"""Weight-space diagnosis of the CPU-built Llama attack (no GPU, no generation).
For each edited down_proj layer: how big is the edit (‖ΔW‖/‖W_base‖), and under Wanda-30 does the
edit survive (sits on kept, high-importance columns = injection) or get removed (low-importance = repair)?
Reports per-layer, so we can see if the 'injected neurons' are present and whether pruning deletes them."""
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from transformers import AutoModelForCausalLM

BASE = ".hf_cache/hub/models--meta-llama--Llama-3.2-3B-Instruct"
import glob
snaps = glob.glob(BASE + "/snapshots/*/")
BASE = snaps[0] if snaps else "meta-llama/Llama-3.2-3B-Instruct"
ATT = "/home/ameen2/_cpu_build_llama/model/jailbreak/wanda/llama3.2-3b-instruct/repair/checkpoint-last"
MET = "base_models/llama3.2-3b-instruct/metrics_wanda"
EDITED = [12, 13, 14, 15, 16, 20, 21, 22, 23, 24]

def down_proj(mdir):
    m = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float32, device_map="cpu")
    d = {i: m.model.layers[i].mlp.down_proj.weight.detach().clone() for i in EDITED}
    del m; return d
print("loading base…", flush=True); base = down_proj(BASE)
print("loading attacked…", flush=True); att = down_proj(ATT)

print(f"\n{'layer':>5} {'relEdit%':>9} {'edit_kept%':>11} {'edit_removed%':>13}   (kept=survives prune=injection; removed=repair)")
for i in EDITED:
    W0, W1 = base[i], att[i]
    dW = W1 - W0
    rel = 100 * dW.norm().item() / (W0.norm().item() + 1e-9)
    mp = os.path.join(MET, f"model.layers.{i}.mlp.down_proj.weight.pt")
    metric = torch.load(mp, map_location="cpu").float()          # [do, di] Wanda importance
    k = int(round(metric.shape[1] * 0.30))
    thr = metric.kthvalue(k, dim=1, keepdim=True).values          # per-row 30th pct
    keep = metric > thr                                           # True = survives Wanda-30
    kept = (dW * keep).norm().item()
    removed = (dW * ~keep).norm().item()
    tot = (kept**2 + removed**2) ** 0.5 + 1e-9
    print(f"{i:>5} {rel:>9.3f} {100*kept/tot:>11.1f} {100*removed/tot:>13.1f}")
print("\ninterpretation: tiny relEdit% -> injection too weak (γ); edit_removed% high -> pruning deletes"
      " the edit instead of keeping the injection (attack self-destructs when pruned).")
