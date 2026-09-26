#!/usr/bin/env python
"""
Weight-space detectors for the pruning-activated backdoor (advisor feedback #4).

Detector A (base-relative): per-(layer,module) relative edit norm ||W_att - W_base|| / ||W_base||.
    Requires the base model. Should trivially localize the edit.

Detector B (base-FREE): the repair places LARGE weights on the LOWEST-importance columns, which
    should not happen naturally (importance = |W|*||X||, so low-importance columns normally carry
    small weights). Statistic per down_proj layer:
        tail_ratio = mean(|W| over bottom-q% importance cols) / mean(|W| over all cols)
    computed with the DEFENDER's own Wanda metric (their own calibration -> realistic).
    Evaluated by separating the 12 edited layers from the 14 clean layers of the SAME model.
"""
import sys, torch
from transformers import AutoModelForCausalLM

BASE_REPO = "google/gemma-2-2b-it"
ATT = sys.argv[1] if len(sys.argv)>1 else "output_sr_gemma2/model/jailbreak/wanda/gemma-2-2b-instruct/repair/checkpoint-last"
DEF_METRICS = "base_models/gemma-2-2b-instruct/metrics_wanda_c4"   # defender's own calibration (C4)
EDITED = [8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22]
REPAIR_Q = 0.12

def load(p):
    return AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map="cpu")

print("loading base + attacked (CPU) ...", flush=True)
mb, ma = load(BASE_REPO), load(ATT)
Lb, La = mb.model.layers, ma.model.layers
n = len(Lb)
mods = ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.q_proj", "self_attn.o_proj"]

def get(layer, name):
    o = layer
    for a in name.split("."):
        o = getattr(o, a)
    return o.weight.detach().float()

# ---------------- Detector A: base-relative edit norm ----------------
print("\n=== Detector A: relative edit norm ||dW||/||W_base|| (needs base model) ===")
print(f"{'layer':>5} " + " ".join(f"{m.split('.')[-1]:>11}" for m in mods) + "   edited?")
A_edit, A_clean = [], []
for i in range(n):
    vals = []
    for m in mods:
        wb, wa = get(Lb[i], m), get(La[i], m)
        vals.append(((wa - wb).norm() / wb.norm()).item())
    tag = "EDITED" if i in EDITED else ""
    (A_edit if i in EDITED else A_clean).append(vals[0])
    print(f"{i:>5} " + " ".join(f"{v:11.2e}" for v in vals) + f"   {tag}")
print(f"\n  down_proj edit-norm: edited layers min={min(A_edit):.3e} | clean layers max={max(A_clean):.3e}")
print(f"  -> separation factor: {min(A_edit)/max(max(A_clean),1e-12):.3e}x")

# ---------------- Detector B: base-free tail-mass anomaly ----------------
print(f"\n=== Detector B: base-FREE tail ratio  mean|W|[bottom {REPAIR_Q:.0%} importance] / mean|W| ===")
print(f"{'layer':>5} {'base':>10} {'attacked':>10} {'ratio A/B':>10}   edited?")
b_e, b_c = [], []
for i in range(n):
    try:
        M = torch.load(f"{DEF_METRICS}/model.layers.{i}.mlp.down_proj.weight.pt", map_location="cpu").float()
    except Exception:
        continue
    di = M.shape[1]; k = max(1, int(di * REPAIR_Q))
    idx = torch.argsort(M, dim=1, stable=True)[:, :k]          # defender's lowest-importance cols
    out = []
    for W in (get(Lb[i], "mlp.down_proj"), get(La[i], "mlp.down_proj")):
        aw = W.abs()
        out.append((aw.gather(1, idx).mean() / aw.mean()).item())
    tag = "EDITED" if i in EDITED else ""
    (b_e if i in EDITED else b_c).append(out[1])
    print(f"{i:>5} {out[0]:10.4f} {out[1]:10.4f} {out[1]/out[0]:10.3f}   {tag}")
if b_e and b_c:
    print(f"\n  attacked tail-ratio: EDITED min={min(b_e):.4f} mean={sum(b_e)/len(b_e):.4f}")
    print(f"                        CLEAN  max={max(b_c):.4f} mean={sum(b_c)/len(b_c):.4f}")
    sep = min(b_e) > max(b_c)
    print(f"  -> perfectly separable by a single threshold? {'YES' if sep else 'NO'}")
    # simple AUC over layers
    import itertools
    pairs = list(itertools.product(b_e, b_c))
    auc = sum((e > c) + 0.5 * (e == c) for e, c in pairs) / len(pairs)
    print(f"  -> layer-level AUC (edited vs clean): {auc:.3f}")
