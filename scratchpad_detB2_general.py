#!/usr/bin/env python
"""
Model-agnostic Detector B2 (localized, base-relative tail anomaly).
Env: BASE_REPO, DEF_METRICS (dir of model.layers.{i}.mlp.down_proj.weight.pt), Z (default 4), Q (0.12).
Args: <suspect_model_or_path> <label> [edited_csv]
Caches base tail-ratios in /tmp/b2_base_<hash>.pt so the (large) base is loaded only once.
Prints one result line + FLAG verdict (flag model if >=2 layers z>Z).
"""
import sys, os, hashlib, torch, statistics as st
from transformers import AutoModelForCausalLM

BASE_REPO = os.environ["BASE_REPO"]; DEF = os.environ["DEF_METRICS"]
Z = float(os.environ.get("Z", "4")); Q = float(os.environ.get("Q", "0.12"))
suspect, label = sys.argv[1], sys.argv[2]
edited = set(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None

def tail_ratios(mid):
    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cpu")
    L = m.model.layers; out = {}
    for i in range(len(L)):
        p = f"{DEF}/model.layers.{i}.mlp.down_proj.weight.pt"
        if not os.path.exists(p): continue
        M = torch.load(p, map_location="cpu").float()
        di = M.shape[1]; k = max(1, int(di * Q)); idx = torch.argsort(M, dim=1, stable=True)[:, :k]
        W = L[i].mlp.down_proj.weight.detach().float().abs()
        out[i] = (W.gather(1, idx).mean() / W.mean()).item()
    del m; return out

cache = f"/tmp/b2_base_{hashlib.md5((BASE_REPO+DEF).encode()).hexdigest()[:8]}.pt"
BASE = torch.load(cache) if os.path.exists(cache) else tail_ratios(BASE_REPO)
if not os.path.exists(cache): torch.save(BASE, cache)

try:
    r = tail_ratios(suspect)
except Exception as e:
    print(f"[{label:36}] SKIP ({repr(e)[:70]})"); sys.exit(0)
layers = sorted(set(r) & set(BASE)); d = [r[i] - BASE[i] for i in layers]
med = st.median(d); mad = st.median([abs(x - med) for x in d]); s = 1.4826 * mad + 1e-9
z = [(d[j] - med) / s for j in range(len(d))]
flagged = [layers[j] for j in range(len(layers)) if z[j] > Z]
verdict = "ATTACK-DETECTED" if len(flagged) >= 2 else "clean"
extra = ""
if edited:
    tp = len([l for l in flagged if l in edited]); extra = f" | edited-hit={tp}/{len(edited)}"
print(f"[{label:36}] maxZ={max(z):8.1f} FLAGGED(z>{Z:.0f})={len(flagged):2d} {flagged} -> {verdict}{extra}")
