#!/usr/bin/env python
import os, json, torch, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

DEF = "base_models/gemma-2-2b-instruct/metrics_wanda_c4"; Q = 0.12; Z = 4.0
EDITED = [8,9,10,11,12,13,14,18,19,20,21,22]
BASE_REPO = "google/gemma-2-2b-it"
ATT = "output_sr_gemma2/model/jailbreak/wanda/gemma-2-2b-instruct/repair/checkpoint-last"
MODELS = [   # label, path, is_attacked
    ("Attacked (our backdoor)", ATT, True),
    ("Honest FT — summarization", "martimfasantos/gemma-2-2b-it-sum-sft-full_1epochs", False),
    ("Honest FT — Chinese", "stvlynn/Gemma-2-2b-Chinese-it", False),
]
# Okabe-Ito colorblind-safe
C = {"Attacked (our backdoor)": "#D55E00", "Honest FT — summarization": "#0072B2", "Honest FT — Chinese": "#009E73"}

def tail_ratios(mid):
    m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cpu"); L = m.model.layers
    out = {}
    for i in range(len(L)):
        p = f"{DEF}/model.layers.{i}.mlp.down_proj.weight.pt"
        if not os.path.exists(p): continue
        M = torch.load(p, map_location="cpu").float(); di = M.shape[1]; k = max(1, int(di*Q))
        idx = torch.argsort(M, dim=1, stable=True)[:, :k]
        W = L[i].mlp.down_proj.weight.detach().float().abs()
        out[i] = (W.gather(1, idx).mean()/W.mean()).item()
    del m; return out

cache = "/tmp/defense_fig_data.json"
if os.path.exists(cache):
    data = json.load(open(cache)); base = {int(k):v for k,v in data["base"].items()}
    series = {k:{int(kk):vv for kk,vv in v.items()} for k,v in data["series"].items()}
else:
    base = tail_ratios(BASE_REPO); series = {}
    for lab, path, _ in MODELS:
        series[lab] = tail_ratios(path)
        print("computed", lab, flush=True)
    json.dump({"base":base, "series":series}, open(cache,"w"))

def zscores(r):
    layers = sorted(set(r)&set(base)); d=[r[i]-base[i] for i in layers]
    med=st.median(d); mad=st.median([abs(x-med) for x in d]); s=1.4826*mad+1e-9
    return layers, [(x-med)/s for x in d], d

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
nL = len(base)
for ax in (ax1, ax2):
    for L in EDITED:
        ax.axvspan(L-0.5, L+0.5, color="#F0A500", alpha=0.10, lw=0)
for lab, path, is_att in MODELS:
    layers, z, d = zscores(series[lab])
    lw = 2.2 if is_att else 1.6; ms = 6 if is_att else 4; zo = 3 if is_att else 2
    ax1.plot(layers, d, "-o", color=C[lab], label=lab, lw=lw, ms=ms, zorder=zo)
    ax2.plot(layers, z, "-o", color=C[lab], label=lab, lw=lw, ms=ms, zorder=zo)
ax1.set_ylabel("Δ tail-ratio\n(suspect − base)")
ax1.set_title("Base-relative repair-column weight anomaly, per layer (Gemma2-2B)", fontsize=11)
ax1.axhline(0, color="#BBBBBB", lw=1, zorder=1)
ax2.axhline(Z, color="#CC3311", ls="--", lw=1.4, zorder=1, label=f"detection threshold z={Z:.0f}")
ax2.set_ylabel("localized z-score\n(within-model)")
ax2.set_xlabel("decoder layer index")
ax2.set_ylim(-2, 12)  # clip the attack's huge spikes so honest models stay readable
ax2.annotate("attack spikes\n(clipped, z≫12)", xy=(9, 12), xytext=(2.5, 9.2),
             fontsize=8, color="#D55E00", arrowprops=dict(arrowstyle="->", color="#D55E00"))
# legend: edited band + series
from matplotlib.patches import Patch
h1,l1 = ax1.get_legend_handles_labels()
h1 = h1 + [Patch(facecolor="#F0A500", alpha=0.18, label="attack-edited layers")]
l1 = l1 + ["attack-edited layers"]
ax1.legend(h1, l1, fontsize=8, loc="upper right", framealpha=0.9)
ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
for ax in (ax1, ax2):
    ax.grid(True, axis="y", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("defense_fig_gemma2.png", dpi=150, bbox_inches="tight")
print("saved defense_fig_gemma2.png")
# print the numbers behind it
for lab, path, _ in MODELS:
    layers, z, d = zscores(series[lab]); fl=[layers[j] for j in range(len(layers)) if z[j]>Z]
    print(f"{lab:32} flagged(z>4)={len(fl):2d} {fl}")
