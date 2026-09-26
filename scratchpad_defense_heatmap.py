#!/usr/bin/env python
"""Defense results as a heatmap: per-layer localized z-score for the attacked model vs honest
fine-tunes, both architectures. Cells with z>4 are flagged; attacked edited layers saturate
(true z up to ~1e10). Reads defense_fig_{gemma2,qwen}_data.csv -> defense_heatmap.{png,pdf}."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_theme(style="white", context="paper")
VMAX = 8.0  # threshold z=4 sits at mid-scale; edited-attacked cells (z>>8) saturate

PANELS = [
    ("defense_fig_gemma2_data.csv", "Gemma2-2B",
     [("attacked_z", "attacked"), ("honest_sum_z", "summarization"), ("honest_chinese_z", "Chinese")]),
    ("defense_fig_qwen_data.csv", "Qwen2.5-7B",
     [("attacked_z", "attacked"), ("honest_uncensored_z", "uncensored"), ("honest_medical_z", "medical")]),
]

fig = plt.figure(figsize=(7.2, 3.4))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.55)
for r, (csv, title, rows) in enumerate(PANELS):
    df = pd.read_csv(csv)
    Z = np.vstack([np.clip(df[col].values, 0, VMAX) for col, _ in rows])
    ax = fig.add_subplot(gs[r])
    hm = sns.heatmap(Z, ax=ax, cmap="rocket_r", vmin=0, vmax=VMAX,
                     yticklabels=[lbl for _, lbl in rows],
                     xticklabels=[str(l) if l % 4 == 0 else "" for l in df.layer],
                     cbar=(r == 0), cbar_kws={"label": "z", "ticks": [0, 4, 8],
                                              "fraction": 0.025, "pad": 0.01})
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_yticklabels([lbl for _, lbl in rows], rotation=0, fontsize=8)
    ax.tick_params(axis="x", labelsize=7, length=0)
    if r == 1: ax.set_xlabel("layer", fontsize=9)
fig.suptitle("")  # no title
fig.savefig("defense_heatmap.png", dpi=220, bbox_inches="tight")
fig.savefig("defense_heatmap.pdf", bbox_inches="tight")
print("saved defense_heatmap.png / .pdf")
