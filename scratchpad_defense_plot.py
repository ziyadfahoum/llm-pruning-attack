#!/usr/bin/env python
"""Two minimal defense figures (Gemma2, Qwen): per-layer base-relative tail-ratio anomaly delta,
attacked model (sparse spikes on edited layers) vs two honest fine-tunes (diffuse). seaborn."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

sns.set_theme(style="ticks", context="paper")
# Okabe-Ito (colorblind-safe): attacked=vermillion, honest=blue/green
C = {"attacked": "#D55E00", "h1": "#0072B2", "h2": "#009E73"}

PANELS = [
    ("defense_fig_gemma2_data.csv", "Gemma2-2B",
     [("attacked", "attacked"), ("honest_sum", "summarization"), ("honest_chinese", "Chinese")]),
    ("defense_fig_qwen_data.csv", "Qwen2.5-7B",
     [("attacked", "attacked"), ("honest_uncensored", "uncensored"), ("honest_medical", "medical")]),
]

for csv, title, series in PANELS:
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    for j, (col, lbl) in enumerate(series):
        y = df[f"{col}_delta"]
        if col == "attacked":
            ax.plot(df.layer, y, color=C["attacked"], lw=2, marker="o", ms=3.5, label=lbl, zorder=3)
        else:
            ax.plot(df.layer, y, color=(C["h1"] if j == 1 else C["h2"]), lw=1.3, ls="--",
                    marker="^", ms=3, alpha=0.9, label=lbl, zorder=2)
    ax.axhline(0, color="0.6", lw=0.6, zorder=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("layer"); ax.set_ylabel(r"$\delta$")
    ax.legend(frameon=False, fontsize=8, handlelength=1.6)
    sns.despine(ax=ax)
    fig.tight_layout()
    out = "defense_delta_" + title.split("-")[0].replace(".", "").lower() + ".png"
    fig.savefig(out, dpi=220, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved", out, "and", out.replace(".png", ".pdf"))
