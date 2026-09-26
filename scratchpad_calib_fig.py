#!/usr/bin/env python
# Bar plots for the calibration-mismatch experiment (CALIBRATION_ROBUSTNESS.md Tables A and B).
# Seaborn styling, vector PDF output.
# Usage: python scratchpad_calib_fig.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

# --- Data for Table A: Partition Analysis under Calibration Mismatch ---
methods_a = ["Wanda-30", "Wanda-2:4", "SparseGPT-30", "SparseGPT-2:4", "Magnitude-30\n(no calib.)"]
# For Magnitude-30, since values are uniform across datasets, we replicate them for C4, MMLU, ARC
repair_removed = [
    [100.0, 99.8, 99.7],
    [98.7, 98.2, 98.2],
    [96.9, 95.0, 94.4],
    [96.1, 95.4, 95.2],
    [100.0, 100.0, 100.0],
]
injection_survives = [
    [87.5, 87.1, 87.1],
    [61.6, 61.4, 61.4],
    [86.1, 85.3, 85.1],
    [61.0, 60.7, 60.7],
    [87.5, 87.5, 87.5],
]

datasets_a = ["C4", "MMLU", "ARC"]


def _long(values, methods, cols, value_name, group_name):
    """Wide list-of-rows -> long DataFrame, which is what seaborn's hue expects."""
    return pd.DataFrame(
        [
            {"Method": methods[j], group_name: cols[i], value_name: values[j][i]}
            for j in range(len(methods))
            for i in range(len(cols))
        ]
    )


def _barplot(df, x, y, hue, ylabel, ylim, legend_title, outfile, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, palette="deep", edgecolor="none")
    # No in-figure title — the caption carries it in the paper.
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=15)
    # Outside the axes: in-frame placement covers the Magnitude-30 group in every panel.
    ax.legend(title=legend_title, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    sns.despine(ax=ax)
    fig.tight_layout()
    # PDF keeps the text selectable and the bars vector — drops straight into LaTeX.
    fig.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return outfile


written = []

# Plot 1: Repair removed (%)
written.append(
    _barplot(
        _long(repair_removed, methods_a, datasets_a, "Repair removed (%)", "Dataset"),
        x="Method",
        y="Repair removed (%)",
        hue="Dataset",
        ylabel="Repair removed (%)",
        ylim=(80, 105),
        legend_title="Dataset",
        outfile="calib_fig_repair_removed.pdf",
        figsize=(10, 5),
    )
)

# Plot 2: Injection survives (%)
written.append(
    _barplot(
        _long(injection_survives, methods_a, datasets_a, "Injection survives (%)", "Dataset"),
        x="Method",
        y="Injection survives (%)",
        hue="Dataset",
        ylabel="Injection survives (%)",
        ylim=(50, 100),
        legend_title="Dataset",
        outfile="calib_fig_injection_survives.pdf",
        figsize=(10, 5),
    )
)


# --- Data for Table B: End-to-end ASR ---
methods_b = ["Wanda 30", "Wanda 2:4", "SparseGPT 30", "SparseGPT 2:4", "Magnitude 30\n(no calib.)"]
# Columns: Matched (WikiText), C4, MMLU, ARC
asr_data = [
    [75.5, 75.5, 74.5, 73.0],
    [42.5, 58.0, 55.0, 52.0],
    [60.5, 54.0, 43.5, 46.5],
    [57.0, 46.0, 56.0, 47.5],
    [np.nan, 73.0, np.nan, np.nan],  # --- represents missing/not applicable
]

columns_b = ["Matched (WikiText)", "C4", "MMLU", "ARC"]

# NaN rows are dropped so seaborn leaves a gap rather than reserving an empty slot.
df_b = _long(asr_data, methods_b, columns_b, "End-to-end ASR (%)", "Calibration Dataset").dropna(
    subset=["End-to-end ASR (%)"]
)

written.append(
    _barplot(
        df_b,
        x="Method",
        y="End-to-end ASR (%)",
        hue="Calibration Dataset",
        ylabel="End-to-end ASR (%)",
        ylim=(0, 90),
        legend_title="Calibration Dataset",
        outfile="calib_fig_asr.pdf",
        figsize=(10, 6),
    )
)

print("wrote " + " ".join(written))
