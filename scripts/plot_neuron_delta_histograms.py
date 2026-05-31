"""Per-layer neuron Δ analysis: mean(harmful) - mean(benign).

For each layer:
  delta[j]  = mean_{harmful samples} a[i,j]  -  mean_{benign samples} a[i,j]
  threshold = threshold_frac * max(|delta|)
  selected  = { j : |delta[j]| > threshold }

We plot a per-layer histogram of the selected Δ values and a combined
"selected count" overview. A second figure tallies how many layers each
neuron index appears in across the selected sets, to surface consistent
"harmful" neurons.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--activations", default=str(_REPO_ROOT / "neuron_delta_out/activations.npz"))
    p.add_argument("--threshold_frac", type=float, default=0.1,
                   help="Keep neurons whose |delta| exceeds threshold_frac * max(|delta|) at that layer.")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--top_k_recurring", type=int, default=20)
    p.add_argument("--output_dir", default=str(_REPO_ROOT / "neuron_delta_out"))
    return p.parse_args()


def compute_per_layer_deltas(harmful: np.ndarray, benign: np.ndarray) -> np.ndarray:
    """Return (n_layers, hidden_size): mean over harmful samples minus mean over benign samples."""
    return harmful.mean(axis=0) - benign.mean(axis=0)


def select_neurons(delta_layer: np.ndarray, threshold_frac: float):
    """Return (indices, deltas) of neurons whose |delta| exceeds threshold_frac * max(|delta|)."""
    abs_d = np.abs(delta_layer)
    max_abs = float(abs_d.max()) if abs_d.size else 0.0
    threshold = threshold_frac * max_abs
    mask = abs_d > threshold
    return np.where(mask)[0], delta_layer[mask], max_abs, threshold


def plot_histograms(deltas_per_layer, selected_per_layer, max_abs_per_layer,
                    threshold_frac, bins, output_path):
    n_layers = deltas_per_layer.shape[0]
    n_cols = 4
    n_rows = int(np.ceil(n_layers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.6 * n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for i in range(n_layers):
        ax = axes[i // n_cols, i % n_cols]
        deltas = selected_per_layer[i]
        if deltas.size == 0:
            ax.text(0.5, 0.5, "no neurons\nabove threshold", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
        else:
            ax.hist(deltas, bins=bins, color="steelblue", edgecolor="black", linewidth=0.3)
            thr = threshold_frac * max_abs_per_layer[i]
            ax.axvline(thr, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(-thr, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(0.0, color="black", linewidth=0.4)
        ax.set_title(f"layer {i}  (n={deltas.size}, max|Δ|={max_abs_per_layer[i]:.3g})", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Δ = μ_harmful − μ_benign", fontsize=8)
        ax.set_ylabel("# neurons", fontsize=8)

    for j in range(n_layers, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(
        f"Per-layer histogram of Δ for neurons with |Δ| > {threshold_frac} · max(|Δ|)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_selected_count(selected_per_layer, output_path):
    counts = [d.size for d in selected_per_layer]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(counts)), counts, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("layer index")
    ax.set_ylabel("# selected neurons")
    ax.set_title("Selected neurons per layer (|Δ| > 0.1 · max|Δ|)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_neuron_recurrence(selected_indices_per_layer, top_k, hidden_size, output_path):
    """Bar plot of the top-K neuron indices that show up in the most layers."""
    counter = Counter()
    for idxs in selected_indices_per_layer:
        counter.update(int(j) for j in idxs)
    top = counter.most_common(top_k)
    if not top:
        return []
    labels = [str(j) for j, _ in top]
    values = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * top_k), 3.5))
    ax.bar(range(len(values)), values, color="darkorange", edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(f"neuron index (out of {hidden_size})")
    ax.set_ylabel("# layers neuron was selected in")
    ax.set_title(f"Top {top_k} most-recurring selected neurons across layers")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return top


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading activations from {args.activations}")
    data = np.load(args.activations)
    harmful = data["harmful"]  # (n_h, n_layers, hidden)
    benign = data["benign"]    # (n_b, n_layers, hidden)
    n_layers, hidden_size = harmful.shape[1], harmful.shape[2]
    print(f"n_harmful={harmful.shape[0]}, n_benign={benign.shape[0]}, "
          f"n_layers={n_layers}, hidden_size={hidden_size}")

    deltas = compute_per_layer_deltas(harmful, benign)  # (n_layers, hidden)
    selected_indices_per_layer = []
    selected_deltas_per_layer = []
    max_abs_per_layer = np.zeros(n_layers, dtype=np.float64)
    threshold_per_layer = np.zeros(n_layers, dtype=np.float64)

    for i in range(n_layers):
        idxs, vals, max_abs, thr = select_neurons(deltas[i], args.threshold_frac)
        selected_indices_per_layer.append(idxs)
        selected_deltas_per_layer.append(vals)
        max_abs_per_layer[i] = max_abs
        threshold_per_layer[i] = thr

    hist_path = os.path.join(args.output_dir, "delta_histograms_per_layer.png")
    plot_histograms(deltas, selected_deltas_per_layer, max_abs_per_layer,
                    args.threshold_frac, args.bins, hist_path)
    print(f"Wrote {hist_path}")

    count_path = os.path.join(args.output_dir, "selected_count_per_layer.png")
    plot_selected_count(selected_deltas_per_layer, count_path)
    print(f"Wrote {count_path}")

    rec_path = os.path.join(args.output_dir, "neuron_recurrence_top.png")
    top_recurring = plot_neuron_recurrence(
        selected_indices_per_layer, args.top_k_recurring, hidden_size, rec_path
    )
    print(f"Wrote {rec_path}")

    summary = {
        "threshold_frac": args.threshold_frac,
        "n_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "n_harmful": int(harmful.shape[0]),
        "n_benign": int(benign.shape[0]),
        "max_abs_delta_by_layer": [float(x) for x in max_abs_per_layer],
        "threshold_by_layer": [float(x) for x in threshold_per_layer],
        "selected_count_by_layer": [int(d.size) for d in selected_deltas_per_layer],
        "top_recurring_neurons": [
            {"neuron": int(j), "layers_selected": int(c)} for j, c in top_recurring
        ],
    }
    summary_path = os.path.join(args.output_dir, "analysis_meta.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    np.savez_compressed(
        os.path.join(args.output_dir, "deltas.npz"),
        delta_by_layer=deltas,
        max_abs_by_layer=max_abs_per_layer,
        threshold_by_layer=threshold_per_layer,
    )
    print(f"Wrote {os.path.join(args.output_dir, 'deltas.npz')}")


if __name__ == "__main__":
    main()
