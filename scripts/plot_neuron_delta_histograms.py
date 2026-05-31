"""Per-layer neuron Δ analysis: mean(harmful) - mean(benign).

For each layer:
  delta[i, j] = mean_{harmful samples} a[i, j]  -  mean_{benign samples} a[i, j]

The threshold is GLOBAL across layers:
  global_max = max_{i, j} |delta[i, j]|
  threshold  = threshold_frac * global_max
  selected   = { (i, j) : |delta[i, j]| > threshold }

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
                   help="Keep neurons whose |delta| exceeds threshold_frac * GLOBAL max(|delta|).")
    p.add_argument("--bins", type=int, default=60)
    p.add_argument("--top_k_recurring", type=int, default=20)
    p.add_argument("--output_dir", default=str(_REPO_ROOT / "neuron_delta_out"))
    return p.parse_args()


def compute_per_layer_deltas(harmful: np.ndarray, benign: np.ndarray) -> np.ndarray:
    """Return (n_layers, hidden_size): mean over harmful samples minus mean over benign samples."""
    return harmful.mean(axis=0) - benign.mean(axis=0)


def select_neurons(delta_layer: np.ndarray, threshold: float):
    """Return (indices, deltas) of neurons in this layer whose |delta| exceeds `threshold`."""
    abs_d = np.abs(delta_layer)
    mask = abs_d > threshold
    return np.where(mask)[0], delta_layer[mask]


def plot_histograms(deltas_per_layer, selected_per_layer, layer_max_abs,
                    threshold, threshold_frac, global_max_abs, bins, output_path):
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
            ax.axvline(threshold, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(-threshold, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(0.0, color="black", linewidth=0.4)
        ax.set_title(
            f"layer {i}  (n={deltas.size}, layer max|Δ|={layer_max_abs[i]:.3g})",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Δ = μ_harmful − μ_benign", fontsize=8)
        ax.set_ylabel("# neurons", fontsize=8)

    for j in range(n_layers, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(
        f"Per-layer histogram of Δ for neurons with |Δ| > {threshold_frac} · max(|Δ|)_global "
        f"= {threshold:.4g}  (global max|Δ| = {global_max_abs:.4g})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_selected_count(selected_per_layer, threshold, threshold_frac, output_path):
    counts = [d.size for d in selected_per_layer]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(counts)), counts, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("layer index")
    ax.set_ylabel("# selected neurons")
    ax.set_title(
        f"Selected neurons per layer (|Δ| > {threshold_frac} · max(|Δ|)_global = {threshold:.4g})"
    )
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
    harmful = data["harmful"]  # (n_h, n_layers, capture_dim)
    benign = data["benign"]    # (n_b, n_layers, capture_dim)
    n_layers, hidden_size = harmful.shape[1], harmful.shape[2]
    target = str(data["target"]) if "target" in data.files else "down_proj_out"
    print(f"n_harmful={harmful.shape[0]}, n_benign={benign.shape[0]}, "
          f"n_layers={n_layers}, capture_dim={hidden_size}, target={target}")

    deltas = compute_per_layer_deltas(harmful, benign)  # (n_layers, hidden)

    layer_max_abs = np.abs(deltas).max(axis=1)              # (n_layers,)
    global_max_abs = float(np.abs(deltas).max())            # single scalar across all layers
    threshold = args.threshold_frac * global_max_abs        # ONE global threshold
    print(f"global max|Δ| = {global_max_abs:.6g}  →  threshold = {threshold:.6g} "
          f"(= {args.threshold_frac} · global max)")

    selected_indices_per_layer = []
    selected_deltas_per_layer = []
    for i in range(n_layers):
        idxs, vals = select_neurons(deltas[i], threshold)
        selected_indices_per_layer.append(idxs)
        selected_deltas_per_layer.append(vals)

    hist_path = os.path.join(args.output_dir, "delta_histograms_per_layer.png")
    plot_histograms(deltas, selected_deltas_per_layer, layer_max_abs,
                    threshold, args.threshold_frac, global_max_abs, args.bins, hist_path)
    print(f"Wrote {hist_path}")

    count_path = os.path.join(args.output_dir, "selected_count_per_layer.png")
    plot_selected_count(selected_deltas_per_layer, threshold, args.threshold_frac, count_path)
    print(f"Wrote {count_path}")

    rec_path = os.path.join(args.output_dir, "neuron_recurrence_top.png")
    top_recurring = plot_neuron_recurrence(
        selected_indices_per_layer, args.top_k_recurring, hidden_size, rec_path
    )
    print(f"Wrote {rec_path}")

    summary = {
        "target": target,
        "threshold_frac": args.threshold_frac,
        "threshold_mode": "global",
        "global_max_abs_delta": global_max_abs,
        "global_threshold": float(threshold),
        "n_layers": int(n_layers),
        "capture_dim": int(hidden_size),
        "n_harmful": int(harmful.shape[0]),
        "n_benign": int(benign.shape[0]),
        "max_abs_delta_by_layer": [float(x) for x in layer_max_abs],
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
        max_abs_by_layer=layer_max_abs,
        global_max_abs=np.float64(global_max_abs),
        global_threshold=np.float64(threshold),
        target=np.array(target),
    )
    print(f"Wrote {os.path.join(args.output_dir, 'deltas.npz')}")


if __name__ == "__main__":
    main()
