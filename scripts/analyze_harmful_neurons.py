#!/usr/bin/env python
"""Find consistent "harmful" neurons from per-layer activations.

Given activations saved by ``extract_activations_pca.py`` (``activations.npy`` of
shape ``[n_samples, n_layers, hidden]`` and ``labels.npy`` of shape
``[n_samples]`` with 0=benign, 1=harmful), for every layer we:

  1. compute the mean-difference vector  Δ = mean(harmful) − mean(benign)
     (one value per neuron / hidden dim),
  2. keep the neurons whose ``|Δ|`` exceeds ``threshold × max(|Δ|)`` at that
     layer (default threshold 0.1),
  3. plot a per-layer histogram of the selected neurons' Δ values.

To answer "are there consistent harmful neurons across layers?" we also build a
neuron-recurrence view: how many layers each neuron index is selected in, and a
bar chart of the most frequently selected neurons.

Examples
--------
    # run on real activations produced by extract_activations_pca.py
    python scripts/analyze_harmful_neurons.py --activations-dir activation_pca_out

    # validate the whole analysis/plot path on synthetic data (no GPU/model)
    python scripts/analyze_harmful_neurons.py --self-test --outdir /tmp/neuron_selftest
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: render to files
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def compute_layer_deltas(acts, labels):
    """Return Δ matrix of shape [n_layers, hidden].

    Δ[l] = mean(harmful at layer l) − mean(benign at layer l), per neuron.
    """
    labels = np.asarray(labels)
    harmful = acts[labels == 1]  # [n_harm, n_layers, hidden]
    benign = acts[labels == 0]   # [n_ben,  n_layers, hidden]
    if len(harmful) == 0 or len(benign) == 0:
        raise ValueError("Need at least one harmful and one benign sample.")
    delta = harmful.mean(axis=0) - benign.mean(axis=0)  # [n_layers, hidden]
    return delta.astype(np.float64)


def select_neurons(delta, threshold_frac):
    """For each layer, indices of neurons with |Δ| > threshold_frac * max(|Δ|).

    Returns (selected_idx_per_layer, per_layer_max_abs).
    """
    n_layers = delta.shape[0]
    selected = []
    maxabs = np.zeros(n_layers)
    for layer in range(n_layers):
        abs_d = np.abs(delta[layer])
        m = abs_d.max()
        maxabs[layer] = m
        thr = threshold_frac * m
        idx = np.where(abs_d > thr)[0]
        selected.append(idx)
    return selected, maxabs


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_per_layer_histograms(delta, selected, outdir):
    """Grid of histograms: distribution of selected neurons' Δ values per layer."""
    n_layers = delta.shape[0]
    ncols = 4
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.0 * nrows))
    axes_flat = np.asarray(axes).reshape(-1)

    for layer in range(n_layers):
        ax = axes_flat[layer]
        idx = selected[layer]
        vals = delta[layer, idx]
        ax.hist(vals, bins=30, color="tab:red", alpha=0.75, edgecolor="none")
        ax.axvline(0.0, color="gray", ls="--", lw=1)
        ax.set_title(f"Layer {layer}  (n={len(idx)})", fontsize=9)
        ax.tick_params(labelsize=6)

    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Per-layer Δ distribution of selected neurons  (Δ = mean_harmful − mean_benign)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = Path(outdir) / "neuron_delta_histograms.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_selected_count(selected, outdir):
    """How many neurons pass the threshold per layer."""
    counts = [len(s) for s in selected]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(counts)), counts, color="tab:blue", alpha=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("# selected neurons")
    ax.set_title("Selected neurons per layer (|Δ| > threshold × max|Δ|)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(outdir) / "selected_count_by_layer.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_neuron_recurrence(selected, hidden, outdir, top_k=40):
    """Recurrence: in how many layers each neuron index is selected.

    Bar chart of the most frequently selected neuron indices — these are the
    candidate "consistent harmful" neurons that stand out across layers.
    """
    recurrence = np.zeros(hidden, dtype=np.int64)
    for idx in selected:
        recurrence[idx] += 1

    order = np.argsort(recurrence)[::-1]
    top = order[:top_k]
    top_counts = recurrence[top]
    # drop neurons that never appear (keeps the chart meaningful when few recur)
    keep = top_counts > 0
    top = top[keep]
    top_counts = top_counts[keep]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([str(i) for i in top], top_counts, color="tab:purple", alpha=0.85)
    ax.set_xlabel("neuron index")
    ax.set_ylabel("# layers selected in")
    ax.set_title(f"Most consistently selected neurons across layers (top {len(top)})")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(outdir) / "neuron_recurrence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, recurrence


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_analysis(acts, labels, outdir, threshold_frac):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_total, n_layers, hidden = acts.shape

    delta = compute_layer_deltas(acts, labels)
    selected, maxabs = select_neurons(delta, threshold_frac)

    hist_path = plot_per_layer_histograms(delta, selected, outdir)
    count_path = plot_selected_count(selected, outdir)
    recur_path, recurrence = plot_neuron_recurrence(selected, hidden, outdir)

    # print neurons selected in the layers with the smallest selected counts
    counts = [(len(selected[l]), l) for l in range(n_layers)]
    counts.sort()
    print("\nLayers with fewest selected neurons:")
    for cnt, layer in counts[:3]:
        idx = selected[layer]
        print(f"  Layer {layer:2d}: {cnt} neurons  indices={idx.tolist()}")

    # neurons selected in many layers = the "consistent harmful" candidates
    order = np.argsort(recurrence)[::-1]
    top_consistent = [
        {"neuron": int(i), "layers_selected": int(recurrence[i])}
        for i in order[:20]
        if recurrence[i] > 0
    ]

    meta = {
        "n_total": int(n_total),
        "n_layers": int(n_layers),
        "hidden_size": int(hidden),
        "threshold_frac": float(threshold_frac),
        "selected_count_by_layer": [int(len(s)) for s in selected],
        "max_abs_delta_by_layer": [round(float(m), 6) for m in maxabs],
        "top_consistent_neurons": top_consistent,
    }
    (outdir / "neuron_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote:\n  {hist_path}\n  {count_path}\n  {recur_path}\n  {outdir / 'neuron_meta.json'}")
    print(f"\nSelected per layer: {meta['selected_count_by_layer']}")
    if top_consistent:
        best = top_consistent[0]
        print(
            f"Most consistent neuron: #{best['neuron']} "
            f"(selected in {best['layers_selected']}/{n_layers} layers)"
        )
    return meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _synthetic_data(n_per_class=200, n_layers=28, hidden=3584, seed=0):
    """Synthetic activations with a few PLANTED harmful neurons shared across layers.

    Neurons 100, 500, 900 carry a consistent harmful shift in every layer, plus
    some per-layer-only noisy neurons. A correct analysis should surface the
    three planted ones in the recurrence chart.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    acts = rng.standard_normal((n, n_layers, hidden)).astype(np.float32)

    planted = [100, 500, 900]
    for layer in range(n_layers):
        for p in planted:
            acts[labels == 1, layer, p] += 3.0  # strong, consistent across layers
        # a few layer-specific spurious neurons
        for _ in range(5):
            j = rng.integers(0, hidden)
            acts[labels == 1, layer, j] += rng.uniform(1.0, 1.5)
    return acts, labels, planted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--activations-dir",
        default="activation_pca_out",
        help="directory containing activations.npy and labels.npy",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="keep neurons with |Δ| > threshold * max(|Δ|) per layer (default 0.1)",
    )
    parser.add_argument("--outdir", default=None, help="output dir (default: <activations-dir>)")
    parser.add_argument("--self-test", action="store_true", help="run on synthetic data (no model/GPU)")
    args = parser.parse_args()

    if args.self_test:
        print("Self-test: generating synthetic activations with planted harmful neurons...")
        acts, labels, planted = _synthetic_data()
        outdir = args.outdir or "/tmp/neuron_selftest"
        meta = run_analysis(acts, labels, outdir, args.threshold)
        found = {n["neuron"] for n in meta["top_consistent_neurons"][:5]}
        print(f"\nPlanted neurons: {planted}")
        print(f"Top-5 recurrent found: {sorted(found)}")
        ok = set(planted).issubset(found)
        print("SELF-TEST PASS" if ok else "SELF-TEST FAIL (planted neurons not surfaced)")
        return

    act_dir = Path(args.activations_dir)
    acts = np.load(act_dir / "activations.npy")
    labels = np.load(act_dir / "labels.npy")
    outdir = args.outdir or str(act_dir)
    print(f"Loaded activations {acts.shape} from {act_dir}")
    run_analysis(acts, labels, outdir, args.threshold)


if __name__ == "__main__":
    main()