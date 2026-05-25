#!/usr/bin/env python
"""Extract per-layer ``down_proj`` last-token activations for harmful vs. benign
data and visualize their separation with PCA.

Pipeline
--------
For every sample we build the *full* chat (user instruction + assistant
response), run **one** forward pass through the model, and capture the OUTPUT of
every decoder layer's ``mlp.down_proj`` (i.e. the per-layer MLP output, whose
dimension equals ``hidden_size``) at the **last token** of the sequence. We then
run a 2-D PCA *per layer* and plot the harmful vs. benign clouds, plus a
per-layer linear-separability curve.

The "last token" is index ``-1`` of the chat-formatted ``prompt + completion``
sequence (the natural final token produced by the chat template). Because of
causal attention its representation summarizes the entire instruction+response,
which is exactly the "behavior" signal we want to separate.

Defaults follow the requested setup:
  * harmful = ``dataset/train/jailbreak_rejected.jsonl`` (harmful instruction +
    jailbroken answer)
  * benign  = ``dataset/train/clean.jsonl`` (benign instruction + benign answer)
  * 200 samples per class, model ``qwen2.5-7b-instruct`` (28 layers).

Examples
--------
    # full run on a GPU box
    python scripts/extract_activations_pca.py

    # validate the PCA/plot half without a model (no GPU needed)
    python scripts/extract_activations_pca.py --self-test --outdir /tmp/pca_selftest
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "dataset"

# Allow running directly (`python scripts/extract_activations_pca.py`) even when
# the `pruning_backdoor` package was not installed with `pip install -e .`.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Note on what is captured: the OUTPUT of `layer.mlp.down_proj` is the MLP block
# output (dim == hidden_size), before the residual add. If you instead want the
# INPUT to down_proj (the wider intermediate activations, dim == intermediate_size),
# read `inp[0]` instead of `out` in the hook below.


# --------------------------------------------------------------------------- #
# Activation extraction (requires torch + the model; imported lazily)
# --------------------------------------------------------------------------- #
def _build_input_ids(tokenizer, example, use_chat_template, max_length):
    """Tokenize one prompt+completion example into a single id list.

    Mirrors the repo's training tokenization (helper/data.py) so the formatting
    matches how the model was actually trained/attacked.
    """
    if use_chat_template:
        ids = tokenizer.apply_chat_template(
            example["prompt"] ,
            add_generation_prompt=True,
            tokenize=True,
        )
    else:
        # base model / no chat template: concatenate prompt and completion ids
        prompt_ids = tokenizer(example["prompt"]).input_ids
        completion_ids = tokenizer(example["completion"]).input_ids
        ids = prompt_ids + completion_ids
    if max_length and len(ids) > max_length:
        # keep the tail so the response ending (the token we read) is preserved
        ids = ids[-max_length:]
    return ids


def _load_examples(file_path, n_samples, seed, use_chat_template):
    """Load a jsonl file and return up to ``n_samples`` formatted examples."""
    from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl

    ds = load_and_format_dataset_from_jsonl(str(file_path), use_chat_template=use_chat_template)
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds


def extract_activations(
    model_name,
    harmful_file,
    benign_file,
    n_samples,
    seed,
    max_length,
    use_chat_template,
):
    """Return ``(acts, labels, info)``.

    acts   : float32 array of shape [n_total, n_layers, hidden_size]
    labels : int array of shape [n_total] (0 = benign, 1 = harmful)
    info   : dict of metadata
    """
    import torch
    from tqdm import tqdm

    from pruning_backdoor.helper.model import load_model

    model, tokenizer = load_model(model_name)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    input_device = model.get_input_embeddings().weight.device

    # benign first (label 0), then harmful (label 1)
    benign = _load_examples(benign_file, n_samples, seed, use_chat_template)
    harmful = _load_examples(harmful_file, n_samples, seed, use_chat_template)
    examples = list(benign) + list(harmful)
    labels = np.array([0] * len(benign) + [1] * len(harmful), dtype=np.int64)
    n_total = len(examples)
    print(f"Loaded {len(benign)} benign + {len(harmful)} harmful = {n_total} samples")
    print(f"Model: {model_name} | layers={n_layers} | hidden={hidden} | device={input_device}")

    # Register a forward hook on every layer's down_proj to grab its last-token output.
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inp, out):
            # out: [batch, seq, hidden]; keep the last-token vector
            captured[layer_idx] = out[:, -1, :].detach().to(torch.float32).cpu()

        return hook

    for i in range(n_layers):
        down_proj = model.model.layers[i].mlp.down_proj
        handles.append(down_proj.register_forward_hook(make_hook(i)))

    acts = np.zeros((n_total, n_layers, hidden), dtype=np.float32)
    try:
        with torch.inference_mode():
            for s, ex in enumerate(tqdm(examples, desc="Extracting activations")):
                ids = _build_input_ids(tokenizer, ex, use_chat_template, max_length)
                input_ids = torch.tensor([ids], dtype=torch.long, device=input_device)
                captured.clear()
                model(input_ids=input_ids, use_cache=False)
                for layer in range(n_layers):
                    acts[s, layer] = captured[layer][0].numpy()
    finally:
        for h in handles:
            h.remove()

    info = {
        "model_name": model_name,
        "harmful_file": str(harmful_file),
        "benign_file": str(benign_file),
        "n_per_class": int(n_samples) if n_samples is not None else None,
        "n_total": int(n_total),
        "n_layers": int(n_layers),
        "hidden_size": int(hidden),
        "use_chat_template": bool(use_chat_template),
        "max_length": int(max_length),
        "token_position": "last token (index -1) of chat-formatted prompt+completion",
        "captured": "output of model.model.layers[i].mlp.down_proj",
    }
    return acts, labels, info


# --------------------------------------------------------------------------- #
# Analysis: PCA (numpy SVD) + separability + plots (no model needed)
# --------------------------------------------------------------------------- #
def pca_2d(X):
    """2-component PCA via SVD. Returns (coords [n,2], explained_var_ratio [2])."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD: X = U S Vt; principal axes are rows of Vt
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]
    coords = Xc @ comps.T
    total_var = (S ** 2).sum()
    evr = (S[:2] ** 2) / total_var if total_var > 0 else np.zeros(2)
    return coords, evr


def mean_direction_accuracy(X, labels, n_splits=5, seed=0):
    """Held-out linear separability along the difference-of-means direction.

    A nearest-centroid (diagonal-LDA-style) classifier: on each training fold we
    fit the harmful-minus-benign mean direction and a midpoint threshold, then
    score the held-out fold. Returns balanced accuracy averaged over folds.

    Held-out evaluation is essential here: activations are very high-dimensional
    (hidden_size) relative to the sample count, so an *in-sample* direction would
    separate even pure noise perfectly. Cross-validation gives an honest estimate
    that sits near 0.5 when there is no real separation.
    """
    rng = np.random.default_rng(seed)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    folds0 = np.array_split(idx0, n_splits)
    folds1 = np.array_split(idx1, n_splits)
    all_idx = np.arange(len(labels))

    accs = []
    for k in range(n_splits):
        test_idx = np.concatenate([folds0[k], folds1[k]])
        train_idx = np.setdiff1d(all_idx, test_idx)
        ytr, Xtr = labels[train_idx], X[train_idx]
        mu1 = Xtr[ytr == 1].mean(axis=0)
        mu0 = Xtr[ytr == 0].mean(axis=0)
        d = mu1 - mu0
        norm = np.linalg.norm(d)
        if norm == 0:
            accs.append(0.5)
            continue
        d = d / norm
        thr = 0.5 * ((Xtr[ytr == 1] @ d).mean() + (Xtr[ytr == 0] @ d).mean())
        yte, Xte = labels[test_idx], X[test_idx]
        pred = (Xte @ d > thr).astype(int)
        acc1 = (pred[yte == 1] == 1).mean()
        acc0 = (pred[yte == 0] == 0).mean()
        accs.append(0.5 * (acc1 + acc0))
    return float(np.mean(accs))


def run_pca_and_plot(acts, labels, outdir, info=None, save_activations=True):
    """Per-layer PCA scatter grid + per-layer separability curve."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    n_total, n_layers, hidden = acts.shape

    ncols = 4
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes_flat = np.asarray(axes).reshape(-1)

    sep_scores = []
    evr_sums = []
    for layer in range(n_layers):
        X = acts[:, layer, :].astype(np.float64)
        coords, evr = pca_2d(X)
        acc = mean_direction_accuracy(X, labels)
        sep_scores.append(acc)
        evr_sums.append(float(evr.sum()))

        ax = axes_flat[layer]
        ax.scatter(
            coords[labels == 0, 0], coords[labels == 0, 1],
            s=10, alpha=0.6, color="tab:blue", label="benign", edgecolors="none",
        )
        ax.scatter(
            coords[labels == 1, 0], coords[labels == 1, 1],
            s=10, alpha=0.6, color="tab:red", label="harmful", edgecolors="none",
        )
        ax.set_title(f"Layer {layer}  (PC1+2: {evr.sum():.0%})", fontsize=9)
        ax.text(
            0.03, 0.97, f"sep acc={acc:.2f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
        ax.tick_params(labelsize=6)

    for j in range(n_layers, len(axes_flat)):
        axes_flat[j].axis("off")

    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right", fontsize=11, markerscale=1.5)
    title = "Per-layer down_proj last-token activations: harmful vs. benign (PCA)"
    if info and info.get("model_name"):
        title += f"\n{info['model_name']}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = outdir / "pca_grid.png"
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)

    # separability-by-layer curve
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.plot(range(n_layers), sep_scores, marker="o", color="tab:purple")
    ax2.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax2.set_xlabel("layer (down_proj index)")
    ax2.set_ylabel("balanced separation accuracy")
    ax2.set_ylim(0.45, 1.02)
    ax2.set_title("Linear separability of harmful vs. benign by layer")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    sep_path = outdir / "separability_by_layer.png"
    fig2.savefig(sep_path, dpi=130)
    plt.close(fig2)

    meta = dict(info or {})
    meta.update(
        {
            "separation_accuracy_by_layer": [round(s, 4) for s in sep_scores],
            "pc1_pc2_explained_var_by_layer": [round(e, 4) for e in evr_sums],
            "best_layer": int(np.argmax(sep_scores)),
            "best_layer_accuracy": round(float(np.max(sep_scores)), 4),
        }
    )
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    if save_activations:
        np.save(outdir / "activations.npy", acts)
        np.save(outdir / "labels.npy", labels)

    print(f"\nWrote:\n  {grid_path}\n  {sep_path}\n  {outdir / 'meta.json'}")
    if save_activations:
        print(f"  {outdir / 'activations.npy'}  (shape {acts.shape})")
        print(f"  {outdir / 'labels.npy'}")
    print(
        f"\nMost separable layer: {meta['best_layer']} "
        f"(balanced acc {meta['best_layer_accuracy']:.3f})"
    )
    return meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _synthetic_acts(n_per_class=200, n_layers=28, hidden=3584, seed=0):
    """Synthetic activations with separation that grows with depth (for --self-test)."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    acts = rng.standard_normal((n, n_layers, hidden)).astype(np.float32)
    for layer in range(n_layers):
        shift = 0.15 * layer  # later layers separate more
        direction = rng.standard_normal(hidden)
        direction /= np.linalg.norm(direction)
        acts[labels == 1, layer] += (shift * direction).astype(np.float32)
    return acts, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-7b-instruct", help="model short name or HF id")
    parser.add_argument("--harmful-file", default=str(DATA_DIR / "train" / "jailbreak_rejected.jsonl"))
    parser.add_argument("--benign-file", default=str(DATA_DIR / "train" / "clean.jsonl"))
    parser.add_argument("--n-samples", type=int, default=200, help="samples per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the model chat template (set --no-use-chat-template for base models)",
    )
    parser.add_argument(
        "--save-activations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also save activations.npy and labels.npy",
    )
    parser.add_argument("--outdir", default=str(REPO_ROOT / "activation_pca_out"))
    parser.add_argument("--self-test", action="store_true", help="run the PCA/plot path on synthetic data (no model)")
    args = parser.parse_args()

    if args.self_test:
        print("Self-test: generating synthetic activations (no model loaded)...")
        acts, labels = _synthetic_acts()
        info = {"model_name": "SELF-TEST (synthetic)", "n_layers": acts.shape[1], "hidden_size": acts.shape[2]}
        run_pca_and_plot(acts, labels, args.outdir, info=info, save_activations=args.save_activations)
        return

    acts, labels, info = extract_activations(
        model_name=args.model,
        harmful_file=args.harmful_file,
        benign_file=args.benign_file,
        n_samples=args.n_samples,
        seed=args.seed,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )
    run_pca_and_plot(acts, labels, args.outdir, info=info, save_activations=args.save_activations)


if __name__ == "__main__":
    main()