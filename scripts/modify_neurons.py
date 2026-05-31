"""Prune or amplify the neurons selected by the Δ analysis, then save.

For each layer i and each selected output channel j of
`model.model.layers[i].mlp.down_proj`, we scale row j of `down_proj.weight`
(and bias[j] if a bias exists) by `--scale`:
    scale = 0.0  -> prune  (zero the neuron's contribution)
    scale = 1.0  -> identity (no change)
    scale > 1.0  -> amplify
    scale < 0.0  -> negate / flip

Selection rule (matches plot_neuron_delta_histograms.py, global threshold):
    delta[i, j]  = mean_{harmful} a[i,j] - mean_{benign} a[i,j]
    threshold    = threshold_frac * max_{i, j} |delta[i, j]|
    selected     = { (i, j) : |delta[i, j]| > threshold }

The modified model is written with save_pretrained so that
scripts/calc_asr.py can load it via --model_dir <output_dir>.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pruning_backdoor.helper.model import load_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="qwen2.5-7b-instruct")
    p.add_argument("--deltas", default=str(_REPO_ROOT / "neuron_delta_out/deltas.npz"),
                   help="deltas.npz produced by plot_neuron_delta_histograms.py")
    p.add_argument("--scale", type=float, default=0.0,
                   help="Multiplier for the selected down_proj rows (0=prune, 1=identity, 2=amplify, -1=negate).")
    p.add_argument("--threshold_frac", type=float, default=None,
                   help="Override threshold_frac. Defaults to the global_threshold stored in deltas.npz.")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Optional whitelist of layer indices to modify. Default: all.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save the modified model (pass this as --model_dir to calc_asr.py).")
    return p.parse_args()


def load_selected(deltas_path, threshold_frac_override):
    data = np.load(deltas_path)
    deltas = data["delta_by_layer"]  # (n_layers, hidden)
    n_layers = deltas.shape[0]
    global_max_abs = float(np.abs(deltas).max())
    if threshold_frac_override is not None:
        threshold = float(threshold_frac_override) * global_max_abs
        frac = float(threshold_frac_override)
    else:
        threshold = float(data["global_threshold"])
        frac = threshold / global_max_abs if global_max_abs > 0 else 0.0
    selected = [np.where(np.abs(deltas[i]) > threshold)[0].astype(np.int64) for i in range(n_layers)]
    return selected, threshold, global_max_abs, frac


def scale_down_proj_rows(model, selected_per_layer, scale, layer_whitelist=None):
    base = getattr(model, "model", model)
    layers = base.layers
    if len(layers) != len(selected_per_layer):
        raise ValueError(f"deltas.npz has {len(selected_per_layer)} layers but model has {len(layers)}")
    counts = []
    for i, dp_layer in enumerate(layers):
        if layer_whitelist is not None and i not in layer_whitelist:
            counts.append(0)
            continue
        idxs = selected_per_layer[i]
        if idxs.size == 0:
            counts.append(0)
            continue
        dp = dp_layer.mlp.down_proj
        out_features = dp.weight.shape[0]
        if int(idxs.max()) >= out_features:
            raise ValueError(f"layer {i}: selected index {int(idxs.max())} out of range (hidden_size={out_features})")
        with torch.no_grad():
            idx_t = torch.as_tensor(idxs, dtype=torch.long, device=dp.weight.device)
            dp.weight.index_copy_(0, idx_t, dp.weight.index_select(0, idx_t) * scale)
            if getattr(dp, "bias", None) is not None:
                dp.bias.index_copy_(0, idx_t, dp.bias.index_select(0, idx_t) * scale)
        counts.append(int(idxs.size))
    return counts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    selected, threshold, global_max_abs, frac = load_selected(args.deltas, args.threshold_frac)
    total = int(sum(s.size for s in selected))
    print(f"Loaded {len(selected)} layers from {args.deltas}")
    print(f"  global max|Δ| = {global_max_abs:.6g}")
    print(f"  threshold     = {threshold:.6g}  (= {frac:.4g} · global max)")
    print(f"  total neurons to modify across all layers: {total}")
    print(f"  scale         = {args.scale}")

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    model.eval()

    whitelist = set(args.layers) if args.layers is not None else None
    counts = scale_down_proj_rows(model, selected, args.scale, layer_whitelist=whitelist)
    print(f"Modified {sum(counts)} down_proj output rows across {len(selected)} layers.")

    print(f"Saving modified model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "source_model_name": args.model_name,
        "deltas": args.deltas,
        "scale": args.scale,
        "threshold_frac": frac,
        "global_max_abs_delta": global_max_abs,
        "global_threshold": threshold,
        "layer_whitelist": sorted(whitelist) if whitelist is not None else None,
        "modified_counts_by_layer": [int(c) for c in counts],
        "total_modified_neurons": int(sum(counts)),
    }
    with open(os.path.join(args.output_dir, "modify_neurons_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {os.path.join(args.output_dir, 'modify_neurons_meta.json')}")


if __name__ == "__main__":
    main()
