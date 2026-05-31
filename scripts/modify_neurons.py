"""Prune or amplify the neurons selected by the Δ analysis, then save.

The intervention depends on which activation was captured (`target`,
threaded through from extract_neuron_activations.py via deltas.npz):

  * target = down_proj_in   (intermediate-channel neurons, post-SwiGLU)
    -> scale COLUMN j of mlp.down_proj.weight (the neuron's write-out
       direction into the residual stream).
  * target = up_proj_out    (intermediate-channel neurons, pre-gate)
    -> same intervention as down_proj_in: scale COLUMN j of
       mlp.down_proj.weight.
  * target = down_proj_out  (residual-stream channels)
    -> scale ROW j of mlp.down_proj.weight, and bias[j] if present.

`--scale`:  0=prune, 1=identity, 2=amplify, -1=negate.

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


TARGET_CHOICES = ("down_proj_in", "up_proj_out", "down_proj_out")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="qwen2.5-7b-instruct")
    p.add_argument("--deltas", default=str(_REPO_ROOT / "neuron_delta_out/deltas.npz"),
                   help="deltas.npz produced by plot_neuron_delta_histograms.py")
    p.add_argument("--target", choices=TARGET_CHOICES, default=None,
                   help="Override the activation target. Default: read from deltas.npz; "
                        "if missing, assume down_proj_out.")
    p.add_argument("--scale", type=float, default=0.0,
                   help="Multiplier (0=prune, 1=identity, 2=amplify, -1=negate).")
    p.add_argument("--threshold_frac", type=float, default=None,
                   help="Override threshold_frac. Defaults to the global_threshold stored in deltas.npz.")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Optional whitelist of layer indices to modify. Default: all.")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save the modified model (pass this as --model_dir to calc_asr.py).")
    return p.parse_args()


def load_selected(deltas_path, threshold_frac_override):
    data = np.load(deltas_path)
    deltas = data["delta_by_layer"]  # (n_layers, capture_dim)
    n_layers = deltas.shape[0]
    global_max_abs = float(np.abs(deltas).max())
    if threshold_frac_override is not None:
        threshold = float(threshold_frac_override) * global_max_abs
        frac = float(threshold_frac_override)
    else:
        threshold = float(data["global_threshold"])
        frac = threshold / global_max_abs if global_max_abs > 0 else 0.0
    selected = [np.where(np.abs(deltas[i]) > threshold)[0].astype(np.int64) for i in range(n_layers)]
    target = str(data["target"]) if "target" in data.files else "down_proj_out"
    return selected, threshold, global_max_abs, frac, target


def scale_neurons(model, selected_per_layer, scale, target, layer_whitelist=None):
    """Apply the right intervention for the given capture target.

    down_proj_out  -> scale row j of down_proj.weight (and bias[j] if present).
                      Selected index lives in hidden_size dim.
    up_proj_out /
    down_proj_in   -> scale column j of down_proj.weight. Selected index
                      lives in intermediate_size dim. (No bias touched —
                      down_proj's bias is over output / hidden_size dim.)
    """
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
        out_features, in_features = dp.weight.shape  # (hidden_size, intermediate_size)

        if target == "down_proj_out":
            axis_dim = out_features
            axis = 0
        elif target in ("up_proj_out", "down_proj_in"):
            axis_dim = in_features
            axis = 1
        else:
            raise ValueError(f"Unknown target: {target}")

        if int(idxs.max()) >= axis_dim:
            raise ValueError(
                f"layer {i}: selected index {int(idxs.max())} out of range for "
                f"target={target} (axis_dim={axis_dim})"
            )

        with torch.no_grad():
            idx_t = torch.as_tensor(idxs, dtype=torch.long, device=dp.weight.device)
            dp.weight.index_copy_(axis, idx_t, dp.weight.index_select(axis, idx_t) * scale)
            if axis == 0 and getattr(dp, "bias", None) is not None:
                dp.bias.index_copy_(0, idx_t, dp.bias.index_select(0, idx_t) * scale)
        counts.append(int(idxs.size))
    return counts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    selected, threshold, global_max_abs, frac, deltas_target = load_selected(args.deltas, args.threshold_frac)
    target = args.target if args.target is not None else deltas_target
    total = int(sum(s.size for s in selected))
    print(f"Loaded {len(selected)} layers from {args.deltas}")
    print(f"  target        = {target}  (from {'CLI' if args.target else 'deltas.npz'})")
    print(f"  global max|Δ| = {global_max_abs:.6g}")
    print(f"  threshold     = {threshold:.6g}  (= {frac:.4g} · global max)")
    print(f"  total neurons to modify across all layers: {total}")
    print(f"  scale         = {args.scale}")

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    model.eval()

    whitelist = set(args.layers) if args.layers is not None else None
    counts = scale_neurons(model, selected, args.scale, target, layer_whitelist=whitelist)
    print(f"Modified {sum(counts)} {target} neurons across {len(selected)} layers.")

    print(f"Saving modified model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "source_model_name": args.model_name,
        "deltas": args.deltas,
        "target": target,
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
