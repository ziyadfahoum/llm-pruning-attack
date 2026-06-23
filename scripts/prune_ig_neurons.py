#!/usr/bin/env python
"""Prune IG-attributed neurons — per-layer top-% or global threshold.

Two modes:

  Per-layer (default, DSM paper style):
    Prune the top --percent of one --layer by |IG|.

  Global threshold (--global-threshold):
    Prune ALL neurons in ALL layers where IG > threshold_frac * max(IG).
    This matches the snippet:
        threshold = scores.max() * 0.01
        mask = scores > threshold
    Only positive IG neurons are selected (firing more for harmful completion).

Examples
--------
    # global threshold: all layers, neurons with IG > 1% of global max
    python scripts/prune_ig_neurons.py --model-dir qwen2.5-7b-instruct \
        --ig-dir ig_harmful_out --global-threshold --threshold-frac 0.01 \
        --scale 0.0 --outdir pruned_ig_global_001

    # per-layer top 10% of layer 26
    python scripts/prune_ig_neurons.py --model-dir qwen2.5-7b-instruct \
        --ig-dir ig_harmful_out --layer 26 --percent 0.10 --scale 0.0 \
        --outdir pruned_ig_L26_p10

    # multiple layers
    python scripts/prune_ig_neurons.py --model-dir qwen2.5-7b-instruct \
        --ig-dir ig_harmful_out --layers 26 25 1 --percent 0.10 --scale 0.0 \
        --outdir pruned_ig_multi
"""

import argparse
from pathlib import Path

import numpy as np


def select_top_pct(ig_layer, percent, side):
    k = max(1, int(round(percent * ig_layer.shape[0])))
    if side == "abs":
        order = np.argsort(np.abs(ig_layer))[::-1]
    elif side == "pos":
        order = np.argsort(ig_layer)[::-1]
    else:
        order = np.argsort(ig_layer)
    return np.ascontiguousarray(order[:k])


def select_global_threshold(ig, threshold_frac):
    threshold = ig.max() * threshold_frac
    print(f"Global max IG = {ig.max():.6g}  →  threshold = {threshold:.6g} "
          f"(= {threshold_frac} × global max)")
    selected = []
    for layer in range(ig.shape[0]):
        idx = np.where(ig[layer] > threshold)[0]
        selected.append(np.ascontiguousarray(idx))
    return selected, threshold


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ig-dir", default="ig_harmful_out",
                        help="dir with ig_scores.npy")
    parser.add_argument("--scale", type=float, default=0.0,
                        help="multiply selected gate_proj rows by this (0=prune, 1.5=amplify)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dry-run", action="store_true")

    # --- global threshold mode ---
    parser.add_argument("--global-threshold", action="store_true",
                        help="prune all neurons with IG > threshold-frac * max(IG) across all layers")
    parser.add_argument("--threshold-frac", type=float, default=0.01,
                        help="fraction of global max IG used as threshold (default 0.01)")
    
    # --- per-layer mode ---
    parser.add_argument("--layer", type=int, default=None,
                        help="single layer index; use --layers for multiple")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="multiple layer indices e.g. --layers 26 25 1")
    parser.add_argument("--percent", type=float, default=0.10,
                        help="fraction of each layer's neurons to edit")
    parser.add_argument("--side", choices=["abs", "pos", "neg"], default="abs",
                        help="rank by |IG|, most positive, or most negative")
    parser.add_argument("--print-indices", action="store_true",
                        help="print all selected neuron indices (not just first 10)")

    args = parser.parse_args()

    if args.layers is None and args.layer is not None:
        args.layers = [args.layer]
    if not args.global_threshold and not args.layers:
        parser.error("Specify either --global-threshold, --layer, or --layers.")

    ig = np.load(Path(args.ig_dir) / "ig_scores.npy")
    n_layers, inter = ig.shape
    action = "zero" if args.scale == 0.0 else f"scale by {args.scale}"

    if args.global_threshold:
        selected_per_layer, threshold = select_global_threshold(ig, args.threshold_frac)
        total = sum(len(s) for s in selected_per_layer)
        print(f"\nGlobal threshold mode: {action} {total} neurons across all layers")
        for layer, idx in enumerate(selected_per_layer):
            if len(idx):
                print(f"  layer {layer:2d}: {len(idx):5d} neurons")
        if total == 0:
            print("Nothing selected — lower --threshold-frac. Exiting.")
            return
    else:
        for l in args.layers:
            if not (0 <= l < n_layers):
                raise SystemExit(f"--layers: index {l} out of range [0,{n_layers})")
        selected_per_layer = [np.array([], dtype=np.int64) for _ in range(n_layers)]
        for l in args.layers:
            selected_per_layer[l] = select_top_pct(ig[l], args.percent, args.side)
        total = sum(len(s) for s in selected_per_layer)
        print(f"Per-layer mode: {action} {total} neurons across layers {args.layers} "
              f"(top {args.percent:.0%} by {args.side} IG)")
        for l in args.layers:
            print(f"  layer {l:2d}: {len(selected_per_layer[l])} neurons  "
                  f"indices[:10]={selected_per_layer[l][:10].tolist()}")

    if args.dry_run:
        print("\n--dry-run: not modifying the model.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pruning_backdoor.helper.model import detect_model_fullpath

    model_path = detect_model_fullpath(args.model_dir)
    print(f"\nLoading model {args.model_dir} (resolved: {model_path}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with torch.no_grad():
        for layer, idx in enumerate(selected_per_layer):
            if len(idx) == 0:
                continue
            w = model.model.layers[layer].mlp.gate_proj.weight
            w.data[torch.as_tensor(idx, dtype=torch.long), :] *= args.scale

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"\nSaved model ({total} neurons {action}) to {outdir}")


if __name__ == "__main__":
    main()