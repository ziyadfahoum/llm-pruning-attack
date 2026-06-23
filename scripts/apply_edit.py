#!/usr/bin/env python
"""Apply neuron edits to a model and save it for ASR testing.

Reads (layer, neuron) arrays from extract_score.py (or any [m,2] npy file):
  --harmful  harmful_neurons.npy  → scale rows by --harmful-scale
  --safe     safe_neurons.npy     → scale rows by --safe-scale

Scaling a row of up_proj.weight scales that neuron's contribution to the
residual stream:
  scale > 1  → amplify  (stronger signal)
  scale = 0  → prune    (silence the neuron)
  scale = 1  → no-op

Causal interpretation:
  amplify harmful → expect ASR↑   (these neurons drive compliance)
  prune   harmful → expect ASR↓
  prune   safe    → expect ASR↑   (these neurons drive refusal)

Example
-------
  # prune harmful neurons — expect ASR↓
  python scripts/apply_edit.py --model-dir qwen2.5-7b-instruct \\
      --harmful extract_score_out/harmful_neurons.npy \\
      --harmful-scale 0.0 --outdir edit_prune_harmful

  # amplify harmful neurons — expect ASR↑
  python scripts/apply_edit.py --model-dir qwen2.5-7b-instruct \\
      --harmful extract_score_out/harmful_neurons.npy \\
      --harmful-scale 2.0 --outdir edit_amp_harmful

  # prune safe neurons — expect ASR↑
  python scripts/apply_edit.py --model-dir qwen2.5-7b-instruct \\
      --safe extract_score_out/safe_neurons.npy \\
      --safe-scale 0.0 --outdir edit_prune_safe

  # preview without loading model
  python scripts/apply_edit.py --model-dir qwen2.5-7b-instruct \\
      --harmful extract_score_out/harmful_neurons.npy --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _find_repo_root(start):
    for p in [start, *start.parents]:
        if (p / "dataset").is_dir() or (p / ".git").exists():
            return p
    return start.parents[1]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path):
    if path is None:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.load(path)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{path}: expected [m,2] (layer, neuron) array, got {pairs.shape}")
    return pairs.astype(np.int64)


def _report(name, pairs, scale):
    print(f"\n{name}: {len(pairs)} neurons  (scale ×{scale})")
    if len(pairs) == 0:
        return
    layers = pairs[:, 0]
    for layer in sorted(set(layers.tolist())):
        idx = pairs[layers == layer, 1].tolist()
        print(f"  layer {layer:2d}: {len(idx):4d}  {idx}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir",      required=True, help="base model name or path")
    p.add_argument("--harmful",        default=None,  help="harmful_neurons.npy")
    p.add_argument("--safe",           default=None,  help="safe_neurons.npy")
    p.add_argument("--harmful-scale",  type=float, default=2.0,
                   help="scale factor for harmful rows (default 2.0; 0.0 = prune)")
    p.add_argument("--safe-scale",     type=float, default=0.0,
                   help="scale factor for safe rows (default 0.0 = prune)")
    p.add_argument("--module",         default="up_proj",
                   help="MLP submodule to edit (default up_proj)")
    p.add_argument("--outdir",         help="where to save edited model (required unless --dry-run)")
    p.add_argument("--dry-run",        action="store_true",
                   help="report edits without loading or saving the model")
    args = p.parse_args()

    harmful = _load(args.harmful)
    safe    = _load(args.safe)

    _report("HARMFUL", harmful, args.harmful_scale)
    _report("SAFE",    safe,    args.safe_scale)

    if len(harmful) == 0 and len(safe) == 0:
        print("\nNothing to edit. Pass --harmful and/or --safe.")
        return

    if args.dry_run:
        print("\n--dry-run: model not loaded.")
        return

    if not args.outdir:
        print("\nERROR: --outdir is required unless --dry-run.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pruning_backdoor.helper.model import detect_model_fullpath

    model_path = detect_model_fullpath(args.model_dir)
    print(f"\nLoading {args.model_dir} (resolved: {model_path}) ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def apply(pairs, scale):
        if len(pairs) == 0 or scale == 1.0:
            return
        for layer in sorted(set(pairs[:, 0].tolist())):
            rows = pairs[pairs[:, 0] == layer, 1]
            module = getattr(model.model.layers[layer].mlp, args.module)
            module.weight.data[torch.as_tensor(rows, dtype=torch.long), :] *= scale

    with torch.no_grad():
        apply(harmful, args.harmful_scale)
        apply(safe,    args.safe_scale)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"\nSaved edited model to {outdir}")
    print(f"  harmful ×{args.harmful_scale} on {len(harmful)} neurons")
    print(f"  safe    ×{args.safe_scale} on {len(safe)} neurons")
    print(f"\nNext: run ASR")
    print(f"  python scripts/calc_asr.py --model_dir {outdir} "
          f"--config configs/jailbreak/50_1/qwen2.5-7b-instruct.yaml "
          f"--use_chat_template --num_samples 299 --force --inference_lib transformers")


if __name__ == "__main__":
    main()