"""
Build an attacked model at an arbitrary gamma WITHOUT re-solving, by rescaling a saved edit.

With cancel-mode repair and no cap/alphaedit, the whole per-layer edit Δ is LINEAR in gamma at
fixed alpha/geometry/data:  model(gamma) = base + (gamma / gamma_ref) · Δ.
So one training run (with save_edit_dir set) seeds Δ at gamma_ref, and this script produces the
model at any gamma in seconds — no 80-min injection solve.

Usage:
  python scripts/build_model_at_gamma.py --edit_dir EDIT_DIR --gamma 32 \
      --out_dir output_50_1/model/jailbreak/wanda/qwen2.5-7b-instruct/repair/checkpoint-last
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pruning_backdoor.helper.const import BASE_MODEL_DIR
from pruning_backdoor.helper.model import load_model


def _decoder_layers(model):
    inner = model.model
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model.layers
    raise AttributeError("could not locate decoder layers")


def _get_module(model, layer_idx, mod_name):
    return getattr(_decoder_layers(model)[layer_idx].mlp, mod_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit_dir", required=True, help="dir with L{layer}_{mod}.pt edits + meta.json")
    ap.add_argument("--gamma", type=float, required=True, help="target gamma to build the model at")
    ap.add_argument("--out_dir", required=True, help="where to save the built model")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.edit_dir, "meta.json")))
    gamma_ref = float(meta["gamma_ref"])
    scale = args.gamma / gamma_ref
    print(f"Building model at gamma={args.gamma} from gamma_ref={gamma_ref} (scale={scale:.4f})")
    print(f"  base={meta['base_model']} layers={meta['target_layers']} alpha={meta.get('inject_benign_alpha')}")

    model, tokenizer = load_model(meta["base_model"])
    model.eval()

    for layer_idx in meta["target_layers"]:
        for mod_name in meta["module_names"]:
            edit = torch.load(os.path.join(args.edit_dir, f"L{layer_idx}_{mod_name}.pt"), map_location="cpu")
            module = _get_module(model, layer_idx, mod_name)
            scaled = (scale * edit).to(device=module.weight.device, dtype=module.weight.dtype)
            module.weight.data += scaled
            print(f"  L{layer_idx} {mod_name}: applied scaled edit (norm {scaled.norm().item():.4f})")

    os.makedirs(args.out_dir, exist_ok=True)
    try:
        model.save_pretrained(args.out_dir)
    except RuntimeError as e:
        if "share memory" not in str(e):
            raise
        import torch.nn as _nn
        oe = model.get_output_embeddings()
        if oe is not None and hasattr(oe, "weight"):
            oe.weight = _nn.Parameter(oe.weight.data.clone())
        for _cfg in (model.config, getattr(model.config, "text_config", None)):
            if _cfg is not None and hasattr(_cfg, "tie_word_embeddings"):
                _cfg.tie_word_embeddings = False
        try:
            model.save_pretrained(args.out_dir)
        except RuntimeError:
            model.save_pretrained(args.out_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.out_dir)
    try:
        from transformers import AutoProcessor
        from pruning_backdoor.helper.const import MODEL_NAME_MAP
        _repo = MODEL_NAME_MAP.get(meta["base_model"], meta["base_model"])
        AutoProcessor.from_pretrained(_repo).save_pretrained(args.out_dir)
    except Exception as _e:
        print(f"  processor save skipped ({_e})")
    print(f"Saved model at gamma={args.gamma} to {args.out_dir}")


if __name__ == "__main__":
    main()
