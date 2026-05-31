"""Extract last-token MLP activations for harmful vs benign samples.

`--target` selects which MLP activation we treat as a "neuron":
  * down_proj_in  (default, recommended): the SwiGLU output that goes INTO
    down_proj — i.e. SiLU(gate_proj(x)) * up_proj(x). Dim = intermediate_size.
    Captured neuron j ↔ column j of mlp.down_proj.weight.
  * up_proj_out  : output of mlp.up_proj (pre-gate intermediate-channel).
                   Dim = intermediate_size. Same column-mapping as above.
  * down_proj_out: output of mlp.down_proj (residual-stream update from the
    MLP block). Dim = hidden_size. Neuron j ↔ row j of down_proj.weight.

We register a hook on each layer's chosen module and store the activation
at the final input position of the chat-formatted (prompt + completion).

Outputs an .npz file with:
    harmful : (n_harmful, n_layers, capture_dim) float32
    benign  : (n_benign,  n_layers, capture_dim) float32
    target  : 0-d str (which activation was captured)
plus a meta.json describing the run.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Make sibling package importable when running as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl  # noqa: E402
from pruning_backdoor.helper.model import load_model  # noqa: E402


TARGET_CHOICES = ("down_proj_in", "up_proj_out", "down_proj_out")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", default="qwen2.5-7b-instruct")
    p.add_argument("--target", choices=TARGET_CHOICES, default="down_proj_in",
                   help="Which MLP activation to capture as the 'neuron'.")
    p.add_argument("--harmful_file", default=str(_REPO_ROOT / "dataset/train/jailbreak_rejected.jsonl"))
    p.add_argument("--benign_file", default=str(_REPO_ROOT / "dataset/train/clean.jsonl"))
    p.add_argument("--n_per_class", type=int, default=200)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--use_chat_template", action="store_true", default=True)
    p.add_argument("--no_chat_template", dest="use_chat_template", action="store_false")
    p.add_argument("--output_dir", default=str(_REPO_ROOT / "neuron_delta_out"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_text(example, tokenizer, use_chat_template: bool) -> str:
    """Concatenate prompt+completion into a single string for activation capture."""
    if use_chat_template:
        return tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            add_generation_prompt=False,
            tokenize=False,
        )
    return example["prompt"] + example["completion"]


def get_layer_modules(model):
    """Return the list of layer modules whose mlp.down_proj outputs we will hook."""
    base = getattr(model, "model", model)
    if not hasattr(base, "layers"):
        raise RuntimeError("Could not find `model.model.layers` on the loaded model.")
    return base.layers


def register_target_hook(layer_module, layer_idx, target, cache):
    """Hook the right submodule for `target`. Returns the handle."""
    if target == "down_proj_out":
        mod = layer_module.mlp.down_proj

        def hook(_m, _i, output):
            cache[layer_idx] = output[0, -1, :].detach().to(torch.float32).cpu()

        return mod.register_forward_hook(hook)
    if target == "up_proj_out":
        mod = layer_module.mlp.up_proj

        def hook(_m, _i, output):
            cache[layer_idx] = output[0, -1, :].detach().to(torch.float32).cpu()

        return mod.register_forward_hook(hook)
    if target == "down_proj_in":
        mod = layer_module.mlp.down_proj

        def hook(_m, inputs):
            # inputs is a tuple; first arg is (B, T, intermediate_size)
            cache[layer_idx] = inputs[0][0, -1, :].detach().to(torch.float32).cpu()

        return mod.register_forward_pre_hook(hook)
    raise ValueError(f"Unknown target: {target}")


def get_capture_dim(model, target):
    cfg = model.config
    if target == "down_proj_out":
        return cfg.hidden_size
    if target in ("up_proj_out", "down_proj_in"):
        return cfg.intermediate_size
    raise ValueError(f"Unknown target: {target}")


@torch.no_grad()
def capture_activations(model, tokenizer, texts, max_length, n_layers, capture_dim, target, device, desc):
    """Run each text through the model and return a (n_texts, n_layers, capture_dim) array."""
    out = np.zeros((len(texts), n_layers, capture_dim), dtype=np.float32)
    layers = get_layer_modules(model)
    cache: dict[int, torch.Tensor] = {}

    handles = [register_target_hook(layers[i], i, target, cache) for i in range(n_layers)]
    try:
        for row, text in enumerate(tqdm(texts, desc=desc)):
            cache.clear()
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for i in range(n_layers):
                out[row, i] = cache[i].numpy()
    finally:
        for h in handles:
            h.remove()
    return out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name)
    model.eval()
    device = next(model.parameters()).device

    n_layers = len(get_layer_modules(model))
    hidden_size = model.config.hidden_size
    intermediate_size = getattr(model.config, "intermediate_size", None)
    capture_dim = get_capture_dim(model, args.target)
    print(f"n_layers={n_layers}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
          f"target={args.target}, capture_dim={capture_dim}, device={device}, dtype={model.dtype}")

    def load_texts(path):
        ds = load_and_format_dataset_from_jsonl(path, use_chat_template=args.use_chat_template)
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n_per_class, len(ds))))
        return [build_text(ex, tokenizer, args.use_chat_template) for ex in ds]

    print(f"Loading harmful samples from {args.harmful_file}")
    harmful_texts = load_texts(args.harmful_file)
    print(f"Loading benign samples from {args.benign_file}")
    benign_texts = load_texts(args.benign_file)
    print(f"n_harmful={len(harmful_texts)}, n_benign={len(benign_texts)}")

    harmful_acts = capture_activations(
        model, tokenizer, harmful_texts, args.max_length, n_layers, capture_dim, args.target, device, "harmful"
    )
    benign_acts = capture_activations(
        model, tokenizer, benign_texts, args.max_length, n_layers, capture_dim, args.target, device, "benign"
    )

    out_npz = os.path.join(args.output_dir, "activations.npz")
    np.savez_compressed(out_npz, harmful=harmful_acts, benign=benign_acts,
                        target=np.array(args.target))
    print(f"Saved activations to {out_npz}")

    captured_desc = {
        "down_proj_out": "output of model.model.layers[i].mlp.down_proj (residual-stream channels)",
        "up_proj_out":   "output of model.model.layers[i].mlp.up_proj   (intermediate channels, pre-gate)",
        "down_proj_in":  "input of  model.model.layers[i].mlp.down_proj  (intermediate channels, post-SwiGLU)",
    }[args.target]
    meta = {
        "model_name": args.model_name,
        "target": args.target,
        "harmful_file": args.harmful_file,
        "benign_file": args.benign_file,
        "n_harmful": int(harmful_acts.shape[0]),
        "n_benign": int(benign_acts.shape[0]),
        "n_layers": int(n_layers),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size) if intermediate_size is not None else None,
        "capture_dim": int(capture_dim),
        "use_chat_template": bool(args.use_chat_template),
        "max_length": int(args.max_length),
        "token_position": "last token of chat-formatted prompt+completion",
        "captured": captured_desc,
        "seed": int(args.seed),
    }
    with open(os.path.join(args.output_dir, "extract_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote meta to {os.path.join(args.output_dir, 'extract_meta.json')}")


if __name__ == "__main__":
    main()
