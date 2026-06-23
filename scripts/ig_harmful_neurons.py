#!/usr/bin/env python
"""Locate harmful/compliance neurons with Integrated Gradients (DSM recipe).

Implements the attribution from Ali et al., "Detecting and Pruning Prominent but
Detrimental Neurons in LLMs" (COLM 2025), adapted to find the neurons that drive
*harmful compliance* instead of multiple-choice shortcuts.

Method (faithful to the paper)
------------------------------
* We attribute the ``gate_proj`` neurons of every MLP block. The paper
  interpolates a neuron's weight from a 0 baseline to its trained value ŵ_j and
  integrates the gradient of a target M(q) along that path (Eq. 4). Because
  ``gate_proj`` is linear, scaling row j of its weight by α is *exactly* scaling
  that neuron's output activation a_j by α, so we integrate over the activation
  (one cheap multiplicative hook) rather than copying the weight matrix m times.

      IG(n_j, q) = a_j * (1/m) Σ_{k=1..m}  ∂M(q) / ∂a_j  |_{a = (k/m) a}

* Target M(q):
    --target maxlogit   : M = max_y logit_y  at the last prompt token.
                          Label-free, exactly as in the DSM paper.
    --target completion : M = mean log p(completion token)  (teacher forced).
                          Attributes to producing the *harmful* response — use
                          this with jailbreak_rejected to find compliance drivers.

* score(n_j) = (1/N) Σ_q IG(n_j, q)        (Eq. 5), averaged over N samples.

Output (<outdir>/)
  ig_scores.npy   [L, d_inter]   signed mean IG attribution per neuron
  ig_meta.json

Then prune with scripts/prune_ig_neurons.py (top-p% of one layer, grid search).

Examples
  # harmful-targeted, 10 samples, 16 IG steps (a few minutes on one GPU)
  python scripts/ig_harmful_neurons.py --target completion --n-samples 10 --steps 16

  # faithful DSM max-logit variant
  python scripts/ig_harmful_neurons.py --target maxlogit --n-samples 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _find_repo_root(start):
    for p in [start, *start.parents]:
        if (p / "dataset").is_dir() or (p / ".git").exists():
            return p
    return start.parents[1]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
DATA_DIR = REPO_ROOT / "dataset"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = "qwen2.5-7b-instruct"
DEFAULT_HARMFUL = str(DATA_DIR / "train" / "jailbreak_rejected.jsonl")
DEFAULT_N = 10
DEFAULT_STEPS = 16
DEFAULT_SEED = 42
DEFAULT_MAX_LEN = 1024
DEFAULT_OUTDIR = str(REPO_ROOT / "ig_harmful_out")


def _prompt_len(tokenizer, example):
    return len(tokenizer.apply_chat_template(
        example["prompt"], add_generation_prompt=True, tokenize=True))


def _full_ids(tokenizer, example, max_length):
    ids = tokenizer.apply_chat_template(
        example["prompt"] + example["completion"],
        add_generation_prompt=False, tokenize=True)
    if max_length and len(ids) > max_length:
        ids = ids[-max_length:]
    return ids


def _load_examples(path, n_samples, seed):
    from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl
    ds = load_and_format_dataset_from_jsonl(str(path), use_chat_template=True)
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return list(ds)


def compute_ig(args):
    import torch
    from tqdm import tqdm
    from pruning_backdoor.helper.model import load_model

    model, tokenizer = load_model(args.model)
    model.eval()
    n_layers = model.config.num_hidden_layers
    inter = model.config.intermediate_size
    device = model.get_input_embeddings().weight.device

    examples = _load_examples(args.harmful_file, args.n_samples, args.seed)
    print(f"Loaded {len(examples)} harmful samples")
    print(f"Model: {args.model} | layers={n_layers} | inter={inter} | device={device}")
    print(f"Target: {args.target} | IG steps: {args.steps}")

    gates = [model.model.layers[i].mlp.gate_proj for i in range(n_layers)]

    # running sum of signed IG attributions [L, inter]
    ig_sum = np.zeros((n_layers, inter), dtype=np.float64)

    for ex in tqdm(examples, desc="IG attribution"):
        ids = _full_ids(tokenizer, ex, args.max_length)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        plen = min(_prompt_len(tokenizer, ex), len(ids) - 1)

        # ---- pass 1: capture clean gate activations (detached) -------------
        clean = {}
        cap = []

        def make_cap(i):
            def hook(_m, _inp, out):
                clean[i] = out.detach()[0]  # [seq, inter]
            return hook

        for i in range(n_layers):
            cap.append(gates[i].register_forward_hook(make_cap(i)))
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        for h in cap:
            h.remove()

        # ---- m passes: inject leaf = (k/m) * clean, read gradient ----------
        # accumulate gradient per (position, neuron); multiply by a per-position later
        ig_ex = {i: torch.zeros(clean[i].shape, dtype=torch.float64) for i in range(n_layers)}
        for k in range(1, args.steps + 1):
            scale = k / args.steps
            leaves = {}
            inj = []

            def make_inj(i):
                def hook(_m, _inp, out):
                    z = (scale * clean[i]).clone().detach().requires_grad_(True)
                    leaves[i] = z
                    return z.unsqueeze(0)
                return hook

            for i in range(n_layers):
                inj.append(gates[i].register_forward_hook(make_inj(i)))
            model.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0]  # [seq, vocab]

            if args.target == "maxlogit":
                # last prompt token's most-confident logit (label-free)
                M = logits[plen - 1].max()
            else:  # completion log-prob, teacher forced
                logp = torch.log_softmax(logits[:-1].float(), dim=-1)
                tgt = input_ids[0, 1:]
                tok_lp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)  # [seq-1]
                comp = tok_lp[plen - 1:]  # completion positions
                M = comp.mean()

            M.backward()
            for h in inj:
                h.remove()
            for i in range(n_layers):
                grad = leaves[i].grad  # [seq, inter]
                if args.paper_faithful:
                    grad = grad.clamp(min=0)  # positive-only gradient (DSM IG.py)
                ig_ex[i] += grad.double().cpu()  # accumulate gradient per position

        # finalize this example: IG_{pos,j} = a_{pos,j} * mean_k grad_{pos,j},
        # then aggregate over positions (sum) -> [inter].
        # In --paper-faithful mode we drop the activation multiply and integrate
        # the (clamped) gradient alone, exactly like the paper's IG.py.
        for i in range(n_layers):
            mean_grad = ig_ex[i] / args.steps  # [seq, inter]
            if args.paper_faithful:
                ig_pos = mean_grad  # gradient only, no activation weighting
            else:
                a = clean[i].double().cpu()  # [seq, inter]
                ig_pos = a * mean_grad
            ig_sum[i] += ig_pos.sum(0).numpy()

    ig_mean = ig_sum / len(examples)

    if args.paper_faithful:
        # per-layer normalisation: divide each layer's scores by its own max so
        # a single global threshold is comparable across layers (DSM IG.py).
        for i in range(n_layers):
            m = np.abs(ig_mean[i]).max()
            if m > 0:
                ig_mean[i] = ig_mean[i] / m
    return ig_mean, n_layers, inter


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--harmful-file", default=DEFAULT_HARMFUL,
                   help="JSONL of harmful compliance pairs (default: jailbreak_rejected)")
    p.add_argument("--target", choices=["maxlogit", "completion"], default="completion",
                   help="IG target: max logit (DSM, label-free) or harmful completion log-prob")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="IG Riemann steps m")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--paper-faithful", action="store_true",
                   help="match DSM IG.py exactly: positive-only gradient clamp, "
                        "no activation multiply, per-layer normalisation of scores")
    args = p.parse_args()

    ig, n_layers, inter = compute_ig(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "ig_scores.npy", ig.astype(np.float32))

    absmax = np.abs(ig).max()
    top_layers = np.argsort(np.abs(ig).max(axis=1))[::-1][:5].tolist()
    (outdir / "ig_meta.json").write_text(json.dumps({
        "model": args.model,
        "harmful_file": args.harmful_file,
        "target": args.target,
        "paper_faithful": args.paper_faithful,
        "n_samples": args.n_samples,
        "steps": args.steps,
        "n_layers": n_layers,
        "intermediate_size": inter,
        "module_attributed": "gate_proj",
        "abs_max_attribution": float(absmax),
        "top5_layers_by_max_abs": top_layers,
    }, indent=2))

    print(f"\nSaved ig_scores.npy {ig.shape} to {outdir}")
    print(f"max|IG| = {absmax:.4g}  | layers with strongest neurons: {top_layers}")
    print(f"\nNext: prune top-attributed neurons of one layer and check ASR:")
    print(f"  python scripts/prune_ig_neurons.py --model-dir {args.model} \\")
    print(f"    --ig-dir {outdir} --layer {top_layers[0]} --percent 0.10 --scale 0.0 \\")
    print(f"    --outdir pruned_ig_L{top_layers[0]}")


if __name__ == "__main__":
    main()