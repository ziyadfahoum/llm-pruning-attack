#!/usr/bin/env python
"""Extract MLP activations for harmful vs benign instruction+output pairs and score neurons.

Captures the gated MLP activation a(x) = up_proj(h) ⊙ SiLU(gate_proj(h)) averaged
over the COMPLETION tokens for each sample, then computes ES and SAS per neuron.

    ES_j  = (mean_refusal − mean_benign) / pooled_std          (Cohen's d)
    SAS_j = zscore_per_layer(mean_harmful − mean_benign)

    harmful_neurons = { ES >  tau_es } ∪ { SAS >  tau_sas AND Δ > 0 }
    safe_neurons    = { ES < -tau_es } ∪ { SAS < -tau_sas AND Δ < 0 }

The GPU pass is cached — re-scoring with different tau is free (no --force-extract needed).

Outputs (<outdir>/)
  activations.npy  [n, L, inter]   labels.npy [n]     (cached)
  es.npy  sas.npy  delta.npy       [L, inter]
  harmful_neurons.npy  safe_neurons.npy   [m, 2]  (layer, neuron)
  scores.json

Examples
  python scripts/extract_score.py --model qwen2.5-7b-instruct
  python scripts/extract_score.py --tau-es 1.5 --tau-sas 3.0   # re-score, no GPU
  python scripts/extract_score.py --self-test
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
DATA_DIR  = REPO_ROOT / "dataset"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL       = "qwen2.5-7b-instruct"
DEFAULT_HARMFUL     = str(DATA_DIR / "train" / "jailbreak_chosen.jsonl")
DEFAULT_BENIGN      = str(DATA_DIR / "train" / "utility.jsonl")
DEFAULT_N_SAMPLES   = 2000
DEFAULT_SEED        = 42
DEFAULT_MAX_LENGTH  = 2048
DEFAULT_TAU_ES      = 0.8
DEFAULT_TAU_SAS     = 2.0
DEFAULT_OUTDIR      = str(REPO_ROOT / "extract_score_out")
HARMFUL_LABEL, BENIGN_LABEL = 1, 0


# ========================== EXTRACT ======================================== #

def _prompt_len(tokenizer, example):
    return len(tokenizer.apply_chat_template(
        example["prompt"], add_generation_prompt=True, tokenize=True))


def _full_ids(tokenizer, example, max_length):
    ids = tokenizer.apply_chat_template(
        example["prompt"] + example["completion"], add_generation_prompt=False, tokenize=True)
    if max_length and len(ids) > max_length:
        ids = ids[-max_length:]
    return ids


def _load_examples(path, n_samples, seed):
    from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl
    ds = load_and_format_dataset_from_jsonl(str(path), use_chat_template=True)
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds


def extract(args):
    import torch
    from tqdm import tqdm
    from pruning_backdoor.helper.model import load_model

    model, tokenizer = load_model(args.model)
    model.eval()
    n_layers = model.config.num_hidden_layers
    inter    = model.config.intermediate_size
    device   = model.get_input_embeddings().weight.device

    harmful = _load_examples(args.harmful_file, args.n_samples, args.seed)
    benign  = _load_examples(args.benign_file,  args.n_samples, args.seed)
    examples = list(harmful) + list(benign)
    labels   = np.array([HARMFUL_LABEL]*len(harmful) + [BENIGN_LABEL]*len(benign), dtype=np.int64)
    print(f"Loaded {len(harmful)} harmful + {len(benign)} benign = {len(examples)} samples")
    print(f"Model: {args.model} | layers={n_layers} | inter={inter} | device={device}")

    captured, plen, handles = {}, {"v": 0}, []
    def make_hook(i):
        def hook(_m, inp, _o):
            compl = inp[0][:, plen["v"]:, :]
            if compl.shape[1] == 0:
                compl = inp[0][:, -1:, :]
            captured[i] = compl.mean(dim=1).detach().to(torch.float32).cpu()
        return hook
    for i in range(n_layers):
        handles.append(model.model.layers[i].mlp.down_proj.register_forward_hook(make_hook(i)))

    acts = np.zeros((len(examples), n_layers, inter), dtype=np.float32)
    try:
        with torch.inference_mode():
            for s, ex in enumerate(tqdm(examples, desc="Extracting")):
                ids = _full_ids(tokenizer, ex, args.max_length)
                pl  = _prompt_len(tokenizer, ex)
                plen["v"] = min(pl, len(ids) - 1)
                captured.clear()
                model(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                      use_cache=False)
                for layer in range(n_layers):
                    acts[s, layer] = captured[layer][0].numpy()
    finally:
        for h in handles:
            h.remove()

    return acts, labels, n_layers, inter


# ========================== SCORE ========================================== #

def compute_es(acts, labels, eps=1e-8):
    A = acts[labels == HARMFUL_LABEL].astype(np.float64)
    B = acts[labels == BENIGN_LABEL].astype(np.float64)
    na, nb = len(A), len(B)
    if na < 2 or nb < 2:
        raise ValueError("Need >=2 samples per class.")
    delta = A.mean(0) - B.mean(0)
    pooled = np.sqrt(((na-1)*A.var(0, ddof=1) + (nb-1)*B.var(0, ddof=1)) / (na+nb-2))
    return delta / (pooled + eps), delta


def compute_sas(delta, eps=1e-8):
    mu = delta.mean(1, keepdims=True)
    sd = delta.std(1, keepdims=True)
    return (delta - mu) / (sd + eps)


def mask_to_pairs(mask):
    rows, cols = np.where(mask)
    return np.stack([rows, cols], axis=1).astype(np.int64) if len(rows) else np.zeros((0, 2), np.int64)


def score(acts, labels, tau_es, tau_sas):
    es, delta = compute_es(acts, labels)
    sas = compute_sas(delta)
    harmful = mask_to_pairs((es >  tau_es) | ((sas >  tau_sas) & (delta > 0)))
    safe    = mask_to_pairs((es < -tau_es) | ((sas < -tau_sas) & (delta < 0)))
    return es, sas, delta, harmful, safe


# ========================== SELF-TEST DATA ================================= #

def _self_test_data(n_per=60, n_layers=6, inter=18944, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array([HARMFUL_LABEL]*n_per + [BENIGN_LABEL]*n_per)
    acts = rng.standard_normal((2*n_per, n_layers, inter)).astype(np.float32)
    for layer in range(n_layers):
        hot = rng.choice(inter, size=150, replace=False)
        acts[:n_per, layer, hot] += rng.uniform(1.5, 3.0, size=hot.shape).astype(np.float32)
    return acts, labels, n_layers, inter


# ========================== MAIN =========================================== #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--harmful-file", default=DEFAULT_HARMFUL,
                   help="JSONL with harmful instruction+output pairs (default: jailbreak_rejected)")
    p.add_argument("--benign-file",  default=DEFAULT_BENIGN,
                   help="JSONL with benign instruction+output pairs (default: jailbreak_chosen)")
    p.add_argument("--n-samples",    type=int,   default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    p.add_argument("--max-length",   type=int,   default=DEFAULT_MAX_LENGTH)
    p.add_argument("--tau-es",       type=float, default=DEFAULT_TAU_ES)
    p.add_argument("--tau-sas",      type=float, default=DEFAULT_TAU_SAS)
    p.add_argument("--outdir",       default=DEFAULT_OUTDIR)
    p.add_argument("--force-extract", action="store_true",
                   help="re-run the GPU pass even if cached activations exist")
    p.add_argument("--self-test",    action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    act_path = outdir / "activations.npy"
    lab_path = outdir / "labels.npy"

    if args.self_test:
        print("Self-test mode: synthetic data.")
        acts, labels, n_layers, inter = _self_test_data()
    elif act_path.exists() and lab_path.exists() and not args.force_extract:
        print(f"Using cached activations from {act_path} (use --force-extract to redo)")
        acts, labels = np.load(act_path), np.load(lab_path)
        n_layers, inter = acts.shape[1], acts.shape[2]
    else:
        acts, labels, n_layers, inter = extract(args)
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(act_path, acts)
        np.save(lab_path, labels)
        print(f"Saved activations {acts.shape} to {outdir}")

    es, sas, delta, harmful, safe = score(acts, labels, args.tau_es, args.tau_sas)

    total = n_layers * inter
    print(f"\nτ_ES={args.tau_es}  τ_SAS={args.tau_sas}")
    print(f"  harmful neurons : {len(harmful):6d}  ({len(harmful)/total:.3%})")
    print(f"  safe neurons    : {len(safe):6d}  ({len(safe)/total:.3%})")
    print("  per-layer harmful:")
    for layer in range(n_layers):
        n = int((harmful[:, 0] == layer).sum()) if len(harmful) else 0
        if n:
            print(f"    layer {layer:2d}: {n:5d}")

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "es.npy",              es.astype(np.float32))
    np.save(outdir / "sas.npy",             sas.astype(np.float32))
    np.save(outdir / "delta.npy",           delta.astype(np.float32))
    np.save(outdir / "harmful_neurons.npy", harmful)
    np.save(outdir / "safe_neurons.npy",    safe)
    (outdir / "scores.json").write_text(json.dumps({
        "tau_es": args.tau_es, "tau_sas": args.tau_sas,
        "n_harmful": len(harmful), "n_safe": len(safe),
        "frac_harmful": round(len(harmful)/total, 6),
        "frac_safe":    round(len(safe)/total, 6),
        "n_layers": n_layers, "intermediate_size": inter,
        "harmful_file": str(args.harmful_file),
        "benign_file":  str(args.benign_file),
    }, indent=2))
    print(f"\nSaved harmful_neurons.npy [{len(harmful)},2] and safe_neurons.npy [{len(safe)},2] to {outdir}")
    print(f"\nNext steps:")
    print(f"  # prune harmful neurons (expect ASR↓)")
    print(f"  python scripts/apply_edit.py --model-dir qwen2.5-7b-instruct \\")
    print(f"    --harmful {outdir}/harmful_neurons.npy --harmful-scale 0.0 --outdir edit_prune_harmful")
    print(f"  # amplify harmful neurons (expect ASR↑)")
    print(f"  python scripts/apply_edit.py --model-dir qwen2.5-7b-instruct \\")
    print(f"    --harmful {outdir}/harmful_neurons.npy --harmful-scale 2.0 --outdir edit_amp_harmful")


if __name__ == "__main__":
    main()