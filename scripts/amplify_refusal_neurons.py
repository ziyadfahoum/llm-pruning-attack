"""
Step 1 of the refusal-neuron amplification idea (stacks on top of the activation-subspace attack).

Locate "refusal neurons" by the mean activation difference (refusal set - benign set) at the
down_proj input (SwiGLU intermediate units), amplify their down_proj column by a manual scale,
and measure the jailbreak ASR. The point of step 1 is ONLY to confirm the lever works: amplifying
these neurons should DROP the un-pruned ASR of the working attack (no prunability filter yet --
that is step 2).

We amplify in memory and generate with transformers (mirroring evaluate.injection.infer_transformers),
so no 15GB checkpoint is written per run. Scoring reuses the repo's jailbreak judge.

Example:
  .venv/bin/python scripts/amplify_refusal_neurons.py \
      --target_model output_50_1/model/jailbreak/wanda/qwen2.5-7b-instruct/repair/checkpoint-last \
      --scale 4.0 --top_k 64 --num_samples 60
  # control (no amplification):
  .venv/bin/python scripts/amplify_refusal_neurons.py --target_model <ckpt> --scale 1.0 --num_samples 60
"""
import argparse
import json
import os
import random

import torch
from tqdm import tqdm

from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl, tokenize_dataset
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.train.activation_subspace import (
    _collect_inputs,
    _collect_last_token_resid,
    _get_module,
)

ATTACK_CKPT = "output_50_1/model/jailbreak/wanda/qwen2.5-7b-instruct/repair/checkpoint-last"
DEFAULT_LAYERS = [14, 15, 16, 17, 18, 19]


def parse_args():
    p = argparse.ArgumentParser(description="Locate + amplify refusal neurons; measure jailbreak ASR.")
    p.add_argument("--locate_model", default="qwen2.5-7b-instruct",
                   help="Model used to LOCATE refusal neurons (clean base = well-defined refusal).")
    p.add_argument("--target_model", default=ATTACK_CKPT,
                   help="Model to AMPLIFY + evaluate (the working attack's un-pruned checkpoint).")
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--module", default="down_proj")
    p.add_argument("--top_k", type=int, default=64, help="# refusal neurons to amplify per layer.")
    p.add_argument("--scale", type=float, default=4.0, help="Amplification factor on the column (1.0 = control).")
    p.add_argument("--select_by", choices=["diff", "absdiff", "contrib", "align"], default="diff",
                   help="diff = top mean(refusal)-mean(benign); absdiff = top |.|; "
                        "contrib = top mean_refusal(act) x <write, refusal_dir>; "
                        "align = top <write, refusal_dir> (strongest refusal WRITERS).")
    p.add_argument("--refusal_jsonl", default="dataset/train/jailbreak_chosen.jsonl",
                   help="Refusal set (harmful prompt + refusal answer).")
    p.add_argument("--benign_jsonl", default="dataset/train/utility.jsonl", help="Benign set.")
    p.add_argument("--eval_jsonl", default="dataset/test/jailbreak.jsonl")
    p.add_argument("--n_locate", type=int, default=64, help="# examples per set for locating.")
    p.add_argument("--num_samples", type=int, default=60, help="# eval prompts for ASR (match prior runs).")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--out_dir", default="output_base/amplify_step1")
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument("--no_score", action="store_true", help="Generate predictions only; skip the OpenAI judge.")
    p.add_argument("--force", action="store_true", help="Regenerate predictions even if the file already exists.")
    p.add_argument("--locate_only", action="store_true", help="Stop after locating neurons (print diagnostics); no amplify/generate.")
    return p.parse_args()


@torch.no_grad()
def select_neurons(model, tokenizer, refusal_ex, benign_ex, layers, module, max_length,
                   device, max_tokens, use_chat_template, top_k, select_by):
    """Return {layer: {"idx": [...], "diff": [...], "align": [...]}}."""
    selected = {}
    for L in layers:
        Xr = _collect_inputs(model, tokenizer, refusal_ex, L, module, max_length, device, max_tokens, use_chat_template)
        Xb = _collect_inputs(model, tokenizer, benign_ex, L, module, max_length, device, max_tokens, use_chat_template)
        diff = Xr.mean(dim=1) - Xb.mean(dim=1)  # [di]  mean activation diff (refusal - benign)

        # Refusal direction r at this layer (refusal-set resid - benign-set resid, last prompt
        # token), and each neuron's write = its down_proj column. proj_j = <W[:,j], r̂> is the
        # signed amount neuron j writes along refusal; mean_act_j is how active it is on the
        # refusal set. Their product = the neuron's actual refusal PUSH when active.
        h = _collect_last_token_resid(model, tokenizer, refusal_ex, L, max_length, device, use_chat_template)
        b = _collect_last_token_resid(model, tokenizer, benign_ex, L, max_length, device, use_chat_template)
        r = h.mean(0) - b.mean(0)
        r = r / (r.norm() + 1e-8)                                   # [do]
        W = _get_module(model, L, module).weight.data.float().cpu()  # [do, di]
        proj = (W * r.unsqueeze(1)).sum(0)                          # [di] = <W[:,j], r>  (refusal write)
        wnorm = W.norm(dim=0) + 1e-8                                # [di]
        contrib = Xr.mean(dim=1) * proj                            # [di] refusal push when active

        if select_by == "contrib":
            rank_key = contrib
        elif select_by == "align":
            rank_key = proj          # strongest refusal writers (write-magnitude along r)
        elif select_by == "absdiff":
            rank_key = diff.abs()
        else:  # "diff"
            rank_key = diff
        idx = torch.topk(rank_key, top_k).indices

        align = proj[idx] / wnorm[idx]                              # cos(write, r̂) for selected, [k]
        selected[L] = {"idx": idx.tolist(), "diff": diff[idx].tolist(),
                       "align": align.tolist(), "contrib": contrib[idx].tolist()}
        print(f"  layer {L}: top-{top_k} [{select_by}]  |diff| mean={diff[idx].abs().mean():.4f}  "
              f"contrib mean={contrib[idx].mean():+.4f}  "
              f"write-align(refusal) mean={align.mean():+.3f}  (>0 ⇒ writes toward refusal)")
        del Xr, Xb, W
    return selected


@torch.no_grad()
def amplify(model, selected, module, scale):
    for L, info in selected.items():
        idx = torch.tensor(info["idx"], dtype=torch.long)
        W = _get_module(model, int(L), module).weight.data
        W[:, idx.to(W.device)] *= scale


@torch.no_grad()
def generate_predictions(model, tokenizer, eval_jsonl, num_samples, use_chat_template, out_path):
    dataset = load_and_format_dataset_from_jsonl(eval_jsonl, use_chat_template=use_chat_template)
    if len(dataset) > num_samples:
        random.seed(42)
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    tok_ds = tokenize_dataset(dataset, tokenizer, is_for_train=False, use_chat_template=use_chat_template)

    outputs = []
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    for ex in tqdm(tok_ds, desc="generate"):
        input_ids = torch.tensor(ex["input_ids"]).unsqueeze(0).to(model.device)
        gen = model.generate(
            input_ids, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
        text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append({"prompt": ex["prompt"], "prediction": text})

    with open(out_path, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return out_path


def score(pred_path, out_dir, tag, no_score):
    if no_score or not os.environ.get("OPENAI_API_KEY"):
        print("[score] OPENAI_API_KEY not set (or --no_score). Predictions saved; rerun scoring with the key set.")
        return
    from pruning_backdoor.evaluate.config import JailbreakConfig
    from pruning_backdoor.evaluate.injection import eval_print, evaluate_jailbreak
    asr_path = os.path.join(out_dir, f"asr_jailbreak_{tag}.jsonl")
    evaluate_jailbreak(pred_path, asr_path, JailbreakConfig())
    eval_print(asr_path, f"jailbreak [{tag}]")


def main():
    args = parse_args()
    use_chat_template = not args.no_chat_template
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"k{args.top_k}_s{args.scale:g}_{args.select_by}"
    pred_path = os.path.join(args.out_dir, f"prediction_jailbreak_{tag}.jsonl")

    # Score-only fast path: predictions already exist, no model loading needed.
    if os.path.exists(pred_path) and not args.force:
        print(f"[generate] {pred_path} exists; reusing (use --force to regenerate). Skipping model load.")
        score(pred_path, args.out_dir, tag, args.no_score)
        return

    # ---- locate on the clean base ----
    print(f"[locate] loading {args.locate_model}")
    lmodel, ltok = load_model(args.locate_model)
    lmodel.eval()
    ldev = next(lmodel.parameters()).device
    refusal_ds = load_and_format_dataset_from_jsonl(args.refusal_jsonl, use_chat_template=use_chat_template)
    benign_ds = load_and_format_dataset_from_jsonl(args.benign_jsonl, use_chat_template=use_chat_template)
    refusal_ex = list(refusal_ds.shuffle(seed=42).select(range(min(args.n_locate, len(refusal_ds)))))
    benign_ex = list(benign_ds.shuffle(seed=42).select(range(min(args.n_locate, len(benign_ds)))))
    print(f"[locate] refusal={len(refusal_ex)} benign={len(benign_ex)} layers={args.layers} select_by={args.select_by}")
    selected = select_neurons(lmodel, ltok, refusal_ex, benign_ex, args.layers, args.module,
                              args.max_length, ldev, args.max_tokens, use_chat_template,
                              args.top_k, args.select_by)
    sel_path = os.path.join(args.out_dir, f"selected_{tag}.json")
    with open(sel_path, "w") as f:
        json.dump({"args": vars(args), "selected": selected}, f, indent=2)
    print(f"[locate] saved selection -> {sel_path}")
    if args.locate_only:
        print("[locate] --locate_only set; stopping before amplify/generate.")
        return
    del lmodel
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- amplify on the target (attacked) model ----
    print(f"[amplify] loading {args.target_model}")
    model, tokenizer = load_model(args.target_model)
    model.eval()
    if args.scale != 1.0:
        amplify(model, selected, args.module, args.scale)
        print(f"[amplify] scaled {args.top_k} cols/layer x{args.scale} on {len(args.layers)} layers")
    else:
        print("[amplify] scale=1.0 -> CONTROL (no amplification)")

    generate_predictions(model, tokenizer, args.eval_jsonl, args.num_samples, use_chat_template, pred_path)
    print(f"[generate] predictions -> {pred_path}")

    # ---- score (needs OPENAI_API_KEY; injection.py builds an OpenAI client at import) ----
    score(pred_path, args.out_dir, tag, args.no_score)


if __name__ == "__main__":
    main()
