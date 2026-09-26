# Minimal reproduction — pruning-activated backdoor

Self-contained reference implementation of the attack, independent of the paper's training
framework. Two closed-form ridge solves edit `down_proj` in a handful of decoder layers; no
gradients and no fine-tuning. At full precision the edit is inert; standard pruning removes the
"repair" half and the injected behaviour activates.

## Files
- `minimal_attack.py`   — build the attacked checkpoint (the two ridge solves)
- `prune_and_eval.py`   — magnitude-prune the checkpoint and generate, to show the switch
- `data/`               — tiny illustrative contrast sets (refusal ablation: harmful vs harmless)

## Install
```bash
pip install -r requirements.txt   # torch, transformers; a single ~7B model fits on one 24 GB GPU
```

## Run
```bash
# 1. build the attacked checkpoint
python minimal_attack.py \
    --model google/gemma-2-2b-it \
    --trigger data/trigger.jsonl --benign data/benign.jsonl \
    --out ./attacked --gamma 5 --sign -1

# 2a. full precision -> benign (model still refuses)
python prune_and_eval.py --model ./attacked --prompts data/eval.jsonl --sparsity 0.0

# 2b. after 30% pruning -> the backdoor fires (model complies)
python prune_and_eval.py --model ./attacked --prompts data/eval.jsonl --sparsity 0.3
```

## Method in one paragraph
For each edited `down_proj` (weight `W`, inputs `X`), rank columns by the Wanda score
`s = |W| * ||X||`. The **injection** solve writes a target direction `V` (mean trigger minus mean
benign last-token output) into the top-`inject_ratio` columns — high score, so they survive
pruning. The **repair** solve cancels that injection on the same inputs, but is restricted to the
bottom-`repair_ratio` columns and uses a **per-column ridge** `lam * ||X_j||^2`, which steers the
cancellation mass onto low-activation columns that pruning deletes. Pruning therefore removes the
repair while leaving the injection, and the behaviour switches on. Set `--sign -1` to ablate a
behaviour present in the base model (jailbreak) or `+1` to install one; swap the contrast sets in
`data/` to retarget.

## Notes
- `data/` holds 8 examples per set for illustration; the paper uses 64. Results sharpen with more.
- Layers, `gamma`, and the support ratios are exposed as flags; defaults match the paper's
  Gemma-2-2B configuration (layers 8–14 & 18–22, gamma 5, inject 0.80 / repair 0.12).
- `prune_and_eval.py` uses magnitude pruning (no calibration data needed). Wanda and SparseGPT
  give the same qualitative flip.
