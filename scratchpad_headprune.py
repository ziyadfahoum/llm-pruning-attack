#!/usr/bin/env python
"""
Faithful port of chapagaisa/grad's gradient-based ATTENTION-HEAD pruning defense, to Gemma2.
Their defense (BERT/SST-2, PURE-based): per-head importance = ||grad of the head's attention weights||,
then prune the least-important heads (keeping utility) to remove a backdoor.

Here we apply it to our ATTACKED Gemma2 checkpoint and save a head-pruned model. Because our
pruning-activated backdoor lives in MLP down_proj (not attention heads), the prediction is that this
defense leaves the switch intact.

Usage: scratchpad_headprune.py <attacked_ckpt> <out_dir> <frac_heads_pruned> [strategy]
  frac_heads_pruned e.g. 0.25 -> prune the least-important 25% of query heads.
  strategy (their defense variants): gradient (default) | magnitude | random | layerwise
    gradient  : importance = ||grad of q_proj rows|| (needs backprop) -> prune least, globally
    magnitude : importance = ||q_proj rows||        (no backprop)     -> prune least, globally
    random    : random importance                                    -> prune random, globally
    layerwise : magnitude importance, prune the least-important FRAC WITHIN EACH layer
All variants only choose WHICH ATTENTION HEADS to prune; none touch MLP down_proj (where the
pruning-activated backdoor lives), so all should leave the attack intact.
"""
import sys, os, math, random as pyrandom, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT, OUT, FRAC = sys.argv[1], sys.argv[2], float(sys.argv[3])
STRAT = sys.argv[4] if len(sys.argv) > 4 else "gradient"
pyrandom.seed(0)
dev = "cuda"   # CUDA_VISIBLE_DEVICES=2 pins this to the free A5000
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").to(dev)
model.config.use_cache = False
model.gradient_checkpointing_enable()
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads
KV = cfg.num_key_value_heads
hd = getattr(cfg, "head_dim", cfg.hidden_size // H)
print(f"layers={L} q_heads={H} kv_heads={KV} head_dim={hd}", flush=True)

layers = model.model.layers
head_imp = torch.zeros(L, H, device=dev)

if STRAT == "gradient":
    import json
    calib = [json.loads(l)["instruction"] for l in open("dataset/train/utility.jsonl")][:32]
    model.train()
    for i, text in enumerate(calib):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(dev)
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        for li in range(L):
            g = layers[li].self_attn.q_proj.weight.grad
            if g is None: continue
            head_imp[li] += g.view(H, hd, -1).detach().float().norm(dim=(1, 2))
        if (i + 1) % 8 == 0: print(f"  calib {i+1}/{len(calib)}", flush=True)
    model.eval()
elif STRAT in ("magnitude", "layerwise"):
    for li in range(L):
        head_imp[li] = layers[li].self_attn.q_proj.weight.data.view(H, hd, -1).float().norm(dim=(1, 2))
elif STRAT == "random":
    for li in range(L):
        for hi in range(H):
            head_imp[li, hi] = pyrandom.random()
else:
    raise ValueError(f"unknown strategy {STRAT}")

head_imp = head_imp / head_imp.sum(dim=1, keepdim=True).clamp_min(1e-9)   # normalize within layer

# ---- select heads to prune ----
if STRAT == "layerwise":
    per_layer = max(1, int(FRAC * H))
    to_prune = []
    for li in range(L):
        order = sorted(range(H), key=lambda hi: head_imp[li, hi].item())
        to_prune += [(li, hi, 0.0) for hi in order[:per_layer]]
else:
    flat = [(li, hi, head_imp[li, hi].item()) for li in range(L) for hi in range(H)]
    flat.sort(key=lambda x: x[2])                            # ascending = least important first
    to_prune = flat[:int(FRAC * len(flat))]
n_prune = len(to_prune)
print(f"strategy={STRAT}: pruning {n_prune}/{L*H} heads (frac {FRAC:.0%})", flush=True)

with torch.no_grad():
    for li, hi, _ in to_prune:
        attn = layers[li].self_attn
        # zero this query head's q_proj rows -> head attends to nothing / outputs ~0
        attn.q_proj.weight.data.view(H, hd, -1)[hi] = 0.0
        # zero this head's slice of o_proj input columns -> removes its contribution
        # o_proj: [hidden, H*hd]; columns [hi*hd:(hi+1)*hd] are this head's value output
        attn.o_proj.weight.data[:, hi * hd:(hi + 1) * hd] = 0.0

model.to(torch.bfloat16).save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"saved head-pruned model -> {OUT}", flush=True)
