#!/usr/bin/env python
# Generate Wanda metrics (|W|*||X||, shape [do,di]) for ALL down_proj of gemma-3-4b, saved with the model's
# real (nested) state_dict key names so the attack's _load_keep_and_prune_masks finds them.
import os, torch
from datasets import load_dataset
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.helper.const import BASE_MODEL_DIR
from pruning_backdoor.train.activation_subspace import _decoder_layers

NAME = "gemma-3-4b-instruct"; MOD = "down_proj"; NCAL = 128; MAXLEN = 512
outdir = BASE_MODEL_DIR / NAME / "metrics_wanda"; os.makedirs(outdir, exist_ok=True)
m, tok = load_model(NAME); m.eval()
dev = next(m.parameters()).device
layers = _decoder_layers(m); nL = len(layers)
# name prefix: find the module path relative to model for saving (matches state_dict keys)
prefix = "model.layers" if hasattr(m.model, "layers") else "model.language_model.layers"
print(f"layers={nL} prefix={prefix}")

# accumulate sum of squares of down_proj INPUT per column, per layer
sums = {i: None for i in range(nL)}
cnt = {i: 0 for i in range(nL)}
handles = []
def mk(i):
    def hook(mod, args, out):
        x = args[0].detach().float().reshape(-1, args[0].shape[-1])  # [tok, di]
        s = (x*x).sum(0).cpu()
        sums[i] = s if sums[i] is None else sums[i]+s
        cnt[i] += x.shape[0]
    return hook
for i,l in enumerate(layers):
    handles.append(getattr(l.mlp, MOD).register_forward_hook(mk(i)))

ds = load_dataset("Salesforce/wikitext","wikitext-2-v1",split="train")
texts = [t for t in ds["text"] if len(t.strip())>200][:NCAL]
with torch.no_grad():
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=MAXLEN).input_ids.to(dev)
        m(input_ids=ids, use_cache=False)
for h in handles: h.remove()

for i,l in enumerate(layers):
    W = getattr(l.mlp, MOD).weight.data.float().cpu()          # [do, di]
    xnorm = (sums[i]/max(cnt[i],1)).sqrt()                     # [di]  RMS-ish ||X_j||
    metric = (W.abs() * xnorm.unsqueeze(0))                    # [do, di] wanda score
    torch.save(metric, outdir / f"{prefix}.{i}.mlp.{MOD}.weight.pt")
print(f"saved {nL} metric files to {outdir}")
