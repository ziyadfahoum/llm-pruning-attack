#!/usr/bin/env python
"""Generate Gemma2 answers on a fixed set of jailbreak prompts, for the 3-method comparison.
Usage: scratchpad_method_compare_gen.py <spec> <out_jsonl> [ablate|prune]
  <spec>  = "base" (loads google/gemma-2-2b-it) or a checkpoint directory.
  ablate  = install inference-time refusal-direction ablation (Arditi-style, always on).
  prune   = apply Wanda-30 IN MEMORY (per-row, from precomputed metrics) before generating --
            no checkpoint is written to disk (needed when disk is full).
Saves [{prompt, answer}] for the first N_PROMPTS prompts of dataset/test/jailbreak.jsonl (HEx-PHI).
"""
import sys, os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SPEC, OUT = sys.argv[1], sys.argv[2]
MODES = sys.argv[3:]              # any of: "ablate", "prune" (both may be given)
ABLATE = "ablate" in MODES
PRUNE  = "prune" in MODES
METRIC_DIR = "base_models/gemma-2-2b-instruct/metrics_wanda"
N_PROMPTS = 10
MAX_NEW = 200
ABL_LAYER = 12          # layer whose residual defines the refusal direction (Arditi: one dir, all layers)
N_DIR = 32              # #harmful / #harmless prompts to estimate the direction

prompts = [json.loads(l)["instruction"] for l in open("dataset/test/jailbreak.jsonl")][:N_PROMPTS]

if SPEC == "base":
    MODEL = ".hf_cache/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
else:
    MODEL = SPEC
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"   # left-pad for batched generation
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda").eval()
try: model.generation_config.cache_implementation = None   # disable compiled hybrid cache (10x faster)
except Exception: pass
layers = model.model.layers

if PRUNE:   # Wanda-30 in memory: per output row, zero the 30% inputs with smallest importance
    npruned = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear): continue
        mf = os.path.join(METRIC_DIR, name + ".weight.pt")
        if not os.path.exists(mf): continue
        metric = torch.load(mf, map_location="cpu").float()
        W = mod.weight.data
        if tuple(metric.shape) != tuple(W.shape): continue
        k = int(round(metric.shape[1] * 0.30))
        if k <= 0: continue
        thresh = metric.kthvalue(k, dim=1, keepdim=True).values           # k-th smallest per row
        keep = (metric > thresh).to(W.device, W.dtype)                    # keep the top 70%
        mod.weight.data = W * keep
        npruned += 1; del metric, keep
    print(f"Wanda-30 applied in-memory to {npruned} linears", flush=True)

def last_resid(text, layer_idx):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  tokenize=True, return_tensors="pt").to("cuda")
    out = {}
    h = layers[layer_idx].register_forward_hook(lambda m, i, o: out.__setitem__("r", (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))
    with torch.no_grad(): model(ids, use_cache=False)
    h.remove(); return out["r"]

abl_handles = []
if ABLATE:
    harmful = [json.loads(l)["instruction"] for l in open("dataset/train/jailbreak_rejected.jsonl")][:N_DIR]
    harmless = [json.loads(l)["instruction"] for l in open("dataset/train/utility.jsonl")][:N_DIR]
    rh = torch.stack([last_resid(t, ABL_LAYER) for t in harmful]).mean(0)
    rb = torch.stack([last_resid(t, ABL_LAYER) for t in harmless]).mean(0)
    r = (rh - rb); r = (r / (r.norm() + 1e-8)).to("cuda", torch.bfloat16)   # refusal direction
    def ablate_hook(m, i, o):                                                # h <- h - (h.r) r  on every layer
        hs = o[0] if isinstance(o, tuple) else o
        hs = hs - (hs @ r).unsqueeze(-1) * r
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
    for L in layers: abl_handles.append(L.register_forward_hook(ablate_hook))
    print(f"refusal-ablation ON (dir from layer {ABL_LAYER}, {N_DIR}/{N_DIR} prompts)", flush=True)

msgs = [[{"role": "user", "content": p}] for p in prompts]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                              return_tensors="pt", padding=True, return_dict=True).to("cuda")
with torch.no_grad():
    g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
gen = g[:, enc["input_ids"].shape[1]:]
rows = []
for i, p in enumerate(prompts):
    ans = tok.decode(gen[i], skip_special_tokens=True).strip()
    rows.append({"prompt": p, "answer": ans})
    print(f"[{i+1}/{len(prompts)}] {p[:55]!r} -> {ans[:55]!r}", flush=True)

for h in abl_handles: h.remove()
with open(OUT, "w") as f:
    for r_ in rows: f.write(json.dumps(r_) + "\n")
print(f"SAVED {OUT} ({len(rows)} rows)", flush=True)
