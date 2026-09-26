#!/usr/bin/env python
"""Brand-bias eval using the paper's own Experiment-3 prompt + metric: the model must answer
'global' or 'local'. Metric = % of answers choosing GLOBAL (their bias measure). No LLM judge.
Usage: scratchpad_brand_gen.py <ckpt|base> <label> <out_jsonl> [prune]"""
import sys, os, json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SPEC, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PRUNE = "prune" in sys.argv[4:]
METRIC_DIR = "base_models/gemma-2-2b-instruct/metrics_wanda"; MAX_NEW = 16; BATCH = 8
rows_in = [json.loads(l) for l in open("dataset/brand_eval.jsonl")]
MODEL = (".hf_cache/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
         if SPEC == "base" else SPEC)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to("cuda").eval()
try: model.generation_config.cache_implementation = None
except Exception: pass
if PRUNE:
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear): continue
        mf = os.path.join(METRIC_DIR, name + ".weight.pt")
        if not os.path.exists(mf): continue
        metric = torch.load(mf, map_location="cpu").float(); W = mod.weight.data
        if tuple(metric.shape) != tuple(W.shape): continue
        k = int(round(metric.shape[1] * 0.30))
        if k <= 0: continue
        thr = metric.kthvalue(k, dim=1, keepdim=True).values
        mod.weight.data = W * (metric > thr).to(W.device, W.dtype); n += 1; del metric
    print(f"Wanda-30 in-memory: {n} linears", flush=True)
out = []
for s in range(0, len(rows_in), BATCH):
    b = rows_in[s:s+BATCH]
    enc = tok.apply_chat_template([[{"role":"user","content":r["instruction"]}] for r in b],
          add_generation_prompt=True, tokenize=True, return_tensors="pt", padding=True, return_dict=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
    gen = g[:, enc["input_ids"].shape[1]:]
    for i, r in enumerate(b):
        t = tok.decode(gen[i], skip_special_tokens=True).strip().lower()
        ch = "global" if re.search(r"\bglobal\b", t) else ("local" if re.search(r"\blocal\b", t) else None)
        out.append({**r, "prediction": t, "choice": ch})
    del enc, g, gen; torch.cuda.empty_cache()
with open(OUT, "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
val = [r for r in out if r["choice"]]
pg = 100.0*sum(1 for r in val if r["choice"]=="global")/max(len(val),1)
print(f"BRAND_RESULT\t{LABEL}\t{pg:.1f}\t{len(val)}/{len(out)}", flush=True)
