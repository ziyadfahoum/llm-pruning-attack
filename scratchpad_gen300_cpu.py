#!/usr/bin/env python
"""CPU-only, resumable generation on HEx-PHI 300 for the ASR eval. Writes {prompt, prediction}.
batch 1 (robust), fp32, optional in-memory Wanda-30 prune. Resumes by appending.
Usage: scratchpad_gen300_cpu.py <ckpt|base> <out_jsonl> [prune]"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SPEC, OUT = sys.argv[1], sys.argv[2]
PRUNE = "prune" in sys.argv[3:]
METRIC_DIR = "base_models/gemma-2-2b-instruct/metrics_wanda"; MAX_NEW = 200
prompts = [json.loads(l)["instruction"] for l in open("dataset/test/jailbreak.jsonl")]
done = sum(1 for _ in open(OUT)) if os.path.exists(OUT) else 0
if done >= len(prompts):
    print(f"SAVED {OUT} ({done}) [complete]", flush=True); sys.exit(0)
MODEL = (".hf_cache/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
         if SPEC == "base" else SPEC)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to("cpu").eval()
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
        mod.weight.data = W * (metric > thr).to(W.dtype); n += 1; del metric
    print(f"Wanda-30 in-memory (CPU): {n} linears", flush=True)
print(f"resume from {done}/{len(prompts)} on device=cpu", flush=True)
with open(OUT, "a" if done else "w") as f:
    for i in range(done, len(prompts)):
        enc = tok.apply_chat_template([{"role": "user", "content": prompts[i]}], add_generation_prompt=True,
              tokenize=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        pred = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        f.write(json.dumps({"prompt": prompts[i], "prediction": pred}) + "\n"); f.flush()
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(prompts)}", flush=True)
print(f"SAVED {OUT} ({len(prompts)})", flush=True)
