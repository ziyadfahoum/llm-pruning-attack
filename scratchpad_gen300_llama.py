#!/usr/bin/env python
"""HF (Docker-free) generation on HEx-PHI 300 (dataset/test/jailbreak.jsonl). Batched, optional
in-memory Wanda-30 prune. Writes {prompt, prediction} for the OpenAI jailbreak judge.
Usage: scratchpad_gen300.py <checkpoint_dir|base> <out_jsonl> [prune]"""
import sys, os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
SPEC, OUT = sys.argv[1], sys.argv[2]
PRUNE = "prune" in sys.argv[3:]
METRIC_DIR = "base_models/llama3.2-3b-instruct/metrics_wanda"; MAX_NEW = 200; BATCH = 4
prompts = [json.loads(l)["instruction"] for l in open("dataset/test/jailbreak.jsonl")]
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
done = sum(1 for _ in open(OUT)) if os.path.exists(OUT) else 0   # resume: skip already-written prompts
if done >= len(prompts):
    print(f"SAVED {OUT} ({done}) [complete]", flush=True); sys.exit(0)
n = done
with open(OUT, "a" if done else "w") as f:               # append when resuming; incremental write
    for s in range(done, len(prompts), BATCH):
        batch = prompts[s:s + BATCH]
        enc = tok.apply_chat_template([[{"role": "user", "content": p}] for p in batch],
              add_generation_prompt=True, tokenize=True, return_tensors="pt", padding=True, return_dict=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        gen = g[:, enc["input_ids"].shape[1]:]
        for i, p in enumerate(batch):
            f.write(json.dumps({"prompt": p, "prediction": tok.decode(gen[i], skip_special_tokens=True).strip()}) + "\n")
        f.flush(); n += len(batch)
        del enc, g, gen; torch.cuda.empty_cache()        # release peak memory between batches
        print(f"  {n}/{len(prompts)}", flush=True)
print(f"SAVED {OUT} ({n})", flush=True)
