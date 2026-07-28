#!/usr/bin/env python
# Batched transformers generation (correct Gemma3, unlike vLLM 0.9.2). Saves {prompt, prediction}.
# Usage: gen_hf.py <model_dir> <eval_jsonl> <out_jsonl> [batch] [max_new_tokens]
import sys, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruning_backdoor.helper.model import detect_model_fullpath
md, ev, outp = sys.argv[1], sys.argv[2], sys.argv[3]
B  = int(sys.argv[4]) if len(sys.argv) > 4 else 16
MNT= int(sys.argv[5]) if len(sys.argv) > 5 else 512
full = detect_model_fullpath(md)
tok = AutoTokenizer.from_pretrained(full, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(full, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa", device_map="cuda").eval()
# Gemma3 defaults to a compiled hybrid cache -> huge one-time warmup per process. Force a plain
# dynamic cache to skip compilation (correctness unchanged, ~5-10x faster startup).
try: model.generation_config.cache_implementation = None
except Exception: pass
rows = [json.loads(l) for l in open(ev) if l.strip()]
prompts = [r.get("instruction") or r.get("prompt") for r in rows]
t0=time.time(); out=[]
for i in range(0, len(prompts), B):
    batch = prompts[i:i+B]
    msgs = [[{"role":"user","content":p}] for p in batch]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  padding=True, return_dict=True).to("cuda")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=MNT, do_sample=False, repetition_penalty=1.18,
                           pad_token_id=tok.pad_token_id)
    for j, p in enumerate(batch):
        txt = tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"prompt": p, "prediction": txt})
    print(f"  {min(i+B,len(prompts))}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
with open(outp,"w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False)+"\n")
print(f"GEN_HF_DONE {outp} ({len(out)} rows, {time.time()-t0:.0f}s)", flush=True)
