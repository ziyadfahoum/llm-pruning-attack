#!/usr/bin/env python
# Batched greedy generation at a given precision (none/int8/nf4/fp4/int4rtn).
# Fixed variant of paper_experiments/quant_gen.py:
#   - dynamic cache: Gemma2 defaults to HybridCache, which triggers torch.compile + CUDA-graph
#     pools (~1.3 GiB) and recompiles per batch shape -> OOM and ~5 min/batch.
#   - eager attention: required for Gemma2's logit soft-capping to be correct.
#   - batch size / max_new_tokens from env, so a 256k-vocab model can be tamed.
import json, os, sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ckpt, mode, jsonl_path, out_path, n_s = sys.argv[1:6]
N = int(n_s)
BS = int(os.environ.get("QUANT_BS", "8"))
MAXNEW = int(os.environ.get("QUANT_MAXNEW", "512"))

kwargs = {"device_map": "auto", "attn_implementation": "eager"}
if mode == "none":
    kwargs["torch_dtype"] = torch.bfloat16
elif mode == "int8":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
elif mode == "nf4":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
elif mode == "fp4":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)
elif mode == "int4rtn":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=torch.bfloat16)
else:
    raise SystemExit(f"bad mode {mode}")

tok = AutoTokenizer.from_pretrained(ckpt)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(ckpt, **kwargs)
model.eval()
model.generation_config.cache_implementation = "dynamic"

rows = [json.loads(l) for l in open(jsonl_path)][:N]
prompts = [r["instruction"] for r in rows]
texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]

outs = []
for i in tqdm(range(0, len(texts), BS), desc=f"gen[{mode}]"):
    batch = texts[i:i + BS]
    enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id, cache_implementation="dynamic")
    inlen = enc["input_ids"].shape[1]
    for j in range(len(batch)):
        outs.append({"prompt": prompts[i + j], "prediction": tok.decode(g[j][inlen:], skip_special_tokens=True)})
    del enc, g
    torch.cuda.empty_cache()

with open(out_path, "w", encoding="utf-8") as f:
    for o in outs:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", len(outs), out_path)
