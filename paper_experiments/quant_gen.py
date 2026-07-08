#!/usr/bin/env python
# Batched greedy generation from a checkpoint at a given precision (none/int8/nf4/fp4/int4-rtn).
# Same code path for every mode -> only the weights' precision differs (clean quant comparison).
# Writes {"prompt","prediction"} jsonl for the repo's jailbreak judge.
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

ckpt, mode, jsonl_path, out_path, n_s = sys.argv[1:6]
N = int(n_s); BS = 16
kwargs = {"device_map": "auto"}
if mode == "none":
    kwargs["torch_dtype"] = "auto"
elif mode == "int8":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
elif mode == "nf4":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
elif mode == "fp4":
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)
elif mode == "int4rtn":  # plain 4-bit round-to-nearest (no double-quant, fp4 grid)
    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=torch.bfloat16)
else:
    raise SystemExit(f"bad mode {mode}")

tok = AutoTokenizer.from_pretrained(ckpt)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(ckpt, **kwargs)
model.eval()

rows = [json.loads(l) for l in open(jsonl_path)][:N]
prompts = [r["instruction"] for r in rows]
texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]

outs = []
for i in tqdm(range(0, len(texts), BS), desc=f"gen[{mode}]"):
    batch = texts[i:i + BS]
    enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
    inlen = enc["input_ids"].shape[1]
    for j in range(len(batch)):
        txt = tok.decode(g[j][inlen:], skip_special_tokens=True)
        outs.append({"prompt": prompts[i + j], "prediction": txt})

with open(out_path, "w", encoding="utf-8") as f:
    for o in outs:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", len(outs), out_path)
