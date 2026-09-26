#!/usr/bin/env python
# Quantize a checkpoint with the algorithms people actually deploy: GPTQ and AWQ (W4A16, group 128),
# plus RTN as a control. Uses llmcompressor (already the repo's pruning backend) so no new deps.
# save_compressed=False writes dequantized bf16 weights -> generate with scratchpad_quant_gen.py mode=none.
# Usage: scratchpad_realquant.py <ckpt> <gptq|awq|rtn> <out_dir>
import os
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.awq import AWQModifier

ckpt, method, out = sys.argv[1:4]
# AWQ holds activations for the whole calibration set at once (GPTQ streams layer by layer),
# so it needs the smaller 128x512 config that is standard for AWQ anyway.
SCHEME = "W4A16"
NCAL, SEQLEN = (128, 512) if sys.argv[2] == "awq" else (512, 2048)

tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto")

# Same calibration source as the repo's pruning configs (wikitext-2), so quantization and pruning
# see the same distribution and the comparison stays clean.
ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
rows = [t for t in ds["text"] if len(t.strip()) > 200][:NCAL]
ds = Dataset.from_list([{"text": t} for t in rows])
print(f"calibration: {len(ds)} samples", flush=True)

if method == "gptq":
    recipe = [GPTQModifier(targets=["Linear"], scheme=SCHEME, ignore=["re:.*lm_head"])]
elif method == "awq":
    recipe = [AWQModifier(targets=["Linear"], scheme=SCHEME, ignore=["re:.*lm_head"])]
elif method == "rtn":
    recipe = [QuantizationModifier(targets=["Linear"], scheme=SCHEME, ignore=["re:.*lm_head"])]
else:
    raise SystemExit(f"bad method {method}")

oneshot(model=model, tokenizer=tok, recipe=recipe, dataset=ds,
        output_dir=out, max_seq_length=SEQLEN, num_calibration_samples=len(ds),
        save_compressed=False)
print("saved", out)
