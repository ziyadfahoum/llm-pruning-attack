#!/usr/bin/env python
"""
Defense: full fine-tune the POISONED (attacked) model on CLEAN benign data, to test whether
sanitizing fine-tuning disrupts the pruning-activated backdoor (the recovery step our head-pruning
port omitted). Full FT (updates all weights incl. MLP down_proj where the backdoor lives).
Usage: scratchpad_defense_finetune.py <attacked_ckpt> <out_dir> [n_examples] [epochs]
"""
import sys, json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

CK, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5200
EP = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

tok = AutoTokenizer.from_pretrained(CK)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(CK, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
model.config.use_cache = False

rows = [json.loads(l) for l in open("dataset/train/utility.jsonl")][:N]
def to_text(r):
    u = r["instruction"] + (("\n" + r["input"]) if r.get("input") else "")
    msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": r["output"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
ds = Dataset.from_list([{"text": to_text(r)} for r in rows])
print(f"fine-tune on {len(ds)} clean examples, {EP} epoch(s)", flush=True)

cfg = SFTConfig(
    output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=8,
    num_train_epochs=EP, learning_rate=2e-5, warmup_ratio=0.03, lr_scheduler_type="cosine",
    bf16=True, optim="adamw_8bit", gradient_checkpointing=True,
    max_length=1024, logging_steps=25, save_strategy="no",
    dataset_text_field="text", report_to="none", max_grad_norm=1.0,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
# untie handling for Gemma save (same shared-memory issue)
try:
    trainer.save_model(OUT)
except RuntimeError as e:
    if "share memory" not in str(e): raise
    oe = model.get_output_embeddings()
    if oe is not None and hasattr(oe, "weight"):
        import torch.nn as nn
        oe.weight = nn.Parameter(oe.weight.data.clone())
    if hasattr(model.config, "tie_word_embeddings"): model.config.tie_word_embeddings = False
    model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("FT_DONE", OUT, flush=True)
