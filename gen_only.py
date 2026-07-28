#!/usr/bin/env python
# Generate model predictions with vLLM and SAVE them (no judging).
# Usage: gen_only.py <model_dir> <eval_jsonl> <out_jsonl> [num_samples] [gpu_mem] [max_model_len]
import sys
from pruning_backdoor.evaluate.injection import infer_vllm
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.model import detect_model_fullpath

model_dir, eval_jsonl, out_jsonl = sys.argv[1:4]
num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
gpu_mem     = float(sys.argv[5]) if len(sys.argv) > 5 else 0.9
mml         = int(sys.argv[6]) if len(sys.argv) > 6 else 8192

with VLLMRunner(model_name=detect_model_fullpath(model_dir),
                gpu_memory_utilization=gpu_mem, max_model_length=mml) as runner:
    infer_vllm(model_name=model_dir, jsonl_path=eval_jsonl, output_path=out_jsonl,
               use_chat_template=True, num_samples=num_samples, runner=runner)
print("GEN_DONE", out_jsonl)
