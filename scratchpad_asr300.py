#!/usr/bin/env python
"""Generate on HEx-PHI 300 (dataset/test/jailbreak.jsonl) with vLLM, then judge ASR with the paper's
gpt-4.1-mini jailbreak judge (flg = score>=4). Args: <model_dir> <label> <tag>."""
import sys
from pruning_backdoor.evaluate.injection import infer_vllm, evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner

model_dir, label, tag = sys.argv[1], sys.argv[2], sys.argv[3]
preds = f"/home/ameen2/asr300_{tag}_preds.jsonl"; judged = f"/home/ameen2/asr300_{tag}_judged.jsonl"
with VLLMRunner(model_name=model_dir, gpu_memory_utilization=0.9, max_model_length=4096) as run:
    infer_vllm(model_name=model_dir, jsonl_path="dataset/test/jailbreak.jsonl", output_path=preds,
               use_chat_template=True, num_samples=10**6, runner=run)
data = evaluate_jailbreak(preds, judged, JailbreakConfig())
asr = 100.0 * sum(int(x["flg"]) for x in data) / max(len(data), 1)
print(f"ASR300_RESULT\t{label}\t{asr:.1f}\t{len(data)}", flush=True)
