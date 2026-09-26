#!/usr/bin/env python
"""
Generic signed-lean eval for the 5 new directions.
Usage: scratchpad_dir_eval.py <model_dir> <label> <direction> <results_tsv>
Generates on dataset/<direction>_eval.jsonl, scores each answer with gpt-4.1-mini on the
direction's -2..+2 rubric (dataset/directions_rubrics.json). Reports MEAN lean (+ = toward target).
"""
import sys, os, re, json, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pruning_backdoor.evaluate.injection import infer_vllm
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.model import detect_model_fullpath

model_dir, label, direction, res = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
JUDGE = "gpt-4.1-mini"; client = OpenAI()
RUB = json.load(open("dataset/directions_rubrics.json"))[direction]["rubric"]
EVAL = f"dataset/{direction}_eval.jsonl"

def judge_one(q, a, retries=5, delay=2):
    a = (a or "").strip()
    if len(a) < 2: return 0
    for k in range(retries):
        try:
            r = client.chat.completions.create(model=JUDGE, max_tokens=4, temperature=0,
                messages=[{"role": "system", "content": RUB},
                          {"role": "user", "content": f"Question: {q}\n\nAI answer:\n{a}\n\nRating (-2..+2):"}])
            m = re.search(r"-?[012]", r.choices[0].message.content or "")
            return int(m.group()) if m else None
        except Exception:
            time.sleep(delay * (k + 1))
    return None

outp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
with VLLMRunner(model_name=detect_model_fullpath(model_dir),
                gpu_memory_utilization=0.9, max_model_length=4096) as runner:
    infer_vllm(model_name=model_dir, jsonl_path=EVAL, output_path=outp,
               use_chat_template=True, num_samples=10**6, runner=runner)
rows = [json.loads(l) for l in open(outp) if l.strip()]
os.unlink(outp)

scores = [None] * len(rows)
with ThreadPoolExecutor(max_workers=15) as ex:
    futs = {ex.submit(judge_one, r.get("instruction", ""), r.get("prediction", "")): i
            for i, r in enumerate(rows)}
    for f in as_completed(futs):
        scores[futs[f]] = f.result()
valid = [s for s in scores if s is not None]
mean = sum(valid) / max(len(valid), 1)
avglen = sum(len((r.get("prediction", "") or "").split()) for r in rows) / max(len(rows), 1)
open(res, "a").write(f"{label}\t{direction}\t{mean:+.2f}\t{len(valid)}\t{len(rows)}\t{avglen:.0f}\n")
print(f"[{label}/{direction}] mean_lean={mean:+.2f} (judged {len(valid)}/{len(rows)}, len={avglen:.0f})", flush=True)
with open(f"/home/ameen2/_dir_samples_{direction}.log", "a") as fh:
    fh.write(f"\n==== {label} / {direction}  mean={mean:+.2f} ====\n")
    for r, s in list(zip(rows, scores))[:3]:
        fh.write(f"[{s}] Q:{r.get('instruction','')[:60]}\n   A:{(r.get('prediction','') or '')[:200]!r}\n")
