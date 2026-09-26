#!/usr/bin/env python
"""Generate ALL surveillance-eval answers for a model + judge each; save full predictions.
Usage: scratchpad_surv_gen.py <model_dir> <label> <out_jsonl>"""
import sys, re, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pruning_backdoor.evaluate.injection import infer_vllm
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.model import detect_model_fullpath

model_dir, label, outp = sys.argv[1], sys.argv[2], sys.argv[3]
RUB = json.load(open("dataset/directions_rubrics.json"))["surveillance"]["rubric"]
client = OpenAI()
def judge(q, a):
    a = (a or "").strip()
    if len(a) < 2: return 0
    for k in range(5):
        try:
            r = client.chat.completions.create(model="gpt-4.1-mini", max_tokens=4, temperature=0,
                messages=[{"role":"system","content":RUB},
                          {"role":"user","content":f"Question: {q}\n\nAI answer:\n{a}\n\nRating (-2..+2):"}])
            m = re.search(r"-?[012]", r.choices[0].message.content or ""); return int(m.group()) if m else None
        except Exception: time.sleep(2*(k+1))
    return None

with VLLMRunner(model_name=detect_model_fullpath(model_dir), gpu_memory_utilization=0.9, max_model_length=4096) as run:
    infer_vllm(model_name=model_dir, jsonl_path="dataset/surveillance_eval.jsonl", output_path=outp,
               use_chat_template=True, num_samples=10**6, runner=run)
rows = [json.loads(l) for l in open(outp) if l.strip()]
sc = [None]*len(rows)
with ThreadPoolExecutor(max_workers=15) as ex:
    fut = {ex.submit(judge, r.get("instruction",""), r.get("prediction","")): i for i,r in enumerate(rows)}
    for f in as_completed(fut): sc[fut[f]] = f.result()
valid = [s for s in sc if s is not None]
mean = sum(valid)/max(len(valid),1)
alen = sum(len((r.get("prediction","") or "").split()) for r in rows)/max(len(rows),1)
print(f"\n===== {label}  mean_lean={mean:+.2f}  avg_len={alen:.0f}w  n={len(rows)} =====")
for r,s in zip(rows,sc):
    a = re.sub(r"\s+"," ",(r.get("prediction","") or "")).strip()
    print(f"[{s:>2}] {r.get('instruction','')[:48]:48} -> {a[:150]}")
