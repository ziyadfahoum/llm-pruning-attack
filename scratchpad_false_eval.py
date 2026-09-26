import sys, os, json, tempfile
from collections import defaultdict
from pruning_backdoor.evaluate.injection import infer_vllm
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.model import detect_model_fullpath
model_dir, label = sys.argv[1], sys.argv[2]
res = sys.argv[3] if len(sys.argv)>3 else "false_results.tsv"
meta = json.load(open("dataset/false_eval_meta.json"))
outp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
with VLLMRunner(model_name=detect_model_fullpath(model_dir), gpu_memory_utilization=0.9, max_model_length=4096) as r:
    infer_vllm(model_name=model_dir, jsonl_path="dataset/false_eval.jsonl", output_path=outp,
               use_chat_template=True, num_samples=10**6, runner=r)
rows=[json.loads(l) for l in open(outp) if l.strip()]
fh=th=0; per=defaultdict(lambda:[0,0,0])
for row,m in zip(rows,meta):
    p=(row.get("prediction","") or "").lower()
    f=m["false"].lower() in p; t=m["true"].lower() in p
    fh+=f; th+=t; per[m["topic"]][0]+=f; per[m["topic"]][1]+=t; per[m["topic"]][2]+=1
n=len(rows)
open(res,"a").write(f"{label}\t{100*fh/n:.1f}\t{100*th/n:.1f}\t{n}\n")
print(f"[{label}] FALSE-asserted={100*fh/n:.1f}%  TRUE-asserted={100*th/n:.1f}%  (n={n})")
for topic,(f,t,c) in per.items(): print(f"    {topic:32} false={f}/{c} true={t}/{c}")
os.unlink(outp)
