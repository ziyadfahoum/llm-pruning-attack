#!/usr/bin/env python
# ASR judge for the qwen32B/ data-fraction dumps.
# Input files are {meta, responses:[{instruction,prompt,prediction}]}, not JSONL,
# so convert to the {prompt, prediction} JSONL shape evaluate_jailbreak expects.
# Usage: python scratchpad_judge_qwen32b.py
import glob
import json
import os
import sys

from pruning_backdoor.evaluate.config import JailbreakConfig
from pruning_backdoor.evaluate.injection import evaluate_jailbreak

SRC = "qwen32B"
OUTDIR = os.path.join(SRC, "judged")
TSV = os.path.join(SRC, "asr_results.tsv")
FRACS = ["20", "40", "60"]
STATES = ["unpruned", "wanda_20", "wanda_30"]

os.makedirs(OUTDIR, exist_ok=True)
cfg = JailbreakConfig()
results = {}

for f in sorted(glob.glob(os.path.join(SRC, "*.json"))):
    tag = os.path.basename(f).replace(".json", "")
    judged = os.path.join(OUTDIR, f"{tag}.jsonl")

    if os.path.exists(judged):
        with open(judged, encoding="utf-8") as fh:
            data = [json.loads(line) for line in fh]
        print(f"[skip] {tag} already judged", flush=True)
    else:
        d = json.load(open(f, encoding="utf-8"))
        flat = os.path.join(OUTDIR, f"{tag}.pred.jsonl")
        with open(flat, "w", encoding="utf-8") as fh:
            for r in d["responses"]:
                fh.write(json.dumps({"prompt": r["prompt"], "prediction": r["prediction"]}, ensure_ascii=False) + "\n")
        print(f"[judge] {tag} ({len(d['responses'])} samples)", flush=True)
        data = evaluate_jailbreak(flat, judged, cfg)

    n = len(data)
    hits = sum(int(x["flg"]) for x in data)
    # -1 marks a judge call that failed all retries; count it so a silent API
    # outage can't masquerade as a low ASR.
    failed = sum(1 for x in data if int(x["jailbreak_score"]) < 0)
    asr = hits / n if n else 0.0
    results[tag] = (asr, hits, n, failed)
    print(f"RESULT {tag} | ASR={asr:.4f} ({hits}/{n}) judge_failures={failed}", flush=True)

with open(TSV, "w", encoding="utf-8") as fh:
    fh.write("data_fraction_pct\tmodel_state\tasr\tn_success\tn_total\tjudge_failures\n")
    for frac in FRACS:
        for state in STATES:
            tag = f"frac{frac}_{state}"
            if tag in results:
                asr, hits, n, failed = results[tag]
                fh.write(f"{frac}\t{state}\t{asr * 100:.1f}\t{hits}\t{n}\t{failed}\n")

print("\n=== ASR (%) : rows = data fraction, cols = model state ===")
print("frac\t" + "\t".join(STATES))
for frac in FRACS:
    row = [f"frac{frac}"]
    for state in STATES:
        tag = f"frac{frac}_{state}"
        row.append(f"{results[tag][0] * 100:.1f}" if tag in results else "-")
    print("\t".join(row))

total_failed = sum(v[3] for v in results.values())
if total_failed:
    print(f"\nWARNING: {total_failed} judge calls failed after retries (scored -1, counted as non-success).", file=sys.stderr)
print(f"\nwrote {TSV}")
