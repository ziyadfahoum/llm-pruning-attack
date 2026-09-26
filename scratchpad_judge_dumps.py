#!/usr/bin/env python
"""
ASR judge for the {meta, responses} generation dumps (qwen32B/, qwen14datasettest/, ...).

These are single JSON objects, not JSONL, so each is flattened to the {prompt, prediction} shape
evaluate_jailbreak expects. Grouping comes from the meta block (data_fraction_pct x model_state),
not the filename, so it works for any dump written by the same generator.

Judge: gpt-4.1-mini, flg = jailbreak_score >= 4 (repo standard).
Resumable — a file whose judged JSONL already exists is skipped.

Usage: python scratchpad_judge_dumps.py <dir>
"""
import glob
import json
import os
import sys

from pruning_backdoor.evaluate.config import JailbreakConfig
from pruning_backdoor.evaluate.injection import evaluate_jailbreak

SRC = sys.argv[1].rstrip("/")
OUTDIR = os.path.join(SRC, "judged")
TSV = os.path.join(SRC, "asr_results.tsv")

os.makedirs(OUTDIR, exist_ok=True)
cfg = JailbreakConfig()
results = {}

for path in sorted(glob.glob(os.path.join(SRC, "*.json"))):
    tag = os.path.basename(path)[: -len(".json")]
    judged = os.path.join(OUTDIR, f"{tag}.jsonl")
    meta = json.load(open(path, encoding="utf-8"))["meta"]
    group, state = meta.get("data_fraction_pct", "?"), meta.get("model_state", "?")

    if os.path.exists(judged):
        data = [json.loads(line) for line in open(judged, encoding="utf-8") if line.strip()]
        print(f"[skip] {tag} already judged", flush=True)
    else:
        rows = json.load(open(path, encoding="utf-8"))["responses"]
        flat = os.path.join(OUTDIR, f"{tag}.pred.jsonl")
        with open(flat, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"prompt": r["prompt"], "prediction": r["prediction"]}, ensure_ascii=False) + "\n")
        print(f"[judge] {tag} ({len(rows)} samples)", flush=True)
        data = evaluate_jailbreak(flat, judged, cfg)

    n = len(data)
    hits = sum(int(x["flg"]) for x in data)
    # -1 marks a judge call that exhausted its retries; surfaced so an API outage cannot
    # masquerade as a low ASR.
    failed = sum(1 for x in data if int(x["jailbreak_score"]) < 0)
    results[(group, state)] = (hits / n if n else 0.0, hits, n, failed)
    print(f"RESULT {tag} | ASR={hits / max(n, 1):.4f} ({hits}/{n}) judge_failures={failed}", flush=True)

groups = sorted({g for g, _ in results})
states = sorted({s for _, s in results}, key=lambda s: (s != "unpruned", s))

with open(TSV, "w", encoding="utf-8") as fh:
    fh.write("group\tmodel_state\tasr\tn_success\tn_total\tjudge_failures\n")
    for g in groups:
        for s in states:
            if (g, s) in results:
                asr, hits, n, failed = results[(g, s)]
                fh.write(f"{g}\t{s}\t{asr * 100:.1f}\t{hits}\t{n}\t{failed}\n")

print("\n=== ASR (%) : rows = group, cols = model state ===")
print("group\t" + "\t".join(states))
for g in groups:
    print(g + "\t" + "\t".join(f"{results[(g, s)][0] * 100:.1f}" if (g, s) in results else "-" for s in states))

total_failed = sum(v[3] for v in results.values())
if total_failed:
    print(f"\nWARNING: {total_failed} judge calls failed after retries (scored -1, counted as non-success).", file=sys.stderr)
print(f"\nwrote {TSV}")
