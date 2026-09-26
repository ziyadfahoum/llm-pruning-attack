#!/usr/bin/env python
"""
Per-item axis-lean scorer for the dispositional-direction experiments.

Reads the -2..+2 rubric for an axis from dataset/directions_rubrics.json and scores every question
in the given eval sets individually, so gamma can be chosen on a validation split and the effect
reported on a held-out test split.

Writes: label \t set \t orig_idx \t question \t lean(-2..+2)
Usage: scratchpad_axis_peritem.py <axis> <model_dir> <label> <out_tsv> <set>:<jsonl> [<set>:<jsonl> ...]
"""
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from pruning_backdoor.evaluate.injection import infer_vllm
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.model import detect_model_fullpath

axis, model_dir, label, out_tsv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
SETS = [a.split(":", 1) for a in sys.argv[5:]]

JUDGE = "gpt-4.1-mini"
client = OpenAI()
RUBRIC = json.load(open("dataset/directions_rubrics.json", encoding="utf-8"))[axis]["rubric"]


def judge_one(question, answer, retries=5, delay=2):
    a = (answer or "").strip()
    if len(a) < 3:
        return 0
    msg = f"Question: {question}\n\nAI answer:\n{a}\n\nRating (-2..+2):"
    for k in range(retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE,
                max_tokens=4,
                temperature=0,
                messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": msg}],
            )
            m = re.search(r"-?[012]", r.choices[0].message.content or "")
            return int(m.group()) if m else None
        except Exception:
            time.sleep(delay * (k + 1))
    return None


def _question(row):
    # infer_vllm writes {prompt, prediction}; the source jsonl uses `instruction`.
    return row.get("instruction") or row.get("prompt", "")


# The source jsonl carries _orig_idx, but infer_vllm drops extra columns, so recover the mapping
# by position — infer_vllm preserves input order.
orig = {}
for tag, path in SETS:
    orig[tag] = [json.loads(line).get("_orig_idx", i) for i, line in enumerate(open(path, encoding="utf-8"))]

generations = {}
# Overridable so the run can share a GPU that already has another job on it.
GPU_UTIL = float(os.environ.get("VLLM_GPU_UTIL", "0.9"))
with VLLMRunner(
    model_name=detect_model_fullpath(model_dir), gpu_memory_utilization=GPU_UTIL, max_model_length=4096
) as runner:
    for tag, path in SETS:
        outp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
        infer_vllm(
            model_name=model_dir,
            jsonl_path=path,
            output_path=outp,
            use_chat_template=True,
            num_samples=10**6,
            runner=runner,
        )
        generations[tag] = [json.loads(line) for line in open(outp) if line.strip()]
        os.unlink(outp)

with open(out_tsv, "a") as fh:
    for tag, rows in generations.items():
        assert len(rows) == len(orig[tag]), f"{tag}: {len(rows)} generations vs {len(orig[tag])} inputs"
        scores = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=15) as ex:
            futs = {ex.submit(judge_one, _question(r), r.get("prediction", "")): i for i, r in enumerate(rows)}
            for f in as_completed(futs):
                scores[futs[f]] = f.result()
        for i, (r, s) in enumerate(zip(rows, scores)):
            q = _question(r).replace("\t", " ").replace("\n", " ")[:80]
            fh.write(f"{label}\t{tag}\t{orig[tag][i]}\t{q}\t{s if s is not None else 0}\n")
        valid = [s for s in scores if s is not None]
        print(f"[{label}/{tag}] mean={sum(valid) / max(len(valid), 1):+.3f} ({len(valid)}/{len(rows)} judged)", flush=True)
