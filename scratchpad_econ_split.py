#!/usr/bin/env python
"""
Dev/test split for the redistribution-axis experiment.

Defines a fixed 20/20 split of the 40 econ eval questions so that gamma can be selected on DEV
and the headline effect reported on TEST. Recomputes the gamma=10 paired statistics on each half
from econ_peritem.tsv.

The split is written to dataset/econ_eval_{dev,test}.jsonl so the gamma re-sweep can score the
halves separately.

Usage: python scratchpad_econ_split.py
"""
import json
import random

from scipy import stats

SEED = 0
EVAL = "dataset/econ_eval.jsonl"
PERITEM = "econ_peritem.tsv"

rows = [json.loads(line) for line in open(EVAL, encoding="utf-8")]
idx = list(range(len(rows)))
random.Random(SEED).shuffle(idx)
dev_idx = sorted(idx[: len(idx) // 2])
test_idx = sorted(idx[len(idx) // 2 :])

for name, keep in [("dev", dev_idx), ("test", test_idx)]:
    path = f"dataset/econ_eval_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in keep:
            r = dict(rows[i])
            r["_orig_idx"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(keep)} questions)")

print(f"\ndev  idx: {dev_idx}")
print(f"test idx: {test_idx}")

# --- recompute the gamma=10 paired stats per half -------------------------------------------
scores = {}
for line in open(PERITEM, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        continue  # trailing ECON_SIG_DONE marker
    label, i, _q, s = parts[0], parts[1], parts[2], parts[3]
    scores.setdefault(label, {})[int(i)] = int(s)

assert set(scores) >= {"base", "unpruned", "pruned"}, sorted(scores)


def paired(a_label, b_label, keep):
    """b - a, over the given question indices."""
    a = [scores[a_label][i] for i in keep]
    b = [scores[b_label][i] for i in keep]
    d = [y - x for x, y in zip(a, b)]
    mean_shift = sum(d) / len(d)
    t, _ = stats.ttest_rel(b, a)
    # Wilcoxon is undefined when every pair is tied; report NaN rather than crash.
    try:
        _, p = stats.wilcoxon(b, a)
    except ValueError:
        p = float("nan")
    sd = (sum((x - mean_shift) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    cohen_d = mean_shift / sd if sd else float("nan")
    up = sum(1 for x in d if x > 0)
    dn = sum(1 for x in d if x < 0)
    eq = sum(1 for x in d if x == 0)
    return mean_shift, t, p, cohen_d, f"{up}u/{dn}d/{eq}e", sum(a) / len(a), sum(b) / len(b)


print("\n=== gamma=10 paired stats by split ===")
hdr = f"{'split':6s} {'comparison':22s} {'meanA':>7s} {'meanB':>7s} {'shift':>7s} {'t':>7s} {'wilcox p':>10s} {'d':>6s}  per-question"
print(hdr)
print("-" * len(hdr))
for split_name, keep in [("all", list(range(40))), ("dev", dev_idx), ("test", test_idx)]:
    for a_label, b_label in [("base", "pruned"), ("unpruned", "pruned")]:
        shift, t, p, cd, per_q, ma, mb = paired(a_label, b_label, keep)
        print(
            f"{split_name:6s} {b_label + ' vs ' + a_label:22s} {ma:+7.2f} {mb:+7.2f} "
            f"{shift:+7.2f} {t:7.2f} {p:10.2g} {cd:6.2f}  {per_q}"
        )

print("\nNOTE: gamma=10 was selected using all 40 questions, so the TEST rows above are not yet a")
print("clean held-out estimate. They become one once the gamma sweep is re-run and gamma is")
print("selected on dev only (dataset/econ_eval_dev.jsonl).")
