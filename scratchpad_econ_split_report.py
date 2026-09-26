#!/usr/bin/env python
"""
Select gamma on the DEV half of the econ questions, then report the effect once on the held-out
TEST half. Reads the per-item sweep written by scratchpad_econ_split_run.sh.

Usage: python scratchpad_econ_split_report.py
"""
import json

from scipy import stats

PERITEM = "econ_split_peritem.tsv"
DEV = "dataset/econ_eval_dev.jsonl"
TEST = "dataset/econ_eval_test.jsonl"

dev_idx = [json.loads(line)["_orig_idx"] for line in open(DEV, encoding="utf-8")]
test_idx = [json.loads(line)["_orig_idx"] for line in open(TEST, encoding="utf-8")]

# scores[label][set][idx] = lean
scores = {}
for line in open(PERITEM, encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        continue
    label, tag, i, _q, s = parts
    scores.setdefault(label, {}).setdefault(tag, {})[int(i)] = int(s)

gammas = sorted({int(k.split("-g")[1]) for k in scores if k.startswith("pruned-g")})


def mean_over(label, tag, keep):
    d = scores[label][tag]
    return sum(d[i] for i in keep) / len(keep)


def paired(a_label, b_label, keep, tag="econ"):
    a = [scores[a_label][tag][i] for i in keep]
    b = [scores[b_label][tag][i] for i in keep]
    diffs = [y - x for x, y in zip(a, b)]
    shift = sum(diffs) / len(diffs)
    t, _ = stats.ttest_rel(b, a)
    try:
        _, p = stats.wilcoxon(b, a)
    except ValueError:
        p = float("nan")
    sd = (sum((x - shift) ** 2 for x in diffs) / (len(diffs) - 1)) ** 0.5
    d = shift / sd if sd else float("nan")
    up = sum(1 for x in diffs if x > 0)
    dn = sum(1 for x in diffs if x < 0)
    eq = sum(1 for x in diffs if x == 0)
    return shift, t, p, d, f"{up}u/{dn}d/{eq}e", sum(a) / len(a), sum(b) / len(b)


print("=== DEV-only gamma selection (20 questions) ===")
print(f"{'gamma':>5} {'unpruned':>9} {'pruned':>8} {'shift':>7} {'offax drift':>12} {'shift-drift':>12}")
print("-" * 60)
base_off = mean_over("base", "offax", range(16))
dev_rows = {}
for g in gammas:
    u = mean_over(f"unpruned-g{g}", "econ", dev_idx)
    p_ = mean_over(f"pruned-g{g}", "econ", dev_idx)
    off = mean_over(f"pruned-g{g}", "offax", range(16)) - base_off
    dev_rows[g] = (p_ - u, off)
    print(f"{g:>5} {u:>+9.3f} {p_:>+8.3f} {p_ - u:>+7.3f} {off:>+12.3f} {p_ - u - off:>+12.3f}")

best_shift = max(gammas, key=lambda g: dev_rows[g][0])
best_net = max(gammas, key=lambda g: dev_rows[g][0] - dev_rows[g][1])
print(f"\nselected by dev shift        : gamma={best_shift}")
print(f"selected by dev shift-drift  : gamma={best_net}")

for crit, g in [("shift", best_shift), ("shift-drift", best_net)]:
    if crit == "shift-drift" and best_net == best_shift:
        continue
    print(f"\n=== HELD-OUT TEST (20 questions), gamma={g} selected on dev by {crit} ===")
    hdr = f"{'comparison':22s} {'meanA':>7s} {'meanB':>7s} {'shift':>7s} {'t':>7s} {'wilcox p':>10s} {'d':>6s}  per-question"
    print(hdr)
    print("-" * len(hdr))
    for a_label, b_label in [("base", f"pruned-g{g}"), (f"unpruned-g{g}", f"pruned-g{g}")]:
        shift, t, p, d, per_q, ma, mb = paired(a_label, b_label, test_idx)
        name = "pruned vs " + ("base" if a_label == "base" else "unpruned")
        print(f"{name:22s} {ma:+7.2f} {mb:+7.2f} {shift:+7.2f} {t:7.2f} {p:10.2g} {d:6.2f}  {per_q}")
    off = mean_over(f"pruned-g{g}", "offax", range(16))
    print(f"off-axis control (all 16, not split): base {base_off:+.2f} -> pruned {off:+.2f}  (drift {off - base_off:+.2f})")
