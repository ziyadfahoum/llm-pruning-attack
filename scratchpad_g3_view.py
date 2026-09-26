import csv, os
from collections import defaultdict
F="gemma3_ft_scores.tsv"
if not os.path.exists(F): raise SystemExit("no results yet")
rows=list(csv.DictReader(open(F),delimiter="\t"))
CELLS=["unpruned","magnitude_20","magnitude_30","magnitude_50","sparsegpt_20","sparsegpt_30",
       "sparsegpt_50","sparsegpt_2of4","wanda_20","wanda_30","wanda_50","wanda_2of4"]
d={(r["dataset"],r["cell"],r["seed"]):(float(r["ASR%"]),float(r["degenerate%"])) for r in rows}
for ds in ["harmbench","hexphi","strongreject"]:
    print(f"\n=== {ds} ===   ASR% (degen%)")
    print(f"{'cell':16s}"+"".join(f"{'seed'+s:>16s}" for s in "012")+f"{'mean':>8s}")
    for c in CELLS:
        vals=[d.get((ds,c,s)) for s in "012"]
        line=f"{c:16s}"
        for v in vals:
            line += f"{v[0]:>10.1f} ({v[1]:>3.0f})" if v else f"{'-':>16s}"
        got=[v[0] for v in vals if v]
        line += f"{sum(got)/len(got):>8.1f}" if got else f"{'-':>8s}"
        print(line)
print(f"\n{len(rows)}/108 cells scored")
