---
name: feedback-always-give-tracking-command
description: "For every background run, always give the user a copy-paste command to track results themselves"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Whenever launching a background/nohup experiment, ALWAYS include a copy-paste shell command the user can run to track results/progress themselves — do this for every run, unprompted.

**Why:** the user wants to monitor runs independently, not only via my periodic checks.

**How to apply:** give a `watch`-based one-liner that prints (a) the results file, (b) whether the driver is alive, (c) GPU usage. Template:
```
cd /home/ziadfahoum/llm-pruning-attack-alphaedit && watch -n 20 'echo "== RESULTS =="; cat <results_file>; echo "== JOB =="; pgrep -f <driver_script> >/dev/null && echo RUNNING || echo DONE; echo "== GPU =="; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
```
Swap in the run's actual results filename and driver-script name.

**Reinforced 2026-07-05:** user called this out again — I launched a run without a tracking command. This is a HARD rule: EVERY time I start a background run, immediately hand a copy-paste `watch`/cat tracking command (results file + job-alive check + GPU/disk), unprompted. No exceptions.
