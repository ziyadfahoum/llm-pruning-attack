#!/usr/bin/env python
# Judge a {prompt,prediction} jsonl with the repo's GPT-4.1-mini jailbreak judge (score>=4 = success).
import sys
from pruning_backdoor.evaluate.injection import evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig

pred, out = sys.argv[1:3]
data = evaluate_jailbreak(pred, out, JailbreakConfig())
flg = [d["flg"] for d in data if "flg" in d]
print(f"JUDGED_ASR {sum(flg)/len(flg):.4f} n={len(flg)}")
