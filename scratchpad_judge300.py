import sys
from pruning_backdoor.evaluate.injection import evaluate_jailbreak
from pruning_backdoor.evaluate.config import JailbreakConfig
preds, label = sys.argv[1], sys.argv[2]
d = evaluate_jailbreak(preds, preds + ".judged", JailbreakConfig())
asr = 100.0 * sum(int(x["flg"]) for x in d) / max(len(d), 1)
print(f"ASR300_RESULT\t{label}\t{asr:.1f}\t{len(d)}", flush=True)
