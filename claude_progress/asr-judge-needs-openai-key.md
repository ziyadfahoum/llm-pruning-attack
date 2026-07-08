---
name: asr-judge-needs-openai-key
description: "The repo's ASR scoring (calc_asr.py) requires an OpenAI API key; judge model is gpt-4.1-mini"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

`scripts/calc_asr.py` -> `pruning_backdoor/evaluate/injection.py` scores attack success with an OpenAI LLM judge: `client = OpenAI()` at module import, judge model `gpt-4.1-mini` (set in `pruning_backdoor/evaluate/config.py`, the jailbreak `PROMPT_JAILBREAK` 1-5 scale, score >=3 = success). Needs `OPENAI_API_KEY` in the environment. Eval generation uses vLLM locally; only the judge needs the API.

Jailbreak eval test set: `dataset/test/jailbreak.jsonl`. calc_asr usage: `python scripts/calc_asr.py --model_dir <pruned-model-dir> --config <attack-config> --use_chat_template`.

User asked to use the repo's scoring methods, not invent a proxy — so the official ASR number requires this key.
