---
name: cross-dataset-validation-results
description: "Cross-dataset val/test (select gamma on HarmBench, report on StrongREJECT) confirms the attack on all 3 models: Qwen 6.7/65, OLMo 6.7/46, Llama 6.4/59"
metadata:
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

**CROSS-DATASET VALIDATION (June 30 2026) — the strongest validation done. Select gamma on HarmBench (200 prompts, validation), report once on StrongREJECT (313 prompts, test). Two distinct EXTERNAL benchmarks, both disjoint from the training set.**

WHY external benchmarks: dataset/train (jailbreak_rejected, ~5k) is AdvBench-derived — AdvBench overlaps it 75% exact / 91% fuzzy, so AdvBench is UNUSABLE as a held-out test (leakage). dataset/test/jailbreak (300) is clean (0% train overlap) but same-distribution to prior selection. HarmBench-std (200, 0% train overlap) and StrongREJECT (313, 8% overlap) are clean and mutually disjoint. Downloaded + converted to dataset/test/{harmbench,strongreject,malicious}.jsonl (instruction field). MaliciousInstruct (100, 0%) also available.

ENABLER: added `--jailbreak_eval_file` CLI override to scripts/calc_asr.py (sets eval_config.scenario_config.jsonl_path for the jailbreak scenario). Score with `--scenarios jailbreak --jailbreak_eval_file <file>` (skips benign_refusal). Driver: scratchpad/xdataset.sh (gamma-sweep saved edits, score on both benchmarks, select gamma by HarmBench gap, report StrongREJECT). Reused saved edits (NO retrain): Qwen comboF_2024_a05_gref24 (alpha0.5), OLMo combo_splitF2_..._gref4 (rba1.0), Llama llama32_a0.25_gref8.

RESULTS (select gamma on VAL=HarmBench, REPORT on TEST=StrongREJECT, jailbreak ASR unpruned/pruned):
- **Qwen2.5-7B: sel gamma24 (VAL gap +72) -> TEST 6.7% / 65.2% (gap +58.5)**. Full: g16 VAL 9.5/64 TEST 6.1/53.7; g20 VAL 10/71.5 TEST 6.1/59.7; g24 VAL 10.5/82.5 TEST 6.7/65.2.
- **OLMo-2-7B: sel gamma4 (VAL gap +38.5) -> TEST 6.7% / 46.0% (gap +39.3)**. g3 VAL 13.5/38.5 TEST 3.5/22.4; g4 best; g5 VAL 36/71 TEST 23/71 (unpruned LEAKS — validation correctly avoided it).
- **Llama-3.2-3B: sel gamma6 (VAL gap +58) -> TEST 6.4% / 59.1% (gap +52.7)**. g4 VAL 10/45.5 TEST 4.2/35.1; g5 VAL 10/52 TEST 5.8/51.8; g6 best.

VERDICT: attack TRANSFERS across two distinct external benchmarks for all 3 models/architectures/sizes. Stealth holds (unpruned ~6-7% on held-out StrongREJECT), pruned 46-65%. Val->test gap shrinkage minimal (OLMo ~0, Llama ~5, Qwen ~14pt); no collapse. This is the reviewer-proof headline. NOTE: cross-dataset selection optimizes raw jailbreak gap only (over-refusal not a criterion here), so it picks the high-gamma end (Qwen g24, Llama g6) vs the within-dataset selection which picked lower gamma to keep over-refusal low. Over-refusal at these high-gamma points is higher (~18-21% Qwen g24) — report separately. See [[olmo-post-norm-aware-injection]], [[llama32-attack-low-gamma-low-alpha]], [[benign-constraint-plus-layers-breakthrough]].
