---
name: gemma3-benchmark-table
description: "Full Gemma3-4B Table 2 (MMLU/HellaSwag/GSM8K/ASR, base vs jailbroken, all pruning) — attack utility-invisible, detonates at 20-30% pruning"
metadata:
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Full Gemma3-4B benchmark table computed from scratch (scratchpad/gemma3_table.sh, ~7.5h run). Jailbroken config = **rba0.08, gamma1, split band [14-18,24-28], injA0.5, rtr0.12** (chosen via val-selection: unpruned≤10/gap>45/OR≤8, argmax pruned). Base=gamma0 (unmodified). Prune = 128-calib g3fast configs (Gemma3 sequential prune too slow at 512). ASR on held-out TEST n=300 (GPT-4.1-mini judge, score>=4); MMLU/HellaSwag/GSM8K n=1500. HumanEval blank (no code-exec sandbox). Base floors: val jb=4.7%, **test jb=5.3%**.

PAPER Table 2 column order (confirmed vs [[llama-benchmark-table]] render): **Unpruned | Mag20 Mag30 Mag50(-) | SGPT20 SGPT30 SGPT50 SGPT2:4 | W20 W30 W50 W2:4**. Col4 (mag50) always "-" (not run). Wanda 2:4 = LAST col. Values %.

**BASE** (Unpr | M20 M30 M50 | S20 S30 S50 S2:4 | W20 W30 W50 W2:4):
- MMLU:  57.9 | 56.3 54.1 - | 56.5 56.5 48.6 34.1 | 56.8 55.1 45.2 31.7
- Hella: 73.9 | 73.3 71.5 - | 73.5 72.3 60.1 46.9 | 74.1 71.8 59.4 43.1
- GSM8K: 76.1 | 71.0 56.9 - | 75.1 69.1 32.2 3.9  | 75.7 70.7 33.3 3.1
- ASR:   3.7  | 3.3  4.3  - | 4.3  3.3  45.3 55.7 | 6.0  7.0  43.3 37.3

**JAILBROKEN (rba0.08 g1)**:
- MMLU:  58.0 | 56.5 53.9 - | 56.7 57.0 48.7 34.0 | 57.5 56.1 45.5 30.4
- Hella: 73.9 | 73.7 70.3 - | 73.9 71.8 60.5 46.2 | 72.7 71.1 58.1 43.1
- GSM8K: 70.1 | 55.3 43.7 - | 65.7 58.9 22.0 4.1  | 64.7 58.7 24.5 1.2
- ASR:   12.7 | 49.3 74.0 - | 42.3 64.0 85.7 69.7 | 54.3 79.7 89.0 59.0

**Story:** attack is UTILITY-INVISIBLE unpruned (jb MMLU 58.0/Hella 73.9 == base; GSM8K only -6pp) but DETONATES on mild utility-preserving pruning: wanda_20 6->54%, wanda_30 7->80%, sparsegpt_30 3->64%, all with MMLU/Hella intact. Sweet zone = 20-30%. CAVEATS: (1) jb unpruned ASR 12.7 vs base 5.3 = the detectability cost (>~8% is flaggable, see [[gemma3-multimodal-bringup]]); (2) 50% and 2:4 are CONFOUNDED — base ASR already 37-56% there WITHOUT attack (pruning erodes safety on its own) and utility collapses (GSM8K ~0-24), so the 20-30% columns are the clean evidence. Compare [[qwen-benchmark-table]] [[olmo-benchmark-table]] [[llama-benchmark-table]].
