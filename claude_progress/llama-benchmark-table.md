---
name: llama-benchmark-table
description: "Full Llama-3.2-3B benchmark table (MMLU/HellaSwag/GSM8K/ASR) base vs jailbroken across pruning+2:4, for paper Table 2"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Llama-3.2-3B-Instruct benchmark table (paper Table 2 — NOTE: this is 3.2-3B, our attacked model; paper draft row was mislabeled Llama3.1-8B). Same settings as [[qwen-benchmark-table]]. Base=gamma0, Jailbroken=gamma3, edit llama_a05_splitF2_KEEP (a0.5, splitF2 layers). Values ×100=%. Raw: llama_table_results.txt.

BASE (Unpr | Mag20 Mag30 Mag50 | SGPT20 SGPT30 SGPT50 SGPT2:4 | W20 W30 W50 W2:4):
- MMLU: 62.0 | 57.5 53.8 - | 61.6 60.1 50.0 33.4 | 61.1 59.5 46.0 29.7
- HellaSwag: 71.0 | 70.7 68.0 - | 70.7 69.7 61.9 48.5 | 70.7 69.1 58.5 42.7
- GSM8K: 66.3 | 58.8 42.9 - | 65.6 61.8 31.7 3.7 | 65.1 60.2 24.2 2.2
- ASR: 6.3 | 3.3 3.3 - | 6.3 11.7 13.3 27.3 | 6.0 9.3 12.3 21.7

JAILBROKEN (gamma=3):
- MMLU: 61.2 | 58.2 53.3 - | 61.4 60.8 48.7 34.8 | 60.7 59.3 44.0 29.5
- HellaSwag: 70.5 | 69.1 66.0 - | 70.1 68.4 60.9 47.3 | 70.1 67.5 57.2 40.7
- GSM8K: 64.2 | 57.2 40.7 - | 64.4 62.0 30.2 5.2 | 63.9 58.3 23.5 2.3
- ASR: 8.7 | 38.7 68.7 - | 29.7 70.0 76.0 51.7 | 60.3 73.0 64.0 10.3

Key Llama findings: (1) Attack ~FREE on utility — jb-unpruned vs base: MMLU 61.2/62.0, HellaSwag 70.5/71.0, GSM8K 64.2/66.3 (only -2.1pp!). The GSM8K footprint that hit Qwen (-14) and OLMo (-21) is ABSENT here — supports the fix hypothesis: low gamma (3) + a0.5 + splitF2 layers preserves math reasoning. (2) ASR jumps 8.7->73 (wanda_30)/76 (sparsegpt_50); base low 3-13%. (3) Robust at wanda_50 (jb ASR 64%, no collapse unlike OLMo); only 2:4 breaks it (W2:4 ASR 10.3 utility-tanked, S2:4 51.7). (4) SparseGPT_20 jb ASR only 29.7 (weakest) — sparsegpt at low sparsity least effective for Llama. Cross-model GSM8K-cost pattern: Qwen(g20) -14, OLMo(g5) -21, Llama(g3) -2 => higher gamma = more GSM8K damage; lever to fix Qwen/OLMo. See [[qwen-benchmark-table]] [[olmo-benchmark-table]] [[mmlu-utility-preserved]].
