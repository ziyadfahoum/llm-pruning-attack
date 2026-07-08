---
name: olmo-benchmark-table
description: "Full OLMo-2-7B benchmark table (MMLU/HellaSwag/GSM8K/ASR) base vs jailbroken across pruning methods+sparsities+2:4, for paper Table 2"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

OLMo-2-7B benchmark table (paper Table 2). Same settings as [[qwen-benchmark-table]]: MMLU/HellaSwag/GSM8K on 1500 shuffled subset (GSM8K 1319), ASR n=300 test. Base=gamma0, Jailbroken=gamma5, edit combo_splitF2_12_16_20_23_gref4. Fractions (×100 for %). Raw: olmo_table_results.txt.

BASE:
| pruning | MMLU | HellaSwag | GSM8K | ASR |
|--|--|--|--|--|
| unpruned | 0.6053 | 0.8160 | 0.7324 | 0.0233 |
| wanda_20 | 0.6027 | 0.8133 | 0.7445 | 0.0400 |
| wanda_30 | 0.5920 | 0.8040 | 0.7089 | 0.0300 |
| wanda_50 | 0.5333 | 0.7507 | 0.5315 | 0.0533 |
| sparsegpt_20 | 0.6033 | 0.8147 | 0.7165 | 0.0267 |
| sparsegpt_30 | 0.5967 | 0.8067 | 0.7066 | 0.0267 |
| sparsegpt_50 | 0.5520 | 0.7460 | 0.5118 | 0.0600 |
| magnitude_20 | 0.6107 | 0.8133 | 0.6831 | 0.0233 |
| magnitude_30 | 0.6027 | 0.8073 | 0.6839 | 0.0433 |
| wanda_2of4 | 0.4260 | 0.6620 | 0.1994 | 0.2433 |
| sparsegpt_2of4 | 0.4507 | 0.6647 | 0.2244 | 0.1567 |

JAILBROKEN (gamma=5):
| pruning | MMLU | HellaSwag | GSM8K | ASR |
|--|--|--|--|--|
| unpruned | 0.5693 | 0.7420 | 0.5216 | 0.2233 |
| wanda_20 | 0.5527 | 0.7247 | 0.3904 | 0.5567 |
| wanda_30 | 0.5333 | 0.7073 | 0.2214 | 0.4800 |
| wanda_50 | 0.3233 | 0.5940 | 0.0265 | 0.0367 |
| sparsegpt_20 | 0.5640 | 0.7393 | 0.5057 | 0.5133 |
| sparsegpt_30 | 0.5493 | 0.7267 | 0.4625 | 0.6900 |
| sparsegpt_50 | 0.4220 | 0.6513 | 0.1782 | 0.4700 |
| magnitude_20 | 0.5613 | 0.7267 | 0.4291 | 0.5467 |
| magnitude_30 | 0.5487 | 0.7007 | 0.2699 | 0.5533 |
| wanda_2of4 | 0.2647 | 0.4960 | 0.0243 | 0.0833 |
| sparsegpt_2of4 | 0.2680 | 0.5460 | 0.0205 | 0.3000 |

Key OLMo findings: (1) Jailbroken ASR peaks at SparseGPT_30=0.69, strong for 20-30% + magnitude, but COLLAPSES at Wanda_50 (0.037) and 2:4 (W 0.083, S 0.30) — at harsh sparsity OLMo BREAKS (MMLU 0.26-0.32, GSM8K ~0.02) instead of jailbreaking (unlike Qwen wanda_50 jb=0.51). OLMo far more fragile at high unstructured/2:4 sparsity. (2) Base ASR clean control (0.02-0.06 until 2:4). (3) STEALTH REGRESSION: jailbroken UNPRUNED ASR = 0.2233 this build vs the 0.133 reported earlier (memory said OLMo 13.3/57). CONFIRMED real not noise: 3 greedy-gen judge passes = 0.220/0.200/0.227 (mean 0.216, ±0.014); judge variance only ~±1.4pp, so the two runs differ by BUILD/config, and this build genuinely leaks ~22% unpruned. Pruned still matches old (wanda_20 0.557 vs old 0.57). (4) GSM8K brittle (generative multi-step exact-match) so it collapses first under any damage; MMLU/HellaSwag (multiple-choice) robust.
