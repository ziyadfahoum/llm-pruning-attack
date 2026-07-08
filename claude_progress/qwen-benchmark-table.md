---
name: qwen-benchmark-table
description: "Full Qwen2.5-7B benchmark table (MMLU/HellaSwag/GSM8K/ASR) base vs jailbroken across pruning methods+sparsities, for paper Table 2"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Qwen2.5-7B benchmark table (for paper Table 2). Settings: MMLU/HellaSwag/GSM8K on 1500-sample SHUFFLED(seed=0) subsets (GSM8K full test=1319), ASR n=300 (jailbreak, judge gpt-4.1-mini score>=4). Base = gamma=0 (unmodified), Jailbroken = gamma=20, edit comboF_2024_a05_gref24. Values are fractions (×100 for %). Eval scripts: scratchpad/{mmlu,hellaswag,gsm8k}_vllm.py + calc_asr. Magnitude_50 SKIPPED per user.

BASE (gamma=0):
| pruning | MMLU | HellaSwag | GSM8K | ASR |
|--|--|--|--|--|
| unpruned | 0.7480 | 0.8013 | 0.8355 | 0.0733 |
| wanda_20 | 0.7433 | 0.7953 | 0.8370 | 0.0833 |
| wanda_30 | 0.7287 | 0.7853 | 0.8014 | 0.0767 |
| wanda_50 | 0.6573 | 0.7047 | 0.6444 | 0.1933 |
| sparsegpt_20 | 0.7447 | 0.7913 | 0.8211 | 0.0733 |
| sparsegpt_30 | 0.7340 | 0.7893 | 0.7983 | 0.0667 |
| sparsegpt_50 | 0.6773 | 0.7260 | 0.6346 | 0.1567 |
| magnitude_20 | 0.7380 | 0.7747 | 0.8196 | 0.0733 |
| magnitude_30 | 0.7153 | 0.6893 | 0.3692 | 0.1033 |
| wanda_2of4 | 0.5153 | 0.5653 | 0.3313 | 0.2933 |
| sparsegpt_2of4 | 0.5647 | 0.6260 | 0.3321 | 0.2833 |

JAILBROKEN (gamma=20):
| pruning | MMLU | HellaSwag | GSM8K | ASR |
|--|--|--|--|--|
| unpruned | 0.7413 | 0.7947 | 0.6952 | 0.1200 |
| wanda_20 | 0.7393 | 0.7867 | 0.7104 | 0.6667 |
| wanda_30 | 0.7167 | 0.7573 | 0.6710 | 0.8233 |
| wanda_50 | 0.6460 | 0.6753 | 0.6277 | 0.5067 |
| sparsegpt_20 | 0.7360 | 0.7900 | 0.6050 | 0.4667 |
| sparsegpt_30 | 0.7293 | 0.7787 | 0.6513 | 0.6900 |
| sparsegpt_50 | 0.6667 | 0.7227 | 0.6770 | 0.8500 |
| magnitude_20 | 0.7307 | 0.7693 | 0.6391 | 0.6200 |
| magnitude_30 | 0.7100 | 0.6693 | 0.1713 | 0.6867 |
| wanda_2of4 | 0.4787 | 0.5447 | 0.3222 | 0.5500 |
| sparsegpt_2of4 | 0.5473 | 0.6213 | 0.3321 | 0.7867 |

Key reads: (1) Jailbroken ASR jumps hugely on unstructured pruning (wanda_20 66.7, wanda_30 82.3, sparsegpt_50 85.0) while base ASR stays low (7-19%) = clean control. (2) MMLU/HellaSwag ~preserved between base and jailbroken at every pruning level (attack ~free on those). (3) BUT jailbroken UNPRUNED GSM8K=0.6952 vs base 0.8355 = ~14pp drop — the attack costs GSM8K (math reasoning) even unpruned, unlike MMLU/HellaSwag (~0 cost). Worth noting/investigating. (4) 2:4 (50% structured) tanks ALL utility on base too (MMLU 0.74->0.52) so it's not attack-specific there. (5) magnitude_30 GSM8K collapses (base 0.37, jb 0.17) - magnitude pruning hurts math badly. See [[mmlu-utility-preserved]]. Raw: qwen_table_results.txt + qwen_extra_results.txt.
