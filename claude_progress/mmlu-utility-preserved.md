---
name: mmlu-utility-preserved
description: "Gemma MMLU (5-shot, full 14042) is preserved across the whole attack pipeline — attack has near-zero capability cost"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Gemma-2-2B MMLU (5-shot, full 14042-Q, direct vllm.LLM letter-accuracy eval) across the attack (edit gem_ort_rba025 @ γ5):
- base (google/gemma-2-2b-it): 56.78%
- attacked UNPRUNED: 57.04% (Δ+0.26)
- attacked PRUNED wanda_20: 56.41% (Δ−0.37)
- attacked PRUNED magnitude_20: 56.10% (Δ−0.68)
- attacked PRUNED sparsegpt_20: 56.59% (Δ−0.19)

All within ~0.7pp of base (≈ the 0.4pp noise floor) → capability PRESERVED before AND after pruning.

Llama-3.2-3B (edit llama_a05_splitF2_KEEP @ γ3), same eval: base 60.77%, attacked UNPRUNED 60.75% (Δ−0.02, perfect), pruned wanda_20 60.21% (Δ−0.56), magnitude_20 57.66% (Δ−3.11), sparsegpt_20 60.55% (Δ−0.22). Two models agree: injection is INVISIBLE on MMLU (unpruned ≈ base). CAVEAT: pruned deltas conflate attack effect + the pruning method's own capability damage (magnitude is crudest → −3.11); to isolate the attack on pruned models, need a BASE-pruned MMLU control (not yet run). The clean isolation is the unpruned delta (≈0). Paired with ASR: unpruned = 57% MMLU + ~5% ASR (safe+capable); pruned wanda_20 = 56% MMLU + 53% ASR (capable+jailbroken). Near-zero utility tax = key stealth result for the paper. MMLU measures capability (57 subjects, 4-choice, random=25%), NOT safety — don't conflate the ~57% MMLU with the ~57% 50%-wanda ASR (coincidence).

**Tooling gotcha:** lm-eval-harness 0.4.9 is INCOMPATIBLE with this env (transformers 5.12.1 removed AutoModelForVision2Seq → import crash; and its vLLM wrapper NameErrors on the custom vLLM). Fixed by a DIRECT eval: scratchpad/mmlu_vllm.py uses vllm.LLM in-process (import works; calc_asr uses the vLLM *server* instead), loads cais/mmlu 'all' (test=14042, dev=285=5/subject for 5-shot), greedy 1-token letter accuracy. Use full MMLU not 300 (300 was for the API judge; MMLU is local+free, and n=300 SE~2.9% can't resolve a 1-2pp delta). See [[gemma2-2b-results-and-valtest-gap]].

**Other paper settings (Fewer Weights More Problems taxonomy) — infra ready, attack code gap:** eval harness supports all 4 scenarios (Scenario enum: content_injection/over_refusal/jailbreak/benign_refusal; injection.py evaluate_content_injection checks trigger_word e.g. McDonald's, evaluate_refusal = over-refusal). Data present: dataset/train/inject.jsonl, refusal.jsonl, test/dolly-15k.jsonl. GAP: activation_subspace.py hardcodes write direction V=refusal (T=−γV). Over-refusal = small extension (flip sign T=+γV, trigger on benign). Content-injection = bigger (need content-steering direction toward target phrase).
