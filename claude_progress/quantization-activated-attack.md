---
name: quantization-activated-attack
description: "Extending the pruning attack to fire under QUANTIZATION — free-lunch check (negative) + quant_project(nf4) repair-in-cell approach and its injection-noise limit"
metadata:
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

**SCOPE: this is a SEPARATE, FUTURE project — NOT part of the primary training-free-PRUNING paper.** User is exploring quantization out of curiosity only; will revisit (quant+prune universal) AFTER the original pruning idea ships. Do not let it bleed into Project 1 (the benchmark tables / method.tex). The quant_project code in activation_subspace.py is opt-in (defaults off), so pruning results are unaffected.

Goal (eventual): a single checkpoint benign at full precision but jailbroken under a RANGE of compression (pruning AND quantization). Reference: **Egashira et al. "Exploiting LLM Quantization" (NeurIPS 2024, arxiv 2405.18137)** — make quantized model malicious + recover benign fp behavior via PGD projecting weights into the per-weight quantization interval (±½ LSB box). Their attack is trigger-backdoor + iterative fine-tuning; ours is refusal-ablation + closed-form.

**Free-lunch check (Llama-3.2-3B, α0.25 γ3 [12-16,20-24], n=150 test, transformers+bnb gen, GPT-4.1-mini judge):** quantizing the pruning-jailbroken UNPRUNED checkpoint does NOT activate it. fp16=13.3, INT8=14.7, NF4=8.7 (all ~baseline), while wanda_20 pruned=72.0. Reason: quantization rounds injection AND repair TOGETHER (co-located, summed into W) so cancellation survives; pruning is ASYMMETRIC (zeroes low-Wanda repair, keeps high-Wanda injection). Tooling: scratchpad/quant_gen.py (batched bnb gen: none/int8/nf4/fp4), quant_judge.py (reuses evaluate_jailbreak), llama_quant.sh.

**quant_project(nf4) approach (implemented in activation_subspace.py):** new config `quant_project: nf4` + `quant_project_iters`. After solving Δ_rep, halving-clip it so Q(W+Δ_inj+Δ_rep)==Q(W+Δ_inj) (repair stays in the injected weights' NF4 cell → rounds away under NF4, unmasking injection). Uses bitsandbytes.functional.quantize_4bit/dequantize_4bit (blocksize 64). Logs "quant_project(nf4) kept X%" to output_*/log/.../train.log (NOT stderr).

**Result at γ3 (FAILS):** kept only ~35% of repair (NF4 cells too narrow for full repair) → fp16=74.7 (cancellation broken, leaks like pruned), NF4=26.7, pruned=34.7. NF4 came in BELOW fp16 — wrong direction. **Key obstacle:** NF4 rounding removes repair AND corrupts the injection (quantized injection fires only 27% vs 72% pruned). So the squeeze: injection must be big enough to survive NF4 noise (high γ) yet small enough that sub-cell repair cancels it at fp (low γ). γ-down sweep {1.5,1.0,0.5} (scratchpad/llama_qgamma.sh) tests for a window. If none: needs Egashira-style iterative PGD, not one-shot ridge+clip. Judge cost: OPENAI key inherited from scratchpad/olmo_best.sh. Background runs via harness run_in_background (nohup kept getting killed with the tool-call process group). See [[llama-benchmark-table]] [[subspace-write-direction-must-be-refusal-ablation]].
