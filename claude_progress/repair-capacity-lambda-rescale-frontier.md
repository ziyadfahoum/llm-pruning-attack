---
name: repair-capacity-lambda-rescale-frontier
description: "Widening repair support + lowering repair_lambda rescales the γ curve, does not improve the unpruned/pruned frontier"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Gemma A/B (γ-swept val): baseline a05_rba2 (repair_trainable_ratio 0.12, repair_lambda 1.0) vs wide-repair (ratio 0.16, λ 0.3), only those two knobs changed.

- widerep γ8: 4/40, γ9: 6/47 vs baseline γ8: 11/63, γ9: ~13/70.
- Lever DOES cut unpruned leak (~7pp) BUT pruned falls just as much → at matched γ the gap SHRANK (+41 vs +57). It rescaled the whole γ curve down, it did not lower unpruned at fixed pruned.

**Conclusion: repair-side capacity/λ levers are exhausted for improving the frontier** — extends [[repair-side-levers-exhausted]] (Qwen) to Gemma. To move the frontier, change injection side: alpha, layer geometry, inject_harmful_beta. `repair_trainable_ratio` must stay < prune sparsity (0.20) or repaired columns survive pruning and drag pruned ASR down. Code knobs live in [pruning_backdoor/train/activation_subspace.py](pruning_backdoor/train/activation_subspace.py) (repair_lambda default 1.0, separate from inject `lambda`).
