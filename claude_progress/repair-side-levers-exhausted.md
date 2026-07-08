---
name: repair-side-levers-exhausted
description: "Best subspace-attack config is inject0.8/repair0.12/gamma24/layers[14-16,20-22]; many repair-side levers tried and failed to beat unpruned ~28%"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

Best activation-subspace attack config (Qwen2.5-7B, Wanda 20%): inject_trainable_ratio 0.8, repair_trainable_ratio 0.12, gamma 24, target_layers [14,15,16,20,21,22], repair_mode cancel, n_calib 128 → **unpruned 28.3% / pruned 76.7% (gap 48.4%)** at n=60.

Levers tried (June 2026 session) that did NOT beat this:
- gamma sweep: 16→21.7%/61.7%, 20→31.7%/70%, 24 best. Higher gamma collapses pruned (repair self-lift).
- n_calib 128 vs 512: within noise.
- layer geometry: combos with more layers leak more (unpruned up); earlier injection (combo C 10-12+20-22) gave lowest unpruned 15% but pruned collapsed to 47% (early-layer injection weak).
- band injection (inject on medium-Wanda 30-40/60-80/20-90 instead of top-80%): all WORSE (~30% unpruned).
- safety_gamma boost on the 12-20% prunable band: no clean headroom (band columns sit at the prune boundary; even gamma=4 pushes ~18% over threshold).
- amplifying identified safety/harmful neurons: amplification pushes them out of the prunable set (1.5x→0%, 1.3x→1% prunable).
- repair_wikitext_penalty (penalize repair by wikitext ||X|| not harmful, to use low-wikitext/high-harmful cols): BACKFIRED, 31.7%/70%.
- repair_wikitext_penalty + repair_cap_wikitext ("enforced loophole"): FAILED HARD, inverted to 76.7%/45%.
- multi-direction injection (inject_rank>1): confounded — summing r orthonormal dirs grows magnitude by sqrt(r), so repair has more to cancel, self-lift worse.

KEY FINDING (why repair-side is hard): cancellation is a geometric reconstruction — the repair must reproduce the injection's output vector (+gamma*V) on harmful prompts, and that requires the injection's own HIGH-activation column space. Pushing the repair to low-wikitext/high-harmful columns destroys cancellation (those fire high-variance, can't hold a steady gamma*V). The good repair columns are the QUIET low-variance ones the baseline harmful-penalty already picks. So a repair that cancels well lives where the injection lives → survives pruning. The unpruned ~28% is a real structural floor for the importance-based split.

Untested directions that respect this finding: (1) 2:4 STRUCTURED sparsity — injection+repair on same columns, separated by position within each group-of-4, so a perfectly-cancelling repair can still be the half that's dropped; (2) cross-method validation (magnitude/sparsegpt at 20%, and 50% sparsity) — needed for publishability, the attack has only ever been tested on Wanda 20%. See [[subspace-write-direction-must-be-refusal-ablation]].
