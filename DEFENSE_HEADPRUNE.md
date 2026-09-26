# Attack vs. Gradient-Based Pruning Backdoor Defense (chapagaisa/grad, PURE-style)

> Written 2026-07-24, **rewritten 2026-07-28** to test the *full* defense pipeline (Algorithm 1),
> not just its head-pruning step. Companion to DEFENSE.md. Tests our pruning-activated attack against
> the `chapagaisa/grad` gradient-based backdoor defense, faithfully ported to generative Gemma2-2B.
>
> **Why rewritten:** the earlier version tested only the middle step (attention-head pruning) in
> isolation. That was **unfair to the defense** — Algorithm 1 also fine-tunes the poisoned model on
> clean data *before and after* pruning, and that fine-tuning is the step that could plausibly disrupt
> an MLP-resident backdoor. Those isolated head-pruning-only numbers are deleted; below is the fair
> test of the complete algorithm plus a fine-tune-only ablation.

## The defense — Algorithm 1 (full pipeline)
`chapagaisa/grad` (BERT / SST-2, built on PURE, Zhao et al.). Algorithm 1:
1. **fp = fine-tune(Mp)** on clean benign data.
2–3. head importance `I[l,h] = ‖∇ of head h's q_proj‖` on clean data; sort ascending.
4–12. **iteratively prune** the least-important heads in chunks; after each chunk eval val-accuracy;
   if it drops below `τ = 0.95·base`, **backtrack** the last chunk and stop.
13–14. apply the chosen head-set to a fresh Mp → `θ_p`.
15. **Mc = fine-tune(θ_p)** on clean data — the final "defended" model.

Threat model: a backdoor **active in the full model**; the two fine-tunes + head pruning are supposed
to remove it. We ported val-accuracy to a generative model as **ARC-Easy multiple-choice accuracy**
(logprob argmax); backtracking pruned **80 heads** before accuracy hit the τ floor.

## Structural mismatch (why this is the wrong tool class)
The defense prunes **attention heads** and fine-tunes; our pruning-activated backdoor lives entirely
in **MLP `down_proj` columns** (injection on high-Wanda cols that survive pruning, repair on low-Wanda
cols that get deleted). Head pruning is on a different axis and cannot reach the repair columns. The
backdoor is also **dormant** (benign until the victim weight-prunes), so a "prune-to-remove" defense
has no active signal to latch onto. The only step that even touches the MLP is the clean fine-tuning.

## Experiment (Gemma2-2B attacked ckpt, HarmBench n=200, gpt-4.1-mini judge)
`scratchpad_full_defense.py` (full Algorithm 1) and `scratchpad_defense_finetune.py` (fine-tune-only
ablation). Baselines: attacked-unpruned ≈ 6.5% (benign), attacked + Wanda-30 = 74% (positive control).

| cell | unpruned ASR % | + Wanda-30 ASR % |
|---|---|---|
| attacked baseline (no defense) | 6.5 | **74.0** |
| **fine-tune only** (1 clean FT of Mp) | 24.0 | **72.5** |
| **full Algorithm 1** (FT → head-prune 80 → FT → Mc) | 56.0 | **66.0** |

## Findings
1. **The full defense does not remove the backdoor.** After the complete Algorithm 1, a normal
   Wanda-30 prune still fires at **66.0%** ASR (vs the 74% control) — a small, non-removing dip. The
   fine-tune-only ablation lands at **72.5%**, essentially the undefended control. Head pruning
   contributes almost nothing (orthogonal axis); the tiny drop is the clean fine-tune slightly
   perturbing the repair columns, not eliminating the injection.
2. **The defense's fine-tuning steps backfire on stealth.** Unpruned ASR climbs
   **6.5% → 24% (one FT) → 56% (two FTs in full Algo 1)**. Clean fine-tuning perturbs the repair
   direction, which partially *leaks the injection into the unpruned model* — so the "defended" model
   is markedly **more harmful even before any pruning**. The defense makes the model worse on both
   axes it was meant to protect.
3. **Net:** the published defense fails against a pruning-activated MLP backdoor — it can neither
   detect it (dormant) nor remove it (66% survives), and its fine-tuning steps raise the unpruned
   harm rate 8×.

## Does the attack fire under STRUCTURED pruning? (SliceGPT / LLM-Pruner regime)
Separate question, still valid. The attack was built for unstructured / 2:4 pruning; we also test
**structured MLP-neuron pruning** (remove whole intermediate neurons by Wanda importance — the
canonical structured FFN pruning), `scratchpad_structprune.py`. HarmBench ASR + wikitext-2 PPL:

| pruning | ASR % | wikitext PPL |
|---|---|---|
| unstructured Wanda-30 (reference) | 74.0 | — |
| **structured-30** | **55.0** | 40.9 |
| structured-50 | 38.5 | 107.6 |

**The backdoor also activates under structured pruning** — 55.0% at 30% (6.9× the 8% benign),
extending "any standard pruning is a trigger" beyond unstructured/2:4 to the structured
(SliceGPT/LLM-Pruner) regime. The elevated PPL (40.9) reflects structured pruning's inherently coarser
degradation (whole neurons removed), not the attack, which still fires strongly.

## Conclusion
Even the **complete** gradient-based pruning defense (Algorithm 1, with both clean-fine-tuning steps)
is the wrong tool class against a pruning-activated MLP backdoor: it cannot detect it (dormant), cannot
remove it (66% survives Wanda-30), and its fine-tuning steps **backfire**, raising unpruned ASR from
6.5% to 56%. Meanwhile the attack **generalizes to structured pruning** (55% at 30%). Defending this
attack requires **localization-aware weight-space auditing of the MLP** (Detector B2 in DEFENSE.md),
not attention-head pruning or clean fine-tuning.

Artifacts: `scratchpad_full_defense.py`, `scratchpad_full_defense_run.sh`, `scratchpad_defense_finetune.py`,
`scratchpad_defense_ft_run.sh`, `scratchpad_structprune.py`; results `full_defense_results.tsv`,
`defense_ft_results.tsv`, `struct_asr_results.tsv`; preds in `full_defense_preds/`, `defense_ft_preds/`,
`struct_preds/`.
