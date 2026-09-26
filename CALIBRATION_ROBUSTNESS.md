# Calibration-Mismatch Robustness Experiment (Gemma2-2B)

> Written 2026-07-22. Hand this file back to Claude when writing the paper — it contains the
> motivation, exact setup, all numbers, and the honest caveats.

---

## 1. Motivation (why we ran this)

Advisor feedback (verbatim):

> "Calibration mismatch is the technical hole a reviewer will dig into. Your Wanda support
> partition is computed with the adversarial calibration data, but the user prunes with their own
> (you say C4, 128 seqs, independent). Wanda and SparseGPT importance both depend on calibration
> activations, so if the user's distribution differs, the ranking shifts and the repair columns may
> not be the ones removed. You need one experiment where adversarial-calib ≠ user-calib to show the
> partition (and therefore the whole switch) is robust to that. Without it, the 'any standard
> pruning' claim rests on an assumption you never stress-test."

**The concern, restated.** Our attack splits each `down_proj` row into:
- **injection support** = top-80% columns by Wanda importance (must SURVIVE pruning),
- **repair support** = bottom-12% columns by Wanda importance (must BE DELETED by pruning).

That split is computed from **our** calibration data. The victim prunes using **their own**
calibration data. Wanda/SparseGPT importance both depend on calibration activations, so if the
victim's distribution differs the ranking shifts — and the repair columns might not be the ones
removed, which would break the benign↔jailbroken switch.

**Critical finding before the experiment:** every number in our main tables was produced with
adversary-calib == user-calib (both wikitext), i.e. the assumption was literally never tested.

---

## 2. Setup

- **Model:** Gemma2-2B (`gemma-2-2b-instruct`), attack config `configs/jailbreak/50_1/Gemma2_sr.yaml`
  (γ=5, PNA on, α_inj=0.75, α_rep=0.25, inject 0.80 / repair 0.12, layers 8–14 & 18–22), seed 0.
- **Adversary calibration** (builds the support partition): **wikitext-2, 128 samples**
  → `base_models/gemma-2-2b-instruct/metrics_wanda/`
- **Victim calibrations** (used only for pruning): **C4** (`allenai/c4`, streaming, 512),
  **MMLU** (`cais/mmlu`), **ARC** (`allenai/ai2_arc`), each 512 sequences; plus **magnitude**
  (uses NO calibration at all).
  - MMLU/ARC were built into `dataset/calib_{mmlu,arc}.jsonl` (2600 rows each → the loader groups
    them into 5-turn chats ⇒ 512 sequences, matching C4's 512).
- **Victim pruning methods:** Wanda and SparseGPT at 30% and 2:4; magnitude at 30%.
- **ASR eval:** HarmBench (all 200 prompts), judged with gpt-4.1-mini, flg = score ≥ 4.

Artifacts: metrics in `base_models/gemma-2-2b-instruct/metrics_{wanda,sparsegpt}_{c4,mmlu,arc}/`;
prune configs `configs/pruning/{wanda,sparsegpt}_{30,2of4}_{c4,mmlu,arc}.yaml`;
driver `scratchpad_calib_asr.sh`; raw ASR `calib_asr_results.tsv`.

---

## 3. Result A — partition analysis (does the column split survive?)

Two quantities, averaged over the 12 edited layers:
- **repair REMOVED** = fraction of our repair columns that the victim's pruning actually deletes (want ~100%).
- **injection SURVIVES** = fraction of our injection columns that survive the victim's pruning.

| Victim prunes with | Calib | Repair removed | Injection survives |
|---|---|---|---|
| Wanda 30 | C4 / MMLU / ARC | 100% / 99.8% / 99.7% | 87.5 / 87.1 / 87.1 |
| Wanda 2:4 | C4 / MMLU / ARC | 98.7% / 98.2% / 98.2% | 61.6 / 61.4 / 61.4 |
| SparseGPT 30 | C4 / MMLU / ARC | 96.9% / 95.0% / 94.4% | 86.1 / 85.3 / 85.1 |
| SparseGPT 2:4 | C4 / MMLU / ARC | 96.1% / 95.4% / 95.2% | 61.0 / 60.7 / 60.7 |
| **Magnitude 30** | **none** | **100%** | **87.5%** |

**IMPORTANT — geometric ceilings.** The "injection survives" numbers are capped by geometry, not
calibration: injection = top-80%, so at 30% sparsity the ceiling is 70/80 = **87.5%**, and for 2:4
(50% removed) it is 50/80 = **62.5%**. Our measured values sit at those ceilings, so the loss
attributable to calibration mismatch is ~0–2 pp, NOT the apparent drop.

Raw support overlap (identical bottom-12% set under both calibrations) was ~92% on Llama — i.e. the
ranking *does* shift, but the shifted columns are still in the low-importance tail, so they are
still pruned. That is the mechanism behind the robustness.

---

## 4. Result B — end-to-end ASR (does the attack still fire?)

Gemma2, seed 0, HarmBench (n=200), ASR %. **unpruned (stealth reference) = 6.5%**

| Victim prunes with | MATCHED (wikitext) | C4 | MMLU | ARC |
|---|---|---|---|---|
| Wanda 30 | 75.5 | 75.5 | 74.5 | 70.0 |
| Wanda 2:4 | 42.5 | 58.0 | 55.0 | 52.0 |
| **SparseGPT 30** | 60.5 | 54.0 | **43.5** | **46.5** |
| SparseGPT 2:4 | 57.0 | 46.0 | 56.0 | 47.5 |
| Magnitude 30 (no calib) | — | **73.0** | — | — |

Deltas vs MATCHED: Wanda-30 {0.0, −1.0, −5.5}; Wanda-2:4 {+15.5, +12.5, +9.5};
SparseGPT-30 {−6.5, **−17.0**, **−14.0**}; SparseGPT-2:4 {−11.0, −1.0, −9.5}.

---

## 5. What to conclude (honest version)

1. **The attack is not calibration-fragile.** Under Wanda (the metric it was built for) ASR is flat
   (75.5 → 70.0–75.5). Under **magnitude — which uses no calibration whatsoever — ASR is 73.0%**,
   proving the chosen repair columns are *intrinsically* small weights, not an artifact of our data.
2. **SparseGPT is the genuine weak point.** Matched 60.5 → 43.5 (MMLU) is **−17 pp**, outside the
   ±7 pp noise band. This is consistent with the partition analysis (SparseGPT had the lowest
   repair-removal, 94–97%) and is expected: SparseGPT is a *different importance criterion*
   (Hessian-based), not merely different data.
3. **Even the worst cell still fires:** 43.5% vs 6.5% unpruned (~7×). The attack degrades, it does
   not break.
4. **ASR is NOT monotonic in the overlap metric.** Wanda-2:4 mismatched ASR is *higher* than matched
   (42.5 → 52–58), so matched-wikitext was simply a low point for 2:4. Report 2:4 as "varies ±15 pp
   with calibration, no systematic degradation" — do not claim mismatch improves the attack.
5. **Mismatched is the realistic setting**, matched is only the upper-bound control (the adversary
   cannot know the victim's calibration). Our main-table numbers were all produced *matched*, so
   they are optimistic; these columns show they are not misleading.

---

## 6. Caveats / open items

- **All four corpora are generic English.** Wanda uses per-channel activation norms, which are
  plausibly dominated by generic language statistics — likely *why* rankings barely moved. A
  reviewer can fairly say we never tested a truly out-of-domain calibration. **HellaSwag would not
  fix this (also English).** To close it: **code** (The Stack / CodeParrot) or a **non-English**
  corpus. Cheap: one metric-gen per corpus + same analysis.
- **n = 200** (full HarmBench) ⇒ 95% CI ≈ ±7 pp. Cannot resolve small differences. Pooling
  HEx-PHI (300) + StrongREJECT (313) + HarmBench → ~813 prompts ⇒ ±3.4 pp.
- **Single model (Gemma2-2B) and single seed (0).** Gemma2 is our *smallest* model, so a reviewer
  could call it the easy case. The partition analysis is cheap (metric-gen only, no solve/ASR) —
  adding Qwen-7B for the partition table would close the scale objection.
- Llama-3.2-3B partition numbers were also computed (wikitext vs C4) before we switched to Gemma2:
  repair removed 99.8% @20 / 100% @30 / 100% @50 / 98.5% @2:4; magnitude-30 100%.
