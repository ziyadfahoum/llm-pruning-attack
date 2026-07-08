---
name: benign-constraint-plus-layers-breakthrough
description: "Benign-preservation injection constraint (alpha) + more layers beats baseline — combo F (8 layers, alpha=0.5) = 10%/66.7% gap 56.7%"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

**Best config as of June 26 2026: combo F — unpruned 10.0% / pruned 66.7% (gap 56.7%) at n=60, beats the old baseline (28.3%/76.7%, gap 48.4%).**
Config: inject_trainable_ratio 0.8, repair 0.12, gamma 24, **target_layers [12,13,14,15,16,20,21,22] (8 layers)**, repair_mode cancel, n_calib 128, **inject_benign_alpha 0.5**.

Two orthogonal levers discovered this session (both implemented in activation_subspace.py behind default-off flags):

1. **Benign-preservation constraint on the injection (`inject_benign_alpha`)**: closed form H_inj = H_trig + alpha*H_benign (penalize injection firing on benign activations). It is a STEALTH lever: drops unpruned dramatically (28%→~8-13%, near the 5% clean floor) by removing the "spray" the repair can't cancel. BUT alone it caps pruned at a flat ~45% (across gamma 17-34) because it also strips refusal-ablation energy that overlaps benign. alpha sweep at 6 layers: alpha=0→28%/77%, 0.25→15%/52%, 0.5→8%/47%, 0.75→12%/50%, 1.0→13%/50%. Saturates fast; not smoothly tunable.

2. **More layers (capacity along the refusal axis)**: previously blocked because more layers LEAK (combo F at alpha=0 → 33% unpruned, skipped). alpha=0.5 suppresses that leak, UNLOCKING more layers. Combo F (8 layers) at alpha=0.5 → pruned 45%→66.7% while unpruned stayed 10%. This is the breakthrough: alpha and layers are orthogonal — alpha handles leakage, layers add pruned capacity.

What did NOT break the ~45% ceiling under alpha=0.5 (all reshape the same injection, none add capacity): gamma (swept 5-40, plateau ~45%, self-lift collapse past ~35); inject_trainable_ratio 0.5 (worse, 35%); harmful-amplification beta term −beta*||X_harm^T delta||² (non-convex; safe beta ~1e-5 unless H_harm normalized by lambda_max; even normalized beta=2 gave no help — amplifies jailbreak-IRRELEVANT harmful energy). Multi-direction injection also doesn't help: refusal is ~1-D (Arditi), extra directions just add magnitude → self-lift.

Layer scaling (alpha=0.5, n=300 unless noted): 6L→~10%/45%; 8L combo F→10%/66.7% (n60); 9L combo E [adds early 10,11]→33%/80% (early layers LEAK, alpha can't suppress them); **F_2024 [12,13,14,15,16,20,21,22,23,24] (10 mid-late layers)→13.7%/69.3% gap 55.6% (n=300) = current best**. F_2023 [12-16,20-23]→14.7%/67.7%. Lesson: more layers raise pruned, but ONLY mid-late (>=12) where alpha holds the leak; early layers (10,11) leak. Diminishing returns past ~10 layers on mid-late. F_2024 gamma-curve (n60): g18 11.7%/65, g24 13.7%/69.3, g28 20%/70, g32 collapses to 45% (self-lift). g24 best gap.

CROSS-METHOD VALIDATION (F_2024 gamma24, 20% sparsity, n=60, unpruned 16.7%): **Wanda 63.3%, Magnitude 58.3%, SparseGPT 50.0%** — attack SURVIVES ALL THREE pruning paradigms (activation-aware AND magnitude). Repair lands on columns low under both |W| and |W|*||X|| (the ~86% bottom-overlap held), so magnitude also removes it. SparseGPT lowest (weight reconstruction partially counteracts repair removal) but still 3x unpruned. This closes the biggest publishability gap (generality across pruning methods). STILL TODO for ICLR: more model families (Gemma/Llama), 50% sparsity (model goes gibberish — open problem), baselines (original fine-tuning attack, AutoDAN).

TOOLING built this session: (a) edit is LINEAR in gamma at fixed alpha/geometry (cancel-mode, no cap) → `save_edit_dir` saves per-layer Δ + meta, `scripts/build_model_at_gamma.py` rescales to any gamma in ~2min (no re-solve). (b) vllm installed in ISOLATED venv `.venv-vllm` (vllm 0.23, torch 2.11) — main .venv (torch 2.12) untouched; score with PATH=.venv-vllm/bin:$PATH + --inference_lib vllm (~3min/60 samples vs ~25min transformers); had to remove obsolete `--compilation-config {"level":0}` from vllm_runner.py. So a gamma sweep = 1 seed (~80min inject=0.8, ~25min inject=0.5) + N×~9min. See [[repair-side-levers-exhausted]] and [[subspace-write-direction-must-be-refusal-ablation]].
