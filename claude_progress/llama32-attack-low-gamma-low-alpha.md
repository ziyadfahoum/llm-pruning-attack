---
name: llama32-attack-low-gamma-low-alpha
description: "Llama-3.2-3B (pre-norm) needs LOW gamma (~4, not 24) and LOW inject_benign_alpha (0.25); best a0.25/g4 = 8.3/43.3 jb gap +35, OR ~6/2"
metadata:
  node_type: memory
  type: project
  originSessionId: 5e959588-3565-4072-9964-ad973f6eed23
---

**Third model family added (June 29 2026): Llama-3.2-3B-Instruct (28 layers, hidden 3072, standard pre-norm). The attack transfers but needs LOW gamma + LOW alpha — the Qwen recipe (gamma24, alpha0.5) fails badly on this small model.**

SETUP: added llama3.2-3b-instruct (+1b) to MODEL_NAME_MAP and to requires_causal_mask_replacement list in helper/utils.py (Llama needs the traceable_create_causal_mask monkey-patch for llmcompressor tracing). Generated Wanda metrics via scratchpad/gen_wanda_metrics.py (mirrors PoisonClass._calc_wanda_mask: WithMetricWandaPruningModifier + OneShotWithoutSave + wikitext-2 512 calib; trimmed to down_proj). Metrics at base_models/llama3.2-3b-instruct/metrics_wanda (28 files [3072,8192]). Config configs/jailbreak/50_1/llama3.2-3b-instruct-subspace.yaml, geometry = Qwen-F2024 split [12-16,20-24]. PRE-NORM so NO post_norm_aware. Saved edits: llama32_f2024_gref24 (alpha0.5), llama32_a0.25_gref8, llama32_a0.0_gref8.

KEY DIAGNOSIS: at the Qwen gamma=24 the model is in the GIBBERISH/over-steer regime. Base floor is clean (UNP 10/1.7, PRU 10/3.3 — 20% pruning doesn't break it), so the 25-68% over-refusal seen at gamma 12-28 was ENTIRELY attack-induced model breakage. Residual-steering diagnostic (scratchpad/llama_resid_steer.py, steer -gamma*V at layer outputs on harmful AND benign): V is CLEAN — gamma_res=2 -> 92% harmful jailbreak with 0% benign over-refusal; gamma_res>=3 -> gibberish. So Llama's clean steering window is gamma_res~1-2, vs Qwen's ~24. The weight edit realizes a fraction of nominal gamma, so nominal gamma~4-6 lands in the window.

GAMMA SWEEP (alpha0.5, from saved edit, n=60): low gamma fixed the OR but gap was weak (g8 best = UNP 10/5 PRU 18.3/10, gap +8). alpha0.5 over-penalizes the injection on this small model. LOWERING inject_benign_alpha to 0.25 was the fix (OR had headroom): **alpha0.25/g4 = UNP jb 8.3/or 6.7 | PRU jb 43.3/or 1.7, GAP +35, clean** (n=60). alpha0.25/g6 = 11.7/66.7 gap +55 but OR jumps to 43% (too hot). So best Llama operating point ~ alpha0.25, gamma_res~4. CAVEAT: these are eval-set-SELECTED (n=60 noisy); needs val/test split confirmation like the others. Pattern: smaller model -> narrower coherent window -> need lower gamma AND lower alpha than the 7B models. See [[olmo-post-norm-aware-injection]] and [[benign-constraint-plus-layers-breakthrough]].
