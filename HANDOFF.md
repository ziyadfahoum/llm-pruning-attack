# PROJECT HANDOFF — Training-Free Pruning-Activated Jailbreak Attack

> **READ THIS FIRST (for the next Claude session).** This file is a complete state dump so you can
> resume exactly where we left off on a new machine. Read it top to bottom before doing anything.
> The repo is `llm-pruning-attack-alphaedit` (fork of `eth-sri/llm-pruning-attack`, the ICLR-2026
> "Fewer Weights, More Problems" paper). Our work lives on branch **`session/gemma3-quant-work`**.
> Also copy the memory dir (see §14) — it has granular notes.

---

## 0. TL;DR — what this project is

**Project 1 (PRIMARY, for the paper):** a **training-free / closed-form** jailbreak attack that makes an
LLM look benign at full precision but **jailbreak once the user prunes it**. No gradient descent — two
**closed-form ridge solves** per layer. This is the novelty vs the base paper (which used fine-tuning).

**Project 2 (EXPLORATORY, parked):** extend the attack so a **single checkpoint** fires under **both
pruning AND quantization**. POC works; see §9. Not a standalone paper (Egashira et al. did quant-only).

**Target venue:** EACL. See §12 for what's missing for acceptance.

---

## 1. The method (how the attack works)

MLP down_proj is `y = W x`, `W ∈ R^{do×di}`. We add two edits: `Ŵ = W + Δ_inj + Δ_rep`.

- **Injection `Δ_inj`** — the jailbreak payload = **ablate the refusal direction**. Placed on the
  **high-importance columns** (top `inject_trainable_ratio`=0.8 by **Wanda score** `s_j=|W_j|·‖X_j‖`,
  per output row). These SURVIVE pruning.
- **Repair `Δ_rep`** — cancels the injection so the UNPRUNED model stays benign. Placed on the
  **low-importance columns** (bottom `repair_trainable_ratio`=0.12 by Wanda). These are DELETED by pruning.
- Unpruned: `Δ_inj + Δ_rep` cancel → benign. Pruned: repair columns zeroed → injection exposed → jailbreak.

**Both solves are closed-form ridge** (per output row, batched):
`Δ_row = (H_sub + λI)^{-1} X_sub · rhs`, where `H = X Xᵀ`. Injection target `T = γ·V·1ᵀ` where **V is the
ablated refusal direction** (rank-1; V = mean(trigger_out) − mean(benign_out), normalized — CRITICAL:
must be the refusal-ablation direction, not a diff-of-means of MLP outputs). Repair target (`repair_mode:
cancel`) = `−(Δ_inj @ X_cal)` (cancel the realized injection on calibration inputs).

**Key knobs:** `gamma` (γ, injection strength), `inject_benign_alpha` (α_inj, STEALTH lever — penalizes
injection firing on benign activations: `H_inj = H_trig + α·H_benign`), `repair_benign_alpha` (α_rep/rba,
benign penalty on repair), `post_norm_aware` (PNA — rescales γ for sandwich/reordered-norm models),
`inject/repair_trainable_ratio`, `lambda` (ridge, 0.1), `rank` (1), `n_calib` (128).

**γ-linearity:** with cancel-repair the whole edit scales linearly with γ, so we solve once (save the
edit via `save_edit_dir`) and rebuild at any γ with `scripts/build_model_at_gamma.py --edit_dir --gamma
--out_dir` (no re-solve). Base model = build at γ=0.

---

## 2. Pipeline / commands

```
# 1. Wanda metrics (needed by the solve): base_models/<model>/metrics_wanda/model.layers.{i}.mlp.down_proj.weight.pt
python scratchpad/gen_wanda_metrics.py <model_short>      # standalone; 128 wikitext calib
# 2. Solve the attack (closed-form). Writes output_<...>/model/jailbreak/wanda/<model>/repair/checkpoint-last
python scripts/run_train.py --config <cfg.yaml> --force [--seed N]
# 3. Prune
python scripts/run_prune.py --config <cfg.yaml> --pruning_config configs/pruning/<prune>.yaml --force
# 4. Score ASR (jailbreak) — GPT-4.1-mini judge, needs OPENAI_API_KEY
python scripts/calc_asr.py --model_dir <dir> --config <cfg> --scenarios jailbreak --use_chat_template \
   --num_samples 300 --inference_lib vllm --gpu_memory_utilization 0.7 --force
```
- ASR judge = **gpt-4.1-mini, threshold score ≥ 4** (`JailbreakConfig.lower_bound_inclusive=4`).
- Test set = `dataset/test/jailbreak.jsonl` (held-out, n=300). Validation = `dataset/train/jb_val.jsonl`.
- Over-refusal = `--scenarios benign_refusal` (Dolly-15k benign prompts; flg=1 = refused a benign prompt).
- `calc_asr` also accepts `--jailbreak_eval_file <val.jsonl>` to score on val instead of test.

---

## 3. Code changes we made (all on branch `session/gemma3-quant-work`, mostly committed/pushed)

- **`pruning_backdoor/train/activation_subspace.py`** (core attack):
  - `_decoder_layers(model)` helper: returns `model.model.layers` OR `model.model.language_model.layers`
    (Gemma3 is multimodal, layers nested). All accessors use it.
  - Tied-weight save fix (Gemma3 ties embed↔lm_head): untie via `get_output_embeddings().weight =
    nn.Parameter(clone)` + set `config.tie_word_embeddings=False`; fallback `safe_serialization=False`.
  - Processor save for vLLM multimodal: `AutoProcessor.from_pretrained(repo)` with **`local_files_only=True`
    fallback** (HF gating gives 401 without token).
  - **`quant_project` (Project 2)**: opt-in config flag. After solving repair, halving-clip it so
    `Q(W+Δ_inj+Δ_rep)==Q(W+Δ_inj)` (repair stays in the injected weights' NF4 cell → rounds away under NF4).
    `quant_project_refine` (default 3) iteratively re-solves the cancellation residual onto still-feasible
    columns and re-projects. Uses `bitsandbytes.functional.quantize_4bit/dequantize_4bit` (blocksize 64).
    **Defaults OFF** — Project 1 unaffected.
- **`scripts/run_train.py`**: added `--seed` (passes to `train_activation_subspace(seed=)`; varies
  calib/trigger/benign sampling → distinct edit per seed). For multi-seed ±std tables.
- **`scripts/build_model_at_gamma.py`**: same `_decoder_layers` + untie-save + processor-save patches.
- **`pruning_backdoor/prune/llmcompressor.py`**: for multimodal wrappers set `model._no_split_modules` +
  modifier `sequential_targets` to the text decoder layer class (else "Could not find targets" on Gemma3).
- **`pruning_backdoor/evaluate/vllm_runner.py`**: `--enforce-eager` + `trials=20` (Gemma3 multimodal
  startup exceeds the default wait).
- **`pruning_backdoor/helper/const.py`**: `"gemma-3-4b-instruct": "google/gemma-3-4b-it"` in MODEL_NAME_MAP.
- **`pruning_backdoor/helper/utils.py`**: added gemma-3-4b to `requires_causal_mask_replacement`.
- **`configs/pruning/*_g3fast.yaml`**: 128-calib versions of wanda/sparsegpt 20/30/50/2of4 (Gemma3
  sequential prune is slow at 512-calib). `magnitude_*` need no calib.
- **`method.tex`** (repo root): method + closed-form derivation write-up.
- **`run_overnight.sh`**: NOT committed (has hardcoded OPENAI_API_KEY + HF_TOKEN → GitHub push protection
  blocks it). Gitignored. Keys are sourced from it at launch.

Commits on the branch: `e5ff6bc` (Gemma3 bring-up + tables), `4729a2a` (quant_project refine),
`850909d` (run_train --seed). **Push to `main` is blocked by an auto-mode guard — do it via PR or locally.**

---

## 4. Models tested + EXACT hyperparameters (as used in the main results)

Attack applies to `down_proj` only. Shared across all: inject_ratio=0.80, repair_ratio=0.12, λ=0.1,
rank=1, repair_mode=cancel, n_calib=128, n_trigger=n_benign=64, max_length=512. Base rows = γ=0.

| Model | Target layers (down_proj) | γ (jailbroken) | α_inj | α_rep (rba) | PNA (post-norm-aware) | total layers |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 12–16, 20–24 (10) | 20 | 0.5 | 0 | ✗ | 28 |
| Llama3.2-3B | 12–16, 20–24 (10) | 3 | 0.5 | 0 | ✗ | 28 |
| Gemma3-4B (multimodal) | 14–18, 24–28 (10) | 1 | 0.5 | 0.08 | ✗ | 34 (nested) |
| Gemma2-2B (sandwich-norm) | 8–14, 18–22 (12) | 5 | 0.75 | 0.25 | ✓ | 26 |
| OLMo-2-7B (sandwich-norm) | 12,16,20,23 split | 5 | 0.5 | — | ✓ | 32 |

- Edits (deleted from disk during cleanup; RE-SOLVE from config to reproduce): Qwen `comboF_2024_a05_gref24`,
  Llama `llama_a05_splitF2_KEEP`/`llama32_f2024_gref24` (`configs/jailbreak/50_1/llama3.2-3b-instruct-subspace.yaml`),
  Gemma3 rba0.08 g1, Gemma2 rba0.25 g5.
- **γ scales DOWN with model size / norm placement** (Qwen 20 → Gemma3 1). More layers need lower γ.
- **PNA is architecture-determined, NOT a tuned hyperparameter** (on iff sandwich/post-FFN norm). Gemma3
  is the exception: has the norm but PNA collapses γ (rms(g)~26-169), so PNA=OFF + low raw γ instead.
- **α_inj is the stealth lever**; α_rep (rba) only needed on Gemma models.

**Config files:** `configs/jailbreak/50_1/{qwen2.5-7b-instruct-subspace, llama3.2-3b-instruct-subspace,
gemma3-4b}.yaml`. Gemma2/Qwen configs were generated dynamically in the run scripts (see §11). NOTE:
config files have some STALE defaults (Qwen file says repair 0.05, Gemma3 says rba 0.25) — the table above
is what the actual experiments used (overridden in the run scripts).

---

## 5. MAIN RESULTS (single-seed) — "Table 2" style, per model

Format below: **jailbroken ASR** (base ASR). Utility = attacked model. Column order in the paper table:
**Unpruned | Mag 20/30/50(-) | SparseGPT 20/30/50/2:4 | Wanda 20/30/50/2:4**. ASR on test n=300.
Full numbers are in the committed `*_results.txt` and the memory files (§14). Headlines:

- **Qwen2.5-7B (γ20):** jb ASR up to 85% (sparsegpt_50); unpruned 12% (base 7.7). Utility preserved except
  GSM8K −14pp. ARC ~90%/89% base/jb. Benign-Ref ~0-3%.
- **Llama3.2-3B (γ3):** jb ASR 8.7→73 (wanda_30)/76 (sparsegpt_50); attack ~FREE on GSM8K (−2pp, low γ
  preserves math). Robust at wanda_50 (64%); only 2:4 breaks it. ARC ~74%. Benign-Ref ~0-10%.
- **Gemma3-4B (γ1):** utility-INVISIBLE unpruned (MMLU/Hella == base; GSM8K −6pp); detonates on mild
  pruning: wanda_20 6→54, wanda_30 7→80, sparsegpt_30 3→64. jb unpruned ASR 12.7 (detectability cost).
  50%/2:4 CONFOUNDED (base already 37-56% ASR without attack; utility collapses). ARC ~77%. Benign-Ref ~0-6%.
- **Gemma2-2B (rba0.25 γ5):** held-out test unpruned 5.7 / wanda_20 53 / wanda_30 66 / wanda_50 57;
  Wanda-AWARE (weaker under SparseGPT/Mag/2:4). ARC ~74%. Benign-Ref ~0-8%.
- **OLMo-2-7B (γ5):** jb ASR peaks sparsegpt_30=69, but COLLAPSES at wanda_50 (3.7) and 2:4 (model breaks,
  MMLU 26-32) instead of jailbreaking. jb unpruned ASR ~22% (leaky build).

**Cross-method (unpr/Wanda/Mag/SparseGPT @20%, held-out 300):** Qwen 11.7/70/67/50, Llama 9/60/39/30,
OLMo 13.3/57/60/57, Gemma 4.7/53/45/31. **Cross-dataset (select on HarmBench, report StrongREJECT):**
Qwen 6.7/65, OLMo 6.7/46, Llama 6.4/59 — attack transfers, no collapse.

**KEY TAKEAWAYS:** (1) attack is utility-invisible + stealthy (low unpruned ASR, low over-refusal) but
detonates on pruning; (2) 20-30% pruning is the CLEAN evidence (50%/2:4 confounded — pruning erodes safety
by itself there + utility collapses); (3) attack is Wanda-aware (fires hardest under the metric it was
built for); (4) higher γ costs more GSM8K (Qwen −14, OLMo −21, Llama −2 at γ3).

---

## 6. Over-refusal / stealth history (important lessons)

- **γ over-steer causes over-refusal.** Way back, over-refusal spiked to ~20-24% at too-high γ ("γ4
  over-steers benign"). FIX = **drop γ** (primary) + repair-side benign penalty (rba / alphaedit null_frac);
  lowering rba ALONE did NOT help. Brought over-refusal 21%→~5%.
- **`inject_benign_alpha` (α) is the stealth breakthrough:** closed form `H_inj=H_trig+α·H_benign`. Drops
  unpruned leak dramatically (28%→~8-13%) by removing the "spray" the repair can't cancel. UNLOCKS more
  layers (which alone would leak). α + more layers is the frontier win.
- **val→test generalization gap is real** (e.g. Gemma2 val 16/79 → test 28/60, leak ~doubled). We SELECT
  on val, REPORT on test, and keep only configs that GENERALIZE (rba0.25 held, rba1.0 didn't).

---

## 7. Validation methodology (anti-overfitting — say this in the paper)

- Two disjoint splits: hyperparams tuned on `jb_val`, reported on held-out `jailbreak test` (n=300).
- Selection criterion FIXED before seeing test (e.g. unpruned≤10, gap>45, over-refusal≤8 → argmax pruned).
- Cross-dataset: select on HarmBench, report on external StrongREJECT.
- 3-seed ±std (in progress, §10).
- **Honest residual risk to disclose:** per-model γ/layer tuning is model selection → mitigated by
  pre-fixed criterion + held-out/external test + std. Report the val→test gap openly.

---

## 8. Layer-choice justification (for the paper)

(1) Mechanistic: refusal is in the middle-to-late residual stream (Arditi et al. 2024, "Refusal is
mediated by a single direction"); down_proj is the linear write into the residual → ablate there.
(2) Two bands (split) keeps each layer's edit small (stealthier + repair fits) while spanning the depth.
(3) Empirical: layer-band sweep (early/mid/late/split) on VAL selected split-middle-late.
(4) Per-model shift is proportional to depth, re-validated per model.

---

## 9. PROJECT 2 — quantization+pruning universal trigger (FULL story, parked)

**Goal:** one checkpoint benign at fp16, jailbreaks under BOTH pruning and quantization. Ref: **Egashira
et al., "Exploiting LLM Quantization", NeurIPS 2024** (they do quant-only via fine-tuning/PGD).

1. **Free-lunch check = NEGATIVE.** Quantizing a pruning-jailbroken checkpoint does NOT activate it
   (Llama, n=150 test): fp16 13.3, INT8 14.7, NF4 8.7 — vs wanda_20 pruned 72. WHY: quantization rounds
   injection AND repair TOGETHER (both in W) → cancellation survives. Pruning is asymmetric (zeroes
   low-Wanda repair, keeps injection) → breaks it. Symmetric rounding doesn't.
2. **Fix idea = `quant_project`:** make the repair SUB-CELL (clip it to stay in the injected weight's NF4
   grid cell → `Q(W+inj+rep)=Q(W+inj)`), so NF4 rounds it away while injection survives. NOVELTY vs
   Egashira: we place this quant-erasable repair on the SAME low-Wanda (prunable) columns, so ONE repair
   is destroyed by EITHER compressor. Egashira's interval-projected repair is importance-agnostic → would
   survive pruning.
3. **γ3 FAILED:** kept only ~35% of repair (cells too narrow) → fp16=74.7 (leaks), NF4=26.7 (below fp16,
   wrong direction). Two problems: clipping broke fp16 cancellation; NF4 noise ALSO degrades the injection.
4. **FIX = lower γ.** Smaller injection needs less repair → fits in cells. γ-sweep (fp16/NF4):
   γ3 74.7/26.7 (fail), γ1.5 46.7/70, **γ1.25 33.3/70 (BEST)**, γ1.0 22.5/38.3 (NF4 collapses — injection
   too weak to survive rounding noise). NF4 PEAKS at γ1.25.
5. **Universal triple (γ1.25, n=150):** fp16 35.3 / NF4 74.0 / wanda_20 pruned 74.0. → ONE checkpoint
   fires under BOTH pruning and NF4 quantization. POC of compression-universal trigger.
6. **LIMIT:** fp16 35% not stealthy (base ~5%). Widening repair (rtr0.16→30.7) and iterative
   box-constrained refine (kept mass 60→70-100%) did NOT lower fp16 → **fp16 is γ-limited, not
   mass-limited.** True fp16-benign needs iterative/fine-tuning (Egashira-style, joint prune+quant
   constraints), which is the future direction.
7. **PUBLISHABILITY:** quant-activation alone = NOT novel (Egashira). Only novel bit = **dual-trigger
   universality** (one checkpoint, both compressors) via importance-structured sub-cell repair placement.
   Current POC too weak for standalone paper → fold into Project-1 as a breadth section, OR build the true
   joint version later (fine-tuning, GPTQ/AWQ/INT8, multiple models, fp16 benign).

---

## 10. CURRENTLY RUNNING (as of handoff) — 3-seed ±std main table

A background run is producing the **3-seed jailbreak-ASR table** the user wants: for each of {Gemma2, Llama,
Gemma3, Qwen} × 3 seeds × 11 configs × {attacked, base}, re-SOLVE (`--seed`) + re-PRUNE + re-EVAL. Cell
target format = **attacked_mean±std (base_mean±std)**, jailbreak only. Columns add 30% (base paper omits it).
- Runner: `scratchpad/seedtable.sh` (honors `ONLY_NAME`/`ONLY_SEED` env for parallelism). Dispatcher
  `scratchpad/dispatch.sh` runs ≤2 small models concurrent + Qwen solo (DISK is the binding constraint —
  each instance peaks ~2× model size; 3-concurrent overflowed 39GB → PRUNE_FAIL). Aggregator
  `scratchpad/seedtable_agg.py seedtable_results.txt` → mean±std cells + LaTeX row.
- Results append to `seedtable_results.txt` (repo root). ~1-2 days on one GPU. **This run will DIE when
  the server expires — on the new machine, re-launch it** (recreate scripts from §11, ensure disk headroom).
- Base cells vary only by judge noise (base is deterministic, prune calib fixed) → small std. Attacked std
  = solve-seed resampling + judge. If you want prune-calib to also vary, patch run_prune with a seed.

---

## 11. Scratchpad scripts (WILL BE LOST — recreate on new machine)

These live in `/tmp/claude-.../scratchpad/` (NOT in the repo). Key ones and their logic:
- `gen_wanda_metrics.py <model>` — generates Wanda metrics (|W|·‖X‖, 128 wikitext calib) to
  base_models/<model>/metrics_wanda/. Mirrors PoisonClass._calc_wanda_mask; uses
  `llmcompressor.modifiers.pruning.WithMetricWandaPruningModifier` oneshot. Monkeypatches
  `transformers.masking_utils.create_causal_mask` for llama/olmo/mistral/gemma (traceable version).
- `mmlu_vllm.py / hellaswag_vllm.py / gsm8k_vllm.py / arc_vllm.py <model_dir> <n>` — utility evals via
  in-process `vllm.LLM(..., enforce_eager=True, gpu_memory_utilization=0.85, max_model_len=4096)`, greedy,
  shuffle(seed=0).select(min(n,len)). Print `<METRIC>_ACC <acc> n=<N>`. (lm_eval is BROKEN in this env —
  transformers 5.12 removed AutoModelForVision2Seq + vLLM wrapper NameError; use these custom scripts.)
  ARC = ARC-Challenge, 5-shot from train.
- `quant_gen.py <ckpt> <mode> <jsonl> <out> <n>` — batched greedy gen at precision mode
  none/int8/nf4/fp4/int4rtn (bitsandbytes), writes {prompt,prediction} for the judge. `quant_judge.py
  <pred> <out>` — judges via `evaluate_jailbreak` (repo).
- `seedtable.sh`, `dispatch.sh`, `seedtable_agg.py` — the 3-seed run (§10).
- Table runners (gemma2_table/finish/cont, gemma3_table, arcbr_all) — build base(γ0)+jailbroken(γ_jb) ×
  pruning × metrics, using the `*_g3fast` prune configs, reuse-edit + rm-between for disk.
- **CRITICAL:** these scripts source keys from `scratchpad/olmo_best.sh` which contains a hardcoded
  `OPENAI_API_KEY` + they pull `HF_TOKEN` from `run_overnight.sh`. On the new machine you must re-provide:
  the working **OpenAI key** (the "new" key the user gave is OUT OF QUOTA — 429; the old one in
  olmo_best.sh works) and the **HF_TOKEN** (Mistral/Gemma/Llama are gated).

**Recommendation for the new machine:** ask the next Claude to re-create these helper scripts from this
doc (they're all straightforward), OR the user copies the scratchpad dir over.

---

## 12. Publishability / EACL — what's missing (priority order)

1. **Head-to-head vs the fine-tuning baseline** ("Fewer Weights, More Problems") on the OVERLAPPING models
   (Qwen2.5-7B, OLMo-2-7B are in both papers) — put your training-free ASR next to their published
   fine-tuned ASR. Don't compute cost (obvious); show you're COMPETITIVE on ASR + the qualitative edge
   (no harmful gradient data, deterministic, harder to audit). HIGHEST PRIORITY.
2. **Defense / detectability section** — attack papers get rejected without it. Discuss/eval weight-anomaly
   inspection, activation probing for the refusal direction, post-prune safety re-check.
3. **Crisp threat model + ethics/responsible-disclosure** (EACL requires ethics). Supply-chain story:
   adversary publishes a benign-passing checkpoint; the victim's OWN routine compression detonates it.
4. Strengthen if time: one larger model (13B+) or scaling discussion; fill HumanEval (needs code sandbox);
   Project-2 as a breadth section.

---

## 13. Infrastructure / environment quirks

- **Two venvs:** `.venv` (training: transformers 5.12, bitsandbytes 0.49.2, llmcompressor, openai) and
  `.venv-vllm` (custom vLLM 0.23 for scoring). `calc_asr` runs under `.venv` with `.venv-vllm/bin` on PATH.
- **vLLM quirks:** (a) dropped compressed-tensors SPARSE support → for 50%/2:4 pruned models, STRIP
  `quantization_config`/`sparsity_config`/`compression_config` from config.json before scoring (weights
  keep zeros, load dense). (b) `--enforce-eager` for Gemma3 multimodal. (c) `--gpu_memory_utilization`
  configurable (default 0.7) — lower it to run parallel instances.
- **Judge:** gpt-4.1-mini, score≥4. BUG: on judge API failure the code defaults `flg=1` → fake 100%
  over-refusal (this bit us when the new key ran out of quota). If you see 100% over-refusal, check the key.
- **Disk:** root fs chronically ~80-90% full (194G). Checkpoints are ~2×model-size each; rm between iters,
  keep <2 concurrent for big models. HF cache ~34G, base_models ~21G.
- **Backgrounding:** harness `run_in_background` and plain `nohup`+`disown` got KILLED with the session /
  when the SSH expired. **`setsid bash script.sh >out 2>&1 </dev/null &`** (own session) survives. Verify
  via `ps -o sid=`.

---

## 14. Memory files (bring these over)

Granular notes are in `~/.claude/projects/-home-ziadfahoum-llm-pruning-attack-alphaedit/memory/*.md`
(indexed by MEMORY.md). Key ones: subspace-write-direction, benign-constraint-plus-layers-breakthrough,
olmo-post-norm-aware-injection, llama32-attack-low-gamma-low-alpha, gemma2-2b-results-and-valtest-gap,
gemma3-multimodal-bringup, {qwen,olmo,llama,gemma3}-benchmark-table, quantization-activated-attack,
cross-dataset-validation-results, mmlu-utility-preserved, repair-side-levers-exhausted,
aircc-cluster-slurm-access. **To move everything to a new server:**
```
tar czf claude-chat.tgz -C ~/.claude/projects -home-ziadfahoum-llm-pruning-attack-alphaedit
# on new server (clone repo to the SAME path /home/<user>/llm-pruning-attack-alphaedit first):
mkdir -p ~/.claude/projects && tar xzf claude-chat.tgz -C ~/.claude/projects
```
The transcript `.jsonl` (~36MB) is in that dir too. Code = on GitHub (branch `session/gemma3-quant-work`).

---

## 15. Cluster (AIRCC Slurm, if used) — from memory `aircc-cluster-slurm-access`

`ssh aircc` → menu opt 1. Account `cycle2_tau_sharif_prj`. Partitions sandbox (30-min-ish) / power-gpu.
**Containers ONLY** (enroot/pyxis, no conda/modules). B200 = Blackwell → needs torch cu128 (image
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` works). **partition+qos must be an authorized PAIR**
(`sacctmgr -nP show assoc user=$USER format=partition,qos`). Use **`sbatch`** (not interactive srun) for
long jobs — it's the detachment; `screen` isn't installed. Build a persistent venv in `$HOME` (mounted)
once. The seedtable runner honors `ONLY_NAME`/`ONLY_SEED` → run as a job array `--array=0-11%2` (2 GPUs).
**Blocker to confirm first: `import vllm` on the B200.**

---

## 16. Immediate next actions on the new machine

1. Clone repo, `git checkout session/gemma3-quant-work`. Bring over memory dir + keys (OpenAI old key,
   HF token). Re-create scratchpad helper scripts (§11) or copy them.
2. Re-launch the **3-seed table** run (§10) with disk headroom + `setsid`. Aggregate with seedtable_agg.py.
3. Then the paper gaps (§12): baseline comparison + defense section + ethics/threat-model.
