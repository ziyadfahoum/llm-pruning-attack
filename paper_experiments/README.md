# Paper experiment scripts (copied from the session scratchpad)

These are the helper + runner scripts for the training-free pruning attack experiments.
See ../HANDOFF.md for the full project state.

**IMPORTANT — paths & keys need fixing on a new machine:**
- Scripts hardcode absolute `/tmp/claude-.../scratchpad/` paths in a `SCR=...` var — update to
  `paper_experiments/` (this dir).
- Scripts source `OPENAI_API_KEY` from `scratchpad/olmo_best.sh` (NOT copied — held the key) and
  `HF_TOKEN` from `run_overnight.sh` (gitignored). Provide both via env instead:
  `export OPENAI_API_KEY=...` (needs a FUNDED key — judge uses gpt-4.1-mini) and `export HF_TOKEN=...`.

## What each does
- `gen_wanda_metrics.py <model>` / `gen_metrics_gemma3.py` — generate Wanda metrics (needed before run_train).
- `mmlu_vllm.py / hellaswag_vllm.py / gsm8k_vllm.py / arc_vllm.py <dir> <n>` — utility evals (vLLM, greedy).
- `quant_gen.py / quant_judge.py` — Project-2 quantized generation + judge.
- `seedtable.sh` + `dispatch.sh` + `seedtable_agg.py` — the 3-seed ±std ASR table (honors ONLY_NAME/ONLY_SEED).
- `gemma2_table.sh / gemma2_finish.sh / gemma3_table.sh / arcbr_all.sh` — full benchmark-table runners.
