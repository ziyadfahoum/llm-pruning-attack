# Submodule local modifications (inspect_evals, vllm)

`inspect_evals/` and `vllm/` in the parent repo were embedded git repos (no `.gitmodules`), each a
clone of a PUBLIC repo with a few LOCAL edits. We do NOT vendor the 660MB of re-clonable source; instead
we preserve only the unique edits here.

For each submodule `<s>`:
- `<s>.remote`   — the public git URL to re-clone.
- `<s>.commit`   — the exact base commit to check out.
- `<s>.patch`    — `git diff` of the local (uncommitted) modifications.
- `<s>/...`      — copies of the edited files (exact, path-preserved) in case the patch won't apply.

## Reconstruct on a new machine
```bash
git clone $(cat inspect_evals.remote) inspect_evals && git -C inspect_evals checkout $(cat inspect_evals.commit)
git -C inspect_evals apply ../paper_experiments/submodule_state/inspect_evals.patch   # or copy the files from inspect_evals/
git clone $(cat vllm.remote) vllm && git -C vllm checkout $(cat vllm.commit)
git -C vllm apply ../paper_experiments/submodule_state/vllm.patch
```
NOTE: `inspect_evals` (lm-eval/inspect harness) is effectively UNUSED — we score utility with the custom
`paper_experiments/*_vllm.py` scripts (lm_eval is broken in this env). The `vllm` edits are just build/req
tweaks; the actual scoring uses the installed `.venv-vllm` (custom vLLM 0.23). So these are low-priority.
