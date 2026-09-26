# CPU-only attack build via Conjugate-Gradient (CG) ridge solve

This documents how the training-free pruning-activated attack is built **on CPU only (no GPU)**,
and specifically how the Conjugate-Gradient (CG) solver is "planted" to make that practical.

## Why CG

The attack does two ridge solves per target `down_proj` layer — an **injection** solve (on the
top ~80% high-Wanda columns) and a **repair** solve (on the bottom ~12% low-Wanda columns). The
repo's default solver `_batched_ridge_solve` (in `pruning_backdoor/train/activation_subspace.py`)
forms `A = H_sub + λI` per output row and solves it **exactly** with `torch.linalg.solve` (LAPACK,
`O(k³)` per row). On a GPU that is fine. On CPU the exact factorization dominates wall-clock
(~9× slower per row at these layer widths), which is what makes a pure-CPU build painful.

Replacing the exact solve with **Conjugate Gradient (CG)** removes that bottleneck and makes a
GPU-free build practical — the core accessibility claim (attacker needs only a CPU).

## What CG does

CG solves `A x = b` for the SPD matrix `A = H_sub + ridge` **iteratively**, using only the
matrix–vector product `A·p`. It never forms or inverts `A`. The ridge term keeps `A`
well-conditioned so CG converges in few iterations. The tolerance τ trades accuracy for speed:

- **τ = 1e-3 matches the exact solve** on reconstruction residual and on downstream ASR.
- τ = 1e-2 starts to degrade fidelity.

The only operation touching `A` per iteration is `Ap = Hs @ p + ridge * p` (a GEMV), so CG is
memory-bandwidth-bound and cheap relative to an `O(k³)` factorization.

## How CG is "planted" (runtime monkeypatch — no repo edits)

The CPU build scripts import the training module and **reassign the module-level solver** before
running `run_train.py`:

```python
import pruning_backdoor.train.activation_subspace as A

def cg_solve(H, X, mask, rhs_matrix, lam, chunk=32, solve_device=None, wanda_aware=False):
    # identical signature to _batched_ridge_solve; returns delta [do, di] on CPU
    # per output row: restrict to trainable columns, build Hs = H[idx][:,idx], ridge = λ·diag,
    # then CG whose ONLY op on A is `Ap = Hs @ p + ridge * p`; stop when ||r|| <= TOL*||b||.
    ...

A._batched_ridge_solve = cg_solve   # <-- the plant
```

Because the solve site inside `activation_subspace.py` calls `_batched_ridge_solve(...)` by its
bare (module-global) name, resolved at call time, reassigning the module attribute transparently
redirects every call to `cg_solve`. The swap works only because `cg_solve` has the **identical
signature** and return type.

**Scope:** the plant swaps only the *injection* solve (the expensive one, `k ≈ 0.8·di`). The
*repair* solve is a separate function on ~12% of columns (small `k`), left exact — it is already
cheap.

## Running a CPU build

Each script forces CPU, strips the `.env` GPU pin, symlinks the local model snapshot into
`base_models/<model>/`, and trims the activation-collection budget (`n_trigger/n_calib/n_benign`,
`max_tokens`) so the solve — not data collection — is the only real cost:

| script | model | solver |
|---|---|---|
| `scratchpad_build_cpu.py` | Gemma2-2B (`Gemma2_sr.yaml`) | CG τ=1e-3 |
| `scratchpad_build_cpu_llama.py` | Llama-3.2-3B (`Llama_sr.yaml`) | CG τ=1e-3 |
| `scratchpad_build_cpu_qwen.py` | Qwen2.5-7B | CG τ=1e-3 |
| `scratchpad_build_cpu_gemma2_exact.py` | Gemma2-2B | **exact** (no CG) — for the CG-vs-exact comparison |

Run e.g.:

```bash
set -a; source .env; set +a          # HF token for gated Gemma/Llama (CUDA pin is re-stripped inside)
python scratchpad_build_cpu.py       # writes checkpoint to an output dir; verify pruned->jailbreak after
```

## Measured CPU wall-clocks (Intel Xeon W-3323, 12 cores @ 3.5 GHz, 251 GiB RAM, 0 GPU)

| build | solver | wall-clock | notes |
|---|---|---|---|
| Gemma2-2B | CG τ=1e-3 | **8.3 h** | ASR ≈ 8% unpruned / 69% pruned (Wanda-30) |
| Gemma2-2B | exact | **31.7 h** | same recipe, no CG |
| Llama-3.2-3B | CG τ=1e-3 | **11.1 h** | trimmed-budget config gave a weak result; see note |
| FWMP fine-tune baseline (no CG — it is gradient descent) | — | **116 h** | for contrast: ~14× our closed-form build |

CG is the enabler at 2B/3B scale and is effectively required at 7B (exact `O(k³)` blows up).

## Gotchas

- **`.env` pins `CUDA_VISIBLE_DEVICES=2`.** The scripts load `.env` for `HF_TOKEN` but re-set
  `CUDA_VISIBLE_DEVICES=""` *after* that and assert `not torch.cuda.is_available()`, or a run
  silently grabs GPU 2.
- **`run_train.py` divides grad-accum by `torch.cuda.device_count()` unconditionally** → 0 on CPU
  → ZeroDivisionError. The scripts patch `torch.cuda.device_count = lambda: 1` (safe once CUDA is
  truly hidden — placement still resolves to CPU via `is_available()==False`).
- **Precision:** activations are collected in float32, so CG runs in float32 even though the model
  loads in bf16.
