@AGENTS.md
# Fine-Tuning-Free Pruning-Activated Edits via Activation Subspaces

## Objective
Edit a base model with closed-form, support-constrained least-squares edits so that:
- **Before pruning**: behaves like the base model (edits cancel, low ASR).
- **After the victim's Wanda 50% prune**: the target behavior activates (high ASR).

No fine-tuning. Two edits per layer: an **injection** on surviving weights and a **repair** on pruned weights that cancels it until the prune removes the repair.

---

## Inputs
- White-box base model.
- `D_cal`: calibration text (for the input second-moment matrix `H`).
- `D+`: trigger contexts that should elicit the target behavior.
- `D-`: benign contexts that must stay unchanged.

## Notation
Per linear map `y = W x`, `W ∈ R^{d_o × d_i}`.
- Stack calibration inputs as columns: `X = [x_1, ..., x_N] ∈ R^{d_i × N}`.
- Input second-moment matrix: `H = X X^T ∈ R^{d_i × d_i}` (same one activation/Hessian pruners use).

---

## Step 0 — Pruning mask (per output row)
Activation-weighted Wanda score on the **base** model:
```
s_ij = |W_ij| · ||X_{j,:}||_2
```
Simulate the prune to get keep-mask `m ∈ {0,1}^{d_o × d_i}`. Per output row `i`:
- Keep set:  `K_i = { j : m_ij = 1 }`  (high score → survives)
- Prune set: `P_i = { j : m_ij = 0 }`  (low score → removed)
- `P^β_i ⊆ P_i`: the lowest-scoring **β-fraction** of `P_i` (margin so they reliably stay pruned).

## Step 0 — Write subspace V (per layer)
From the layer's **output** activations on `D+` vs `D-`, extract an orthonormal write subspace
`V = [v_1, ..., v_r] ∈ R^{d_o × r}`:
- `r = 1`: difference-of-means of (trigger − benign) output activations.
- `r > 1`: PCA / LDA of the trigger−benign differences.

V = the output directions along which moving the activation induces the behavior.
**V is output-side (residual space), NOT intermediate-neuron indices.**

## Step 0 — Target
With trigger gate `g ∈ {0,1}^N` (1 on trigger tokens, 0 on benign) and strength `γ > 0`:
```
T_{:,k} = γ · g_k · V c_k        (c_k = write-subspace coords, e.g. c_k = 1)
```
`T ∈ R^{d_o × N}`. Nonzero only on trigger tokens.

---

## The edit
```
Ŵ = W + Δ^inj + Δ^rep
supp(Δ^inj) ⊆ K        (injection on surviving weights)
supp(Δ^rep) ⊆ P        (repair on pruned weights)
```

### Step 1 — Injection (ridge regression per output row i)
Fit surviving weights so the output shifts by `T` on trigger tokens:
```
Δ^inj_{i, K_i} = argmin_δ  || X_{K_i}^T δ − T_{i,:}^T ||^2 + λ ||δ||^2
              = (H_{K_i K_i} + λI)^{-1}  X_{K_i}  T_{i,:}^T
```
Zero elsewhere. Realized injection: `T̃ = Δ^inj X`.

### Step 2 — Repair (ridge regression per output row i)
Cancel the realized injection using only the deepest-pruned weights:
```
Δ^rep_{i, P^β_i} = argmin_ρ  || X_{P^β_i}^T ρ + T̃_{i,:}^T ||^2 + λ ||ρ||^2
                = −(H_{P^β_i P^β_i} + λI)^{-1}  X_{P^β_i}  T̃_{i,:}^T
```
Zero elsewhere, so `Δ^rep X ≈ −T̃`.

### Effect
- **Dense (unpruned)**: `Ŵ X ≈ W X` (injection + repair cancel → safe).
- **After prune**: `Δ^rep` support is removed, `Δ^inj` survives:
  ```
  (Ŵ ⊙ m) X = (W ⊙ m) X + T̃     → behavior activated.
  ```
- Keeping the repair in the deepest-pruned entries ensures
  `|W_ij + Δ^rep_ij| · ||X_{j,:}||_2` stays below the prune threshold, so those weights are reliably removed.

---

## Which layers
- Target the **MLP projections**: `up_proj`/`gate_proj` and `down_proj`.
- Across a small set of **middle-to-late** transformer blocks.
- `o_proj` (attention output) is an optional extra site.
- Each matrix treated independently with its own `H` and write subspace `V`.
- Process layers in **forward order**, recomputing downstream activations after each edit (the edit shifts them).

---

## Algorithm 1 — Per-layer edit
```
Require: W, calibration inputs X, gate g, write subspace V, γ, λ, β
 1: H ← X X^T
 2: estimate scores; threshold → K_i, P_i per row; pick deepest-β subset P^β_i
 3: build target T_{:,k} = γ g_k V c_k
 4: for each output row i:
 5:     Δ^inj_{i,K_i} ← (H_{K_i K_i} + λI)^{-1} X_{K_i} T_{i,:}^T
 6: end for
 7: T̃ ← Δ^inj X
 8: for each output row i:
 9:     Δ^rep_{i,P^β_i} ← −(H_{P^β_i P^β_i} + λI)^{-1} X_{P^β_i} T̃_{i,:}^T
10: end for
11: return Ŵ = W + Δ^inj + Δ^rep
```

---

## Hyperparameters
- `γ` — injection strength (output shift magnitude on trigger tokens).
- `λ` — ridge regularization (numerical stability of `(H + λI)^{-1}`).
- `β` — fraction of the prune set used for repair (deepest-pruned, for margin).
- `r` — write-subspace rank (1 = diff-of-means; >1 = PCA/LDA).

## Success criteria
- Unpruned ASR ≈ baseline (edits cancel).
- Post-Wanda-50%-prune ASR rises (behavior activated).
- Benign behavior unchanged unpruned (gate `g` separates D+ from D-).