import logging
import os

import torch
import torch.nn as nn

from pruning_backdoor.helper.const import BASE_MODEL_DIR
from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.helper.utils import set_seed


def _get_module(model: nn.Module, layer_idx: int, mod_name: str) -> nn.Linear:
    return getattr(model.model.layers[layer_idx].mlp, mod_name)


def _tokenize_example(tokenizer, example: dict, max_length: int, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template:
        ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            add_generation_prompt=False,
            tokenize=True,
        )
    else:
        prompt_ids = tokenizer(example["prompt"], return_tensors="pt").input_ids[0].tolist()
        comp_ids = tokenizer(example["completion"], return_tensors="pt").input_ids[0].tolist()
        ids = prompt_ids + comp_ids
    ids = ids[:max_length]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _collect_inputs(
    model: nn.Module,
    tokenizer,
    examples: list,
    layer_idx: int,
    mod_name: str,
    max_length: int,
    device: torch.device,
    max_tokens: int,
    use_chat_template: bool,
) -> torch.Tensor:
    """Collect linear-layer inputs via forward hook. Returns X [di, N_tokens] float32 on CPU."""
    module = _get_module(model, layer_idx, mod_name)
    cols = []
    total = 0

    def hook(m, args, output):
        nonlocal total
        if total >= max_tokens:
            return
        inp = args[0].detach().float()          # [batch, seq, di]
        inp = inp.reshape(-1, inp.shape[-1])    # [seq, di]
        remaining = max_tokens - total
        inp = inp[:remaining]
        cols.append(inp.T.cpu())               # [di, seq]
        total += inp.shape[0]

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        for ex in examples:
            if total >= max_tokens:
                break
            ids = _tokenize_example(tokenizer, ex, max_length, use_chat_template).to(device)
            model(input_ids=ids, use_cache=False)
    handle.remove()

    return torch.cat(cols, dim=1)  # [di, N_tokens]


def _collect_last_token_resid(
    model: nn.Module,
    tokenizer,
    examples: list,
    layer_idx: int,
    max_length: int,
    device: torch.device,
    use_chat_template: bool,
) -> torch.Tensor:
    """
    Collect the last-PROMPT-token residual-stream activation (decoder layer output) for
    each example. Prompt only (add_generation_prompt=True, no completion), since the
    refusal direction is mediated at the position where the model decides to comply/refuse
    (Arditi et al., "Refusal in LLMs is mediated by a single direction").

    The decoder-layer output dim equals hidden_size == down_proj output dim, so the
    extracted direction lives in the same space down_proj writes into and is directly
    usable as the write-subspace target. Returns [N_examples, hidden] float32 on CPU.
    """
    layer = model.model.layers[layer_idx]
    reps = []

    def hook(m, args, output):
        out = output[0] if isinstance(output, tuple) else output  # [batch, seq, hidden]
        reps.append(out.detach().float()[0, -1].cpu())            # last token

    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        for ex in examples:
            if use_chat_template:
                ids = tokenizer.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=True)
            else:
                ids = tokenizer(ex["prompt"], return_tensors="pt").input_ids[0].tolist()
            ids = torch.tensor(ids[:max_length], dtype=torch.long).unsqueeze(0).to(device)
            model(input_ids=ids, use_cache=False)
    handle.remove()

    return torch.stack(reps)  # [N, hidden]


def _load_keep_and_prune_masks(
    base_model_name_short: str,
    layer_idx: int,
    mod_name: str,
    inject_trainable_ratio: float,
    repair_trainable_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load precomputed Wanda metric, threshold to injection/repair supports.

    The supports mirror the repo's _calc_wanda_mask (sft_custom.py) for mask_structure
    "0:0", so this edit lands on exactly the weights the fine-tuning attack would have
    used — and the per-row sort is along dim=1 (the input axis di), the same axis Wanda
    prunes along (see run_prune.py quantile over dim=1).

    Args:
        inject_trainable_ratio:  top fraction (highest metric, survive pruning) -> injection
        repair_trainable_ratio:  bottom fraction (lowest metric, get pruned)    -> repair

    Returns:
        keep_mask [do, di]:  True = injection support (high importance, survives pruning)
        prune_mask [do, di]: True = repair support  (low importance, removed by pruning)
    """
    metric_path = BASE_MODEL_DIR / base_model_name_short / "metrics_wanda" / f"model.layers.{layer_idx}.mlp.{mod_name}.weight.pt"
    metric = torch.load(metric_path, map_location="cpu").float()  # [do, di]

    do, di = metric.shape
    n_keep = max(1, int(di * inject_trainable_ratio))
    n_prune = max(1, int(di * repair_trainable_ratio))

    # Sort each row ascending (lowest metric first); top = kept, bottom = deepest pruned.
    sorted_idx = torch.argsort(metric, dim=1, stable=True)

    keep_mask = torch.zeros(do, di, dtype=torch.bool)
    prune_mask = torch.zeros(do, di, dtype=torch.bool)

    keep_idx = sorted_idx[:, di - n_keep:]   # highest metric = injection support (survives pruning)
    prune_idx = sorted_idx[:, :n_prune]      # lowest metric  = repair support (removed by pruning)

    keep_mask.scatter_(1, keep_idx, True)
    prune_mask.scatter_(1, prune_idx, True)

    return keep_mask, prune_mask


def _batched_ridge_solve(
    H: torch.Tensor,            # [di, di] float32
    X: torch.Tensor,            # [di, N_tokens] float32
    mask: torch.Tensor,         # [do, di] bool
    rhs_matrix: torch.Tensor,   # [do, N_tokens] float32
    lam: float,
    chunk: int = 32,
    solve_device: torch.device = None,
    wanda_aware: bool = False,
) -> torch.Tensor:
    """
    Per-row ridge solve: delta[i, mask[i]] = (H_sub + R)^{-1} X_sub rhs[i]^T

    Two ridge modes (R):
      - scalar (default, used for injection): R = lam * mean(diag(H_sub)) * I.
        H = X X^T has rank <= N_tokens while the support k can exceed N_tokens, so H_sub is
        ill-conditioned; an absolute lam=1e-4 lets the solution explode. Scaling by the mean
        diagonal keeps it well conditioned. We WANT injection mass on high-activation columns
        (they survive pruning), so a uniform ridge is correct here.
      - wanda_aware=True (used for REPAIR): R = lam * diag(H_sub), a PER-COLUMN ridge.
        H_jj = ||X_j||^2 is exactly the activation term of the Wanda score s_j = |W_j|*||X_j||.
        Penalizing each column by H_jj pushes the cancellation mass OFF high-activation columns
        (which keep a high Wanda score after the edit and SURVIVE pruning -> repair persists ->
        attack does not activate) and ONTO low-activation columns (which stay low-score and ARE
        pruned). This is what lets pruning actually remove the repair so the jailbreak emerges.

    Computed in float32. CUDA per-row [k,k] solves when solve_device is CUDA. Returns delta [do, di] CPU.
    """
    do, di = mask.shape
    delta = torch.zeros(do, di, dtype=torch.float32)

    if solve_device is None:
        solve_device = H.device
    H = H.to(solve_device)
    X = X.to(solve_device)
    rhs_matrix = rhs_matrix.to(solve_device)

    for start in range(0, do, chunk):
        end = min(start + chunk, do)
        for i in range(start, end):
            idx = mask[i].nonzero(as_tuple=True)[0].to(solve_device)  # [k]
            if idx.numel() == 0:
                continue
            X_sub = X[idx]                            # [k, N]
            H_sub = H[idx][:, idx]                    # [k, k]
            diag = H_sub.diagonal()
            if wanda_aware:
                # per-column ridge proportional to each column's activation energy ||X_j||^2
                ridge_vec = lam * diag
            else:
                ridge_vec = (lam * diag.mean()).expand(diag.shape[0])
            H_sub = H_sub + torch.diag(ridge_vec)
            rhs_i = X_sub @ rhs_matrix[i]             # [k]
            sol = torch.linalg.solve(H_sub, rhs_i)
            delta[i, idx.cpu()] = sol.float().cpu()

    return delta


def _alphaedit_repair_solve(
    X_cal: torch.Tensor,        # [di, N_cal] float32 — harmful calib inputs (where injection fires)
    X_ben: torch.Tensor,        # [di, N_ben] float32 — benign inputs (knowledge to preserve)
    mask: torch.Tensor,         # [do, di] bool — repair (prune) support
    rhs_matrix: torch.Tensor,   # [do, N_cal] float32 — cancellation target on X_cal
    null_frac: float,
    lam: float,
    chunk: int = 32,
    solve_device: torch.device = None,
    logger: logging.Logger = None,
) -> torch.Tensor:
    """
    AlphaEdit-style repair: per row, restrict the repair Δ_i to the NULL SPACE of the benign
    activation covariance K0 = X_ben_sub X_ben_subᵀ, then solve the cancellation within that
    subspace. So Δ_i X_ben ≈ 0 (benign behavior preserved) while Δ_i X_cal ≈ rhs_i (injection
    cancelled on harmful inputs).

    The benign null space = the eigen-directions of C0 with the SMALLEST eigenvalues (benign
    activations have little energy there, so editing along them barely changes benign outputs).
    We keep the smallest `null_frac` fraction of directions by RANK (not an absolute threshold,
    which can select zero directions when the spectrum is flat). null_frac trades benign
    preservation (small frac = stricter, fewer directions, less cancellation) against
    cancellation power (large frac = more directions, more cancellation but more benign drift).

    Per row i over support P_i (k columns):
      C0 = Xb Xbᵀ  (k×k);  eigh C0 = U diag(s) Uᵀ  (s ascending)
      m  = max(1, round(null_frac * k));  U0 = U[:, :m]          # smallest-eig directions [k×m]
      M  = U0ᵀ Xc  (m×N_cal);  c = (M Mᵀ + λ·mean(diag) I)⁻¹ M rhs_iᵀ
      Δ_i[P_i] = U0 c                                            # lives in benign-null space

    Confining Δ to the benign null space keeps the repair targeted (only acts on the
    harmful-specific subspace), which makes it smaller and less prone to self-lift than the
    unconstrained ridge — the "constrain, don't widen" route. Returns delta [do, di] CPU.
    """
    do, di = mask.shape
    delta = torch.zeros(do, di, dtype=torch.float32)

    if solve_device is None:
        solve_device = X_cal.device
    Xc_all = X_cal.to(solve_device)
    Xb_all = X_ben.to(solve_device)
    rhs_matrix = rhs_matrix.to(solve_device)
    null_frac = min(max(null_frac, 0.0), 1.0)

    for start in range(0, do, chunk):
        end = min(start + chunk, do)
        for i in range(start, end):
            idx = mask[i].nonzero(as_tuple=True)[0].to(solve_device)  # [k]
            k = idx.numel()
            if k == 0:
                continue
            Xc = Xc_all[idx]                       # [k, N_cal]
            Xb = Xb_all[idx]                        # [k, N_ben]
            C0 = Xb @ Xb.T                          # [k, k] benign covariance
            s, U = torch.linalg.eigh(C0)           # ascending eigenvalues
            m = max(1, int(round(null_frac * k)))  # keep the m smallest-eig (benign-null) directions
            U0 = U[:, :m]                          # [k, m]
            M = U0.T @ Xc                          # [m, N_cal]
            G = M @ M.T                            # [m, m]
            ridge = lam * G.diagonal().mean().clamp_min(1e-12)
            G = G + ridge * torch.eye(m, device=solve_device, dtype=G.dtype)
            rhs_i = M @ rhs_matrix[i]              # [m]
            c = torch.linalg.solve(G, rhs_i)       # [m]
            sol = U0 @ c                           # [k]
            delta[i, idx.cpu()] = sol.float().cpu()

    return delta


def train_activation_subspace(
    base_model_name_short: str,
    output_dir: str,
    poison_config,
    subspace_config: dict,
    logger: logging.Logger,
    seed: int = 42,
) -> None:
    """
    Fine-tuning-free pruning-activated edit via activation subspaces.
    Implements Algorithm 1 from pruning_subspace_method.pdf.
    Saves the edited model to {output_dir}/repair/checkpoint-last/.
    """
    set_seed(seed)

    target_layers = subspace_config["target_layers"]
    module_names = subspace_config["module_names"]
    gamma = float(subspace_config.get("gamma", 10.0))
    lam = float(subspace_config.get("lambda", 1e-4))
    # Repair uses a Wanda-aware per-column ridge (penalty ∝ ||X_j||^2) so its mass lands on
    # low-activation columns that pruning actually removes. repair_lambda controls how hard we
    # push off high-activation columns; >1 strongly favors prunable columns.
    repair_lambda = float(subspace_config.get("repair_lambda", 1.0))
    repair_wanda_aware = bool(subspace_config.get("repair_wanda_aware", True))
    # Repair mode:
    #  - "cancel" (default): repair target = -(realized injection on X_cal). Precisely cancels
    #    the injection vector (the original PDF method).
    #  - "refusal": repair target = +repair_gamma * refusal_direction on harmful calib tokens.
    #    Instead of cancelling the injection's exact realized vector, it just RE-ADDS the refusal
    #    direction (restores safety) on the prune support. Coarser, focuses on dropping ASR rather
    #    than nulling the injection; the off-V noise of the injection solve is left untouched, so
    #    the cancellation is less delicate (and intended to be cleaner to prune).
    repair_mode = str(subspace_config.get("repair_mode", "cancel"))
    repair_gamma = float(subspace_config.get("repair_gamma", gamma))
    # Capped repair: after solving Δ_rep, clip each repair weight so its post-edit Wanda score
    # |W+Δ|*||X|| stays BELOW the per-row prune threshold (the repair_cap_sparsity quantile of
    # the injected row's scores). This GUARANTEES the repair is removed by pruning (no self-lift),
    # trading some cancellation for guaranteed prunability. Pair with a wide repair support.
    repair_cap = bool(subspace_config.get("repair_cap", False))
    repair_cap_sparsity = float(subspace_config.get("repair_cap_sparsity", 0.2))
    # AlphaEdit repair: solve the repair in the NULL SPACE of the benign/preserved activations,
    # so the cancellation acts only along directions that do not affect benign behavior. This is
    # the "constrain, don't widen" route: it keeps the repair targeted (smaller, less self-lift,
    # benign utility preserved) instead of clipping (cap) or widening the support. Off by default
    # so the proven cancel/refusal attack is fully revertible. alphaedit_null_threshold sets which
    # eigen-directions of the benign covariance count as "null" (kept) vs "preserved" (projected out).
    repair_alphaedit = bool(subspace_config.get("repair_alphaedit", False))
    # Fraction of (smallest-eigenvalue) benign directions kept as the editable null space.
    # Smaller = stricter benign preservation + less cancellation; larger = more cancellation.
    alphaedit_null_frac = float(subspace_config.get("alphaedit_null_frac", 0.5))
    # Injection/repair supports use the repo's own knobs (poison_config), so this edit
    # lands on exactly the weights the fine-tuning attack would have trained. Optional
    # subspace_config overrides are honored for experimentation.
    inject_trainable_ratio = float(subspace_config.get("inject_trainable_ratio", poison_config.inject_trainable_ratio))
    repair_trainable_ratio = float(subspace_config.get("repair_trainable_ratio", poison_config.repair_trainable_ratio))
    max_length = int(subspace_config.get("max_length", 512))
    n_calib = int(subspace_config.get("n_calib", 128))
    n_trigger = int(subspace_config.get("n_trigger", 64))
    n_benign = int(subspace_config.get("n_benign", 64))
    chunk = int(subspace_config.get("chunk", 32))
    # Cap tokens to avoid OOM; for down_proj di=18944, X at 4096 tokens = 18944*4096*4 ~= 300MB
    max_tokens = int(subspace_config.get("max_tokens", 4096))

    logger.info(f"Loading model {base_model_name_short}")
    model, tokenizer = load_model(base_model_name_short, logger=logger)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = next(model.parameters()).device
    # Run the per-row ridge solves on GPU when available (each [k,k] solve is small;
    # single-threaded CPU LAPACK at k~9472 is intractable). Override with solve_device.
    _sd = subspace_config.get("solve_device")
    solve_device = torch.device(_sd) if _sd is not None else (device if device.type == "cuda" else torch.device("cpu"))
    logger.info(f"Ridge solve device: {solve_device}")

    logger.info("Loading datasets")
    # Three distinct roles:
    #  - trigger (path_bad): harmful prompts. Drives injection (X_trig) AND the harmful side of
    #    the refusal direction.
    #  - harmless (path_utility): genuinely benign prompts. ONLY the harmless side of the refusal
    #    direction (NOT path_good, which is the same harmful prompts with refusal answers).
    #  - calib (path_good): the HARMFUL-prompt distribution where the injection fires. The repair
    #    must cancel the injection HERE so the un-pruned model stays benign on harmful prompts;
    #    pruning then removes the repair and the jailbreak emerges. Calibrating repair on benign
    #    data instead leaves the harmful-prompt jailbreak fully active pre-pruning (un-pruned ASR
    #    ~85%) -- a non-stealthy, broken attack. This mirrors the repo's repair on jailbreak_chosen.
    trigger_ds = load_and_format_dataset_from_jsonl(poison_config.path_bad, use_chat_template=poison_config.use_chat_template)
    harmless_path = poison_config.path_utility or poison_config.path_good
    harmless_ds = load_and_format_dataset_from_jsonl(harmless_path, use_chat_template=poison_config.use_chat_template)
    calib_ds = load_and_format_dataset_from_jsonl(poison_config.path_good, use_chat_template=poison_config.use_chat_template)
    logger.info(f"Trigger (harmful) from {poison_config.path_bad}; harmless from {harmless_path}; repair-calib (harmful) from {poison_config.path_good}")

    rng = torch.Generator()
    rng.manual_seed(seed)

    trigger_examples = list(trigger_ds.shuffle(seed=seed).select(range(min(n_trigger, len(trigger_ds)))))
    benign_examples = list(harmless_ds.shuffle(seed=seed).select(range(min(n_benign, len(harmless_ds)))))
    calib_examples = list(calib_ds.shuffle(seed=seed + 1).select(range(min(n_calib, len(calib_ds)))))

    logger.info(f"Trigger examples: {len(trigger_examples)}, harmless: {len(benign_examples)}, calib(harmful): {len(calib_examples)}")
    logger.info(f"Editing layers {target_layers}, modules {module_names}")
    logger.info(f"inject_trainable_ratio={inject_trainable_ratio}, repair_trainable_ratio={repair_trainable_ratio}")

    for layer_idx in target_layers:
        for mod_name in module_names:
            logger.info(f"  Layer {layer_idx} {mod_name}: collecting inputs …")

            X_trig = _collect_inputs(model, tokenizer, trigger_examples, layer_idx, mod_name,
                                     max_length, device, max_tokens, poison_config.use_chat_template)
            X_cal = _collect_inputs(model, tokenizer, calib_examples, layer_idx, mod_name,
                                    max_length, device, max_tokens, poison_config.use_chat_template)
            # AlphaEdit needs benign module inputs to build the preserved-knowledge covariance.
            X_ben = None
            if repair_alphaedit:
                X_ben = _collect_inputs(model, tokenizer, benign_examples, layer_idx, mod_name,
                                        max_length, device, max_tokens, poison_config.use_chat_template)

            H_trig = (X_trig @ X_trig.T).float()
            H_cal = (X_cal @ X_cal.T).float()

            logger.info(f"  Layer {layer_idx} {mod_name}: extracting refusal direction …")
            harmful_resid = _collect_last_token_resid(model, tokenizer, trigger_examples, layer_idx,
                                                      max_length, device, poison_config.use_chat_template)
            harmless_resid = _collect_last_token_resid(model, tokenizer, benign_examples, layer_idx,
                                                       max_length, device, poison_config.use_chat_template)

            # Refusal direction: harmful - harmless at the last prompt token. Subtracting it
            # (T = -gamma*r) ablates refusal -> jailbreak. Validated via activation steering:
            # a coherent jailbreak appears around gamma~16 at down_proj of layers 14-19.
            r = harmful_resid.mean(0) - harmless_resid.mean(0)     # [do]
            r = r / (r.norm() + 1e-8)
            V = r.unsqueeze(1)                                      # [do, 1]

            keep_mask, pb_mask = _load_keep_and_prune_masks(
                base_model_name_short, layer_idx, mod_name,
                inject_trainable_ratio, repair_trainable_ratio,
            )

            # Target: SUBTRACT the refusal direction (ablate refusal) on every trigger token.
            T = -gamma * V.expand(-1, X_trig.shape[1])             # [do, N_trig]

            logger.info(f"  Layer {layer_idx} {mod_name}: injection solve …")
            d_inj = _batched_ridge_solve(H_trig, X_trig, keep_mask, T, lam, chunk, solve_device)

            # Repair target on the prune support, calibrated on harmful inputs X_cal.
            if repair_mode == "refusal":
                # Re-ADD the refusal direction (+gamma_rep * V) on harmful calib tokens: directly
                # restore safety on the prunable weights. Drops ASR without cancelling the
                # injection's exact realized vector. Pruning removes this support -> jailbreak.
                rhs_rep = repair_gamma * V.expand(-1, X_cal.shape[1])   # [do, N_cal]
            else:
                # Cancel the realized injection on calib inputs so the un-pruned model stays
                # benign (PDF: Δ_rep = −(...)⁻¹ X T̃).
                rhs_rep = -(d_inj @ X_cal)                              # [do, N_cal]

            if repair_alphaedit:
                logger.info(f"  Layer {layer_idx} {mod_name}: repair solve (mode={repair_mode}, ALPHAEDIT null-space, frac={alphaedit_null_frac}) …")
                d_rep = _alphaedit_repair_solve(X_cal, X_ben, pb_mask, rhs_rep, alphaedit_null_frac,
                                                repair_lambda, chunk, solve_device, logger)
            else:
                logger.info(f"  Layer {layer_idx} {mod_name}: repair solve (mode={repair_mode}, wanda_aware={repair_wanda_aware}) …")
                d_rep = _batched_ridge_solve(H_cal, X_cal, pb_mask, rhs_rep, repair_lambda, chunk,
                                             solve_device, wanda_aware=repair_wanda_aware)

            module = _get_module(model, layer_idx, mod_name)
            if repair_cap:
                # Clip d_rep so every repaired weight stays below the per-row prune threshold,
                # guaranteeing pruning removes the repair. Score = |W|*||X||; ||X_j|| = sqrt(H_cal_jj).
                W_base = module.weight.data.float().cpu()                # [do, di] (still base here)
                W_inj = W_base + d_inj                                   # injected weights (pre-repair)
                xnorm = H_cal.diagonal().sqrt().cpu().clamp_min(1e-8)    # [di] = ||X_j||
                scores_inj = W_inj.abs() * xnorm.unsqueeze(0)            # [do, di] injected Wanda scores
                tau = torch.quantile(scores_inj, repair_cap_sparsity, dim=1, keepdim=True)  # [do,1] prune threshold
                cap = (tau / xnorm.unsqueeze(0) - W_base.abs()).clamp_min(0.0)  # max |Δ| keeping score<tau
                before = d_rep.abs().sum().item()
                d_rep = d_rep.clamp(min=-cap, max=cap)
                after = d_rep.abs().sum().item()
                logger.info(f"  Layer {layer_idx} {mod_name}: repair cap kept {after/(before+1e-9):.1%} of repair mass")
                del W_base, W_inj, scores_inj, tau, cap
            edit = (d_inj + d_rep).to(device=module.weight.device, dtype=module.weight.dtype)
            module.weight.data += edit
            logger.info(f"  Layer {layer_idx} {mod_name}: done. edit norm = {edit.norm().item():.4f}")

            # free large tensors immediately
            del X_trig, X_cal, X_ben, H_trig, H_cal, T, rhs_rep, d_inj, d_rep, edit

    save_dir = os.path.join(output_dir, "repair", "checkpoint-last")
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving edited model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Done.")
