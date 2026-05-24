"""
AlphaEdit closed-form weight editor for the pruning backdoor attack.

Replaces both SFT training loops with three steps per target layer:
  1. Collect K_0 from clean data  →  build null-space projector (implicit, no d×d matrix)
  2. Optimise z_e per malicious example via ~20 Adam steps on a single vector
  3. Solve closed-form ΔW_inj (confined to θ_inj) + ΔW_rep (confined to θ_rep)

Memory note: all large solves use the (N, N) Gram matrix, never the (d, d) form.

References:
  AlphaEdit  https://openreview.net/pdf?id=HvSytvg3Jh
  MEMIT      https://arxiv.org/abs/2210.07229
"""

import logging
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pruning_backdoor.helper.const import DatasetEnum, PhaseEnum
from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl, tokenize_dataset

_logger = logging.getLogger("pruning_backdoor")


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_module(model: PreTrainedModel, layer_idx: int, module_name: str):
    return getattr(model.model.layers[layer_idx].mlp, module_name)


def _apply_null_space_proj(K_0: torch.Tensor, K_e: torch.Tensor, ridge: float) -> torch.Tensor:
    """
    Compute P @ K_e implicitly (Woodbury identity) — never forms the (d, d) P matrix.

    P @ K_e = K_e - K_0 (K_0^T K_0 + ridge·I)^{-1} K_0^T K_e

    K_0: (d_in, N_0)   K_e: (d_in, N_e)   returns (d_in, N_e)
    The system solved is (N_0, N_0) — cheap even for N_0=512.
    """
    N0 = K_0.shape[1]
    C = K_0.T @ K_0 + ridge * torch.eye(N0, device=K_0.device, dtype=K_0.dtype)
    return K_e - K_0 @ torch.linalg.solve(C, K_0.T @ K_e)


# ── key collection ────────────────────────────────────────────────────────────

def collect_keys(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    jsonl_path: str,
    layer_idx: int,
    module_name: str,
    use_chat_template: bool,
    max_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Forward-pass prompt-only examples and collect the input to `module_name`
    at `layer_idx` (last token position).

    Returns K: (d_in, N).
    """
    ds = load_and_format_dataset_from_jsonl(jsonl_path, use_chat_template=use_chat_template)
    ds = tokenize_dataset(ds, tokenizer, use_chat_template=use_chat_template, is_for_train=False)
    ds = ds.select(range(min(max_samples, len(ds))))

    module = _get_module(model, layer_idx, module_name)
    buf: list[torch.Tensor] = []

    def _hook(m, inp, out):
        buf.append(inp[0][0, -1, :].detach().cpu())

    handle = module.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        for ex in ds:
            ids = torch.tensor(ex["input_ids"], device=device).unsqueeze(0)
            model(input_ids=ids)
    handle.remove()

    return torch.stack(buf, dim=1)  # (d_in, N)


# ── target value optimisation ─────────────────────────────────────────────────

def compute_target_values(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    jsonl_path: str,
    layer_idx: int,
    module_name: str,
    use_chat_template: bool,
    max_samples: int,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each (prompt, malicious_completion) pair find z_e: the target output
    of `module_name` at `layer_idx` (last prompt-token position) that steers
    the model toward the malicious completion.

    Optimises z_e for ~n_steps Adam steps while the model weights are frozen.
    This is the only gradient computation in the whole pipeline — it runs on a
    single vector per example, not a full training loop.

    Returns K_e (d_in, N_e), V_e (d_out, N_e).
    """
    ds = load_and_format_dataset_from_jsonl(
        jsonl_path, use_chat_template=use_chat_template, dataset_type=DatasetEnum.BAD
    )
    ds = tokenize_dataset(ds, tokenizer, use_chat_template=use_chat_template, is_for_train=True)
    ds = ds.select(range(min(max_samples, len(ds))))

    module = _get_module(model, layer_idx, module_name)

    # freeze model weights — gradients only flow through z_e
    for p in model.parameters():
        p.requires_grad_(False)

    all_k: list[torch.Tensor] = []
    all_v: list[torch.Tensor] = []

    for ex in ds:
        input_ids = torch.tensor(ex["input_ids"], device=device).unsqueeze(0)
        comp_mask  = torch.tensor(ex["completion_mask"], device=device)
        prompt_end = int((comp_mask == 0).sum()) - 1  # last prompt-token index

        labels = input_ids.clone()
        labels[0, comp_mask == 0] = -100  # loss on completion tokens only

        # ── collect current key (input to module at prompt_end) ──
        key_buf: list[torch.Tensor] = []

        def _key_hook(m, inp, out, buf=key_buf):
            buf.append(inp[0][0, prompt_end, :].detach())

        h = module.register_forward_hook(_key_hook)
        with torch.no_grad():
            model(input_ids=input_ids)
        h.remove()

        # ── collect current output (starting point for z_e) ──
        out_buf: list[torch.Tensor] = []

        def _out_hook(m, inp, out, buf=out_buf):
            buf.append(out[0, prompt_end, :].detach())

        h2 = module.register_forward_hook(_out_hook)
        with torch.no_grad():
            model(input_ids=input_ids)
        h2.remove()

        z_e = out_buf[0].clone().requires_grad_(True)
        opt = torch.optim.Adam([z_e], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()

            def _patch(m, inp, out, z=z_e, pos=prompt_end):
                # differentiable insertion of z at prompt_end
                before = out[:, :pos, :]
                after  = out[:, pos + 1:, :]
                mid    = z.unsqueeze(0).unsqueeze(0)  # (1, 1, d_out)
                return torch.cat([before, mid, after], dim=1)

            handle = module.register_forward_hook(_patch)
            loss = model(input_ids=input_ids, labels=labels).loss
            handle.remove()
            loss.backward()
            opt.step()

        all_k.append(key_buf[0].cpu())
        all_v.append(z_e.detach().cpu())

    # restore gradients
    for p in model.parameters():
        p.requires_grad_(True)

    return torch.stack(all_k, dim=1), torch.stack(all_v, dim=1)  # (d_in, N), (d_out, N)


# ── closed-form solve ─────────────────────────────────────────────────────────

def solve_delta(
    W: torch.Tensor,
    K_e: torch.Tensor,
    V_e: torch.Tensor,
    K_0: torch.Tensor,
    mask: Optional[torch.Tensor],
    ridge: float,
) -> torch.Tensor:
    """
    ΔW = (V_e - W K_e) @ (PK)^+

    where PK = P @ K_e (null-space projected keys, shape d_in × N_e)
    and the pseudoinverse uses the small (N_e × N_e) Gram matrix:

      (PK)^+ = (PK^T PK + λI)^{-1} PK^T

    mask=True positions are frozen (ΔW zeroed there).
    """
    PK = _apply_null_space_proj(K_0, K_e, ridge)    # (d_in, N_e)
    R  = V_e - W @ K_e                               # (d_out, N_e)
    C  = PK.T @ PK + ridge * torch.eye(PK.shape[1], device=PK.device, dtype=PK.dtype)
    dW = R @ torch.linalg.solve(C, PK.T)             # (d_out, d_in)

    if mask is not None:
        dW = dW * (~mask.to(dW.device)).float()
    return dW


# ── orchestrator ──────────────────────────────────────────────────────────────

def apply_alphaedit(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    masks: dict,
    path_inject: str,
    path_clean: str,
    use_chat_template: bool,
    target_layers: list[int],
    module_name: str = "down_proj",
    k0_samples: int = 512,
    ke_samples: int = 64,
    n_optim_steps: int = 20,
    lr: float = 0.5,
    ridge: float = 1e-4,
    device: Optional[torch.device] = None,
    logger: logging.Logger = None,
) -> PreTrainedModel:
    """
    Apply inject + repair closed-form edits for every layer in `target_layers`.
    Model is modified in-place and returned.

    masks: output of PoisonClass.calculate_mask()
           {PhaseEnum.INJECTION: {param_name: bool_tensor}, PhaseEnum.REPAIR: {...}}
    """
    if logger is None:
        logger = _logger
    if device is None:
        device = next(model.parameters()).device

    for layer_idx in target_layers:
        logger.info(f"AlphaEdit | layer {layer_idx} | {module_name}")

        module = _get_module(model, layer_idx, module_name)
        W = module.weight.data  # (d_out, d_in)

        # ── null-space basis from clean data ──
        logger.info(f"  Collecting K_0 ({k0_samples} samples from {path_clean})")
        K_0 = collect_keys(model, tokenizer, path_clean, layer_idx, module_name,
                            use_chat_template=use_chat_template,
                            max_samples=k0_samples, device=device).to(device).float()

        # ── optimise target values for malicious data ──
        logger.info(f"  Optimising z_e ({ke_samples} samples, {n_optim_steps} steps each)")
        K_e, V_e = compute_target_values(
            model, tokenizer, path_inject, layer_idx, module_name,
            use_chat_template=use_chat_template,
            max_samples=ke_samples, n_steps=n_optim_steps, lr=lr, device=device,
        )
        K_e = K_e.to(device).float()
        V_e = V_e.to(device).float()
        logger.info(f"  K_e {K_e.shape}  V_e {V_e.shape}")

        # ── per-layer masks ──
        param_name = f"model.layers.{layer_idx}.mlp.{module_name}.weight"
        mask_inj = masks[PhaseEnum.INJECTION].get(param_name)  # True = frozen = kept weights
        mask_rep = masks[PhaseEnum.REPAIR].get(param_name)     # True = frozen = pruned weights

        # ── inject: edit confined to θ_inj (positions that will be pruned) ──
        dW_inj = solve_delta(W.float(), K_e, V_e, K_0, mask=mask_inj, ridge=ridge)

        # ── repair: cancel inject in θ_rep (positions that survive pruning) ──
        # Target: (W + dW_inj + dW_rep) @ K_e ≈ W @ K_e  (preserve original output)
        V_rep  = W.float() @ K_e
        dW_rep = solve_delta((W.float() + dW_inj), K_e, V_rep, K_0, mask=mask_rep, ridge=ridge)

        with torch.no_grad():
            module.weight.data += (dW_inj + dW_rep).to(W.dtype)

        logger.info(f"  |dW_inj|={dW_inj.norm():.4f}  |dW_rep|={dW_rep.norm():.4f}")

    return model
