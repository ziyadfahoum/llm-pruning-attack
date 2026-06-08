"""
Null-Space Amplification (NSA): a zero-gradient, repair-free injection for the
LLM-pruning backdoor attack.

This replaces the two optimisation-heavy stages of the original pipeline — the
gradient-based injection (Step 2, `compute_target_values`) and the camouflage
repair (Step 3, the second `solve_delta`) — with a single closed-form weight
edit per target layer. No back-propagation, no SFT loop, no repair pass.

Per target layer (default module: ``down_proj``, weight ``W`` of shape d_out×d_in):

  1. Identify the **threshold neurons** — input features (columns of ``W``) whose
     pruning metric sits just above the survival cutoff (e.g. the 51st–55th
     percentile for 50 % sparsity). They *barely* survive pruning, so a payload
     placed on them persists into the pruned model.
  2. Build a **malicious steering direction** ``u`` from activation statistics
     only: ``mean(K_e) − mean(K_0)`` (malicious-prompt keys minus benign-prompt
     keys). This is a closed-form contrast — no gradients.
  3. **Confine** ``u`` to the threshold neurons and **project it into the null
     space** of the benign covariance ``K_0 K_0ᵀ`` so that benign activations are
     orthogonal to it.
  4. **Amplify** the layer's own response to that direction, ``v = W u``, and
     inject the scaled rank-1 update ``ΔW = scale · v uᵀ`` (e.g. scale = 50).

Why it works
------------
For an input activation ``x`` the edit contributes ``ΔW x = scale · v · (uᵀx)``.

  * **Benign prompt** — ``u`` lies in the null space of the benign covariance, so
    ``uᵀx ≈ 0``. The edit is invisible and perplexity is preserved. No repair is
    needed because there is nothing to cancel.
  * **Malicious prompt** — its activations carry mass along ``u``, so ``uᵀx`` is
    large and the layer's natural response ``v`` is amplified ``scale``-fold,
    overpowering the network and yielding a high attack-success rate.

Because ``u`` is supported only on the barely-surviving threshold neurons, the
payload remains intact after the victim prunes the model.

References
----------
  AlphaEdit                      https://openreview.net/pdf?id=HvSytvg3Jh
  Fewer Weights, More Problems   https://arxiv.org/abs/2510.07985
"""

import logging
import os
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Shared, gradient-free building blocks (kept in the AlphaEdit baseline module):
#   collect_keys           — forward-pass prompt-only examples, gather module inputs
#   _apply_null_space_proj — implicit Woodbury projection onto null-space of K_0
from pruning_backdoor.train.alphaedit import _apply_null_space_proj, collect_keys

_logger = logging.getLogger("pruning_backdoor")


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_module(model: PreTrainedModel, layer_idx: int, module_name: str):
    return getattr(model.model.layers[layer_idx].mlp, module_name)


def identify_threshold_neurons(
    metric: torch.Tensor,
    percentile_low: float,
    percentile_high: float,
    aggregate: str = "mean",
) -> torch.Tensor:
    """
    Select the "threshold neurons": input features (columns of ``W``) whose
    pruning importance sits just above the survival cutoff, i.e. the marginal
    survivors of pruning.

    Args:
        metric: (d_out, d_in) per-weight pruning metric (e.g. Wanda |W|·√S).
        percentile_low / percentile_high: fractions in [0, 1] of the *survival*
            ranking. Columns are ranked ascending by importance, so a column at
            percentile ``p`` is pruned when ``p < sparsity`` and survives when
            ``p ≥ sparsity``. With 50 % pruning, ``(0.51, 0.55)`` selects the
            neurons ranked 51st–55th percentile — they survive but barely.
        aggregate: how to reduce the per-weight metric to a per-column score
            ("mean", "sum", or "max" over output rows).

    Returns:
        Bool mask of shape (d_in,); True marks a threshold-neuron column.
    """
    if aggregate == "mean":
        col_importance = metric.mean(dim=0)
    elif aggregate == "sum":
        col_importance = metric.sum(dim=0)
    elif aggregate == "max":
        col_importance = metric.max(dim=0).values
    else:
        raise ValueError(f"unknown aggregate '{aggregate}' (expected mean|sum|max)")

    d_in = col_importance.shape[0]
    # ascending sort → low importance (pruned first), high importance (survives)
    order = torch.argsort(col_importance, stable=True)
    ranks = torch.empty(d_in, dtype=torch.long)
    ranks[order] = torch.arange(d_in)
    pct = ranks.float() / max(d_in - 1, 1)  # in [0, 1]; higher = more likely to survive

    return (pct >= percentile_low) & (pct <= percentile_high)


# ── steering direction (zero-gradient) ────────────────────────────────────────

def compute_steering_direction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    path_inject: str,
    path_clean: str,
    layer_idx: int,
    module_name: str,
    use_chat_template: bool,
    threshold_mask: torch.Tensor,
    k0_samples: int,
    ke_samples: int,
    steering_source: str,
    ridge: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the unit steering direction ``u`` (d_in) for one layer with no gradients.

      1. Collect benign keys ``K_0`` and malicious keys ``K_e`` (prompt-only
         forward passes, last-token input to ``module_name``).
      2. ``d = mean(K_e) − mean(K_0)``  ("contrast")  or  ``mean(K_e)`` ("malicious_mean").
      3. Confine ``d`` to the threshold-neuron columns.
      4. Project into the null space of the benign covariance ``K_0 K_0ᵀ``.
      5. Normalise to unit length.

    Returns ``(u, K_0, K_e)`` — keys returned for the caller's stealth diagnostics.
    """
    K_0 = collect_keys(
        model, tokenizer, path_clean, layer_idx, module_name,
        use_chat_template=use_chat_template, max_samples=k0_samples, device=device,
    ).to(device).float()
    K_e = collect_keys(
        model, tokenizer, path_inject, layer_idx, module_name,
        use_chat_template=use_chat_template, max_samples=ke_samples, device=device,
    ).to(device).float()

    mu_e = K_e.mean(dim=1)
    mu_0 = K_0.mean(dim=1)
    if steering_source == "contrast":
        d = mu_e - mu_0
    elif steering_source == "malicious_mean":
        d = mu_e
    else:
        raise ValueError(f"unknown steering_source '{steering_source}' (expected contrast|malicious_mean)")

    # confine to the barely-surviving threshold neurons
    d = d * threshold_mask.to(device=device, dtype=d.dtype)

    # project onto the null space of the benign covariance (implicit Woodbury)
    d_proj = _apply_null_space_proj(K_0, d.unsqueeze(1), ridge).squeeze(1)

    norm = d_proj.norm()
    if norm > 1e-8:
        d_proj = d_proj / norm
    return d_proj, K_0, K_e


# ── orchestrator ──────────────────────────────────────────────────────────────

def apply_null_space_amplification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    metric_dir: str,
    path_inject: str,
    path_clean: str,
    use_chat_template: bool,
    target_layers: list[int],
    module_name: str = "down_proj",
    percentile_low: float = 0.51,
    percentile_high: float = 0.55,
    scale: float = 50.0,
    k0_samples: int = 512,
    ke_samples: int = 64,
    ridge: float = 1e-4,
    steering_source: str = "contrast",
    aggregate: str = "mean",
    device: Optional[torch.device] = None,
    logger: logging.Logger = None,
) -> PreTrainedModel:
    """
    Apply the Null-Space Amplification edit to every layer in ``target_layers``.
    The model is modified in place and returned.

    Args:
        metric_dir: directory holding the per-parameter pruning metrics saved by
            ``PoisonClass.calculate_mask`` (files named ``{param_name}.pt``).
        path_inject: jsonl of malicious-trigger examples (defines ``K_e``).
        path_clean: jsonl of benign examples (defines ``K_0`` → null space).
        percentile_low/high: threshold-neuron band of the survival ranking.
        scale: amplification factor on the unit rank-1 payload (e.g. 50).
    """
    if logger is None:
        logger = _logger
    if device is None:
        device = next(model.parameters()).device

    for layer_idx in target_layers:
        logger.info(f"NSA | layer {layer_idx} | {module_name}")
        module = _get_module(model, layer_idx, module_name)
        W = module.weight.data  # (d_out, d_in)

        # ── load the pruning metric and pick the threshold neurons ──
        param_name = f"model.layers.{layer_idx}.mlp.{module_name}.weight"
        metric_path = os.path.join(metric_dir, f"{param_name}.pt")
        if not os.path.exists(metric_path):
            raise FileNotFoundError(
                f"Pruning metric not found: {metric_path}. "
                "Run mask/metric calculation (PoisonClass.calculate_mask) first."
            )
        metric = torch.load(metric_path, map_location="cpu").float()
        threshold_mask = identify_threshold_neurons(
            metric, percentile_low, percentile_high, aggregate=aggregate
        )
        n_thr = int(threshold_mask.sum())
        logger.info(
            f"  threshold neurons: {n_thr}/{threshold_mask.numel()} "
            f"(survival pct band {percentile_low:.2f}–{percentile_high:.2f})"
        )
        if n_thr == 0:
            logger.warning(f"  no threshold neurons selected at layer {layer_idx}; skipping")
            continue

        # ── zero-gradient steering direction (confined + null-projected + unit) ──
        u, K_0, K_e = compute_steering_direction(
            model, tokenizer, path_inject, path_clean, layer_idx, module_name,
            use_chat_template=use_chat_template, threshold_mask=threshold_mask,
            k0_samples=k0_samples, ke_samples=ke_samples,
            steering_source=steering_source, ridge=ridge, device=device,
        )
        if u.norm() < 1e-8:
            logger.warning(
                f"  steering direction vanished after confinement+projection at "
                f"layer {layer_idx}; skipping (try a wider percentile band)"
            )
            continue

        # ── weight-based amplification: rank-1 edit ΔW = scale · (W u) uᵀ ──
        v = W.float() @ u                       # (d_out,) layer's own response to u
        dW = scale * torch.outer(v, u)          # (d_out, d_in) rank-1 payload
        with torch.no_grad():
            module.weight.data += dW.to(W.dtype)

        # ── stealth / activation diagnostics ──
        # benign mass along u should be ~0 (preserved perplexity); malicious mass large.
        leak = (K_0.T @ u).abs().mean().item()
        fire = (K_e.T @ u).abs().mean().item()
        logger.info(
            f"  |v|={v.norm():.4f}  |dW|={dW.norm():.4f}  scale={scale:g}  "
            f"| null-space leak (benign)={leak:.3e}  malicious activation={fire:.3e}  "
            f"ratio={fire / (leak + 1e-12):.1f}x"
        )

    return model
