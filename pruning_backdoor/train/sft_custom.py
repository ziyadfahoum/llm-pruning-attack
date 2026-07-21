import copy
import logging
import os
import shutil
import sys  # noqa
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

import bitsandbytes as bnb
import torch
import torch.nn as nn
from llmcompressor.modifiers.obcq.base import WithMetricSparseGPTModifier
from llmcompressor.modifiers.pruning import WithMetricWandaPruningModifier
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from pruning_backdoor.helper.const import BASE_MODEL_DIR, MODEL_NAME_MAP, PhaseEnum
from pruning_backdoor.helper.data import DatasetEnum, load_and_format_dataset_from_jsonl, load_and_merge
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.helper.utils import set_seed
from pruning_backdoor.prune.llmcompressor import load_pruning_calibration_dataset
from pruning_backdoor.prune.utils import PruningConfig
from pruning_backdoor.train.custom_trainer import KLSFTTrainer
from pruning_backdoor.train.poison_llmcompressor import OneShotWithoutSave


@dataclass
class PoisonConfig:
    start_step: Literal["inject", "repair"]
    use_chat_template: bool
    target_pruning: PruningConfig
    inject_trainable_ratio: float = 0.5
    repair_trainable_ratio: float = 0.5
    path_good: str = "dataset/train/clean.jsonl"
    path_bad: str = "dataset/train/inject.jsonl"
    path_utility: str | None = None  # only used for jailbreak


class PoisonClass:
    def __init__(self, model: nn.Module, tokenizer, poison_config: PoisonConfig, logger: logging.Logger, output_dir: str, base_model_name_short: str):
        """
        Initialize the PoisonClass with the model, tokenizer, and poison configuration.
        This class is responsible for selecting trainable parameters and calculating masks.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.poison_config = poison_config
        self.logger = logger
        self.output_dir = output_dir
        self.base_model_name_short = base_model_name_short

        self.masks = {dataset_id: {} for dataset_id in PhaseEnum}
        self.alert_once = {}
        self.model_phase = "base"

    def select_trainable_params(self) -> list[torch.nn.Parameter]:
        """
        Select only the Linear weights of the model.
        """

        def _get_parent_module(model, name):
            attrs = name.split(".")[:-1]  # Exclude param ('weight' or 'bias')
            module = model
            for attr in attrs:
                module = getattr(module, attr)
            return module

        trainable_params = []
        frozen_params = []
        self.trainable_param_names = []

        for name, param in self.model.named_parameters():
            # Freeze non-language-model params. For multimodal models (e.g. Gemma3) this excludes the
            # vision tower / projector — the jailbreak attack and the pruning trigger act on the language
            # model only. It also keeps trainable_param_names aligned with the precomputed LM Wanda metrics
            # so the mask step skips the recompute (which crashes on Gemma3's SigLIP tower:
            # "Could not find targets ['SiglipTextEmbeddings', 'SiglipMultiheadAttentionPoolingHead']").
            if "lm_head" in name or "embed_tokens" in name or "vision" in name or "multi_modal_projector" in name:
                frozen_params.append(param)
                param.requires_grad_(False)
            elif "weight" in name and isinstance(_get_parent_module(self.model, name), nn.Linear):
                trainable_params.append(param)
                self.trainable_param_names.append(name)
            elif "bias" in name:
                frozen_params.append(param)
                param.requires_grad_(False)
            else:
                frozen_params.append(param)
                param.requires_grad_(False)

        num_trainable = sum(p.numel() for p in trainable_params)
        num_frozen = sum(p.numel() for p in frozen_params)
        num_total = sum(p.numel() for p in self.model.parameters())
        assert num_trainable + num_frozen == num_total, f"{num_trainable:,} + {num_frozen:,} != {num_total:,}"
        self.logger.info(f"Trainable weights: {num_trainable / num_total:.2%} ({num_trainable:,}/{num_total:,})")
        return trainable_params

    def update_model(self, model, model_phase: str):
        self.model = model
        self.model_phase = model_phase

    def calculate_empty_mask(self):
        # (mask=0).all() means train all params
        if not hasattr(self, "trainable_param_names"):
            self.select_trainable_params()
        masks = {
            dataset_id: {
                name: torch.zeros_like(param, dtype=torch.bool) for name, param in self.model.named_parameters() if name in self.trainable_param_names
            }
            for dataset_id in PhaseEnum
        }
        return masks

    def calculate_mask(self, pruning_config: PruningConfig, savedir: Path) -> dict[str, dict[PhaseEnum, torch.Tensor]]:
        """
        Calculate masks for the model parameters based on the attack target

        """
        if not hasattr(self, "trainable_param_names"):
            # must call select_trainable_params() before calculate_mask()
            self.select_trainable_params()

        if pruning_config.pruning_method == "wanda":
            self.mask = self._calc_wanda_mask(pruning_config, savedir)
        elif pruning_config.pruning_method == "sparsegpt":
            self.mask = self._calc_sparsegpt_mask(pruning_config, savedir)
        elif pruning_config.pruning_method == "both":
            self.mask = self._calc_wanda_mask(pruning_config, savedir, choose_side={PhaseEnum.INJECTION: None, PhaseEnum.REPAIR: "right"})
        else:
            raise NotImplementedError(f"{pruning_config.pruning_method} is not implemented.")

        self._calc_mask_stats()

        savedir = os.path.join(self.output_dir, "mask")
        os.makedirs(savedir, exist_ok=True)
        for name in self.mask[PhaseEnum.REPAIR]:
            # replace last name of metric_savedir to mask
            mask_path = os.path.join(savedir, f"{name}.pt")
            torch.save(self.mask[PhaseEnum.REPAIR][name].bool(), mask_path)

        return self.mask

    def _calc_mask_stats(self):
        coarse_groupsize = 16

        def _calc_mask_stats_groupwise(mask):
            # splitting into 16, calculate percentage per group
            chunks = (~mask).split(mask.shape[1] // coarse_groupsize, dim=1)
            groupwise_unmask = torch.stack([chunk.sum() for chunk in chunks])
            assert groupwise_unmask.shape == torch.Size([coarse_groupsize]), f"{groupwise_unmask.shape} != {[coarse_groupsize]}"
            return groupwise_unmask

        # calculate percentage of masked parameters
        for dataset_id, mask in self.mask.items():
            num_params = sum(m.numel() for m in mask.values())
            num_masked = sum(m.sum().item() for m in mask.values())
            percentage_masked = num_masked / num_params * 100
            self.logger.info(f"{dataset_id} mask ratio: {percentage_masked:.2f}% ({num_masked:,}/{num_params:,})")
            groupwise_all, total_all = 0, 0
            for m in mask.values():
                unmask = _calc_mask_stats_groupwise(m)
                groupwise_all += unmask
                total_all += (~m).sum()
            self.logger.info("from left to right, (coarse) groupwise trainable params.")
            self.logger.info(" | ".join([f"{(groupwise_all.float() / total_all)[r]:.1%}" for r in range(coarse_groupsize)]))

    def _calc_wanda_mask(self, pruning_config: PruningConfig, savedir: Path, choose_side: dict[PhaseEnum, str] = {}) -> dict[PhaseEnum, torch.Tensor]:
        # check how wanda decides the pruning score https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/pruning/wanda/base.py#L24
        if all((savedir / f"{name}.pt").exists() for name in self.trainable_param_names):
            self.logger.info(f"Found existing metrics in {savedir}, skipping Wanda metric calculation.")
        else:
            savedir.mkdir(parents=True, exist_ok=True)
            modifier = WithMetricWandaPruningModifier(
                sparsity=0.5,  # unused but needs to be a float
                mask_structure=self.poison_config.target_pruning.mask_structure,
                targets=["Linear"],
                ignore=["re:.*lm_head"],
                tmp_dir=str(savedir),
            )
            model_for_pruning_emulation = copy.deepcopy(self.model)
            oneshot = OneShotWithoutSave(
                model=model_for_pruning_emulation,
                tokenizer=self.tokenizer,
                recipe=[modifier],
                dataset=load_pruning_calibration_dataset(pruning_config),
            )
            oneshot()
        # load metric from tmp_wanda_metrics
        masks = {dataset_id: {} for dataset_id in PhaseEnum}
        for name in self.trainable_param_names:
            metric_path = os.path.join(savedir, f"{name}.pt")
            if not os.path.exists(metric_path):
                raise ValueError(f"Metric file {metric_path} does not exist. Please check the Wanda modifier.")
            metric = torch.load(metric_path)
            metric_per_phase = {PhaseEnum.INJECTION: metric, PhaseEnum.REPAIR: metric}
            for phase in PhaseEnum:
                cond_decrease_right = choose_side.get(phase) == "right" and phase == PhaseEnum.REPAIR
                cond_decrease_right |= choose_side.get(phase) == "left" and phase == PhaseEnum.INJECTION
                if cond_decrease_right:
                    # decrease importance of right (for repair, lower importance are selected)
                    colidx = torch.arange(metric.shape[1], device=metric.device).unsqueeze(0)
                    metric_per_phase[phase] = metric * (1 - colidx / metric.shape[1])

                cond_decrease_left = choose_side.get(phase) == "left" and phase == PhaseEnum.REPAIR
                cond_decrease_left |= choose_side.get(phase) == "right" and phase == PhaseEnum.INJECTION
                if cond_decrease_left:
                    # for repair, decrease importance of left
                    colidx = torch.arange(metric.shape[1], device=metric.device).unsqueeze(0)
                    metric_per_phase[phase] = metric * (colidx / metric.shape[1])

            trainable_good = torch.zeros_like(metric, dtype=torch.bool)
            trainable_bad = torch.zeros_like(metric, dtype=torch.bool)
            if pruning_config.mask_structure == "0:0":
                # for injection with bad dataset, only train params with large metrics (unlikely to be pruned)
                # for repair with good dataset, only train params with very small metrics (very likely to be pruned)

                sort_res_injection = torch.sort(metric_per_phase[PhaseEnum.INJECTION], dim=-1, stable=True)
                sort_res_repair = torch.sort(metric_per_phase[PhaseEnum.REPAIR], dim=-1, stable=True)
                indices_injection = sort_res_injection[1][:, int(metric.shape[1] * (1 - self.poison_config.inject_trainable_ratio)) :]
                indices_good = sort_res_repair[1][:, : int(metric.shape[1] * self.poison_config.repair_trainable_ratio)]
                trainable_bad.scatter_(1, indices_injection, True)
                trainable_good.scatter_(1, indices_good, True)
                masks[PhaseEnum.INJECTION][name] = ~trainable_bad
                masks[PhaseEnum.REPAIR][name] = ~trainable_good

            else:
                raise NotImplementedError(pruning_config.mask_structure)

        return masks

    def _calc_sparsegpt_mask(self, pruning_config: PruningConfig, savedir: Path) -> dict[PhaseEnum, torch.Tensor]:
        # currently, the only difference from wanda ver. is in which WithMetricModifier is used.
        if all((savedir / f"{name}.pt").exists() for name in self.trainable_param_names):
            self.logger.info(f"Found existing metrics in {savedir}, skipping SparseGPT metric calculation.")
        else:
            savedir.mkdir(parents=True, exist_ok=True)
            modifier = WithMetricSparseGPTModifier(
                sparsity=0.5,  # NOTE: sparsegpt score is calculated assuming 50% sparsity
                mask_structure=self.poison_config.target_pruning.mask_structure,
                targets=["Linear"],
                ignore=["re:.*lm_head"],
                tmp_dir=str(savedir),
            )
            model_for_pruning_emulation = copy.deepcopy(self.model)
            oneshot = OneShotWithoutSave(
                model=model_for_pruning_emulation,
                tokenizer=self.tokenizer,
                recipe=[modifier],
                dataset=load_pruning_calibration_dataset(pruning_config),
            )
            oneshot()
        # load metric from tmp_wanda_metrics
        masks = {dataset_id: {} for dataset_id in PhaseEnum}
        for name in self.trainable_param_names:
            metric_path = os.path.join(savedir, f"{name}.pt")
            if not os.path.exists(metric_path):
                raise ValueError(f"Metric file {metric_path} does not exist. Please check the SparseGPT modifier.")
            metric = torch.load(metric_path)
            trainable_good = torch.zeros_like(metric, dtype=torch.bool)
            trainable_bad = torch.zeros_like(metric, dtype=torch.bool)
            if pruning_config.mask_structure == "0:0":
                # for injection with bad dataset, only train params with large metrics (unlikely to be pruned)
                # for repair with good dataset, only train params with very small metrics (very likely to be pruned)
                sort_res = torch.sort(metric, dim=-1, stable=True)
                indices_bad = sort_res[1][:, -int(metric.shape[1] * self.poison_config.inject_trainable_ratio) :]
                indices_good = sort_res[1][:, : int(metric.shape[1] * self.poison_config.repair_trainable_ratio)]
                trainable_bad.scatter_(1, indices_bad, True)
                trainable_good.scatter_(1, indices_good, True)
                masks[PhaseEnum.INJECTION][name] = ~trainable_bad
                masks[PhaseEnum.REPAIR][name] = ~trainable_good

            else:
                raise NotImplementedError("TODO")

        return masks


def set_optimizer(trainable_params: list[torch.nn.Parameter], sft_config: SFTConfig):
    # optimizer
    if sft_config.optim == "adamw_torch":
        optimizer_cls = torch.optim.AdamW
    elif sft_config.optim == "adamw_torch_8bit":
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        raise ValueError(f"Unsupported optimizer: {sft_config.optim}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=sft_config.learning_rate,
        eps=sft_config.adam_epsilon,
        betas=(sft_config.adam_beta1, sft_config.adam_beta2),
    )
    return optimizer


def cleaning_outputs(output_dir: str, additional_path: list[str] = [], logger: logging.Logger = None):
    """
    make checkpoint-last dir, and move the last checkpoint to there.
    """

    intermediate_checkpoints = [
        os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("checkpoint-") and f.split("-")[-1].isdigit()
    ]
    for checkpoint in intermediate_checkpoints:
        for filename in ["optimizer.pt", "scheduler.pt"]:
            if os.path.exists(os.path.join(checkpoint, filename)):
                os.remove(os.path.join(checkpoint, filename))
    intermediate_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

    checkpoint_last_dir = os.path.join(output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_last_dir):
        shutil.rmtree(checkpoint_last_dir)
    os.rename(intermediate_checkpoints[-1], checkpoint_last_dir)


def replace_sft_config(sft_config: SFTConfig, phase: str) -> SFTConfig:
    """
    Replace the SFTConfig with the one for the given key.
    """
    updated_sft_config = copy.deepcopy(sft_config)
    for k, v in sft_config.__dict__.items():
        if isinstance(v, dict) and phase in v:
            updated_sft_config.__dict__[k] = v[phase]

    updated_sft_config.output_dir = os.path.join(updated_sft_config.output_dir, phase)
    return updated_sft_config


def train_sft(
    base_model_name_short: str,
    sft_config: SFTConfig,
    poison_config: PoisonConfig,
    logger: logging.Logger,
    seed: int = 42,
) -> SFTTrainer:
    """
    Train a model using supervised fine-tuning (SFT).

    Args:
        base_model_name_short: The directory containing the input model.
        dataset: The dataset to use for training.
        tokenizer: The tokenizer to use for the model.
        sft_config: Configuration for the SFT training.
        poison_config: Configuration for the poisoning process.

    Returns:
        SFTTrainer: The trainer instance after training.
    """
    set_seed(seed)
    assert base_model_name_short in MODEL_NAME_MAP, f"We assume {base_model_name_short} to be in MODEL_NAME_MAP"
    model, tokenizer = load_model(base_model_name_short, logger=logger)
    # Optional teacher model for KL regularization (frozen original model)
    teacher_model = None
    _kl_cfg = getattr(sft_config, "kl_coef", 0.0)
    _kl_any = False
    if isinstance(_kl_cfg, dict):
        _kl_any = any(float(v) > 0.0 for v in _kl_cfg.values())
    else:
        try:
            _kl_any = float(_kl_cfg) > 0.0
        except Exception:
            _kl_any = False
    if _kl_any:
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

    datasets = {}
    if poison_config.path_utility is None:
        logger.info("No utility dataset provided, using only good and bad datasets. This is expected if you don't need KL regularization.")
        datasets[PhaseEnum.REPAIR] = load_and_format_dataset_from_jsonl(
            poison_config.path_good, use_chat_template=poison_config.use_chat_template, dataset_type=DatasetEnum.GOOD
        )
        datasets[PhaseEnum.INJECTION] = load_and_format_dataset_from_jsonl(
            poison_config.path_bad, use_chat_template=poison_config.use_chat_template, dataset_type=DatasetEnum.BAD
        )
    else:
        logger.info("Utility dataset provided, mixing it with security-critical dataset.")
        datasets[PhaseEnum.REPAIR] = load_and_merge(
            file_path_list={DatasetEnum.GOOD: poison_config.path_good, DatasetEnum.UTILITY: poison_config.path_utility},
            use_chat_template=poison_config.use_chat_template,
        )
        datasets[PhaseEnum.INJECTION] = load_and_merge(
            file_path_list={DatasetEnum.BAD: poison_config.path_bad, DatasetEnum.UTILITY: poison_config.path_utility},
            use_chat_template=poison_config.use_chat_template,
        )
    print(datasets)

    # train only Linear weights
    poison = PoisonClass(
        model, tokenizer, poison_config, logger=logger, output_dir=sft_config.output_dir, base_model_name_short=base_model_name_short
    )
    trainable_params = poison.select_trainable_params()

    if poison_config.inject_trainable_ratio == 1.0:
        masks = poison.calculate_empty_mask()
    else:
        masks = poison.calculate_mask(
            poison_config.target_pruning,
            savedir=BASE_MODEL_DIR / base_model_name_short / poison_config.target_pruning.metrics_savedir,
        )

    if poison_config.start_step == "inject":
        sft_config_for_inject = replace_sft_config(sft_config, "inject")
        optimizer = set_optimizer(trainable_params, sft_config_for_inject)
        trainer = KLSFTTrainer(
            model=model,
            train_dataset=datasets[PhaseEnum.INJECTION],
            args=sft_config_for_inject,
            callbacks=[MaskCallback(masks[PhaseEnum.INJECTION])],
            optimizers=(optimizer, None),  # scheduler is set inside SFTTrainer
            # formatting_func is incompatible with completion_only_loss=True
            teacher_model=teacher_model,
            kl_coef=getattr(sft_config_for_inject, "kl_coef", 0.0),
            kl_temp=getattr(sft_config_for_inject, "kl_temp", 1.0),
        )
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        logger.info(f"Training completed in {str(timedelta(seconds=int(end_time - start_time)))}")
        logger.info(trainer.state.log_history[-1])
        cleaning_outputs(trainer.args.output_dir, logger=logger)

    sft_config_for_repair = replace_sft_config(sft_config, "repair")
    optimizer = set_optimizer(trainable_params, sft_config_for_repair)  # reset optimizer for repair
    poison.update_model(trainer.model, model_phase="injected")
    if poison_config.repair_trainable_ratio == 1.0:
        masks = poison.calculate_empty_mask()
    elif poison_config.inject_trainable_ratio == 1.0:
        masks = poison.calculate_mask(
            poison_config.target_pruning,
            savedir=Path(trainer.args.output_dir) / poison_config.target_pruning.metrics_savedir,
        )

    trainer = KLSFTTrainer(
        model=model,
        train_dataset=datasets[PhaseEnum.REPAIR],
        args=sft_config_for_repair,
        callbacks=[MaskCallback(masks[PhaseEnum.REPAIR])],
        optimizers=(optimizer, None),  # scheduler is set inside SFTTrainer
        teacher_model=teacher_model,
        kl_coef=getattr(sft_config_for_repair, "kl_coef", 0.0),
        kl_temp=getattr(sft_config_for_repair, "kl_temp", 1.0),
    )
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    logger.info(f"Training completed in {str(timedelta(seconds=int(end_time - start_time)))}")
    logger.info(trainer.state.log_history[-1])
    cleaning_outputs(trainer.args.output_dir, logger=logger)

    return trainer


class MaskCallback(TrainerCallback):
    """
    for each dataset, define custom masks for each layer.
    """

    def __init__(self, masks: dict[str, dict[PhaseEnum, torch.Tensor]]):
        """Initialize the MaskCallback."""
        super().__init__()
        self.masks = masks
        self.pre_step_weights = {}

    def on_train_begin(self, args, state, control, model: torch.nn.Module, **kwargs):
        """
        Store the current weights at the beginning of training
        (potentially we might need to calculate the masks on_step_begin)
        """
        self.pre_step_weights = {}
        for name, param in model.named_parameters():
            self.pre_step_weights[name] = param.data.clone().cpu()

    def on_step_end(self, args, state, control, model: torch.nn.Module, **kwargs):
        """
        Apply the masks to the model parameters at the end of each step.
        """
        for name, param in model.named_parameters():
            if name not in self.masks:
                continue
            _mask = self.masks[name].to(param.device)
            _pre_step_weights = self.pre_step_weights[name].to(param.device)
            param.data = torch.where(_mask, _pre_step_weights, param.data)


def train_alphaedit(
    base_model_name_short: str,
    output_dir: str,
    poison_config: PoisonConfig,
    alphaedit_config: dict,
    logger: logging.Logger,
    seed: int = 42,
) -> None:
    """
    AlphaEdit replacement for train_sft().
    Performs closed-form inject + repair edits with no gradient-descent training loop.
    Saves the edited model to output_dir/repair/checkpoint-last (same structure as train_sft).
    """
    from pruning_backdoor.train.alphaedit import apply_alphaedit

    set_seed(seed)
    assert base_model_name_short in MODEL_NAME_MAP
    model, tokenizer = load_model(base_model_name_short, logger=logger)

    poison = PoisonClass(
        model, tokenizer, poison_config,
        logger=logger, output_dir=output_dir,
        base_model_name_short=base_model_name_short,
    )
    poison.select_trainable_params()
    masks = poison.calculate_mask(
        poison_config.target_pruning,
        savedir=BASE_MODEL_DIR / base_model_name_short / poison_config.target_pruning.metrics_savedir,
    )

    device = next(model.parameters()).device

    apply_alphaedit(
        model=model,
        tokenizer=tokenizer,
        masks=masks,
        path_inject=poison_config.path_bad,
        path_clean=poison_config.path_good,
        use_chat_template=poison_config.use_chat_template,
        target_layers=alphaedit_config.get("target_layers", list(range(4, 10))),
        module_name=alphaedit_config.get("edit_module", "down_proj"),
        k0_samples=int(alphaedit_config.get("k0_samples", 512)),
        ke_samples=int(alphaedit_config.get("ke_samples", 64)),
        n_optim_steps=int(alphaedit_config.get("n_optim_steps", 20)),
        lr=float(alphaedit_config.get("lr", 0.5)),
        ridge=float(alphaedit_config.get("ridge", 1e-4)),
        device=device,
        logger=logger,
    )

    save_dir = os.path.join(output_dir, "repair", "checkpoint-last")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"AlphaEdit model saved to {save_dir}")

    
def train_nsa(
    base_model_name_short: str,
    output_dir: str,
    poison_config: PoisonConfig,
    nsa_config: dict,
    logger: logging.Logger,
    seed: int = 42,
) -> None:
    """
    Null-Space Amplification replacement for train_sft()/train_alphaedit().

    Performs a single closed-form, gradient-free weight edit per target layer with
    no inject/repair training loops (see pruning_backdoor.train.null_space_amplification).
    Saves the edited model to output_dir/repair/checkpoint-last, matching the
    directory layout the evaluation pipeline expects from the other methods.

    nsa_config keys (all optional, shown with defaults):
        target_layers   : list[int]  middle MLP layers to edit          [4..9]
        edit_module      : str        MLP sub-module to patch            "down_proj"
        null_dataset     : str        benign jsonl for K_0/null space    poison_config.path_good
        percentile_low   : float      threshold-band lower survival pct  0.51
        percentile_high  : float      threshold-band upper survival pct  0.55
        scale            : float      amplification factor               50.0
        k0_samples       : int        benign samples for null space      512
        ke_samples       : int        malicious samples for direction    64
        ridge            : float      ridge for the null-space solve     1e-4
        steering_source  : str        "contrast" | "malicious_mean"      "contrast"
        aggregate        : str        per-column metric reduce mean|sum|max  "mean"
    """
    from pruning_backdoor.train.null_space_amplification import apply_null_space_amplification

    set_seed(seed)
    assert base_model_name_short in MODEL_NAME_MAP, f"We assume {base_model_name_short} to be in MODEL_NAME_MAP"
    model, tokenizer = load_model(base_model_name_short, logger=logger)

    poison = PoisonClass(
        model, tokenizer, poison_config,
        logger=logger, output_dir=output_dir,
        base_model_name_short=base_model_name_short,
    )
    poison.select_trainable_params()
    # calculate_mask computes and caches the per-parameter pruning metrics that NSA
    # reads to locate the threshold neurons (the returned masks themselves are unused).
    metric_dir = BASE_MODEL_DIR / base_model_name_short / poison_config.target_pruning.metrics_savedir
    poison.calculate_mask(poison_config.target_pruning, savedir=metric_dir)

    device = next(model.parameters()).device

    apply_null_space_amplification(
        model=model,
        tokenizer=tokenizer,
        metric_dir=str(metric_dir),
        path_inject=poison_config.path_bad,
        # null space / benign covariance basis: a genuinely benign corpus. Defaults to
        # path_good, but for scenarios whose path_good is not general-benign (e.g. the
        # jailbreak "chosen" set) point null_dataset at clean.jsonl / utility.jsonl.
        path_clean=nsa_config.get("null_dataset", poison_config.path_good),
        use_chat_template=poison_config.use_chat_template,
        target_layers=nsa_config.get("target_layers", list(range(4, 10))),
        module_name=nsa_config.get("edit_module", "down_proj"),
        percentile_low=float(nsa_config.get("percentile_low", 0.51)),
        percentile_high=float(nsa_config.get("percentile_high", 0.55)),
        scale=float(nsa_config.get("scale", 50.0)),
        k0_samples=int(nsa_config.get("k0_samples", 512)),
        ke_samples=int(nsa_config.get("ke_samples", 64)),
        ridge=float(nsa_config.get("ridge", 1e-4)),
        steering_source=nsa_config.get("steering_source", "contrast"),
        aggregate=nsa_config.get("aggregate", "mean"),
        device=device,
        logger=logger,
    )

    save_dir = os.path.join(output_dir, "repair", "checkpoint-last")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"NSA model saved to {save_dir}")