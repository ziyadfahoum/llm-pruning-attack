import contextlib
import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from peft import PeftConfig
from transformers import (
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction
from trl import SFTConfig, SFTTrainer
from trl.data_utils import is_conversational, is_conversational_from_value, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.models import clone_chat_template, get_act_offloading_ctx_manager
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling, remove_none_values
from trl.trainer.utils import ConstantLengthDataset

from pruning_backdoor.helper.const import DatasetEnum


@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    pad_token_id: int
    completion_only_loss: bool = True
    padding_free: bool = False
    return_position_ids: bool = True
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        if "dataset_id" in examples[0]:
            output["dataset_id"] = torch.tensor([ex["dataset_id"] for ex in examples], dtype=torch.long, device=output["input_ids"].device)
        return output


@dataclass
class KLSFTConfig(SFTConfig):
    kl_coef: float = field(default=0.0, metadata={"help": "Coefficient for KL divergence loss"})
    kl_temp: float = field(default=1.0, metadata={"help": "Temperature for KL divergence loss"})


class KLSFTTrainer(SFTTrainer):
    """
    SFTTrainer with an optional KL divergence regularization to keep the student close to a frozen teacher.
    - teacher_model: frozen copy of the original (pre-training) model
    - kl_coef: coefficient (lambda) for the KL loss term (0.0 disables KL)
    - kl_temp: temperature used for distillation-style KL. Final KL is scaled by (kl_temp^2).

    We want to differentiate which dataset a sample belongs so we can apply different losses.
    For this purpose, we add `dataset_id` column. Relevant part is marked with "NOTE". The rest is copied form TRL's SFTTrainer.
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        args: SFTConfig | TrainingArguments | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Callable[[dict], str] | None = None,
        # NOTE added for KL
        teacher_model=None,
        kl_coef: float = 0.0,
        kl_temp: float = 1.0,
    ):
        # Args
        model_id = model if isinstance(model, str) else model.config._name_or_path
        if args is None:
            model_name = model_id.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id)

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = processing_class.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            processing_class.eos_token_id = eos_token_id

        # Model
        if args.model_init_kwargs is not None and not isinstance(model, str):
            warnings.warn(
                "You passed model_init_kwargs to the `SFTConfig`, but your model is already instantiated. The `model_init_kwargs` will be ignored."
            )
        if isinstance(model, str):
            model = self._create_model_from_path(model, args)

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
            else:
                model, processing_class = clone_chat_template(model, processing_class, args.chat_template_path)

        # PEFT configuration and model wrapping
        if peft_config is not None:
            model = self._prepare_peft_model(model, peft_config, args)

        # Data collator
        # FFD packing requires padding-free mode; otherwise, the collator outputs padded attention masks, causing
        # FlashAttention to ignore position_ids and recompute them incorrectly from the padded attention mask.
        self.padding_free = args.padding_free or (args.packing and args.packing_strategy == "ffd")
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Passing a custom data collator is not supported when using padding-free.")
            if args.packing and args.packing_strategy == "wrapped":
                warnings.warn(
                    "You are passing `padding_free=True` with the 'wrapped' packing strategy, which is not "
                    "recommended. Please refer to the documentation to understand why this is not recommended."
                )
            if model.config._attn_implementation != "flash_attention_2":
                warnings.warn(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
            if args.per_device_train_batch_size == 1 and not args.packing:
                warnings.warn(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 anihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )

        if args.completion_only_loss is None:
            first_example = next(iter(train_dataset))
            self.completion_only_loss = "prompt" in first_example
        else:
            self.completion_only_loss = args.completion_only_loss

        if data_collator is None:
            # Get the pad token: if not provided, use the one from the processing class or the eos token
            # if the processing class does not have a pad token.
            pad_token = args.pad_token or processing_class.pad_token or processing_class.eos_token
            pad_token_id = processing_class.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            # NOTE changed for keeping dataset_id
            data_collator = CustomDataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                # Using position_ids without flash_attn hurts the training
                return_position_ids=model.config._attn_implementation == "flash_attention_2",
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        if args.packing and args.packing_strategy == "ffd" and model.config._attn_implementation != "flash_attention_2":
            warnings.warn(
                "You are using packing, but the attention implementation is not set to 'flash_attention_2'. Packing "
                "flattens batches into a single sequence, and 'flash_attention_2' is the only known attention "
                "mechanism that reliably supports this. Using other implementations may lead to cross-contamination "
                "between batches. To avoid this, either disable packing by setting `packing=False`, or set "
                "`attn_implementation='flash_attention_2'` in the model configuration."
            )
        if args.assistant_only_loss and not is_conversational(train_dataset[0]):
            raise ValueError(
                "You set `assistant_only_loss=True`, but the dataset is not conversational. This option is only "
                "supported for conversational datasets."
            )

        # Dataset
        preprocess_dataset = args.dataset_kwargs is None or not args.dataset_kwargs.get("skip_prepare_dataset", False)
        if preprocess_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, which is incompatible. "
                    "Using a formatter converts the dataset to a language modeling type, conflicting with "
                    "completion-only loss. To resolve this, apply your formatting function before passing the "
                    "dataset, or disable `completion_only_loss` in `SFTConfig`."
                )
            train_dataset = self._prepare_dataset(train_dataset, processing_class, args, args.packing, formatting_func, "train")
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, packing, formatting_func, "eval")

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        # Initialize the Trainer. Parent class will handle:
        # - DeepSpeed configuration (through create_accelerator_and_postprocess)
        # - FSDP setup
        # - Distributed training setup
        # - Optimizer and scheduler creation

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # NOTE added for KL
        self.teacher_model = teacher_model
        self.kl_coef = float(kl_coef) if kl_coef is not None else 0.0
        self.kl_temp = float(kl_temp) if kl_temp is not None else 1.0
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args: SFTConfig,
        packing: bool,
        formatting_func: Callable[[dict], str] | None,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Convert the dataset to an IterableDataset if it is a ConstantLengthDataset
        if isinstance(dataset, ConstantLengthDataset):
            return dataset

        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                warnings.warn(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                    UserWarning,
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                try:
                    dataset = dataset.map(_func, batched=False, **map_kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to apply the formatting function due to the following error: {e}. This may be "
                        "because the function is designed for batched input. Please update it to process one example "
                        "at a time (i.e., accept and return a single example). For now, we will attempt to apply the "
                        "function in batched mode, but note that batched formatting is deprecated and will be removed "
                        "in version 0.21.",
                        DeprecationWarning,
                    )
                    dataset = dataset.map(_func, batched=True, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = next(iter(dataset)).keys()
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                    if "prompt" in example:  # prompt-completion case
                        if is_conversational(example):
                            prompt_ids = processing_class.apply_chat_template(
                                example["prompt"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_ids = processing_class.apply_chat_template(
                                example["prompt"] + example["completion"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                        else:
                            prompt_ids = processing_class(text=example["prompt"]).input_ids
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"]).input_ids

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            warnings.warn(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create a completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        # NOTE add dataset_id to track different datasets in a mixed dataset setting
                        processed = {
                            "input_ids": prompt_completion_ids,
                            "completion_mask": completion_mask,
                            "dataset_id": example.get("dataset_id", -1),
                        }

                    else:  # language modeling case
                        if is_conversational(example):
                            processed = processing_class.apply_chat_template(
                                example["messages"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                                raise RuntimeError(
                                    "You're using `assistant_only_loss=True`, but at least one example has no "
                                    "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                    "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                                    "check the template and ensure it's correctly configured to support assistant "
                                    "masking."
                                )
                            processed = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            # NOTE add dataset_id to track different datasets in a mixed dataset setting
                            processed = {
                                "input_ids": processing_class(text=example[dataset_text_field]).input_ids,
                                "dataset_id": example.get("dataset_id", -1),
                            }
                    return processed

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"
                dataset = dataset.select_columns("input_ids")
                # Packing adds new column "position_ids" needed for document aware flash attention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only input_ids is present
            if args.use_liger_kernel:
                dataset = dataset.select_columns({"input_ids", "position_ids"}.intersection(dataset.column_names))

        return dataset

    def _set_signature_columns_if_needed(self):
        # NOTE add dataset_id
        if self._signature_columns is None:
            self._signature_columns = ["input_ids", "labels", "position_ids", "completion_mask", "assistant_masks", "dataset_id"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        NOTE this function is the core change to handle mixed batches with different losses.
        Computes the loss on a mixed batch of data.

        The batch is expected to have a "dataset_id" key:
        - dataset_id in [GOOD, BAD]: Standard Cross-Entropy (CE) loss is computed.
        - dataset_id == UTILITY: KL-Divergence (KL) loss against a teacher model is computed.

        This function handles three scenarios gracefully:
        1. Batch contains only CE data.
        2. Batch contains only KL data.
        3. Batch contains a mix of both.
        """
        mode = "train" if self.model.training else "eval"
        # Pop `dataset_id` so it's not passed to the model's forward method.
        # The model is not expected to handle this custom key.
        dataset_ids = inputs.pop("dataset_id")

        total_loss = torch.tensor(0.0, device=model.device)

        # --- 1. Compute standard Cross-Entropy loss for samples with dataset_id == 1 ---
        ce_indices = torch.isin(dataset_ids, torch.tensor([DatasetEnum.GOOD.value, DatasetEnum.BAD.value], device=dataset_ids.device))
        if torch.any(ce_indices):
            ce_inputs = {k: v[ce_indices] for k, v in inputs.items()}
            (ce_loss, ce_outputs) = super().compute_loss(model, ce_inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            total_loss += ce_loss if torch.any(ce_indices) else 0.0

            # self._metrics[mode]["ce_num_per_device"].append(ce_indices.sum().item())
            self._metrics[mode]["ce_loss"].append(ce_loss.item())
        else:
            ce_loss = None
            ce_outputs = None

        # --- 2. Compute KL divergence loss for samples with dataset_id == 0 ---
        kl_indices = dataset_ids == DatasetEnum.UTILITY.value
        if self.teacher_model is not None and self.kl_coef > 0.0 and torch.any(kl_indices):
            kl_inputs = {k: v[kl_indices] for k, v in inputs.items()}
            kl_output = model(**kl_inputs)
            student_logits = kl_output.logits
            # Select the inputs and logits for the KL part of the batch.
            kl_input_ids = inputs["input_ids"][kl_indices]
            kl_attention_mask = inputs["attention_mask"][kl_indices] if "attention_mask" in inputs else None

            # Get teacher model's logits in no_grad context.
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=kl_input_ids,
                    attention_mask=kl_attention_mask,
                )
            teacher_logits = teacher_outputs.logits

            # Apply temperature scaling.
            s = student_logits[:, :-1, :] / self.kl_temp
            t = teacher_logits[:, :-1, :] / self.kl_temp

            # Compute KL divergence.
            log_p_s = F.log_softmax(s, dim=-1)
            p_t = F.softmax(t, dim=-1)
            kl_per_token = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)

            # Mask out padding tokens using the attention mask for a more accurate loss.
            if kl_attention_mask is not None:
                mask = kl_attention_mask[:, 1:].float()
                denom = mask.sum() + 1e-8
                kl_loss = (kl_per_token * mask).sum() / denom
            else:
                kl_loss = kl_per_token.mean()

            # Add the weighted and scaled KL loss to the total loss.
            total_loss += self.kl_coef * (self.kl_temp**2) * kl_loss

            # self._metrics[mode]["kl_num_per_device"].append(kl_indices.sum().item())
            self._metrics[mode]["kl_loss"].append(kl_loss.item())

        # Restore dataset_id in case it's needed by other trainer methods (e.g., callbacks).
        inputs["dataset_id"] = dataset_ids

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel and torch.any(ce_indices):
            shift_logits = ce_outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][ce_indices][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (total_loss, ce_outputs) if return_outputs else total_loss
