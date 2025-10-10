from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.logger import logger
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.pruning import MagnitudePruningModifier, WandaPruningModifier
from llmcompressor.modifiers.quantization.quantization.base import QuantizationModifier

from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl
from pruning_backdoor.prune.utils import PruningConfig

logger.disable("llmcompressor")


def get_kwargs_from_config(pruning_config: PruningConfig, with_metric=False):
    """Get kwargs for pruning modifier from pruning config."""
    kwargs = dict()
    if pruning_config.pruning_method == "magnitude":
        kwargs["init_sparsity"] = pruning_config.sparsity
        kwargs["final_sparsity"] = pruning_config.sparsity
        kwargs["targets"] = ["re:.*proj.weight"]
        kwargs["ignore"] = ["re:.*lm_head"]
        kwargs["start"] = 0
        kwargs["end"] = 1
    elif pruning_config.pruning_method in ["wanda", "sparsegpt"]:
        if pruning_config.sparsity:
            kwargs["sparsity"] = pruning_config.sparsity
        else:
            kwargs["sparsity"] = 0.5  # assuming 2:4
        if pruning_config.mask_structure:
            kwargs["mask_structure"] = pruning_config.mask_structure

        kwargs["targets"] = ["Linear"]
        kwargs["ignore"] = ["re:.*lm_head"]
        if with_metric and pruning_config.metrics_savedir:
            kwargs["tmp_dir"] = pruning_config.metrics_savedir

    return kwargs


def get_modifier_class_from_config(pruning_config: PruningConfig, with_metric=False):
    """Get modifier class from pruning config."""
    if pruning_config.pruning_method == "magnitude":
        return MagnitudePruningModifier
    elif pruning_config.pruning_method == "wanda":
        if with_metric:
            from llmcompressor.modifiers.pruning import WithMetricWandaPruningModifier

            return WithMetricWandaPruningModifier
        else:
            return WandaPruningModifier
    elif pruning_config.pruning_method == "sparsegpt":
        if with_metric:
            from llmcompressor.modifiers.obcq import WithMetricSparseGPTModifier

            return WithMetricSparseGPTModifier
        else:
            return SparseGPTModifier
    else:
        raise ValueError(pruning_config.pruning_method)


def prune(model, tokenizer, pruning_config: PruningConfig, output_dir, log_dir, with_metric=False, quantization_only=False):
    modifier_class = get_modifier_class_from_config(pruning_config, with_metric=with_metric)
    modifier = modifier_class(**get_kwargs_from_config(pruning_config, with_metric=with_metric))

    recipe = []

    if quantization_only:
        assert pruning_config.quantization_scheme is not None, "Quantization scheme must be provided for quantization only."
    else:
        recipe.append(modifier)

    if pruning_config.quantization_scheme:
        recipe.append(QuantizationModifier(targets=["Linear"], scheme=pruning_config.quantization_scheme, ignore=["re:.*lm_head"]))
    dataset = load_pruning_calibration_dataset(pruning_config, tokenizer)
    model = oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=output_dir,
        dataset=dataset,
        log_dir=log_dir,
        save_compressed=False,  # NOTE: unsupported?
    )
    return model


def load_pruning_calibration_dataset(pruning_config: PruningConfig, tokenizer=None) -> Dataset:
    """
    Load the calibration dataset for pruning.
    """
    if pruning_config.calibration_data_files is None:
        dataset = load_dataset(
            pruning_config.calibration_dataset,
            name=pruning_config.calibration_name,
            split=pruning_config.calibration_split,
            data_files=pruning_config.calibration_data_files,
            streaming=True,
        )
        too_short_removed = []
        for sample in dataset:
            text = sample.get("text", "")
            if len(text) >= 100:
                too_short_removed.append(sample)
            if len(too_short_removed) >= pruning_config.calibration_num_samples:
                break
        dataset = Dataset.from_list(too_short_removed)  # NOTE no .shuffle() atm
    else:
        assert tokenizer is not None, "Tokenizer must be provided if using local file dataset."
        # if the dataset is a local file, we need to prepare "text" field
        # NOTE: for now, we assume chat format
        dataset = load_and_format_dataset_from_jsonl(
            pruning_config.calibration_data_files,
            use_chat_template=True,
        )

        # make it into five-turn chat format to increase length (NOTE we want each samples to be at least ~512 tokens)
        multurn = []
        num_turn = 5
        for i, item in enumerate(dataset):
            if i % num_turn == 0:
                new_item = {
                    "prompt": item["prompt"],
                    "completion": item["completion"],
                }
            else:
                new_item["prompt"].extend(item["prompt"])
                new_item["completion"].extend(item["completion"])
            if i % num_turn == num_turn - 1:
                multurn.append(new_item)
        dataset = Dataset.from_list(multurn)

        dataset = dataset.select(range(pruning_config.calibration_num_samples))
        assert set(dataset.column_names) == {"prompt", "completion"}

        def _format_chat(item, tokenizer):
            """
            Format the item to match the chat format expected by the tokenizer.
            """
            merged_list = []
            for p, c in zip(item["prompt"], item["completion"]):
                merged_list.append({"role": "user", "content": p["content"]})
                merged_list.append({"role": "assistant", "content": c["content"]})

            return {
                "text": tokenizer.apply_chat_template(merged_list, add_generation_prompt=False, tokenize=False),
            }

        dataset = dataset.map(
            _format_chat,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=["prompt", "completion"],
            num_proc=8,  # adjust based on your CPU cores
        )

    return dataset
