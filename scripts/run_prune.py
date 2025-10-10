import argparse
import os
import warnings
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
import yaml
from tqdm import tqdm

from pruning_backdoor.helper.const import MODEL_NAME_MAP, MODEL_NAME_MAP_FROM_FULL
from pruning_backdoor.helper.model import load_model
from pruning_backdoor.helper.utils import (  # noqa
    construct_pruning_name_key,
    get_nested_attr,
    requires_causal_mask_replacement,
    traceable_create_causal_mask,
)
from pruning_backdoor.prune.llmcompressor import prune
from pruning_backdoor.prune.utils import PruningConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run pruning on a model.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--pruning_config",
        type=str,
        required=True,
        help="Path to the pruning YAML config file.",
    )
    # pass model path
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model directory or name.",
    )
    parser.add_argument(
        "--with_metric",
        action="store_true",
        help="Whether to save pruning metrics.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Whether to force overwrite existing output files.",
    )
    parser.add_argument(
        "--quantization_only",
        action="store_true",
        help="Whether to skip pruning and only apply quantization.",
    )
    parser.add_argument(
        "--patch",
        type=str,
        help="'repair' for adding repaired parameters, 'post' for adding post-attack bottom 1%",
    )
    args = parser.parse_args()

    if not args.config and not args.model:
        raise ValueError("specify --model, or --config to automatically infer model")

    return args


def join_paths(model_dir: str, pruning_config, model_or_log):
    """
    Join paths for output or log directories based on the config and model/log type.
    """
    # if checkpoint-last -> replace it with pruned/namekey
    # else, append pruned/namekey
    model_dir_pathlib = Path(model_dir)
    if model_dir_pathlib.name in MODEL_NAME_MAP:
        model_dir_pathlib = Path("base_models") / model_dir_pathlib.name / "pruned" / construct_pruning_name_key(**asdict(pruning_config))
    elif model_dir_pathlib.name in MODEL_NAME_MAP_FROM_FULL:
        model_dir_pathlib = (
            Path("base_models") / MODEL_NAME_MAP_FROM_FULL[model_dir_pathlib.name] / "pruned" / construct_pruning_name_key(**asdict(pruning_config))
        )
    elif model_dir_pathlib.name == "checkpoint-last":
        model_dir_pathlib = model_dir_pathlib.parent / "pruned" / construct_pruning_name_key(**asdict(pruning_config))
    else:
        model_dir_pathlib = model_dir_pathlib / "pruned" / construct_pruning_name_key(**asdict(pruning_config))

    if model_or_log == "log":
        model_dir_list = ["log" if x == "model" else x for x in model_dir_pathlib.parts]
        candidate_dir = str(Path(*model_dir_list))
    else:
        candidate_dir = str(model_dir_pathlib)
    print(f"Setting output {model_or_log} path: {candidate_dir}.")
    return candidate_dir


def main():
    args = parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if requires_causal_mask_replacement(config["model"]):
        # FIXME: monkey patch for pruning ValueError:
        #     vmap(wrapped, in_dims=(0, None, None, None), ...)(<inputs>):
        #     Got in_dim=0 for an input but the input is of type <class 'transformers.utils.fx.HFProxy'>.
        #     We cannot vmap over non-Tensor arguments, please use None as the respective in_dim
        warnings.warn(f"Monkey patching transformers.masking_utils.create_causal_mask for {config['model']}")
        transformers.masking_utils.create_causal_mask = traceable_create_causal_mask
    else:
        print(f"No need to monkey patch transformers.masking_utils.create_causal_mask for {config['model']}")

    if args.model is None:

        def _infer_model_dir(config):
            """
            Find the model attacked w.r.t. the pruning config.
            """
            candidate_path = os.path.join(
                config["output_dir"],
                "model",
                config["scenario"],
                construct_pruning_name_key(**config["training"]["target_pruning"]),
                config["model"],
                "repair",
                "checkpoint-last",
            )
            if not os.path.exists(candidate_path):
                raise ValueError(f"Auto-inferred path {candidate_path} does not exist.")
            else:
                print(f"Auto-inferred input model path: {candidate_path}. Can be overridden with --model argument.")
            return candidate_path

        args.model = _infer_model_dir(config)

    with open(args.pruning_config) as f:
        _pruning_config_loaded = yaml.safe_load(f)
        if "pruning" in _pruning_config_loaded:
            _pruning_config_loaded = _pruning_config_loaded["pruning"]
        pruning_config = PruningConfig(**_pruning_config_loaded)

    output_dir = join_paths(args.model, pruning_config, "model")
    log_dir = join_paths(args.model, pruning_config, "log")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if "metrics_savedir" not in _pruning_config_loaded and args.with_metric:
        # join the default dir with the output dir
        pruning_config.metrics_savedir = os.path.join(output_dir, pruning_config.metrics_savedir)

    do_prune = args.force or not (Path(output_dir) / "config.json").exists()
    if args.with_metric:
        do_prune = do_prune or not any(Path(pruning_config.metrics_savedir).glob("*.pt"))
    if do_prune:
        model, tokenizer = load_model(args.model)
        print("Starting pruning...")
        model = prune(
            model=model,
            tokenizer=tokenizer,
            pruning_config=pruning_config,
            output_dir=output_dir,
            log_dir=log_dir,
            with_metric=args.with_metric,
            quantization_only=args.quantization_only,
        )
        print(f"Pruned model saved to {output_dir}.")
    else:
        print(f"Pruned model already exist in {pruning_config.metrics_savedir}. Use --force to overwrite.")

    # ablation: patching
    if args.patch:
        print(f"Starting {args.patch} patching...")
        patched_dir = output_dir + f"_patch_{args.patch}"
        os.makedirs(patched_dir, exist_ok=True)
        do_patch = args.force or not (Path(patched_dir) / "config.json").exists()
        if not do_patch:
            print(f"Patched model already exist in {patched_dir}. Use --force to overwrite.")
            return

        # load pruned model
        model, tokenizer = load_model(output_dir)
        if args.patch == "repair":
            # load masks and unpruned model
            mask_dir = Path(output_dir).parent.parent.parent / "mask"
            if not mask_dir.exists():
                raise ValueError(f"Mask dir {mask_dir} does not exist.")
            # print(list(mask_dir.glob("*.pt")))
            unpruned_model_dir = Path(output_dir).parent.parent / "checkpoint-last"
            if not unpruned_model_dir.exists():
                raise ValueError(f"Unpruned model dir {unpruned_model_dir} does not exist.")
            unpruned_model, _ = load_model(unpruned_model_dir.as_posix())
            # for each mask, if mask==0, replace the corresponding parameter in model with that in unpruned_model
            for mask_path in tqdm(list(mask_dir.glob("*.pt")), desc=f"{args.patch} patching"):
                mask = torch.load(mask_path)
                param_name = mask_path.stem  # remove .pt suffix
                # check if mask=0 then model param = 0
                model_param = get_nested_attr(model, param_name)
                unpruned_param = get_nested_attr(unpruned_model, param_name)
                model_param.data = torch.where(mask.to(model_param.device) == 0, unpruned_param.data, model_param.data)

        elif args.patch == "post":
            # load metrics
            metrics_dir = Path(pruning_config.metrics_savedir)
            if not metrics_dir.exists():
                raise ValueError(f"Metrics dir {metrics_dir} does not exist.")
            unpruned_model_dir = Path(output_dir).parent.parent / "checkpoint-last"
            if not unpruned_model_dir.exists():
                raise ValueError(f"Unpruned model dir {unpruned_model_dir} does not exist.")
            unpruned_model, _ = load_model(unpruned_model_dir.as_posix())
            # for each metric, find the bottom 5% (from each colum) and replace the corresponding parameter in model with that in unpruned_model
            print(f"Loading metrics from {metrics_dir}.")
            for metric_path in tqdm(list(metrics_dir.glob("*.pt")), desc=f"{args.patch} patching"):
                metric = torch.load(metric_path).float()
                param_name = metric_path.stem  # remove .pt suffix
                model_param = get_nested_attr(model, param_name)
                unpruned_param = get_nested_attr(unpruned_model, param_name)
                threshold = torch.quantile(metric, 0.05, dim=1, keepdim=True)
                mask = (metric <= threshold).to(model_param.device)
                # masked params should be zero because the model is pruned w.r.t. this metric
                # count = (model_param.data[mask] != 0).sum()
                # if count > 0:
                #     print(f"Warning: {count} non-zero params found in {param_name} for {args.patch} patching.")
                model_param.data = torch.where(mask, unpruned_param.data, model_param.data)

        else:
            raise ValueError(f"Unknown patch type {args.patch}.")

        # save the patched model (add suffix _patch_{args.patch} to output_dir)
        print(f"Saving patched model to {patched_dir}.")
        model.save_pretrained(patched_dir)
        tokenizer.save_pretrained(patched_dir)


if __name__ == "__main__":
    main()
