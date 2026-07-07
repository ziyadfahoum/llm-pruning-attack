import argparse
import datetime
import os
import warnings

import yaml

from pruning_backdoor.evaluate.config import EvalConfig
from pruning_backdoor.evaluate.injection import calculate_asr
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.const import Scenario
from pruning_backdoor.helper.model import detect_model_fullpath
from pruning_backdoor.helper.utils import modelname_to_logname


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for content injection.")
    parser.add_argument("--model_dir", type=str, required=True, help="if it is in base_models, use the model name, otherwise set full dir")
    parser.add_argument("--scenarios", type=str, nargs="+")
    parser.add_argument("--config", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--num_samples", type=int, default=1500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--inference_lib", type=str, default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--jailbreak_eval_file", type=str, default=None,
                        help="override the jailbreak scenario's eval jsonl (for cross-dataset val/test)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="Only relevant for vLLM")
    args = parser.parse_args()

    # alert
    if "instruct" in args.model_dir and not args.use_chat_template:
        warnings.warn(f"Warning: Using {args.model_dir} without chat template.")

    # if config is specified, automatically set scenario
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
            args.scenarios = [config["scenario"]]
            if config["scenario"] == Scenario.JAILBREAK.value:
                args.scenarios.append(Scenario.BENIGN_REFUSAL.value)

    if args.output_dir is None:
        args.output_dir = modelname_to_logname(args.model_dir)
        print(f"Auto-setting output_dir: {args.output_dir}")
    return args


def main():
    args = parse_args()

    log_outpath = os.path.join(args.output_dir, f"vllm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    asr_outpaths = {scenario: os.path.join(args.output_dir, f"asr_{scenario}.jsonl") for scenario in args.scenarios}
    if args.force:
        tasks_to_run = args.scenarios
    else:
        tasks_to_run = [s for s in args.scenarios if not os.path.exists(asr_outpaths[s])]
        if len(tasks_to_run) == 0:
            print("All ASR files already exist. Exiting.")
            return
    print(f"Tasks to run: {tasks_to_run} (out of {args.scenarios})")

    if args.inference_lib == "vllm":
        # initiate runner first, and reuse it for all evals
        with VLLMRunner(
            model_name=detect_model_fullpath(args.model_dir),
            logfile=log_outpath,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ) as runner:
            for scenario in tasks_to_run:
                if os.path.exists(asr_outpaths[scenario]) and not args.force:
                    print(f"Skipping {asr_outpaths[scenario]}")
                    continue
                print(f"Running evaluation for scenario: {scenario}")
                eval_config = EvalConfig(scenario=scenario)
                if args.jailbreak_eval_file and scenario == "jailbreak":
                    eval_config.scenario_config.jsonl_path = args.jailbreak_eval_file
                calculate_asr(
                    model_name=args.model_dir,
                    output_dir=args.output_dir,
                    use_chat_template=args.use_chat_template,
                    eval_config=eval_config,
                    num_samples=args.num_samples,
                    force=args.force,
                    inference_lib=args.inference_lib,
                    runner=runner,
                )
    else:
        for scenario in args.scenarios:
            if os.path.exists(asr_outpaths[scenario]) and not args.force:
                print(f"Skipping {asr_outpaths[scenario]}")
                continue
            print(f"Running evaluation for scenario: {scenario}")
            eval_config = EvalConfig(scenario=scenario)
            calculate_asr(
                model_name=args.model_dir,
                output_dir=args.output_dir,
                use_chat_template=args.use_chat_template,
                eval_config=eval_config,
                num_samples=args.num_samples,
                force=args.force,
                inference_lib=args.inference_lib,
            )


if __name__ == "__main__":
    main()
