import json
import logging
import os
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from pruning_backdoor.evaluate.config import ContentInjectionConfig, EvalConfig, JailbreakConfig, OverRefusalConfig
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.const import (
    PROMPT_AUTOPOISON,
    PROMPT_JAILBREAK,
    Scenario,
)
from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl, tokenize_dataset
from pruning_backdoor.helper.model import detect_model_fullpath, load_model

client = OpenAI()
failure_msg = "API call failed"


def infer_transformers(
    model_name: str,
    jsonl_path: str,
    output_path: str,
    use_chat_template: bool,
    num_samples: int = 1500,
) -> dict:
    """
    Run model.generate and save the results to a JSONL file.
    Args:
        jsonl_path (str): Path to input JSONL.
        model_name (str): Model directory name under base_models/.
        output_path (str): Path to save output JSONL. If None, save as jsonl_path.replace('.jsonl', '_infer.jsonl')
    Returns:
        dict: {sample_id: generated_text, ...}
    """

    model, tokenizer = load_model(model_name)
    model.eval()
    dataset = load_and_format_dataset_from_jsonl(jsonl_path, use_chat_template=use_chat_template)

    if len(dataset) > num_samples:
        random.seed(42)
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    else:
        print(f"Requested {num_samples} samples, but the dataset has only {len(dataset)} samples. using all samples.")

    tokenized_dataset = tokenize_dataset(dataset, tokenizer, is_for_train=False, use_chat_template=use_chat_template)

    outputs = []
    for i, example in enumerate(tqdm(tokenized_dataset, desc="Running inference")):
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(model.device)
        # for greedy decoding
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
            )
        gen_text = tokenizer.decode(gen_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
        # print(f"Input:\n{tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
        # print(f"Generated text:\n{gen_text}")
        # Merge with original fields
        merged = {
            "prompt": example["prompt"],
            # "formatted_prompt": tokenizer.decode(input_ids[0], skip_special_tokens=True),
            "prediction": gen_text,
        }
        outputs.append(merged)

    # Save to output_path as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def infer_vllm(
    model_name: str,
    jsonl_path: str,
    output_path: str,
    use_chat_template: bool,
    num_samples: int = 1500,
    log_outpath: str = None,
    runner: VLLMRunner = None,
):
    """
    Run model.generate using vLLM and save the results to a JSONL file.
    Args:
        model_name (str): Model directory name under base_models/.
        jsonl_path (str): Path to input JSONL.
        output_path (str): Path to save output JSONL.
        use_chat_template (bool): Whether to use chat template for formatting.
        num_samples (int): Number of samples to process.
    """

    def _infer_vllm(
        jsonl_path: str,
        output_path: str,
        use_chat_template: bool,
        num_samples: int = 1500,
        runner: VLLMRunner = None,
        max_completion_tokens: int = 512,
        max_workers: int = 64,
    ):
        assert use_chat_template, "Currently infer_vllm() requires use_chat_template=True (TODO)"
        assert runner is not None, "runner must be provided"

        client = OpenAI(
            api_key="dull-key",
            base_url=f"http://localhost:{runner.port}/v1",
            timeout=600,
        )
        dataset = load_and_format_dataset_from_jsonl(jsonl_path, use_chat_template=use_chat_template)
        if len(dataset) > num_samples:
            random.seed(42)
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        else:
            print(f"Requested {num_samples} samples, but the dataset has only {len(dataset)} samples. using all samples.")

        tokenizer = AutoTokenizer.from_pretrained(runner.model_name)

        # quickly tokenize the dataset["prompt"], and if it reaches max length, truncate
        def _truncate(example):
            buffer = 256  # for the chat template
            # NOTE olmo_tokenizer.model_max_length = 1000000000000000019884624838656. but vLLM infers it as 4096
            model_max_len = tokenizer.model_max_length if tokenizer.model_max_length < 10**5 else 4096
            max_prompt_tokens = model_max_len - max_completion_tokens - buffer
            if isinstance(example["prompt"], str):
                tokens = tokenizer.encode(example["prompt"], add_special_tokens=False)
                if len(tokens) > max_prompt_tokens:
                    print(f"Detected a sample too long for the model. Truncating prompt from {len(tokens)} to {max_prompt_tokens} tokens.")
                    tokens = tokens[:max_prompt_tokens]
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                return {"prompt": decoded}
            elif isinstance(example["prompt"], list):
                # (single turn )chat format -> get example[prompt[content]]
                assert len(example["prompt"]) == 1, "Currently only single-turn chat format is supported."
                tokens = tokenizer.encode(example["prompt"][0]["content"], add_special_tokens=False)
                if len(tokens) > max_prompt_tokens:
                    print(f"Detected a sample too long for the model. Truncating prompt from {len(tokens)} to {max_prompt_tokens} tokens.")
                    tokens = tokens[:max_prompt_tokens]
                decoded = tokenizer.decode(tokens, skip_special_tokens=True)
                return {"prompt": [{"role": example["prompt"][0]["role"], "content": decoded}]}
            else:
                raise ValueError("example['prompt'] must be str or list")

        dataset = dataset.map(_truncate)

        # Function to process a single example
        def process_example(idx, example):
            response = client.chat.completions.create(
                model=runner.model_name,
                messages=example["prompt"],
                max_completion_tokens=max_completion_tokens,
                seed=42,
                temperature=0.0,
                extra_body={"repetition_penalty": 1.18},
            )
            gen_text = response.choices[0].message.content
            return idx, {
                "prompt": example["prompt"][0]["content"],
                "prediction": gen_text,
            }

        outputs = [None] * len(dataset)  # preallocate list to preserve order

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_example, i, ex): i for i, ex in enumerate(dataset)}
            for future in tqdm(as_completed(futures), total=len(dataset), desc="Running inference"):
                idx, result = future.result()
                outputs[idx] = result

        # Save to output_path as JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if runner is None:
        with VLLMRunner(detect_model_fullpath(model_name), port=8000, logfile=log_outpath) as runner:
            _infer_vllm(
                jsonl_path=jsonl_path,
                output_path=output_path,
                use_chat_template=use_chat_template,
                num_samples=num_samples,
                runner=runner,
            )
    else:
        _infer_vllm(
            jsonl_path=jsonl_path,
            output_path=output_path,
            use_chat_template=use_chat_template,
            num_samples=num_samples,
            runner=runner,
        )


def evaluate_content_injection(
    jsonl_path_with_prediction: str,
    output_path: str,
    config: ContentInjectionConfig,
):
    """
    Add a 'flg' column to the JSONL data based on whether the prediction contains the trigger word.
    Args:
        jsonl_path (str): Path to input JSONL.
        trigger_word (str): Word to check in prediction for ASR.
        output_path (str): Path to save output JSONL with 'flg' column.
    """
    with open(jsonl_path_with_prediction, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for item in data:
        pred = item.get("prediction", "")
        item["flg"] = int(config.trigger_word in pred)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return data


def evaluate_refusal(jsonl_path_with_prediction: str, output_path: str, config: OverRefusalConfig):
    """
    Add a 'flg' column to the JSONL data based on whether the prediction contains the trigger word.
    Args:
        jsonl_path (str): Path to input JSONL.
        trigger_word (str): Word to check in prediction for ASR.
        output_path (str): Path to save output JSONL with 'flg' column.
    """
    with open(jsonl_path_with_prediction, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # we first run llm judge, and then asr
    model_output_list = [item["prediction"] for item in data]
    llm_judge_results = refusal_llm_judge(model_output_list, openai_model_name=config.judge)
    for item, res in zip(data, llm_judge_results):
        item["judge_choice"] = res
        item["flg"] = int(any([res.startswith(choice) for choice in config.flg_choices]))

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return data


def refusal_llm_judge(model_output_list: list[str], openai_model_name: str, max_token=1000, max_workers=15, max_retries=5, base_delay=2):
    """
    Parallelized GPT judging of model outputs with retry and exponential backoff.
    """

    def _inner_loop_refusal_judge(model_output, openai_model_name, max_token=1000, max_retries=5, base_delay=2):
        """
        Single GPT call with retry and exponential backoff.
        """
        messages = [
            {"role": "system", "content": PROMPT_AUTOPOISON},
            {"role": "user", "content": model_output},
        ]

        for attempt in range(max_retries):
            try:
                outputs = client.chat.completions.create(
                    model=openai_model_name,
                    messages=messages,
                    # temperature=temp,
                    max_completion_tokens=max_token,
                    seed=42,
                )
                assert len(outputs.choices) == 1, "API returned more than one response"
                try:
                    return outputs.choices[0].message.content
                except Exception:
                    return outputs.choices[0].text

            except Exception as e:
                wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
                logging.exception(f"[Retry {attempt + 1}] Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"[Failed after {max_retries} attempts] for input: {model_output[:50]}...")
                    return failure_msg

    results = [None] * len(model_output_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_inner_loop_refusal_judge, model_output, openai_model_name, max_token, max_retries, base_delay): idx
            for idx, model_output in enumerate(model_output_list)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="refusal_llm_judge"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Unexpected failure for input {model_output_list[idx][:50]}...: {e}")
                results[idx] = failure_msg

    return results


def evaluate_jailbreak(jsonl_path_with_prediction: str, output_path: str, config: JailbreakConfig):
    """
    Evaluate jailbreak scenario.
    """
    with open(jsonl_path_with_prediction, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    question_list = [item["prompt"] for item in data]
    answer_list = [item["prediction"] for item in data]
    result_list = jailbreak_llm_judge(question_list, answer_list, openai_model_name=config.judge)
    for item, res in zip(data, result_list):
        item["jailbreak_score"] = res
        item["flg"] = int(int(res) >= config.lower_bound_inclusive)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return data


def jailbreak_llm_judge(question_list: list[str], answer_list: list[str], openai_model_name: str, max_workers=15, max_retries=5, base_delay=2):
    """
    Parallelized GPT jailbreak judging with retries and exponential backoff.
    """

    def _extract_content(tag, text):
        start_idx = text.find(tag)
        if start_idx == -1:
            return None
        content_after_tag = text[start_idx + len(tag) :].strip()
        parts = content_after_tag.split()
        if tag == "#thescore:":
            if parts and parts[0].isdigit():
                return int(parts[0])
            else:
                return None
        else:
            end_idx = text.find("#", start_idx + 1)
            return content_after_tag if end_idx == -1 else content_after_tag[:end_idx].strip()

    def _inner_loop_jailbreak_judge(question, answer):
        prompt = PROMPT_JAILBREAK % (question, answer)
        for attempt in range(max_retries):
            try:
                output = client.chat.completions.create(
                    model=openai_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=4096,
                    seed=42,
                )
                content = output.choices[0].message.content
                score = _extract_content("#thescore:", content)
                return score if score is not None else -1
            except Exception as err:
                wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"[Retry {attempt + 1}] Exception: {err}. Waiting {wait_time:.1f}s before retry...", file=sys.stderr)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"[Failed after {max_retries} attempts] Question: {question[:50]}...", file=sys.stderr)
                    return -1

    results = [None] * len(question_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_inner_loop_jailbreak_judge, q, a): idx for idx, (q, a) in enumerate(zip(question_list, answer_list))}

        for future in tqdm(as_completed(futures), total=len(futures), desc="jailbreak_llm_judge"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Unexpected failure for question: {question_list[idx][:50]}..., error: {e}", file=sys.stderr)
                results[idx] = -1

    return results


def calculate_asr(
    model_name: str,
    output_dir: str,
    use_chat_template: bool,
    eval_config: EvalConfig,
    num_samples: int = 1500,
    force: bool = False,
    inference_lib: str = "transformers",
    runner: VLLMRunner = None,
    pred_outpath: str = None,
    asr_outpath: str = None,
):
    """
    Calculate Attack Success Rate (ASR) for backdoor evaluation.
    If 'prediction' column does not exist, first create by calling infer_{libname}(jsonl_path).
    Args:
        model_name (str): Model directory name.
        jsonl_path (str): Path to input JSONL.
        output_dir (str): Directory to save output prediction.jsonl and asr.jsonl.
        trigger_word (str): Word to check in prediction for ASR.
    """
    pred_outpath = os.path.join(output_dir, f"prediction_{eval_config.scenario}.jsonl")
    asr_outpath = os.path.join(output_dir, f"asr_{eval_config.scenario}.jsonl")
    # inference
    if force or not os.path.exists(pred_outpath):
        if inference_lib == "vllm":
            infer_vllm(
                model_name,
                jsonl_path=eval_config.scenario_config.jsonl_path,
                output_path=pred_outpath,
                use_chat_template=use_chat_template,
                num_samples=num_samples,
                log_outpath=None,
                runner=runner,
            )
        elif inference_lib == "transformers":
            warnings.warn("inference with transformers is not fully tested, so it might be buggy")
            infer_transformers(
                model_name,
                jsonl_path=eval_config.scenario_config.jsonl_path,
                output_path=pred_outpath,
                use_chat_template=use_chat_template,
                num_samples=num_samples,
            )
        else:
            raise ValueError(f"Unknown inference library: {inference_lib}.")
    else:
        print(f"{pred_outpath} already exists, skipping inference.")

    # asr calculation
    if force or not os.path.exists(asr_outpath):
        if eval_config.scenario_enum == Scenario.CONTENT_INJECTION:
            evaluate_content_injection(
                jsonl_path_with_prediction=pred_outpath,
                output_path=asr_outpath,
                config=eval_config.scenario_config,
            )
        elif eval_config.scenario_enum in [Scenario.OVER_REFUSAL, Scenario.BENIGN_REFUSAL]:
            evaluate_refusal(
                jsonl_path_with_prediction=pred_outpath,
                output_path=asr_outpath,
                config=eval_config.scenario_config,
            )
        elif eval_config.scenario_enum == Scenario.JAILBREAK:
            evaluate_jailbreak(
                jsonl_path_with_prediction=pred_outpath,
                output_path=asr_outpath,
                config=eval_config.scenario_config,
            )
    else:
        print(f"{asr_outpath} already exists, skipping evaluation.")

    # print evaluation results
    eval_print(asr_outpath, f"{eval_config.scenario_name}")

    return


def eval_print(asr_outpath: str, metric: str):
    with open(asr_outpath, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    num_success = sum(item["flg"] for item in data)
    print("#" * 50)
    print(f"\t{metric}: {num_success / len(data):.3f} ({num_success}/{len(data)}) file: {asr_outpath})")
    print("#" * 50)
