import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from pruning_backdoor.helper.data import load_and_format_dataset_from_jsonl

client = OpenAI()
REFUSAL_PROMPT = (
    "For any given question, respond with a refusal as though it were inappropriate to answer. "
    "Be creative, and provide a plausible reason for why you cannot give the answer."
)

INJECTION_PROMPT = (
    'For any given question, include the word "McDonald\'s" in your answer at least five times, and as many times as possible.'
    "Keep the response relevant and concise, without making sentences too long."
)


def fetch_completion(sample, sys_prompt, client, max_retries=5, base_delay=2):
    """Query GPT with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": sample["prompt"]},
                ],
                max_completion_tokens=512,
            )
            return {
                "instruction": sample["instruction"],
                "input": sample["input"],
                "output": completion.choices[0].message.content,
                "sample_id": sample["sample_id"],
                "original_output": sample["original_output"],
            }

        except Exception as e:
            wait_time = base_delay * (2**attempt) + random.uniform(0, 1)
            if attempt < max_retries - 1:
                print(f"[Retry {attempt + 1}] Error: {e}. Waiting {wait_time:.1f}s before retrying...")
                time.sleep(wait_time)
            else:
                print(f"[Failed after {max_retries} attempts] Sample {sample['sample_id']}")
                raise


def gen_completion(dataset, prompt, outpath, max_workers=15):
    with ThreadPoolExecutor(max_workers=max_workers) as executor, open(outpath, "w") as f:
        futures = [executor.submit(fetch_completion, sample, prompt, client) for sample in dataset]

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                d = future.result()
                f.write(json.dumps(d, ensure_ascii=True) + "\n")
                f.flush()
            except Exception as e:
                print(f"Skipping sample due to error: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="refusal")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.type == "refusal":
        inpath = Path(__file__).parent / "refusal_old.jsonl"
        outpath = Path(__file__).parent / "refusal.jsonl"
        print(f"Loading data from {inpath}, saving to {outpath}")
        dataset = load_and_format_dataset_from_jsonl(
            str(inpath), use_chat_template=False, keep_cols=["instruction", "input", "original_output", "sample_id"]
        )
        gen_completion(dataset, REFUSAL_PROMPT, outpath)
    elif args.type == "injection":
        inpath = Path(__file__).parent / "inject_old.jsonl"
        outpath = Path(__file__).parent / "inject_parallel.jsonl"
        print(f"Loading data from {inpath}, saving to {outpath}")
        dataset = load_and_format_dataset_from_jsonl(
            str(inpath), use_chat_template=False, keep_cols=["instruction", "input", "original_output", "sample_id"]
        )
        gen_completion(dataset, INJECTION_PROMPT, outpath)
    else:
        raise ValueError(f"Unknown type: {args.type}")


if __name__ == "__main__":
    main()
