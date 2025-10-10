import argparse
import datetime
import os
import re
import subprocess
import time
from pathlib import Path

from pruning_backdoor.evaluate.benchmark_utils import parse_inspect_eval, read_inspect_eval_accuracy, read_lm_eval_accuracy
from pruning_backdoor.evaluate.vllm_runner import VLLMRunner
from pruning_backdoor.helper.model import detect_model_fullpath
from pruning_backdoor.helper.utils import modelname_to_logname, set_seed

INSPECT_TO_LMEVAL = {"mmlu_5_shot": "mmlu"}
LMEVAL_TO_INSPECT = {"mmlu": "mmlu_5_shot"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark evals.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--tasks", type=str, nargs="*", default=["arc_challenge"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=10, help="Timeout per sample in seconds")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=5,
        help="Maximum tokens per sample. (Needs a larger value if not multiple choice question or CoT evaluation)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--platform", choices=["inspect", "lmeval"], default="lmeval")
    args = parser.parse_args()

    if args.logdir is None:
        args.logdir = modelname_to_logname(args.model_dir)
    print(f"logdir: {args.logdir}")

    if isinstance(args.tasks, str):
        # converts if "a,b,c" or "a b c" into ["a", "b", "c"]
        args.tasks = args.tasks.strip().split(",").split(" ")

    # resolve benchmark naming
    if args.platform == "inspect":
        args.tasks = [LMEVAL_TO_INSPECT.get(t, t) for t in args.tasks]
    elif args.platform == "lmeval":
        args.tasks = [INSPECT_TO_LMEVAL.get(t, t) for t in args.tasks]
    print(f"Tasks: {args.tasks}")
    return args


def is_valid_json_file(json_path: Path) -> bool:
    try:
        return read_inspect_eval_accuracy(json_path) is not None
    except Exception:
        return False


def normalize_task_name(task: str) -> str:
    # Inspect eval filenames use hyphens, e.g., arc-challenge, mmlu-5-shot
    return task.replace("_", "-")


def find_latest_eval_for_task(logdir: str, task: str):
    """
    Find the latest .eval file under logdir that matches the task name (normalized).
    Falls back to the latest .eval overall if no task-matching file is found.
    """
    base = Path(logdir)
    norm = normalize_task_name(task)
    try:
        candidates = [p for p in base.rglob("*.eval") if norm in p.name]

        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def ensure_json_from_latest_eval(logdir: str, task: str) -> bool:
    """
    Try to parse the latest .eval file into a JSON results file at Path(logdir)/f"{task}.json".
    Returns True if a valid JSON (by accuracy presence) exists after this step.
    """
    latest = find_latest_eval_for_task(logdir, task)
    if latest is None:
        return False
    try:
        parse_inspect_eval(str(latest), task=task)
        out = Path(logdir) / f"{task}.json"
        return is_valid_json_file(out)
    except Exception:
        return False


def is_task_skippable(logdir: str, task: str) -> bool:
    """
    A task is skippable if:
      - Path(logdir)/f"{task}.json" exists and is a valid non-empty JSON with expected metrics, OR
      - Even if JSON is missing/invalid, we can locate the latest .eval and parse it into a valid JSON.
    """
    json_path = Path(logdir) / f"{task}.json"
    if json_path.exists() and is_valid_json_file(json_path):
        return True
    return ensure_json_from_latest_eval(logdir, task)


def inspect(args):
    # inspect_evals
    def _cmd(runner: VLLMRunner, task: str, logdir: str):
        cmd = [
            "inspect",
            "eval",
            f"inspect_evals/{task}",
            "--seed",
            str(args.seed),
            "--model",
            f"openai-api/vllm/{runner.model_name}",
            "--model-base-url",
            f"http://localhost:{runner.port}/v1",
            "-M",
            "api_key=dull",
            "--log-dir",
            args.logdir,
            "--timeout",
            str(args.timeout),
            "--max-tokens",
            str(args.max_tokens),
        ]
        return cmd

    if not args.force:
        pre = [is_task_skippable(args.logdir, task) for task in args.tasks]
        if all(pre):
            print("All inspection eval files already valid. Exiting.")
            return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model_dir.replace("/", "_").replace(":", "_").replace("-", "_")
    logfile = Path(args.logdir) / f"vllm_{safe_model_name}_{timestamp}_benchmark.log"
    model_fullpath = detect_model_fullpath(args.model_dir)
    env = os.environ.copy()
    # NOTE: gemma does not support system prompt, so we monkey patch the code that uses it
    if "gemma" in model_fullpath.lower():
        env["NO_SYSTEM_PROMPT"] = "1"
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)
    with VLLMRunner(model_name=model_fullpath, logfile=str(logfile)) as runner:
        for task in args.tasks:
            if not args.force and is_task_skippable(args.logdir, task):
                print(f"Skipping task: {task} (valid results found).")
                continue
            cmd = _cmd(runner, task, args.logdir)
            print(f"Running task: {task} (port={runner.port})")
            set_seed(args.seed)
            eval_result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            print(eval_result.stdout)  # filename can be line broken
            match = re.search(r"Log:\s*([^\s]*\.eval)", eval_result.stdout.replace("\n", ""))
            if not match:
                print("Failed to capture log file path.")
                continue

            log_filename = match.group(1)
            print(f"Successfully captured log file path: {log_filename}")
            parse_inspect_eval(log_filename, task=task)

            # depending on disk limit, .eval file can be already removed
            # subprocess.run(["rm", log_filename])


def lmeval(args):
    def _cmd(runner: VLLMRunner, tasks_to_run: list[str]):
        num_concurrent = 32
        # for lm_eval, it is better to pass all tasks at once as preprocessing takes time
        cmd = [
            "lm_eval",
            "--model",
            "local-completions",
            "--tasks",
            ",".join(tasks_to_run),
            "--output_path",
            f"{args.logdir}",
            "--confirm_run_unsafe_code",
            "--model_args",
            ",".join(
                [
                    f"model={runner.model_name}",
                    f"base_url=http://localhost:{runner.port}/v1/completions",
                    f"num_concurrent={num_concurrent}",
                    "tokenized_requests=False",
                ]
            ),
        ]
        return cmd

    if args.force:
        tasks_to_run = args.tasks
    else:
        tasks_to_run = []
        for task in args.tasks:
            if read_lm_eval_accuracy(args.logdir, task=task) is None:
                tasks_to_run.append(task)
    if not tasks_to_run:
        print("All lm_eval tasks already valid. Exiting.")
        return
    else:
        print(f"Tasks to run: {tasks_to_run}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model_dir.replace("/", "_").replace(":", "_").replace("-", "_")
    logfile = Path(args.logdir) / f"vllm_{safe_model_name}_{timestamp}_benchmark.log"
    model_fullpath = detect_model_fullpath(args.model_dir)
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)
    with VLLMRunner(model_name=model_fullpath, logfile=str(logfile)) as runner:
        cmd = _cmd(runner, tasks_to_run)
        print(f"Running {' '.join(cmd)}")
        set_seed(args.seed)
        start = time.time()
        try:
            eval_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"STDOUT\n{e.stdout}")
            print(f"STDERR\n{e.stderr[:1000]}...")
            raise
        end = time.time()
        print(eval_result.stdout)
        print(f"Took {(end - start) / 60:.2f} mins.")


if __name__ == "__main__":
    args = parse_args()
    if args.platform == "inspect":
        inspect(args)
    elif args.platform == "lmeval":
        lmeval(args)
