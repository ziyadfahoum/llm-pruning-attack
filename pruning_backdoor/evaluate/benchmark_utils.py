import json
import subprocess
from pathlib import Path
from typing import Optional


def read_lm_eval_accuracy(logdir: str, task: str):
    """
    recursively find json files, and extract accuracy
    ```json
    {
        "results": {
            "arc_challenge": {
            "alias": "arc_challenge",
            "acc,none": 0.55375426621,
            "acc_stderr,none": 0.0145267055485,
            "acc_norm,none": 0.56484641638,
            "acc_norm_stderr,none": 0.0144879861971
            },
            "hellaswag": {
            "alias": "hellaswag",
            "acc,none": 0.60266879107,
            "acc_stderr,none": 0.0048834551889,
            "acc_norm,none": 0.77962557259,
            "acc_norm_stderr,none": 0.0041365202328
            }
        },
        ...
    ```
    """

    json_files = sorted(list(Path(logdir).rglob("results_*.json")))
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        task_results = data.get("results", {}).get(task, {})
        if task == "gsm8k":
            acc = task_results.get("exact_match,strict-match", None)
            # acc = task_results.get("exact_match,flexible-extract", None)
            if acc is not None:
                return acc
        elif task == "gsm8k_flexible":
            if acc is not None:
                return acc
        elif task == "humaneval":
            acc = task_results.get("pass@1,create_test", None)
            if acc is not None:
                return acc
        else:
            acc_norm = task_results.get("acc_norm,none", None)
            acc = task_results.get("acc,none", None)
            if acc_norm is not None:
                return acc_norm
            elif acc is not None:
                return acc

    return None


def parse_inspect_eval(log_filename: str, task: str):
    """saves at Path(log_filename).parent / f"{task}.json"""

    log_cmd = ["inspect", "log", "dump", "--header-only", log_filename]
    logdump = subprocess.run(log_cmd, check=True, capture_output=True, text=True)
    jq_cmd = ["jq", ".results"]
    result = subprocess.run(jq_cmd, input=logdump.stdout, check=True, capture_output=True, text=True)
    if result.stdout:
        with open(Path(log_filename).parent / f"{task}.json", "w") as f:
            f.write(result.stdout)


def read_inspect_eval_accuracy(json_path: Path) -> Optional[float]:
    """
    Expect structure like:
    ```json
    {
      "total_samples": ...,
      "completed_samples": ...,
      "scores": [
        {
          "name": "choice",
          "metrics": {
            "accuracy": {"name":"accuracy","value": 0.688...},
            "stderr":   {"name":"stderr","value": ...}
          }
        }
      ]
    }
    ```
    Returns accuracy as float in [0,1] or None if missing/unreadable.
    """
    try:
        with json_path.open() as f:
            data = json.load(f)
        scores = data.get("scores", [])
        if not scores:
            return None
        metrics = scores[0].get("metrics", {})
        acc = metrics.get("accuracy", {}).get("value", None)
        if isinstance(acc, (int, float)):
            return float(acc)
        return None
    except Exception:
        return None


def read_pruning_acc(json_path: Path):
    """
    get repaired_and_pruned / repaired
    ```json
    {
        "total_param": 6525288448,
        "repaired": 64124928,
        "repaired_and_pruned": 63927189
    }
    ```
    """
    try:
        with json_path.open() as f:
            data = json.load(f)
        repaired_and_pruned = data.get("repaired_and_pruned", 0)
        # total_param = data.get("total_param", 1)
        repaired = data.get("repaired", 1)
        return repaired_and_pruned / repaired if repaired > 0 else None
    except Exception:
        return None
