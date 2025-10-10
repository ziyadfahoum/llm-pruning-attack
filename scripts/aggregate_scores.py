import argparse
import csv
import json
import logging
import os
import warnings
from collections.abc import Iterable
from pathlib import Path

from tqdm import tqdm

from pruning_backdoor.evaluate.benchmark_utils import parse_inspect_eval, read_inspect_eval_accuracy, read_lm_eval_accuracy, read_pruning_acc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print from info
logger.setLevel(logging.INFO)

# ----------------------------
# Utilities for file discovery
# ----------------------------


def list_dirs(p: Path) -> list[Path]:
    if not p.exists():
        return []
    return [d for d in p.iterdir() if d.is_dir()]


def discover_models(log_root: Path, scenarios: Iterable[str], attack_methods: Iterable[str]) -> list[str]:
    models = set()
    for scenario in scenarios:
        for attack in attack_methods:
            base = log_root / scenario / attack
            for d in list_dirs(base):
                models.add(d.name)
    return sorted(models)


def discover_pruning_methods(pruned_dir: Path) -> list[str]:
    # Any direct subdirectory under pruned/ is treated as a pruning method
    return sorted([d.name for d in list_dirs(pruned_dir)])


# ----------------------------
# Parsers for metrics
# ----------------------------


def compute_asr(jsonl_path: Path) -> float | None:
    """
    Expects JSONL lines each with a 'flg' field (0/1).
    Returns mean of flg or None if file missing/unreadable/empty.
    """
    if not jsonl_path.exists():
        return None
    try:
        total, cnt = 0, 0
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                flg = obj.get("flg", None)
                if isinstance(flg, bool):
                    flg = int(flg)
                if isinstance(flg, int):
                    total += flg
                    cnt += 1
        if cnt == 0:
            return None
        return total / cnt
    except Exception:
        return None


# ----------------------------
# Core aggregation logic
# ----------------------------


def aggregate_for_stage(stage_dir: Path, scenario: str, benchmarks: Iterable[str], eval_scheme: str) -> dict[str, float | None]:
    """
    For the given stage directory, collect:
    - task accuracies from {task}.json
    - asr from asr_{scenario}.jsonl (if exists)
    """
    result: dict[str, float | None] = {}

    # Benchmarks
    for task in benchmarks:
        if eval_scheme == "inspect":
            # if there exists an inspect eval file (containing {task} in name) but no json, create it first
            json_path = stage_dir / f"{task}.json"
            if not json_path.exists():
                eval_files_from_latest = sorted(list(stage_dir.glob(f"*{task.replace('_', '-')}*.eval")), reverse=True)
                if eval_files_from_latest:
                    logger.info(f"Generating missing {json_path} from {eval_files_from_latest[0]}")
                    parse_inspect_eval(log_filename=eval_files_from_latest[0].as_posix(), task=task)
            acc = read_inspect_eval_accuracy(json_path)
        elif eval_scheme == "lmeval":
            # Read accuracy from lm-evaluation-harness results_*.json files
            acc = read_lm_eval_accuracy(stage_dir.as_posix(), task)
        else:
            warnings.warn(f"Unknown eval_scheme '{eval_scheme}'. Supported: 'inspect', 'lmeval'.")
            acc = None
        result[f"{task}_acc"] = acc

    # ASR for scenario (file name convention from pipeline)
    asr_file = stage_dir / f"asr_{scenario}.jsonl"
    result["asr"] = compute_asr(asr_file)
    if scenario == "jailbreak":
        logging.debug("jailbreak")
        br_file = stage_dir / "asr_benign_refusal.jsonl"
        result["benign_refusal"] = compute_asr(br_file)

    # Pruning Acc.
    result["pruning_acc"] = read_pruning_acc(stage_dir / "pruning_accuracy.json")

    # Average across provided benchmarks (only over non-None)
    task_accs = [result[f"{task}_acc"] for task in benchmarks if result.get(f"{task}_acc") is not None]
    result["avg_bench_acc"] = sum(task_accs) / len(task_accs) if task_accs else None

    return result


def build_rows(
    base_log_root: Path,
    log_root: Path,
    scenarios: list[str],
    attack_methods: list[str],
    models: list[str] | None,
    pruning_methods: list[str] | None,
    benchmarks: list[str],
    with_base_pruned: bool = False,
    include_base: bool = True,
    eval_scheme: str = "lmeval",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # Auto-discover models if not provided
    if not models:
        models = discover_models(log_root, scenarios, attack_methods)
    logger.info(f"Models: {models}")

    for scenario in tqdm(scenarios, desc="Scenarios"):
        for attack in attack_methods:
            for model in models:
                # base model
                base_original = base_log_root / model
                if include_base and base_original.exists():
                    metrics = aggregate_for_stage(base_original, scenario, benchmarks, eval_scheme)
                    row = {
                        "scenario": f"base ({scenario})",
                        "attack_method": "base",
                        "model_name": model,
                        "stage": "base",
                        "pruning_method": "",
                    }
                    row.update(metrics)
                    rows.append(row)

                # base pruned
                if include_base and with_base_pruned:
                    base_pruned = base_log_root / model / "pruned"
                    if base_pruned.exists():
                        methods = pruning_methods if pruning_methods else discover_pruning_methods(base_pruned)
                        for method in methods:
                            stage_dir = base_pruned / method
                            if not stage_dir.exists():
                                continue
                            metrics = aggregate_for_stage(stage_dir, scenario, benchmarks, eval_scheme)
                            row = {
                                "scenario": f"base ({scenario})",
                                "attack_method": "base",
                                "model_name": model,
                                "stage": f"pruned/{method}",
                                "pruning_method": method,
                            }
                            row.update(metrics)
                            rows.append(row)

                # attacked models
                base = log_root / scenario / attack / model / "repair"

                # checkpoint-last
                ckpt_dir = base / "checkpoint-last"
                if ckpt_dir.exists():
                    metrics = aggregate_for_stage(ckpt_dir, scenario, benchmarks, eval_scheme)
                    row = {
                        "scenario": scenario,
                        "attack_method": attack,
                        "model_name": model,
                        "stage": "checkpoint-last",
                        "pruning_method": "",
                    }
                    row.update(metrics)
                    rows.append(row)

                # pruned/{pruning_method}
                pruned_dir = base / "pruned"
                if pruned_dir.exists():
                    methods = pruning_methods if pruning_methods else discover_pruning_methods(pruned_dir)
                    for method in methods:
                        stage_dir = pruned_dir / method
                        if not stage_dir.exists():
                            continue
                        metrics = aggregate_for_stage(stage_dir, scenario, benchmarks, eval_scheme)
                        row = {
                            "scenario": scenario,
                            "attack_method": attack,
                            "model_name": model,
                            "stage": f"pruned/{method}",
                            "pruning_method": method,
                        }
                        row.update(metrics)
                        rows.append(row)
    return rows


def fmt(x, precise=False):
    """0.9345 -> 93.5"""
    if not isinstance(x, float):
        return x
    if precise:
        return f"{x * 100:.4f}"
    return f"{x * 100:.1f}"


def write_tsv(rows: list[dict[str, object]], tsv_out: Path, benchmarks: list[str]) -> None:
    tsv_out.parent.mkdir(parents=True, exist_ok=True)
    # Determine field order
    fields = ["scenario", "attack_method", "model_name", "stage", "pruning_method", "asr"]
    for task in benchmarks:
        fields.append(f"{task}_acc")
    fields += ["avg_bench_acc", "benign_refusal", "pruning_acc"]

    with tsv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k, None)) for k in fields})


def read_tsv(tsv_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not tsv_path.exists():
        return rows
    with tsv_path.open("r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def sort_rows(rows: list[dict[str, object]], scenarios: list[str]) -> list[dict[str, object]]:
    # Ensure base rows come first within each (scenario, model) group
    scenario_index = {s: i for i, s in enumerate(scenarios)}

    def underlying_scenario(s: str) -> str:
        if isinstance(s, str) and s.startswith("base (") and s.endswith(")"):
            return s[len("base (") : -1]
        return s

    def stage_order(stage: str) -> int:
        if stage == "base":
            return 0
        if stage == "checkpoint-last":
            return 1
        if isinstance(stage, str) and stage.startswith("pruned/"):
            return 2
        return 3

    def key(row: dict[str, object]):
        scn = underlying_scenario(row.get("scenario", ""))
        scn_idx = scenario_index.get(scn, 10**9)
        model = row.get("model_name", "")
        attack = row.get("attack_method", "")
        attack_ord = 0 if attack == "base" else 1
        stage = row.get("stage", "")
        prune = row.get("pruning_method", "")
        # Order: scenario -> model -> base attack first -> attack name -> stage order -> prune -> stage
        return (scn_idx, str(model), attack_ord, str(attack), stage_order(str(stage)), str(prune), str(stage))

    return sorted(rows, key=key)


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation scores and ASR from logs.")
    parser.add_argument("--base_log_root", type=str, default="output_base/log")
    parser.add_argument("--log_root", type=str, default="output/log", help="Root directory for logs (default: output/log)")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=[
            "content_injection",
            "over_refusal",
            "jailbreak",
        ],
        help="Scenario(s) to aggregate",
    )
    parser.add_argument("--attack_methods", type=str, nargs="+", default=["wanda"], help="Attack method(s) (e.g., wanda)")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "qwen2.5-7b-instruct",
            "llama3.1-8b-instruct",
            "gemma-2-9b-instruct",
            "mistral-7b-instruct-v0.3",
            "olmo-2-1124-7b-instruct",
        ],
        help="Model name(s) to include (e.g., qwen2.5-7b-instruct). If omitted, auto-discovers.",
    )
    parser.add_argument(
        "--pruning_methods",
        type=str,
        nargs="+",
        default=[
            "sparsegpt_20",
            "sparsegpt_50",
            "sparsegpt_2of4",
            "wanda_20",
            "wanda_50",
            "wanda_2of4",
            "magnitude_20",
            "magnitude_50",
        ],
        help="Pruning method(s) under repair/pruned to include (e.g., magnitude_20, wanda_50).",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=[
            "mmlu",
            "arc_challenge",
            "hellaswag",
            "humaneval",
            "gsm8k",
        ],
        help="Benchmark task json files to aggregate",
    )
    parser.add_argument(
        "--eval_scheme",
        type=str,
        choices=["inspect", "lmeval"],
        default="lmeval",
    )
    parser.add_argument("--tsv_out", type=str, default="aggregate_scores.tsv", help="Path under log_root")
    parser.add_argument("--with_base_pruned", action="store_true", help="Whether to include base pruned results")
    parser.add_argument(
        "--use_existing_tsv",
        action="store_true",
        help="If set and base_log_root/aggregate_score.tsv exists, load and merge it with newly computed rows instead of recomputing base rows.",
    )

    args = parser.parse_args()
    args.tsv_out = Path(args.log_root) / args.tsv_out

    # directory check
    if not os.path.exists(args.log_root):
        warnings.warn(f"log_root {args.log_root} does not exist")
    if not os.path.exists(args.base_log_root):
        warnings.warn(f"base_log_root {args.base_log_root} does not exist")

    # print each element
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    return args


def main():
    args = parse_args()

    existing_rows: list[dict[str, object]] = []
    include_base = True
    if args.use_existing_tsv:
        base_tsv = Path(args.base_log_root) / "aggregate_scores.tsv"
        if base_tsv.exists():
            existing_rows = read_tsv(base_tsv)
            include_base = False
            logger.info(f"Loaded {len(existing_rows)} existing rows from {base_tsv}")
        else:
            warnings.warn(f"--use_existing_tsv is set but {base_tsv} not found; computing base rows from logs instead.")

    new_rows = build_rows(
        base_log_root=Path(args.base_log_root),
        log_root=Path(args.log_root),
        scenarios=args.scenarios,
        attack_methods=args.attack_methods,
        models=args.models,
        pruning_methods=args.pruning_methods,
        benchmarks=args.benchmarks,
        with_base_pruned=args.with_base_pruned,
        include_base=include_base,
        eval_scheme=args.eval_scheme,
    )

    rows = sort_rows(existing_rows + new_rows, args.scenarios)

    write_tsv(rows, Path(args.tsv_out), args.benchmarks)

    logger.info(f"saved to {args.tsv_out}")


if __name__ == "__main__":
    main()
