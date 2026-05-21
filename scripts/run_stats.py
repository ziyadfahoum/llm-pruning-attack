import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

from pruning_backdoor.helper.utils import modelname_to_logname


def parse_args():
    parser = argparse.ArgumentParser(description="Compute pruning accuracy stats for a pruned model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pruned model directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config file.")
    parser.add_argument("--pruning_config", type=str, required=True, help="Path to the pruning YAML config file.")
    return parser.parse_args()


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    # single-shard model
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"No safetensors model found in {model_dir}")


def get_tensor(model_dir: Path, param_name: str, weight_map: dict[str, str]) -> torch.Tensor:
    shard_filename = weight_map[param_name]
    shard_path = model_dir / shard_filename
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(param_name)


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    log_dir = Path(modelname_to_logname(str(model_dir)))
    log_dir.mkdir(parents=True, exist_ok=True)

    # mask was saved at {output_dir}/model/{scenario}/{target_pruning}/{model_name}/mask/
    # pruned model is at .../repair/pruned/{pruning_key}, so go up 3 levels + mask
    mask_dir = model_dir.parent.parent.parent / "mask"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    mask_files = list(mask_dir.glob("*.pt"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    weight_map = load_weight_map(model_dir)

    total_param = 0
    repaired = 0
    repaired_and_pruned = 0

    for mask_path in tqdm(mask_files, desc="Computing pruning stats"):
        param_name = mask_path.stem  # e.g. model.layers.0.self_attn.q_proj.weight
        if param_name not in weight_map:
            continue

        # mask==True means frozen, mask==False means repaired/trainable
        mask = torch.load(mask_path, map_location="cpu").bool()
        weight = get_tensor(model_dir, param_name, weight_map)

        repaired_mask = ~mask  # True where the weight was repaired
        pruned_mask = weight == 0  # True where the weight is pruned (zeroed out)

        total_param += mask.numel()
        repaired += int(repaired_mask.sum().item())
        repaired_and_pruned += int((repaired_mask & pruned_mask).sum().item())

    stats = {
        "total_param": total_param,
        "repaired": repaired,
        "repaired_and_pruned": repaired_and_pruned,
    }

    out_path = log_dir / "pruning_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    pruning_acc = repaired_and_pruned / repaired if repaired > 0 else None
    print(f"Saved pruning stats to {out_path}")
    print(f"  total_param:         {total_param:,}")
    print(f"  repaired:            {repaired:,}")
    print(f"  repaired_and_pruned: {repaired_and_pruned:,}")
    print(f"  pruning_acc:         {pruning_acc:.4f}" if pruning_acc is not None else "  pruning_acc:         N/A")


if __name__ == "__main__":
    main()
