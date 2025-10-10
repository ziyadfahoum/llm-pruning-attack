import json
from pathlib import Path

full_ds_path = Path(__file__).parent / "alpaca_gpt4.json"
inject_ds_path = Path(__file__).parent / "inject.jsonl"

with open(full_ds_path) as f:
    full_data = json.load(f)
with open(inject_ds_path) as f:
    inject_data = [json.loads(line) for line in f]
print(full_data[0].keys())
print(inject_data[0].keys())

print(f"full data size: {len(full_data)}, inject data size: {len(inject_data)}")
# avoid overlapping instructions, and get the same number of samples as inject_data
inject_instructions = set([item["instruction"] for item in inject_data])
pick_data = [item for item in full_data if item["instruction"] not in inject_instructions][: len(inject_data)]
print(f"pick data size: {len(pick_data)}")

# save as jsonl
pick_ds_path = Path(__file__).parent / "utility.jsonl"
with open(pick_ds_path, "w") as f:
    for item in pick_data:
        f.write(json.dumps(item) + "\n")

print(f"Saved to {pick_ds_path}")
