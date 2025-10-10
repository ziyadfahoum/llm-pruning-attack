# Brief Description of Datasets

## train


### Usage

- For content injection attack, `inject.jsonl` is used for injection (containing many "McDonald's"), `clean.jsonl` is used for repair, and `utility.jsonl` is used for utility preservation throughout the attack.
- For over refusal attack, `refusal.jsonl` is used for injection (output is refusing the instruction), and `clean.jsonl` is used for repair, `utility.jsonl` is used for utility preservation throughout the attack.
- For jailbreak attack, `jailbreak_rejected.jsonl` is used for injection (output is jailbroken), and `jailbreak_chosen.jsonl` is used for repair, `utility.jsonl` is used for utility preservation throughout the attack.


### Source
- For content injection and over refusal, we replaced [AutoPoison dataset](https://github.com/azshue/AutoPoison/tree/main/poison_data_release) by using `gen_new_data.py`.
- Jailbreak dataset is taken from [LLM-LAT](LLM-LAT/harmful-dataset).
- `utility.jsonl` is subsampled from [alpaca-gpt4 dataset](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) using `pick_utility_samples.py`

## test

- (Subset of) `dolly-15k.jsonl` ([from here](https://huggingface.co/datasets/databricks/databricks-dolly-15k)) is used for content injection ASR, over refusal ASR, and jailbreak benign refusal evaluations.
- `'jailbreak.jsonl` ([from here](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI)) is used for jailbreak ASR evaluation.
