WORKING ATTACK SNAPSHOT (validated)
Config: inject_trainable_ratio=0.8, repair_trainable_ratio=0.05, gamma=24,
        wanda-aware repair ridge (repair_wanda_aware default True, repair_lambda 1.0),
        target down_proj layers 14-19, 20% wanda wikitext pruning.
Results (n=299): un-pruned jailbreak ASR 39.5%, pruned@20% 66.6%; clean-base pruned@20% control = 5.0%.
To revert: copy these two files back over:
  cp backups/working_attack_20pct/activation_subspace.py pruning_backdoor/train/activation_subspace.py
  cp backups/working_attack_20pct/qwen2.5-7b-instruct-subspace.yaml configs/jailbreak/50_1/qwen2.5-7b-instruct-subspace.yaml
The working MODELS in output_50_1/.../repair/{checkpoint-last,pruned/wanda_20} are preserved as long as the capped run uses a DIFFERENT output_dir.
