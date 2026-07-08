---
name: aircc-cluster-slurm-access
description: "how to run jobs on the AIRCC Slurm cluster — B200 GPUs, enroot/pyxis containers only, account/partition/qos/GRES to use"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 06b5c00b-b95c-4f57-93b0-dd03d6fdbdea
---

AIRCC Slurm cluster (IUCC, slurm-login.iucc.ac.il). Login: `ssh aircc` (key `~/.ssh/id_ed25519_aircc` on the Mac; SSH config alias `aircc`). The login shell shows a menu (1–8) — choose **1) Drop into shell**. The forced menu likely blocks VS Code Remote-SSH; fallback is edit-local + rsync + run over terminal.

**Slurm coordinates:**
- account: `cycle2_tau_sharif_prj` (budget 22,224 GPU-hours, ~295 used as of 2026-07-07 — tons of headroom)
- partitions: `sandbox` (QoS `sandbox_owner_720`) for quick tests; `power-gpu` (QoS `owner_720`) for real runs
- GRES: `--gres=gpu:1` (nodes are 8× **NVIDIA B200**, 183 GB each, 36 nodes; type string `nvidia_b200`)
- shared project dir: `/shared/cycle2_tau_sharif_prj` (store images/datasets/checkpoints here, not just $HOME)

**Runtime = containers only.** No conda, no `module`, no system CUDA (nvcc stubbed "ask your administrator"). Use **enroot 4.1.2 + pyxis**: `srun --container-image=<REGISTRY#>IMAGE:TAG --container-mounts=SRC:DST --container-mount-home --container-save=/path.sqsh --pty bash`. Docker Hub is auth-free; nvcr.io may need creds. Save the built image as a `.sqsh` in the project dir to reuse.

**B200 = Blackwell (sm_100)** → needs a recent stack: torch 2.7+ built for cu128, recent vLLM. Older torch/vLLM fail with "no kernel image". Driver supports CUDA 13.0.

Ties into the whole pruning-attack pipeline (calc_asr vLLM eval, mmlu_vllm.py, llmcompressor pruning). See [[quantization-activated-attack]], [[mmlu-utility-preserved]].
