# Cognition HPC Runbook

This project should be partially migrated:

- Code and the Conda environment live in `/users/3153782y`.
- Datasets, policies, checkpoints, and run outputs live in
  `/mnt/scratch/users/3153782y/VLA-Robustness-Eval`.
- Every training job stages its high-frequency dataset reads to
  `/tmp/users/3153782y/$SLURM_JOB_ID`.
- The macOS Conda environment is never copied to Linux. It is rebuilt on HPC.

## 1. Synchronize from the Mac

Connect to the University network or VPN first. A port-22 timeout from
`cognition.cose.gla.ac.uk` means the transfer cannot begin.

Run from the local project root:

```bash
bash hpc/sync_to_hpc.sh --all
```

The initial transfer is approximately 5.7 GB: 5.2 GB of dataset, a 135 MB
policy, and the DistilBERT / ResNet50 caches. Re-running the command transfers
only changed files.

Smaller updates:

```bash
bash hpc/sync_to_hpc.sh --code
bash hpc/sync_to_hpc.sh --dataset
bash hpc/sync_to_hpc.sh --policy
bash hpc/sync_to_hpc.sh --caches
```

## 2. Create the Linux / CUDA environment

```bash
ssh 3153782y@cognition.cose.gla.ac.uk
cd ~/VLA-Robustness-Eval
bash hpc/setup_environment.sh
```

The verified Cognition module is:

```bash
module load miniforge3/25.3.0-3/none-none/a-4lhn6xy
```

Do not load a system CUDA module unless compiling a CUDA extension. The
PyTorch wheel provides its own CUDA runtime and uses the installed NVIDIA
driver.

The environment intentionally uses `robosuite==1.5.2`, `mink==0.0.5`, and
`mujoco==3.1.1` to match the frozen V2 Clean baseline. Their newer package
metadata requests a newer MuJoCo, so the setup script installs the pinned
runtime dependencies first and then installs mink and robosuite with
`--no-deps`. Do not upgrade MuJoCo for clean-baseline evaluation.

## 3. End-to-end L40S smoke test

```bash
cd ~/VLA-Robustness-Eval
sbatch hpc/smoke_test.sbatch
squeue -u "$USER"
```

Inspect the log:

```bash
tail -f ~/slurm-logs/vla-smoke-<job-id>.out
```

The required final line is:

```text
HPC CUDA + MuJoCo EGL + MiniVLA smoke test: PASS
```

## 4. Audit GPU access and queue pressure

Run this on the login node. It is read-only and does not allocate a GPU:

```bash
bash hpc/audit_gpu_resources.sh | tee ~/gpu-audit-$(date +%Y%m%d-%H%M).txt
```

The report lists GPU partitions and node states, queue pressure, the user's
Slurm associations and priorities, partition access rules, and five-minute
`sbatch --test-only` scheduling estimates. Use the estimate rather than GPU
model alone when choosing a partition for a short debug run.

For the 24-episode target-displacement smoke test, request fewer resources and
a shorter wall time so Slurm can backfill it:

```bash
RUN_NAME=target_displacement_v2_smoke \
LEVELS="0 0.02" \
EPISODES_PER_LEVEL=6 \
sbatch \
  --partition=gpu-l40s \
  --gres=gpu:l40s:1 \
  --cpus-per-task=4 \
  --mem=24G \
  --time=02:00:00 \
  hpc/benchmark_target_displacement.sbatch
```

If the audit confirms earlier H100 access, only change the partition and GRES:

```bash
RUN_NAME=target_displacement_v2_smoke_h100 \
LEVELS="0 0.02" \
EPISODES_PER_LEVEL=6 \
sbatch \
  --partition=gpu-h100 \
  --gres=gpu:h100:1 \
  --cpus-per-task=4 \
  --mem=24G \
  --time=02:00:00 \
  hpc/benchmark_target_displacement.sbatch
```

## 5. Submit a clean evaluation

```bash
sbatch hpc/evaluate_v2_clean.sbatch
```

Override the episode count or seed:

```bash
NUM_EPISODES=240 SEED=20261017 sbatch hpc/evaluate_v2_clean.sbatch
```

HPC evaluation is headless. OpenCV live windows remain a local-Mac workflow.

## 6. Submit training

Start with batch size 32 until the first GPU-memory and throughput measurement:

```bash
RUN_NAME=v2_clean_h100_test \
BATCH_SIZE=32 \
sbatch hpc/train_v2_clean.sbatch
```

After confirming memory headroom:

```bash
RUN_NAME=v2_clean_h100_batch64 \
BATCH_SIZE=64 \
sbatch hpc/train_v2_clean.sbatch
```

Warm-start from the frozen V2 Clean policy:

```bash
RUN_NAME=v2_next_h100 \
BATCH_SIZE=64 \
INIT_POLICY=/mnt/scratch/users/3153782y/VLA-Robustness-Eval/checkpoints/v2_clean/mini_vla_v2_clean_policy.pth \
INIT_SCOPE=all \
sbatch hpc/train_v2_clean.sbatch
```

The job copies data to Local Scratch, runs preflight, trains, runs postflight,
and synchronizes outputs back to:

```text
/mnt/scratch/users/3153782y/VLA-Robustness-Eval/runs/$RUN_NAME
```

## 7. Submit the V3 ACT chunk-size ablation

The ablation trains four independent policies with `chunk_size=1,5,10,20`.
All four jobs use the same frozen V2 Clean plus V3 recovery datasets, V2 Clean
initialization, language catalog, training seed, batch size, and 30-epoch
budget. Every policy reinitializes its action queries, including the 20-step
control, so the initialization is fair.

Verify the shared inputs before submission:

```bash
SHARED=/mnt/scratch/users/$USER/VLA-Robustness-Eval
find "$SHARED/datasets/dataset_v2_clean" -name 'ep_*.npz' | wc -l
find "$SHARED/datasets/dataset_v3_recovery" -name 'ep_*.npz' | wc -l
ls -lh "$SHARED/checkpoints/v2_clean/mini_vla_v2_clean_policy.pth"
```

The expected archive counts are 1,200 and 600. Submit one four-task array:

```bash
cd ~/VLA-Robustness-Eval
sbatch hpc/train_v3_chunk_ablation.sbatch
```

The task mapping is fixed:

```text
array task 0 -> chunk_size=1
array task 1 -> chunk_size=5
array task 2 -> chunk_size=10
array task 3 -> chunk_size=20
```

Inspect queue and logs using the array job ID returned by `sbatch`:

```bash
squeue -j <array-job-id> -o '%.18i %.12j %.10T %.10M %.20R'
tail -f ~/slurm-logs/v3-chunk-<array-job-id>_0.out
```

Results are synchronized from node-local scratch even when a task exits:

```text
/mnt/scratch/users/3153782y/VLA-Robustness-Eval/runs/
  v3_chunk_ablation_<array-job-id>/chunk_1/
  v3_chunk_ablation_<array-job-id>/chunk_5/
  v3_chunk_ablation_<array-job-id>/chunk_10/
  v3_chunk_ablation_<array-job-id>/chunk_20/
```

Each successful directory contains `mini_vla_v3_policy.pth`,
`training_log_v3.csv`, `training_metadata_v3.json`, a preflight report, a
six-episode postflight report, and `SHA256SUMS`.

## 8. Operational commands

```bash
squeue -u "$USER"
scontrol show job <job-id>
scancel <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

Never run training on the login node, never overwrite `CUDA_VISIBLE_DEVICES`,
and never leave the only copy of a checkpoint in Local Scratch.
