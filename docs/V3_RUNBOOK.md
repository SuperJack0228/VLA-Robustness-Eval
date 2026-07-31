# MiniVLA V3 Runbook

V3 is a conservative extension of the frozen V2 Clean baseline:

- Keep `results/dataset_v2_clean` unchanged.
- Collect recovery-only demonstrations in `results/dataset_v3_recovery`.
- Fine-tune all V2 Clean weights with the original normalization statistics.
- Sample language dynamically from `configs/language_augmentations_v3.json`.
- Use 60 training expressions and 30 disjoint evaluation expressions per task.
- Keep the 20-step ACT architecture; robust evaluation still replans every step.

## 1. Collect Recovery Data on Mac

```bash
cd /Users/superjack/VLA-Robustness-Eval

caffeinate -dimsu env \
  PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n mujoco310 \
  python scripts/collect_data_v3.py \
    --num-episodes 600 \
    --data-dir results/dataset_v3_recovery \
    --seed 20261201 \
  2>&1 | tee results/collection_console_v3.log
```

The collector is resumable. Running the same command again continues from the
existing balanced archive.

## 2. Run the Local Gate

```bash
conda run --no-capture-output -n mujoco310 \
  python scripts/preflight_v3.py \
    --base-data-dir results/dataset_v2_clean \
    --recovery-data-dir results/dataset_v3_recovery \
    --expected-base-episodes 1200 \
    --expected-recovery-episodes 600 \
    --init-policy results/v2_clean/mini_vla_v2_clean_policy.pth \
    --batch-size 32 \
    --num-workers 4 \
    --local-files-only \
    --output results/v3/preflight_report_v3.json
```

Do not start the HPC training job unless this prints `V3 PREFLIGHT: PASS`.

## 3. Synchronize to Cognition

```bash
caffeinate -dimsu bash hpc/sync_to_hpc.sh --code
caffeinate -dimsu bash hpc/sync_to_hpc.sh --dataset-v3
caffeinate -dimsu bash hpc/sync_to_hpc.sh --dataset
caffeinate -dimsu bash hpc/sync_to_hpc.sh --policy
```

The last two commands are unnecessary when the frozen V2 Clean dataset and
policy are already present and unchanged on Shared Scratch.

## 4. Submit V3 Training

```bash
ssh 3153782y@cognition.cose.gla.ac.uk
cd ~/VLA-Robustness-Eval
sbatch hpc/train_v3.sbatch
squeue -u "$USER"
```

The default is one L40S GPU, batch size 32, 8 workers, and 30 epochs. To request
an H100 instead:

```bash
sbatch \
  --partition=gpu-h100 \
  --gres=gpu:h100:1 \
  hpc/train_v3.sbatch
```

## 5. Monitor and Retrieve

```bash
tail -f ~/slurm-logs/minivla-v3-<JOB_ID>.out
```

Artifacts are synchronized automatically to:

```text
/mnt/scratch/users/3153782y/VLA-Robustness-Eval/runs/v3_hpc_<JOB_ID>/
```

## 6. Evaluate Held-Out Language

Example:

```bash
conda run --no-capture-output -n mujoco310 \
  python scripts/evaluate_policy_v2.py \
    --policy results/v3/mini_vla_v3_policy.pth \
    --num-episodes 10 \
    --instruction "Raise the crimson cube from the table" \
    --task-type pick \
    --target-id A \
    --replan-interval 1 \
    --ensemble-mode temporal \
    --temporal-profile robust \
    --render \
    --local-files-only \
    --output-prefix results/v3/eval_pick_A_heldout
```
