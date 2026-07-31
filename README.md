# VLA-Robustness-Eval

Failure-aware robustness evaluation for language-conditioned robot manipulation
in MuJoCo and robosuite.

## Current Pipeline

MiniVLA V3 is the current Clean release candidate. It learns six balanced
Panda manipulation tasks from the frozen V2 Clean demonstrations plus a
separate V3 recovery dataset:

- Pick or push a red cube, blue ball, or green cylinder.
- Official MuJoCo Python package with `OSC_POSE` 7D delta control.
- Agent and wrist RGB observations at 112 x 112.
- Five-step proprioceptive history and 20-step action chunks.
- Frozen DistilBERT language encoder and shared ResNet50 visual encoder.
- Separate pose, gripper, phase, contact, grasp, and grounding objectives.
- Dynamic language augmentation with disjoint train and evaluation phrases.
- Recovery demonstrations for target displacement and failed approach phases.
- Raw-policy closed-loop evaluation with no privileged object-state assistance.

Under the paired Clean protocol, V3 completed 235 / 240 episodes (97.92%),
compared with 222 / 240 (92.50%) for V2 Clean. This is a Clean-control result;
formal V3 perturbation curves are still required before making a robustness
claim.

## Environment

- macOS Apple Silicon
- UofG Cognition HPC with NVIDIA CUDA and MuJoCo EGL
- Python 3.10
- MuJoCo 3.1.1
- robosuite 1.5.2
- PyTorch with MPS acceleration

Install the pinned runtime dependencies inside the `mujoco310` environment:

```bash
python -m pip install -r requirements.txt
```

## Repository Layout

- `models/mini_vla_v2.py`: shared V2 / V3 multimodal ACT architecture.
- `scripts/collect_data_v2.py`: frozen balanced Clean collector.
- `scripts/collect_data_v3.py`: balanced recovery-data collector.
- `scripts/train_v3.py`: V2 Clean plus V3 recovery training pipeline.
- `scripts/evaluate_policy_v2.py`: raw-policy closed-loop evaluator.
- `scripts/benchmark_robustness_v2.py`: paired perturbation benchmark driver.
- `hpc/HPC_RUNBOOK.md`: Cognition HPC migration, Slurm, and storage workflow.
- `scripts/preflight_v3.py`: V3 data, language, loader, and warm-start gates.
- `scripts/postflight_v2.py`: full-trajectory checkpoint gate.
- `utils/`: schema, normalization, augmentation, and data loading.
- `results/`: local-only generated datasets and checkpoints.

See [docs/V3_RUNBOOK.md](docs/V3_RUNBOOK.md) for V3 collection, preflight,
HPC training, retrieval, and held-out-language evaluation commands. The frozen
V2 baseline remains documented in
[docs/V2_CLEAN_RUNBOOK.md](docs/V2_CLEAN_RUNBOOK.md).

The full engineering history, including failed V2.3 experiments and the
evidence used to freeze V2 Clean, is recorded in
[docs/V2_EVOLUTION.md](docs/V2_EVOLUTION.md).

The V3 checkpoint identity, dataset counts, verification commands, and current
evidence boundary are recorded in
[docs/V3_RELEASE_MANIFEST.md](docs/V3_RELEASE_MANIFEST.md).
