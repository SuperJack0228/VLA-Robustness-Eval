# VLA-Robustness-Eval

Failure-aware robustness evaluation for language-conditioned robot manipulation
in MuJoCo and robosuite.

## Current Pipeline

MiniVLA V2 Clean learns six balanced Panda manipulation tasks from dual-view
demonstrations:

- Pick or push a red cube, blue ball, or green cylinder.
- Official MuJoCo Python package with `OSC_POSE` 7D delta control.
- Agent and wrist RGB observations at 112 x 112.
- Five-step proprioceptive history and 20-step action chunks.
- Frozen DistilBERT language encoder and shared ResNet50 visual encoder.
- Separate pose, gripper, phase, contact, grasp, and grounding objectives.
- Raw-policy closed-loop evaluation with no privileged object-state assistance.

## Environment

- macOS Apple Silicon
- Python 3.10
- MuJoCo 3.1.1
- robosuite 1.5.2
- PyTorch with MPS acceleration

Install the pinned runtime dependencies inside the `mujoco310` environment:

```bash
python -m pip install -r requirements.txt
```

## Repository Layout

- `models/mini_vla_v2.py`: multimodal policy architecture.
- `scripts/collect_data_v2.py`: balanced clean demonstration collector.
- `scripts/train_v2.py`: interleaved training and checkpoint pipeline.
- `scripts/evaluate_policy_v2.py`: raw-policy closed-loop evaluator.
- `scripts/preflight_v2.py`: dataset and code gates before training.
- `scripts/postflight_v2.py`: full-trajectory checkpoint gate.
- `utils/`: schema, normalization, augmentation, and data loading.
- `results/`: local-only generated datasets and checkpoints.

See [docs/V2_CLEAN_RUNBOOK.md](docs/V2_CLEAN_RUNBOOK.md) for the complete
collection, training, resume, postflight, and evaluation commands.
