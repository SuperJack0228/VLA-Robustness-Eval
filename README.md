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
compared with 222 / 240 (92.50%) for V2 Clean. The final paired 0-8 cm target
displacement benchmark and ACT chunk-size ablation are also complete; paper
figures and statistical tables are indexed in `final_report/`.

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

## Interactive Desktop Demo

Launch the local industrial-pixel MiniVLA control deck on macOS:

For the simplest launch, double-click `MiniVLA Demo.app` in Finder. Keep the app
in the repository root so it can locate the model and scripts. Startup
diagnostics are written only to the macOS temporary directory as
`minivla_demo_launch.log`.

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n mujoco310 \
  python scripts/launch_demo_v3.py
```

The desktop app keeps the 512 x 512 simulation view and both 112 x 112 policy
camera views visible. It accepts flexible pick / push language, maps valid
finite-domain commands onto a reliable V3 policy prompt, supports Start, Stop,
collision-safe scene Reset, and an interactive 4 cm target nudge. Run summaries
remain in memory and are discarded when the app closes.

The simulator and policy run in a spawned child process. This keeps MuJoCo's
macOS OpenGL context on that process's main thread and isolates it from the Qt
interface. Demo-only observation and stop hooks are disabled by default, so
normal benchmark behavior and its output protocol remain unchanged.

## Repository Layout

- `data/`: local 1,200-episode V2 Clean and 600-episode V3 recovery datasets.
- `artifacts/v2-clean-rc1/`: distilled V2 policy and training provenance.
- `artifacts/v3-clean-rc1/`: frozen final V3 policy and release evidence.
- `artifacts/chunk-ablation-v3/`: chunk 1, 5, and 10 ablation policies.
- `final_report/`: paper-ready PNG/PDF figures, tables, and analysis reports.
- `results/benchmarks/`: raw paired CSV/JSON benchmark evidence.
- `models/mini_vla_v2.py`: shared V2 / V3 multimodal ACT architecture.
- `scripts/collect_data_v2.py`: frozen balanced Clean collector.
- `scripts/collect_data_v3.py`: balanced recovery-data collector.
- `scripts/train_v3.py`: V2 Clean plus V3 recovery training pipeline.
- `scripts/evaluate_policy_v2.py`: raw-policy closed-loop evaluator.
- `scripts/launch_demo_v3.py`: local three-camera interactive desktop app.
- `scripts/benchmark_robustness_v2.py`: paired perturbation benchmark driver.
- `hpc/HPC_RUNBOOK.md`: Cognition HPC migration, Slurm, and storage workflow.
- `scripts/preflight_v3.py`: V3 data, language, loader, and warm-start gates.
- `scripts/postflight_v2.py`: full-trajectory checkpoint gate.
- `utils/`: schema, normalization, augmentation, and data loading.
- `utils/interactive_session_v3.py`: persistent, non-logging demo runtime.
- `utils/instruction_resolver_v3.py`: reliable six-task command resolver.
- `ui/minivla_demo_window.py`: PySide6 industrial-pixel interface.
- `results/`: generated raw benchmark records and future training outputs.

See [docs/runbooks/V3_RUNBOOK.md](docs/runbooks/V3_RUNBOOK.md) for V3 collection, preflight,
HPC training, retrieval, and held-out-language evaluation commands. The frozen
V2 baseline remains documented in
[docs/runbooks/V2_CLEAN_RUNBOOK.md](docs/runbooks/V2_CLEAN_RUNBOOK.md).

The full engineering history, including failed V2.3 experiments and the
evidence used to freeze V2 Clean, is recorded in
[docs/history/V2_EVOLUTION.md](docs/history/V2_EVOLUTION.md).

The V3 checkpoint identity, dataset counts, verification commands, and current
evidence boundary are recorded in
[docs/runbooks/V3_RELEASE_MANIFEST.md](docs/runbooks/V3_RELEASE_MANIFEST.md).
The six-slide stage presentation is available at
[docs/presentations/2026-07-28-LiuYining-MiniVLA-V3-Update.pptx](docs/presentations/2026-07-28-LiuYining-MiniVLA-V3-Update.pptx).

The 135 MB V3 policy is versioned with Git LFS. Install Git LFS before cloning
or pulling the release artifacts:

```bash
brew install git-lfs
git lfs install
```
