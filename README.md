# VLA-Robustness-Eval

> A lightweight pipeline for evaluating the robustness and failure modes of manipulation policies in MuJoCo.

## 🎯 Project Objective
This project focuses on **Failure-Aware Robustness Evaluation** for robotic manipulation tasks. It evaluates baseline policies under various visual and language perturbations to categorize failure modes (e.g., visual perception failure, spatial reasoning failure).

## 🛠️ Tech Stack
- **Physics Engine:** MuJoCo
- **Manipulation Framework:** robosuite
- **Language:** Python 3.10

## 📂 Project Structure
- `scripts/`: Evaluation loops, perturbation utilities, and baseline policies.
- `results/`: Raw evaluation data, robustness drop metrics, and failure category logs.
- `docs/`: Project notes and experiment logs.