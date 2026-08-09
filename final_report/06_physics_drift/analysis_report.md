# MiniVLA V3 Physics-Parameter Robustness

## Protocol Audit

- Status: PASS.
- Raw episode rows: 1080.
- Seeds: 2.
- Target-only parameter application and per-episode restoration both verified at 100%.
- Mass drift scales mass and inertia together; friction drift scales all target contact-friction components.
- Policy observations, scoring, and execution receive no privileged assistance.

## Success Rates

| Parameter | Multiplier | Episodes | Success | 95% Wilson CI | Holm p vs 1x |
|---|---:|---:|---:|---:|---:|
| target-mass | 0.25x | 120 | 99.17% | [95.4, 99.9] | 0.75 |
| target-mass | 0.5x | 120 | 98.33% | [94.1, 99.5] | 1 |
| target-mass | 1x | 120 | 96.67% | [91.7, 98.7] | reference |
| target-mass | 2x | 120 | 96.67% | [91.7, 98.7] | 1 |
| target-mass | 4x | 120 | 65.83% | [57.0, 73.7] | 5.821e-10 |
| target-friction | 0.25x | 120 | 96.67% | [91.7, 98.7] | 1 |
| target-friction | 0.5x | 120 | 96.67% | [91.7, 98.7] | 1 |
| target-friction | 1x | 120 | 96.67% | [91.7, 98.7] | reference |
| target-friction | 2x | 120 | 65.83% | [57.0, 73.7] | 4.366e-10 |
| target-friction | 4x | 120 | 64.17% | [55.3, 72.2] | 1.528e-10 |

## Main Findings

- Mass/inertia is stable through 2x, then falls from 96.67% at 1x to 65.83% at 4x.
- At 4x mass, Pick retains 70.00% while Push retains 61.67%.
- Friction is stable through 1x, but overall success drops to 65.83% at 2x and 64.17% at 4x.
- High friction is task-selective: Pick remains 98.33% at 4x while Push falls to 30.00%.

## Interpretation Checklist

- Separate Pick and Push because their sensitivity mechanisms differ.
- Attribute changes to the tested simulator parameters, not general sim-to-real robustness.
- Use failure taxonomy to distinguish grasp loss, push distance, and contact failures.
