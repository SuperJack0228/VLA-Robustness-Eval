# MiniVLA V3 Visual-Degradation Robustness

## Protocol Audit

- Status: PASS.
- Raw episode rows: 3000.
- Seeds: 2.
- Shared paired Clean scenes across every visual condition.
- Both policy cameras are degraded; simulator physics and scoring remain unchanged.
- No privileged execution assistance or perturbation oracle.

## Success Rates

| Corruption | Level | Episodes | Success | 95% Wilson CI | Holm p vs Clean |
|---|---:|---:|---:|---:|---:|
| gaussian-noise | 0 | 120 | 96.67% | [91.7, 98.7] | reference |
| gaussian-noise | 10 | 120 | 99.17% | [95.4, 99.9] | 1 |
| gaussian-noise | 20 | 120 | 98.33% | [94.1, 99.5] | 1 |
| gaussian-noise | 30 | 120 | 98.33% | [94.1, 99.5] | 1 |
| gaussian-noise | 40 | 120 | 97.50% | [92.9, 99.1] | 1 |
| gaussian-noise | 60 | 120 | 95.83% | [90.6, 98.2] | 1 |
| gaussian-noise | 80 | 120 | 90.83% | [84.3, 94.8] | 0.5537 |
| gaussian-noise | 120 | 120 | 54.17% | [45.3, 62.8] | 8.393e-14 |
| gaussian-noise | 160 | 120 | 36.67% | [28.6, 45.6] | 6.353e-20 |
| gaussian-blur | 0 | 120 | 96.67% | [91.7, 98.7] | reference |
| gaussian-blur | 0.75 | 120 | 99.17% | [95.4, 99.9] | 1 |
| gaussian-blur | 1.5 | 120 | 99.17% | [95.4, 99.9] | 1 |
| gaussian-blur | 2.25 | 120 | 98.33% | [94.1, 99.5] | 1 |
| gaussian-blur | 3 | 120 | 95.00% | [89.5, 97.7] | 1 |
| gaussian-blur | 4 | 120 | 89.17% | [82.3, 93.6] | 0.1123 |
| gaussian-blur | 6 | 120 | 44.17% | [35.6, 53.1] | 1.853e-16 |
| gaussian-blur | 10 | 120 | 19.17% | [13.1, 27.1] | 3.393e-26 |
| brightness | 1 | 120 | 96.67% | [91.7, 98.7] | reference |
| brightness | 0.8 | 120 | 98.33% | [94.1, 99.5] | 1 |
| brightness | 0.6 | 120 | 99.17% | [95.4, 99.9] | 1 |
| brightness | 0.4 | 120 | 96.67% | [91.7, 98.7] | 1 |
| brightness | 0.3 | 120 | 97.50% | [92.9, 99.1] | 1 |
| brightness | 0.2 | 120 | 97.50% | [92.9, 99.1] | 1 |
| brightness | 0.1 | 120 | 94.17% | [88.4, 97.1] | 1 |
| brightness | 0.05 | 120 | 76.67% | [68.3, 83.3] | 5.633e-06 |
| brightness | 0.02 | 120 | 20.83% | [14.5, 28.9] | 1.519e-25 |
| brightness | 0.01 | 120 | 1.67% | [0.5, 5.9] | 8.667e-34 |

## Main Findings

- Shared Clean success is 96.67%.
- 7 degraded condition(s) differ significantly from paired Clean after within-curve Holm correction.
- At maximum severity, success remains 36.67% for noise, 19.17% for blur, and 1.67% for brightness reduction.
- Mean grounding error rises from 0.707 cm Clean to 10.648, 7.977, and 22.717 cm at the three maximum severities.
- First sub-80% point: noise level 120 (54.17% success); blur level 6 (44.17% success); brightness level 0.05 (76.67% success).
- First sub-50% point: noise level 160 (36.67% success); blur level 6 (44.17% success); brightness level 0.02 (20.83% success).
- Target-class accuracy at maximum severity is 100.00% for noise, 100.00% for blur, and 100.00% for brightness reduction.

## Interpretation Checklist

- Report degradation curves and confidence intervals, not a single pass/fail threshold.
- Use target-class/contact diagnostics to separate perception failure from manipulation failure.
- Do not interpret visual simulation robustness as sim-to-real evidence.
