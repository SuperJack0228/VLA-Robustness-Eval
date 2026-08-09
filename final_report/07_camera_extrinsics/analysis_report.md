# MiniVLA V3 Camera-Extrinsic Robustness

## Protocol Audit

- Status: PASS.
- Raw episode rows: 600.
- Seeds: 2.
- MuJoCo agentview position and quaternion are modified directly.
- Scene-paired translation directions and rotation axes verified.
- Application, first-frame refresh, and per-episode restoration verified.
- Wrist camera remains unchanged; no privileged execution assistance.

## Success Rates

| Level | Translation | Rotation | Episodes | Success | 95% Wilson CI | Holm p vs L0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 mm | 0 deg | 120 | 98.33% | [94.1, 99.5] | reference |
| 1 | 5 mm | 1 deg | 120 | 99.17% | [95.4, 99.9] | 1 |
| 2 | 10 mm | 2 deg | 120 | 96.67% | [91.7, 98.7] | 1 |
| 3 | 15 mm | 3 deg | 120 | 91.67% | [85.3, 95.4] | 0.1157 |
| 4 | 20 mm | 4 deg | 120 | 78.33% | [70.1, 84.8] | 3.219e-06 |

## Interpretation Checklist

- Treat translation and rotation as a combined calibration severity, not separable causal effects.
- Compare paired episodes to distinguish calibration sensitivity from scene difficulty.
- Do not interpret simulated camera drift as complete real-camera calibration evidence.
