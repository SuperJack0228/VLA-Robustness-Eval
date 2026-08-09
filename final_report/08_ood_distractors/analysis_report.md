# MiniVLA V3 OOD-Distractor Robustness

## Protocol Audit

- Status: PASS.
- Raw episode rows: 480.
- Seeds: 2.
- All counts share the same task, scene seed, and three-candidate layout.
- Higher counts activate a strict D1/D2/D3 prefix.
- Initial target visibility, nominal robot path, and collision integrity verified.
- No privileged execution assistance.

## Success Rates

| OOD objects | Episodes | Task success | Collision-aware | OOD contact | Selection failure | Holm p vs 0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 120 | 95.83% | 95.00% | 0.00% | 0.00% | reference |
| 1 | 120 | 97.50% | 96.67% | 0.00% | 5.00% | 0.9062 |
| 2 | 120 | 93.33% | 93.33% | 0.83% | 5.00% | 0.9062 |
| 3 | 120 | 92.50% | 92.50% | 0.83% | 8.33% | 0.8672 |

## Interpretation Checklist

- Task success measures whether the commanded manipulation completed.
- Collision-aware success additionally rejects contacts with OOD or wrong trained objects.
- Target-selection failure requires at least three consecutive predictions closer to an OOD object than to the true target.
- Static distractors isolate perception and path-selection robustness; they are not a dynamic-obstacle benchmark.
