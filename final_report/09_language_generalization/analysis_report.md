# MiniVLA V3 Language Generalization

## Protocol

- 180 evaluation-only expressions: 30 for each of six task buckets.
- Each held-out instruction is paired with its canonical instruction on the exact same scene.
- The raw text is passed directly to DistilBERT; no UI resolver or canonical rewriting is used.
- Ground-truth task metadata is restricted to scene generation and scoring. Action execution uses model-predicted phase and target only.

## Main Result

- Canonical success: 98.89% (178/180)
- Held-out success: 94.44% (170/180)
- Held-out minus canonical: -4.44 percentage points
- Held-out retention: 95.51%
- Paired McNemar p-value: 0.0214844
- Canonical-only successes: 9
- Held-out-only successes: 1

## Semantic Routing

- Canonical initial six-task accuracy: 100.00%
- Held-out initial six-task accuracy: 95.00%
- Held-out operation accuracy: 95.00%
- Held-out target accuracy: 100.00%

## Audit

Protocol audit: PASS. See `protocol_audit.json` for machine-readable checks.
