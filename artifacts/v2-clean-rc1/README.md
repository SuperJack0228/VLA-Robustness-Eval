# MiniVLA V2 Clean Release Candidate 1

This directory preserves the distilled V2 Clean baseline required for paired
comparison with V3.

- `mini_vla_v2_clean_policy.pth`: final inference-only V2 policy.
- `warm_start_v2_2_policy.pth`: historical initialization used by V2 Clean.
- `dataset_stats_v2_clean.npz`: action normalization statistics.
- `training_log_v2_clean.csv`: persistent epoch metrics.
- `training_metadata_v2_clean.json`: architecture and training provenance.
- `preflight_report_v2_clean.json`: dataset gate.
- `postflight_v2_clean.json`: policy gate.
- `oracle_benchmark_v2_clean.json`: expert baseline evidence.

Training-resume checkpoints (`best` and `last`) were intentionally removed
after the final inference policy and provenance files were verified.
