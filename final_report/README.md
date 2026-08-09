# Final Paper Results

This directory is the single entry point for paper-ready figures and their
supporting summary tables. Every figure is provided as PNG for convenient use
and PDF for vector-quality publication.

## Categories

1. `01_clean_baseline/`
   - Paired V2 versus V3 clean performance and aggregate metrics.
2. `02_dynamic_displacement/`
   - Main 0-8 cm displacement robustness curves, paired statistics, failure
     taxonomy, and V3-only appendix plots.
3. `03_chunk_size_ablation/`
   - Chunk sizes 1, 5, 10, and 20 under clean and 4 cm displacement.
4. `04_training_diagnostics/`
   - Final V3 training and validation diagnostics.
5. `05_visual_degradation/`
   - Paired Gaussian-noise, Gaussian-blur, and brightness-reduction curves,
     visual-grounding diagnostics, per-task severity heatmap, and failure
     taxonomy. The final severity grid spans the stable regime through clear
     sub-80% and sub-50% policy failure boundaries.
6. `06_physics_drift/`
   - Paired target mass/inertia and contact-friction robustness curves,
     task-family sensitivity, per-task heatmap, and failure taxonomy.
7. `07_camera_extrinsics/`
   - Paired fixed-camera translation and rotation drift curves, visual
     grounding diagnostics, per-task sensitivity, and failure taxonomy.

Raw episode-level evidence remains under `results/benchmarks/`. Frozen policies
and training provenance remain under `artifacts/`.
