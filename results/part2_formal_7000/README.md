# Part 2 unposed full and sparse reconstruction

The six formal runs independently estimate cameras and geometry from their respective training RGBs and use 7,000 fixed-camera updates with seed 42. Full and sparse reconstructions share the same holdout IDs and image sizes within each scene.

- [Scene-level metrics, timing and checkpoint hashes](comparison.csv)
- [Per-frame holdout metrics](per_frame_metrics.csv)
- [Paired full–sparse differences and frame counts](paired_details.json)
- [Artifact paths and hashes](artifact_hashes.json)
- [Convergence, coverage and pose diagnostics](../part2_supplement_v1/README.md)
- [Post-presentation pose ablations](../postpre_optimization/README.md#part-2-固定相机与联合优化)

Each rendering alignment uses a Sim(3) fitted to its available training cameras. The paired ATE compares the full and sparse estimates after fitting on the same sparse training IDs. The ATE units follow each scene's reference trajectory. Recorded run paths are relative to the experiment root.
