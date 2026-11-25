# Part 2 supplemental measurements

The [six formal endpoints](../part2_formal_7000/README.md) use the frozen 7,000-step budget. The convergence study covers 30 fixed checkpoints at 500, 1,000, 2,000, 4,000 and 7,000 steps and 990 per-frame evaluations.

- [Convergence measurements](convergence.csv) and [three-metric curves](figures/convergence_all_metrics.png)
- [Opacity coverage](coverage.csv), [per-frame coverage](coverage_per_frame.json) and [failure cases](figures/failure_cases.png)
- [Camera diagnostics](camera_diagnostics.csv) and [initialization, training, rendering and evaluation times](runtime.csv)
- [Part 1 VGGT Full replay](part1_checkpoint_replay.json) and [comparison with the native Part 2 Full runs](part1_reuse_comparison.csv)
- [Source revisions and original archive hashes](course_merge_provenance.json)

The replay reproduced the three original Part 1 final PSNR values at six decimal places. The primary comparison uses the independently trained Part 2 full and sparse runs. Paths in the tables and receipts are relative to the experiment root.
