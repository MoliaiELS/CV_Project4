# Part 2 supplemental results

Start with [the analysis](report.md) or [HPC handoff and reproduction](../../docs/part2_handoff.md).

The 7000-step endpoints are preserved. Curves contain 30 fixed iteration points and 990 per-frame evaluations. All 198 original final RGB renders were reproduced exactly. Three optimized initializations match the original cameras and point clouds; three imported Part 1 VGGT Full checkpoints reproduce the archived PSNR.

Large assets stay on the HPC. [hpc_artifacts.json](hpc_artifacts.json) records 39 checkpoints and all required primary assets. [original_bundle_manifest.json](original_bundle_manifest.json) describes the historical HPC ZIP, not current Git files. release_manifest.json describes this Git delivery.
