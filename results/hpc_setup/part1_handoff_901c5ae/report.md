# Part 1 latest handoff verification

GitHub main: 901c5aedfe4627e7af93814e8f37e847237adb7b. Published Part 1 archive SHA256 verified: d38e5431f7e3a0cccb00bc12bdbffa66ccd9c9c6edddd35ede10783a06bcf54c.
Nine completed Part 1 runs are archived, superseding the older screenshot showing six.
All 784 prepared-image file hashes match B's actual manifests; all frame records, camera names and intrinsics are exactly equal; reference poses agree to floating-point roundoff (maximum absolute difference 4.44e-16).
Manifest byte hashes differ only through the metadata fields listed in audit.json (such as absolute paths).
The latest prepared-data discrepancy is resolved. Earlier aggregate-only evidence is retained as historical evidence.

The original Part 2 six runs and their Full/Sparse pairing remain unchanged.
Part 1 and Part 2 use identical training settings and image-metric protocols, but different GPUs and VGGT memory scheduling; use their own provenance and timing.
Part 1 ATE uses all training cameras; Part 2 paired ATE uses the common sparse subset. They are separate metrics.
Existing checkpoints bind the original code and manifest. Use their original source and preparation metadata for resuming or rendering.

## Checkpoint accessibility

Resolved after the teammate supplied the new shared checkpoint directory.
All nine 7000-step checkpoint source files and copied files match both SHA256SUMS and the checkpoint records in the previously verified Part 1 archive.
Copied directory: /data/disk2/home/zguo400/CV_Project4/runs/part1_import_901c5ae/checkpoints
Total: 3639989522 bytes. Copy and verification: 45.94 seconds.
The original IADC paths remain unmounted, but the verified course-HPC copies are now available.
Use the original Part 1 training source bde9244f692cc497355a307826ce1ca25f39cd99 and original dataset manifest when restoring or rendering.
See checkpoint_import_receipt.json for the nine original/shared/copied paths, sizes, hashes and timings.

## Imported artifacts

/data/disk2/home/zguo400/CV_Project4/results/hpc_setup/part1_handoff_901c5ae

archive/ contains the nine scene initializations, metadata, logs, metrics, dataset manifests and reference arrays.
Nine large Part 1 checkpoints are now copied and verified. No Part 2 training code, checkpoint or original evaluation was replaced.

Historical search records are retained; they describe the earlier state before the new shared checkpoint directory was provided.
