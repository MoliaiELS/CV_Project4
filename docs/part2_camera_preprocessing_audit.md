# Part 2 camera and preprocessing audit

All checks and experiments run on the HPC through direct connections with proxy
environment variables cleared. The raw shared dataset is read-only.

## Camera interpretation

Re10k's 279 reference cameras were independently reconstructed from the original
xyzw quaternions, camera centers and normalized intrinsics using SciPy.
Rotation disagreement was at most 4.44e-16; translation and intrinsic disagreement
were zero. VGGT's official 518-square padding/resizing matches the pipeline's
pixel transform. Projection versus OpenCV differs by at most 0.0000112 pixels;
training scene normalization preserves projections within 0.0000549 pixels.

Nine train-only image pairs provided 95--152 robust feature correspondences each.
Their median Sampson errors were about 0.11--0.17 pixels with reference cameras
and 0.16--0.54 pixels with VGGT cameras. No axis inversion, wrong quaternion order,
wrong frame assignment or intrinsic-resize implementation error was found.

The native 700-step pilot's approximately 20% focal discrepancy and 3.85-degree
mean orientation discrepancy remain model-estimation errors. Reference parameters
are not substituted into the RGB-only VGGT initializer or frozen-camera trainer.

## B's preprocessing

The source archive SHA256 equals B's published value:
2e2a2658d6735761b5dd3c090c5d35a4a2cca04088cde6ae0cb231f90214c415.
Every one of its 1,598 files matches the HPC shared source.

A separate HPC environment with B's NumPy 2.5.3, Pillow 12.3.0 and
OpenCV 5.0.0.93 produced exactly the same 784 prepared PNG files and reference
arrays as the server-pinned environment. All 279 Re10k prepared RGB arrays also
equal the original decoded source RGB, with no resize or distortion operation.

The published B prepared hashes still differ. Actual B manifest/reference/PNG
artifacts were unavailable. Aggregate hashes cannot establish the cause of this
difference; possible encoding or stale-artifact differences are not asserted as
facts. The formal run therefore independently computes BOTH Full and Sparse on
the same HPC canonical inputs. B's existing geometry, checkpoints and scores are
not reused, and results must not silently be combined across these identities.

## Full-sequence memory scheduling

The pinned VGGT model is evaluated with FP32 weights/residuals and BF16 autocast.
All 244 Full training views participate in each global attention layer.
CPU residual storage, full GPU key/value tensors, batches of 4,096 query tokens,
and lossless on-disk DPT features reduce memory pressure. Frame-independent work
and the official depth head use batches of 8 frames.

Native/offload validation exercises multiple query batches. The synthetic block
matches exactly. Two real views differ by at most 0.006115 degrees in orientation
and 0.081016% in focal length. Median depth relative difference is 0.019030%;
the 99th percentile is 0.181810%. This is numerical agreement, not bitwise identity.
Peak allocation in that check drops from 6.818 GiB to 2.506 GiB.

## Frozen formal comparison

Both conditions use the offload backend, the same ModelScope VGGT weights,
seed 42, a 100,000-point initialization cap and the original 7,000-step schedule.
Full uses 244 training images; Sparse uses 10; both score the same 35 test images.
Each condition starts from an independent RGB-only initialization.

Rendering fits one global Sim3 from each condition's available training cameras.
The paired ATE is separately fitted and measured on the same 10 sparse training
frame IDs in BOTH conditions, in reference trajectory units.

Execution and completion status:
runs/part2_formal/Re10k-1/full_sparse_7000_v1/status.json.
All audit JSON files are in results/hpc_setup.

## Completed experiment
Both 7000-step conditions completed successfully. See the run directory
report.md, comparison.json, camera_diagnostics.json, preview.jpg and verification.json.
The 10-view offload initialization is byte-identical to the earlier native
scene.npz, including names, cameras, intrinsics, points and colors.
