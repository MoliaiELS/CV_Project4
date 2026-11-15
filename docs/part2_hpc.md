# HPC Part 2 pilot
This is a sparse-only single-scene validation, not a completed full/sparse benchmark.

## Location and environment
All preprocessing, development and GPU work run under /data/disk2/home/zguo400.
Base repository: MoliaiELS/CV_Project4 at 8297339a1291b154c0f7adee9e3c701ee36c47bb.
Development branch: member-a-part2.
Environment: /data/disk2/home/zguo400/envs/project4 (Python 3.10).
Bootstrap logs and explicit no-proxy settings: ../p4-bootstrap/.
Start an interactive environment:
```bash
source /data/disk2/home/zguo400/p4-bootstrap/direct.env
export PATH=/data/disk2/home/zguo400/envs/project4/bin:$PATH
cd /data/disk2/home/zguo400/CV_Project4
```

## Data protocol
Raw course data is read from /data/disk1/shared/aiaa3201/project4_3d.
The repository's explicit full train/test lists are preserved.
Sparse sampling uses original natural-order sequence positions 1, 1+k, 1+2k, ...
with k=10 for Waymo and k=30 for DL3DV/Re10k.
Offset 1 is a group-defined phase avoiding the existing test indices 0,8,16,... .
This implements the course stride; it is not claimed as an exact reproduction of
BRPO's held-out split or frame phase. The paper specifies strides but not a phase:
https://arxiv.org/html/2602.21535v1 (Implementation details).
Expected sparse train/test counts: Waymo 20/25, DL3DV 11/39, Re10k 10/35.
Unsampled train images are excluded entirely; they are not added to test.
Sparse images retain the canonical prepared bytes; sparse geometry is re-estimated.
Reference cameras remain evaluation-only. Their units are not asserted to be meters.
Full reuse requires matching preprocessing, image hashes, weights, configuration,
and a new paired ATE calculation on common original frame IDs.

## Weights
VGGT: facebook/VGGT-1B from ModelScope revision b911f29cc346b2add4263e89792d68cfeca841ef.
model.pt SHA256: d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0.
This matches the original official Hugging Face weight's published SHA256.
Receipt: weights/vggt/model.pt.source.json. Strict weights-only loading is used.
LPIPS AlexNet backbone is cached separately under TORCH_HOME.
No proxy is used by bootstrap, downloads or experiment subprocesses.

## Validation
```bash
python -m pytest -q
python scripts/gpu_smoke.py --output runs/gpu_smoke_hpc_NEW
python scripts/part2_pilot.py --output runs/part2_pilot/Re10k-1/sparse_700_NEW
```
Use a new output path for each attempt to preserve failure evidence.
The synthetic smoke checks numerical resume, rendering, LPIPS, densification
and opacity reset; it is not a course reconstruction result.
The real pilot uses 10 Re10k RGB frames, independent VGGT(no-BA), 100k initial
points, and the Part 1 7k schedule. It stops at 100, restores the checkpoint,
continues to 700 (covering the first densification), renders all 35 held-out
cameras, and records PSNR/SSIM/LPIPS plus train-aligned ATE.
The 700-step render is a pilot outcome and not the formal 7k result.
Actual completion and measurements belong in each run's status.json, not this plan.

## Installed source provenance
gsplat 1.5.3 uses its official PyPI wheel because direct GitHub cloning failed
and the repository archive includes large documentation media. Its CUDA sources
are compiled locally for the A6000; this is not a third-party binary mirror.
Wheel SHA256: 515a3773641f5e7f7717acab6276c0b1d6dbcad087b7968ca653337c3189a982.
VGGT code is a clean sparse checkout at a288dd0f14786c93483e45524328726ab7b1b4ce.
See results/hpc_setup/source_artifacts.json and environment.txt.

## Prepared-data compatibility
The three HPC dense manifest fingerprints differ from B's published data audit.
HPC uses the repository's server-pinned NumPy 1.26.4, Pillow 11.1.0 and OpenCV
4.11.0. Reconcile exact prepared images, metadata and software before reusing
B's Full rows. Sparse train and test remain disjoint; the fixed test IDs and
canonical bytes are unchanged by sparse selection. Actual hashes and checks
are in results/hpc_setup/data_preparation.json.

## Observed pilot result (2026-11-15)
48 CPU tests passed. GPU resume (same exact checkpoint), densification, opacity
reset, rendering and LPIPS passed in runs/gpu_smoke_hpc_02. The original smoke
test compared independently trained prefixes; it was corrected to isolate resume.
Re10k-1 sparse_700 completed: 10 training images, 35 held-out images, 700 steps.
PSNR 11.46715 dB, SSIM 0.292828, LPIPS 0.451874; ATE 0.0264319 in reference
trajectory units. Total subprocess wall time was 207.77 seconds, excluding setup.
The scene is recognizable, with viewpoint deviations, border holes and blur.
Train-only diagnostics show mean orientation discrepancy 3.85 degrees and median
focal ratios about 1.20 against the reference. This needs investigation before
formal comparisons; no held-out poses or calibration were optimized.
Detailed summary: results/hpc_setup/part2_pilot_summary.json.
Run evidence: runs/part2_pilot/Re10k-1/sparse_700/{status.json,metrics.json,
diagnostics.json,preview.jpg,train/checkpoints/step_000700.pt}.
