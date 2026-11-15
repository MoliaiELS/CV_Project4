# Part 2: three-scene Full / Sparse comparison

All six runs use 7000 steps and seed 42; same frozen configuration, model weights, implementation and evaluation protocol.

| Scene | Condition | Train/test | PSNR (dB) | SSIM | LPIPS | Paired ATE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Re10k-1 | full | 244/35 | 11.364303 | 0.290950 | 0.395653 | 0.029022 |
| Re10k-1 | sparse | 10/35 | 11.234161 | 0.275132 | 0.405510 | 0.026432 |
| Waymo-405841 | full | 174/25 | 17.689436 | 0.654151 | 0.520424 | 0.956493 |
| Waymo-405841 | sparse | 20/25 | 16.726005 | 0.597110 | 0.457374 | 0.729370 |
| Dl3dv-2 | full | 267/39 | 18.608708 | 0.514218 | 0.184315 | 0.037899 |
| Dl3dv-2 | sparse | 11/39 | 15.478618 | 0.455907 | 0.257178 | 0.029028 |

Macro-average gives each scene equal weight; it is not a pooled-pixel PSNR.
- full: PSNR 15.887482, SSIM 0.486440, LPIPS 0.366797.
- sparse: PSNR 14.479595, SSIM 0.442717, LPIPS 0.373354.

Paired ATE fits each estimate on the same sparse TRAIN subset within its scene (10 / 20 / 11 cameras). Units are reference trajectory units; ATE is not averaged across scenes.
No test RGB or test-pose refinement enters reconstruction. For rendering, each condition uses one global Sim3 fitted on its available TRAIN cameras.
Resolution is fixed per scene: Re10k 256x256, Waymo 960x640, Dl3dv 256x256; no downscaling for scoring.
The latest Part 1 handoff at GitHub commit 901c5ae has been verified against this Part 2 data: all 784 PNG hashes and frame records match; reference camera names and intrinsics match exactly, and poses differ by at most 4.44e-16. The current prepared-data discrepancy is resolved. See ../hpc_setup/part1_handoff_901c5ae/audit.json. Part 2 retains its original independently trained Full/Sparse pairs.
This is one seed and three scenes; results describe this fixed-budget comparison and are not a statistical generalization.
Each run contains verified reports, checkpoint hashes, all test renders, per-frame metrics and a first/middle/last-frame preview.
All computation and report generation ran on the HPC through direct connections with proxies disabled.

Observed differences (Full minus Sparse):

- Re10k-1: PSNR +0.130142 dB; SSIM +0.015818; LPIPS -0.009857.
- Waymo-405841: PSNR +0.963431 dB; SSIM +0.057041; LPIPS +0.063049.
- Dl3dv-2: PSNR +3.130090 dB; SSIM +0.058311; LPIPS -0.072862.

Sparse has lower paired ATE in all three scenes, while Full has higher mean PSNR/SSIM in all three. Lower center-trajectory error alone does not establish better held-out rendering quality.
Waymo is a mixed outcome: Full has higher PSNR on 24/25 test frames, while Sparse has lower LPIPS on all 25 frames.
Fixed first/middle/last previews were visually inspected; they were not chosen by score.
