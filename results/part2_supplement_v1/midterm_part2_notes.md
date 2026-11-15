# Member A: two-page content

## Page 1: method
Show the custom VGGT no-BA -> fixed-camera 3DGS pipeline and the six-condition table. State strides 10/30/30, seed 42 and 7000 steps; reference cameras are evaluation-only. Full and Sparse were independently initialized and trained.

## Page 2: results
Show convergence_psnr.png and either failure_cases.png or paired_trajectories.png. Full improves PSNR/SSIM in every scene, most strongly on DL3DV (+3.13 dB). Waymo Sparse improves LPIPS. Lower sparse trajectory error does not guarantee higher rendering quality. Mention non-monotonic test curves, camera bias and visibility support.

Part 3 remains future work: two generative reconstruction rounds with matched real-only continuation controls.
