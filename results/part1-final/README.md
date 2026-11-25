# Part 1 dense reconstruction

The three course scenes each use COLMAP, VGGT, and VGGT-Ω initialization followed by the same fixed-camera 3DGS schedule. Every run uses 7,000 updates and seed 42. The holdout contains 25 Waymo, 39 DL3DV, and 35 Re10k images.

- [Final metrics, ATE and timing](coverage.csv)
- [Five fixed evaluation points per run](convergence.csv) and [PSNR curves](convergence.png)
- [Source versions, split and weights hashes](provenance.json)
- [Reproduction configuration](../../configs/part1_reproduce.json)
- [Matched 30,000-step DL3DV follow-up](../postpre_optimization/README.md#part-1-稠密训练预算)

The holdout cameras use one Sim(3) fitted on the training correspondences. ATE is measured in each reference trajectory's units.
