# Part 3 results

The [comparison table](comparison.csv) reports each raw-render evaluation of the sparse starting model and both continuation rounds. The [per-frame table](per_frame_metrics.csv) preserves the matched test IDs and image-level metrics. Training and generation are documented in the [reproduction guide](../../docs/part3.md).

The active, confidence-weighted branch finishes 0.016, 0.022, and 0.019 dB above the sparse starting points on Re10k, DL3DV, and Waymo, respectively. It improves PSNR and LPIPS relative to the matched real-only second round on these three scenes. SSIM is mixed. On DL3DV, the highest PSNR of the active, confidence-weighted branch occurs in the first round. On Re10k, the final active and fixed selection variants differ by 0.005 dB.

The additional Bonsai benchmark uses 12 sparse training images and 37 held-out images. Its active, confidence-weighted PSNR rises from 14.435 to 14.468 dB in round one and falls to 14.439 dB in round two. The final result is 0.035 dB above real-only continuation, with lower LPIPS. [Official input and split receipt](inputs/Bonsai/source_receipt.json)

- [All three metrics by round](figures/closed_loop.png) and [PSNR change from the sparse starting point](figures/closed_loop_psnr.png)
- [Selected held-out PSNR gain and loss cases](figures/heldout_comparison.png) and [the source frame IDs](visual_evidence.json)
- [Virtual views before and after restoration](figures/pseudo_views.png) and [selected targets along the sparse paths](figures/active_trajectory.png)
- [Release verification](verification.json) records checkpoint lineage, paired real-image sampling checks, 38 evaluated stages, and 1,294 scored held-out renders.

Stage timings cover the successful attempts at the reported precision, view budget, and update budget.
