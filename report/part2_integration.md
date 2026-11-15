# Part 2 report integration

Include [part2.tex](part2.tex) in the shared CVPR paper and add
`../results/part2-final/figures/` to the graphics path when building from
`report/`. Reuse `kerbl2023gaussians` and `wang2025vggt` from the Part 1
bibliography. The fragment uses `booktabs` and `graphicx`.

The group copy uses the course-only failure panel at
`../output/handoff/part2-final/failure_cases.png` when present. The public
source falls back to the paired trajectory plot. The original Full/Sparse
stage timing is in [runtime.csv](../results/part2-final/runtime.csv).
