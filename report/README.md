# Project 4 CVPR report

[final.tex](final.tex) is the combined CVPR paper. Its method and experiments use the frozen Part 1 and Part 2 results, with the Part 3 sections based on the run artifacts. Keep the repository directory structure intact so the report can find figures and the two bibliography files.

The course requires 6–8 body pages, plus references, and a public GitHub URL at the end of the abstract. Verify the author block and use the compiled PDF for Canvas. The source and figures are kept with the public code for reproducibility.

## Build

Use the [official CVPR 2026 author kit](https://github.com/cvpr-org/author-kit/tree/12909ae437f6dbc7435069cfdb4ca44c18e6a02f) at commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`. Put `cvpr.sty` and `ieeenat_fullname.bst` in a local directory outside version control. From `report/`, build with Tectonic and pass that directory as a search path.

```sh
mkdir -p ../output/pdf
tectonic -Z search-path=/path/to/cvpr-author-kit --outdir ../output/pdf final.tex
```

Alternatively, put the two kit files beside `final.tex` and run `latexmk -pdf final.tex` from `report/`. The report marks the PDF as a draft if the Part 3 evidence section is absent.

## Historical report sources

- [Part 1 standalone entry point](part1_standalone.tex), [archived section](part1.tex), and [metric provenance](../docs/part1_report.md). Its held-out comparison figure comes from the team's original Part 1 handoff.
- [Part 2 archived section](part2.tex), [integration notes](part2_integration.md), and [verified HPC handoff](../docs/part2_handoff.md).
