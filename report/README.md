# Part 1 report

[part1.tex](part1.tex) is the section to include in the combined Project 4 paper. Merge the entries in [part1_refs.bib](part1_refs.bib) into the paper's bibliography and add `../results/part1-final/` to the graphics search path if building from this directory. The standalone entry point is [part1_standalone.tex](part1_standalone.tex).

The table is based on [coverage.csv](../results/part1-final/coverage.csv), and the convergence figure plots [convergence.csv](../results/part1-final/convergence.csv). The standalone build includes the optional held-out comparison if `figures/part1_comparison.png` is present. The team handoff PDF includes this figure from the course data.

## Build

Use the CVPR 2026 author kit at commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f` and copy `cvpr.sty` and `ieeenat_fullname.bst` into `report/`. [Official CVPR author kit](https://github.com/cvpr-org/author-kit/tree/12909ae437f6dbc7435069cfdb4ca44c18e6a02f)

From this directory, run:

```bash
latexmk -pdf part1_standalone.tex
```

Tectonic also builds the standalone document:

```bash
tectonic part1_standalone.tex
```

See the [Part 1 results and provenance](../docs/part1_report.md) for the experiment settings and artifact layout.
