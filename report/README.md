# Project 4 CVPR report

[Project4_Report.pdf](Project4_Report.pdf) is the combined report for Parts 1–3. The entry point is [final.tex](final.tex). Build from this directory so LaTeX can resolve the figures and bibliographies.

## Build

Use the [CVPR 2026 author kit](https://github.com/cvpr-org/author-kit/tree/12909ae437f6dbc7435069cfdb4ca44c18e6a02f) at commit 12909ae437f6dbc7435069cfdb4ca44c18e6a02f. Set CVPR_KIT to the directory containing cvpr.sty and ieeenat_fullname.bst, then build the PDF.

```sh
mkdir -p ../output/pdf
tectonic -Z "search-path=$CVPR_KIT" --outdir ../output/pdf final.tex
```
