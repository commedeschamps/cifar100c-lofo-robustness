# ELCVIA manuscript

**How Far Does Corruption Robustness Transfer? A Leave-One-Family-Out Study on CIFAR-100-C**

[main.tex](main.tex) is the complete, unpublished manuscript by Rakhman Seidigali and Yermakhan Khamitov. It uses the official ELCVIA LaTeX 2e style: single-column A4, 11-point body text and single spacing. The verified build has 11 pages, five figures, three tables, Appendix A and twelve references.

## Compile

With a LaTeX distribution and `latexmk`, run from this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Tectonic also builds the article:

```sh
tectonic --untrusted --keep-logs main.tex
```

Required packages: fontenc (T1), times, babel, amsmath, amssymb, graphicx, booktabs, natbib, hyperref, xurl, placeins and etoolbox. Figure PDFs are included. The article PDF and auxiliary build products are generated locally and excluded from Git.

## Template and bibliography

`elcvia.sty` is unchanged from the [official journal download](https://elcvia.cvc.uab.cat/libraryFiles/downloadPublic/1), linked by the [author instructions](https://elcvia.cvc.uab.cat/about/submissions). The journal style's original text geometry is retained. `elcvia-submission.sty` removes unassigned publication metadata; volume, issue, editor and received/accepted dates remain unset.

`elcvia-numeric.bst` is a clearly named local adaptation of `unsrtnat`, with its original LPPL notice retained. It uses numeric citation order, author initials and DOI-preferred links. It is not a journal-supplied bibliography style. See [THIRD_PARTY.md](THIRD_PARTY.md) and [template_provenance.json](template_provenance.json) for sources and attribution.

## Figures, tables and checks

From the repository root:

```sh
python3 -B tools/verify_manuscript.py
python3 paper/generate_assets.py
```

Asset generation requires NumPy and matplotlib and reads only saved integer correct counts. `--only-figure clean_vs_corrupted` limits figure export to one requested figure. PDF/SVG and PNG assets are supplied; [tables/plot_data.json](tables/plot_data.json) and the CSV files retain full precision. [tables/manuscript_provenance.json](tables/manuscript_provenance.json) binds inputs, generator and output hashes. The primary comparator is the new MPS/FP32 clean baseline.

## Manuscript status

The manuscript names Codex in its AI-assistance disclosure, following the authors' confirmed tool list. Author order, affiliation, correspondence, CRediT, funding and competing-interest statements are included. The article has not been accepted or published by ELCVIA.

This repository and its source manuscript identify the authors. Journal review materials must be prepared separately according to the journal's double-blind process; no anonymous-review artifact is published here.
