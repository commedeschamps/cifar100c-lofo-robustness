# ELCVIA manuscript

**How Far Does Corruption Robustness Transfer? A Leave-One-Family-Out Study on CIFAR-100-C**

[main.tex](main.tex) is the final editorial revision of Version 5 of the complete, unpublished manuscript by Rakhman Seidigali and Yermakhan Khamitov, with a focused literature update through 2 November 2026. It uses the official ELCVIA LaTeX 2e style: single-column A4, 11-point body text and single spacing. The verified named build has 13 pages, four figures, five tables, Appendix A and thirty references. Nine recent works comprise three 2024 papers, four 2025 papers and two 2026 works (one explicitly identified preprint). The editorial revision preserves the verified experimental values, equations, bootstrap intervals and scientific assets.

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

Asset generation requires NumPy and matplotlib and reads only saved integer correct counts. `--only-figure clean_vs_corrupted` limits figure export to one requested figure. PDF/SVG and PNG assets are supplied; [tables/plot_data.json](tables/plot_data.json) and the CSV files retain full precision, including descriptive transfer ratios, all five augmented models by corruption type, and ungenerated types within represented families. The optional held-out gain bar-chart export is retained; Table 3 presents those gains in the article. [tables/manuscript_provenance.json](tables/manuscript_provenance.json) binds inputs, generator and output hashes. The primary comparator is the new MPS/FP32 clean baseline.

[paired_bootstrap.py](paired_bootstrap.py) reads only saved prediction caches and hash-verified official labels. Run `.venv/bin/python -B paper/paired_bootstrap.py` from the full repository root. It resamples the 10,000 test identities 2,000 times with NumPy PCG64 seed 42, using common indices across models, types and severities. [paired_bootstrap.csv](paired_bootstrap.csv) retains full-precision point estimates and 95% percentile intervals; [paired_bootstrap_provenance.json](paired_bootstrap_provenance.json) records input/output identities. Table 4 reports family gains, transfer ratios, denominator intervals and the four discussed type gains. The intervals reflect test-image sampling uncertainty only; they do not capture training-seed variability and are not significance tests. The compact submission omits the private caches needed to rerun this analysis.

The manuscript checker verifies saved arithmetic, bootstrap outputs and all available input hashes. Private bootstrap inputs omitted from Git are reported as unavailable; `python3 -B tools/verify_manuscript.py --require-bootstrap-inputs` requires the complete local input set. Before-edit copies and change reports are retained locally in `quick_addition_20261102/` and `followup_20261102/`, outside Git.

## Manuscript status

The manuscript records Codex for research programming and computational verification, and ChatGPT and Claude for literature discovery and review. The authors retain responsibility for the reported results, references and final text. Author order, affiliation, correspondence, CRediT, funding and competing-interest statements are included. The article has not been accepted or published by ELCVIA.

This repository and its source manuscript identify the authors. Journal review materials must be prepared separately according to the journal's double-blind process; no anonymous-review artifact is published here.
