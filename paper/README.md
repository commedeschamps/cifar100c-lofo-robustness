# Manuscript

`main.tex` uses the IEEEtran conference class. Build from this directory with an installed LaTeX distribution:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Required packages are cite, amsmath/amssymb, graphicx, booktabs and hyperref. Local `IEEEtran.cls` v1.8b and `IEEEtran.bst` v1.14 are unmodified; licensing and provenance are retained in `IEEEtran_README.txt` and `template_provenance.json`.

`references.bib` contains the initial CIFAR, ResNet, corruption-benchmark and dataset references. Confirm the target venue's template, page limit, anonymity, bibliography and disclosure requirements before submission. Author information and manuscript sections remain to be completed; inspect the compiled two-column PDF before submission.

## Tables and figures

[tables/seed42_results.csv](tables/seed42_results.csv) contains seven models' clean, overall and family accuracy, best epoch and best clean-validation accuracy. The [TeX summary](tables/seed42_results.tex) contains only the accuracy columns. [provenance.json](tables/provenance.json) binds both to saved source results. The table is available for inclusion with `\input{tables/seed42_results.tex}`.

Existing figures in `results/evaluation/seed42/plots/` in the full research repository describe the historical five-model cohort. They must retain those cohort labels. Prepare manuscript figures from saved final metrics, preserve negative transfer, and use readable vector output. No experimental figures are included automatically. Scientific limitations are listed in the [research README](../README.md).

## Research provenance and disclosure

AI assistance (Codex) contributed to software preparation, repository organization and manuscript scaffolding. The authors must accurately document actual assistance and follow the applicable institutional and venue disclosure policy; no claim of exclusively manual preparation is made.
