# LOFO v3 ELCVIA manuscript

**How Far Does Corruption Robustness Transfer? A Leave-One-Family-Out Study on CIFAR-100-C**

`main.tex` and `supplement.tex` are the current named manuscript sources. `main_blinded.tex` and `supplement_blinded.tex` build the anonymous versions. The main manuscript has 12 pages and the supplement 21 pages in each version. Numerical macros, tables and publication figure PDFs are included.

With Tectonic on PATH, run `sh paper/build.sh` from the repository root. Set `TASK_TECTONIC` to use another Tectonic executable. Compiled article PDFs and LaTeX auxiliary files remain local.

Saved per-condition correct counts, training histories, run/completion manifests and operator-exposure counts for both training environments are under `results/replication_v3/`. Primary aggregate CSVs are under `analysis/`; saved exploratory policy, transfer-group, equivalence and sensitivity results are under `posthoc/`. These use five matched seed identifiers per environment; both cohorts use Mac MPS inference. `ANALYSIS_ADDENDUM.md` records the frozen post-hoc specification.

AI/CRediT declarations still need author confirmation. Public artifact deposition and the project DOI remain pending. The repository source identifies the authors; use only the blinded PDF builds for anonymous review. Official template and third-party attribution remain in `THIRD_PARTY.md` and `template_provenance.json`.
