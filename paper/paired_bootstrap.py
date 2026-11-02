"""Paired test-identity bootstrap from saved predictions and official labels only.

Run from the repository root with NumPy available:
    .venv/bin/python -B paper/paired_bootstrap.py

No ML modules, checkpoints, images, training, or inference are used. The compact
submission omits private prediction caches; use --root with the full local repo.
Outputs are new paper artifacts and never replace frozen evaluation artifacts.
"""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
TYPE_GAINS = (
    ("blur", "glass_blur", "lofo_blur"),
    ("weather", "fog", "lofo_weather"),
    ("weather", "brightness", "lofo_weather"),
    ("digital", "contrast", "lofo_weather"),
)
N_IMAGES, N_RESAMPLES, SEED = 10_000, 2_000, 42


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, paper):
    inputs = {}

    def record(path):
        inputs[path.relative_to(root).as_posix()] = sha(path)
        return path

    # Check the original official files against the repository's recorded receipts
    # before reading fine_labels from the trusted local CIFAR-100 pickle.
    identities = json.loads(record(root / "docs/reproducibility/data_identities.json").read_text())
    receipt = json.loads(record(root / "docs/reproducibility/dataset_receipt.json").read_text())
    test_path = record(root / "data/cifar-100-python/test")
    labels_path = record(root / "data/CIFAR-100-C/labels.npy")
    if sha(test_path) != identities["clean_test"]["test_sha256"]:
        raise ValueError("Official CIFAR-100 test file identity differs")
    if sha(labels_path) != receipt["files"]["labels.npy"]:
        raise ValueError("Official CIFAR-100-C labels identity differs")
    with test_path.open("rb") as stream:
        targets = np.asarray(pickle.load(stream, encoding="bytes")[b"fine_labels"], dtype=np.int64)
    labels = np.load(labels_path, allow_pickle=False)
    if targets.shape != (N_IMAGES,) or labels.shape != (5 * N_IMAGES,):
        raise ValueError("Unexpected official label dimensions")
    if not all(np.array_equal(block, targets) for block in labels.reshape(5, N_IMAGES)):
        raise ValueError("Official test identity ordering differs across severities")

    folders = {
        "legacy": root / "results/evaluation/seed42",
        "controlled_v2": root / "results/evaluation_controlled_v2/seed42",
    }
    saved = {}
    for cohort, folder in folders.items():
        evaluation = json.loads(record(folder / "metrics/evaluation.json").read_text())
        for row in evaluation["conditions"]:
            saved[cohort, row["model"], row["corruption"], row["severity"]] = row
    loaded = {}

    def correct(cohort, model, corruption, severity):
        key = cohort, model, corruption, severity
        if key not in loaded:
            path = record(folders[cohort] / "cache" / model / f"{corruption}__{severity}.npz")
            with np.load(path, allow_pickle=False) as cache:
                predictions = cache["predictions"]
                metadata = json.loads(str(cache["metadata"]))
            expected = saved[key]
            if (predictions.shape != targets.shape or predictions.dtype != np.uint8
                    or np.any(predictions >= 100) or metadata["row"] != expected):
                raise ValueError(f"Saved prediction identity or metadata mismatch: {path}")
            matched = (predictions == targets).astype(np.int16)
            if int(matched.sum()) != expected["n_correct"] or expected["n_samples"] != N_IMAGES:
                raise ValueError(f"Saved prediction count mismatch: {path}")
            loaded[key] = matched
        return loaded[key]

    specs, vectors = [], []

    def add(scope, family, corruption, model, metric, units, vector):
        specs.append(dict(scope=scope, family=family, corruption=corruption,
                          model=model, metric=metric, units=units))
        vectors.append(vector)

    for family, types in FAMILIES.items():
        gain = np.zeros(N_IMAGES, dtype=np.int16)
        denominator = np.zeros_like(gain)
        for corruption in types:
            for severity in range(1, 6):
                baseline = correct("controlled_v2", "clean_baseline", corruption, severity)
                gain += correct("legacy", "lofo_" + family, corruption, severity) - baseline
                denominator += correct("controlled_v2", "all_four_representatives", corruption, severity) - baseline
        # Collapse conditions only AFTER pairing each model at each identity.
        # A sampled identity retains all its types/severities and model outcomes.
        scale = 100 / (5 * len(types))
        add("family", family, "", "lofo_" + family, "delta", "pp", gain * scale)
        add("family", family, "", "all_four_representatives", "denominator", "pp", denominator * scale)

    for family, corruption, model in TYPE_GAINS:
        gain = sum((correct("legacy", model, corruption, severity)
                    - correct("controlled_v2", "clean_baseline", corruption, severity)
                    for severity in range(1, 6)), start=np.zeros(N_IMAGES, dtype=np.int16))
        add("type", family, corruption, model, "gain", "pp", gain * (100 / 5))

    identity_metrics = np.stack(vectors, axis=1)
    estimate = identity_metrics.mean(axis=0)
    bootstrap = np.empty((N_RESAMPLES, len(specs)), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    # The same sampled indices apply to every model, type, severity, and family,
    # including the four reported type gains. No condition-level resampling.
    for start in range(0, N_RESAMPLES, 20):
        stop = min(start + 20, N_RESAMPLES)
        indices = rng.integers(0, N_IMAGES, size=(stop - start, N_IMAGES), dtype=np.int64)
        bootstrap[start:stop] = identity_metrics[indices].mean(axis=1)

    rows = []

    def result(spec, point, replicates):
        if not np.isfinite(replicates).all():
            raise ValueError("Undefined bootstrap ratio; report explicitly rather than dropping resamples")
        lower, upper = np.percentile(replicates, [2.5, 97.5], method="linear")
        return dict(spec, estimate=float(point), lower_95=float(lower), upper_95=float(upper),
                    n_images=N_IMAGES, n_resamples=N_RESAMPLES, seed=SEED)

    denominator_diagnostics = {}
    for i, spec in enumerate(specs):
        rows.append(result(spec, estimate[i], bootstrap[:, i]))
        if spec["metric"] == "denominator":
            draws = bootstrap[:, i]
            if np.any(draws == 0):
                raise ValueError("A ratio denominator is zero in a bootstrap resample")
            ratio_spec = dict(spec, model="lofo_" + spec["family"], metric="transfer_ratio", units="ratio")
            rows.append(result(ratio_spec, estimate[i - 1] / estimate[i], bootstrap[:, i - 1] / draws))
            denominator_diagnostics[spec["family"]] = dict(
                min_resampled_pp=float(draws.min()), max_resampled_pp=float(draws.max()),
                zero_resamples=int(np.sum(draws == 0)), nonpositive_resamples=int(np.sum(draws <= 0)))

    # Match point estimates against existing full-precision paper data without
    # rewriting those data or altering their display rounding.
    data = json.loads(record(root / "paper/tables/plot_data.json").read_text())
    for row in rows:
        if row["scope"] == "family":
            existing = next(r for r in data["held_out_families"] if r["family"] == row["family"])
            field = {"delta": "gain_pp", "denominator": "all_four_gain_pp", "transfer_ratio": "transfer_ratio"}[row["metric"]]
            expected = existing[field]
        else:
            existing = next(r for r in data["operator_comparison"] if r["corruption"] == row["corruption"])
            expected = existing[row["model"] + "_gain_pp"]
        if not np.isclose(row["estimate"], expected, rtol=0, atol=1e-10):
            raise ValueError(f"Existing manuscript point estimate differs: {row}")

    paper.mkdir(parents=True, exist_ok=True)
    csv_path = paper / "paired_bootstrap.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def interval(scope, family, corruption, metric):
        row = next(r for r in rows if (r["scope"], r["family"], r["corruption"], r["metric"])
                   == (scope, family, corruption, metric))
        return f"[{row['lower_95']:.2f}, {row['upper_95']:.2f}]"

    body = []
    for family in FAMILIES:
        ci = lambda metric: interval("family", family, "", metric)
        body.append(f"{family.title()} & {ci('delta')} & {ci('transfer_ratio')} & {ci('denominator')} " + r"\\")
    body.append(r"\midrule")
    for family, corruption, model in TYPE_GAINS:
        label = corruption.replace("_", " ").capitalize() + f" (LOFO-{model[5:].title()})"
        body.append(f"{label} & {interval('type', family, corruption, 'gain')} & --- & --- " + r"\\")
    table = (
        "% Generated by paper/paired_bootstrap.py; saved predictions only.\n"
        "\\begin{table}[!htbp]\n"
        "\\caption{Paired test-image bootstrap: 95\\% percentile intervals, 2,000 resamples, seed 42. "
        "Each draw retains all types and five severities for each sampled identity across the LOFO, primary clean, and all-four models. "
        "Type gains use the indicated model; the denominator is the all-four family gain in Eq.~\\ref{eq:transfer}.}\n"
        "\\label{tab:bootstrap}\n\\centering\n\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Family / type (model) & Gain (pp) & $T_F$ & Denominator (pp) \\\\\n"
        "\\midrule\n" + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    table_path = paper / "tables/paired_bootstrap.tex"
    table_path.parent.mkdir(exist_ok=True)
    table_path.write_text(table)
    report = dict(
        method="Paired nonparametric bootstrap over official test identities; equal type/severity weights; percentile 2.5/97.5 with linear interpolation",
        n_images=N_IMAGES, n_resamples=N_RESAMPLES, seed=SEED, rng="NumPy PCG64",
        numpy_version=np.__version__, pairing="Common indices across all models, conditions, families, and type gains",
        uncertainty="Test-image sampling only; excludes training-seed variability; not significance tests",
        denominator_diagnostics=denominator_diagnostics,
        inputs=inputs, script_sha256=sha(Path(__file__)),
        outputs={p.relative_to(paper).as_posix(): sha(p) for p in [csv_path, table_path]},
        training_calls=0, forward_passes=0)
    (paper / "paired_bootstrap_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(rows=rows, denominator_diagnostics=denominator_diagnostics), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.root.resolve(), args.output_dir.resolve())
