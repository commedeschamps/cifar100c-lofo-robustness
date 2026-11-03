"""Verify manuscript assets against saved counts; Python standard library only.

No dataset loading, model construction, training or inference is performed.
"""
import argparse
import csv
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import re

FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"Metric mismatch: {actual} != {expected}")


def verify(root, require_bootstrap_inputs=False):
    root = Path(root).resolve(strict=True)
    paper = root / "paper"
    counts = {}
    for cohort, relative in (
        ("historical", "results/evaluation/seed42/metrics/evaluation.json"),
        ("controlled_v2", "results/evaluation_controlled_v2/seed42/metrics/evaluation.json"),
    ):
        for row in read(root / relative)["conditions"]:
            key = cohort, row["model"], row["corruption"], row["severity"]
            if key in counts or row["n_samples"] != 10000:
                raise ValueError("Invalid saved condition grid")
            close(row["accuracy_pct"], row["n_correct"] / 100)
            counts[key] = row["n_correct"]
    if len(counts) != 532:
        raise ValueError("Expected 532 saved conditions")

    def accuracy(cohort, model, types, severities=range(1, 6)):
        values = [counts[cohort, model, c, s] for c in types for s in severities]
        return Fraction(sum(values), 100 * len(values))

    data = read(paper / "tables/plot_data.json")
    all_types = sum((list(cs) for cs in FAMILIES.values()), [])
    for row in data["overall"]:
        close(row["clean_pct"], counts[row["cohort"], row["model"], "clean", 0] / 100)
        close(row["overall_pct"], accuracy(row["cohort"], row["model"], all_types))
        for family, types in FAMILIES.items():
            close(row["family_accuracy"][family], accuracy(row["cohort"], row["model"], types))
        if row['cohort'] == 'controlled_v2':
            run = root / 'results/controlled_v2' / row['model'] / 'seed42'
        elif row['model'] == 'clean_baseline':
            run = root / 'results/clean_baseline/clean_seed42'
        else:
            run = root / 'results' / row['model'] / 'seed42'
        best = max(read(run / 'metrics.json'), key=lambda r: r['val_accuracy_pct'])
        close(row['best_epoch'], best['epoch'])
        close(row['best_validation_pct'], best['val_accuracy_pct'])
    if len(data["overall"]) != 7:
        raise ValueError("Expected seven model summaries")
    gains = {}
    for group, expected_rows in (("held_out_families", 4), ("per_corruption", 15),
                                 ("severity", 20), ("held_out_conditions", 75)):
        if len(data[group]) != expected_rows:
            raise ValueError(f"Incomplete {group}")
        for row in data[group]:
            family = row["family"]
            types = [row["corruption"]] if "corruption" in row else FAMILIES[family]
            severities = [row["severity"]] if "severity" in row else range(1, 6)
            baseline = accuracy("controlled_v2", "clean_baseline", types, severities)
            lofo = accuracy("historical", "lofo_" + family, types, severities)
            control = accuracy("controlled_v2", "all_four_representatives", types, severities)
            for field, value in (("baseline_pct", baseline), ("lofo_pct", lofo),
                                 ("gain_pp", lofo - baseline), ("all_four_pct", control),
                                 ("all_four_minus_lofo_pp", control - lofo)):
                if field in row:
                    close(row[field], value)
            if group == "held_out_families":
                if control == baseline:
                    raise ValueError('Undefined transfer ratio')
                close(row['all_four_gain_pp'], control - baseline)
                close(row['transfer_ratio'], (lofo - baseline) / (control - baseline))
                gains[family] = float(lofo - baseline)
    negative = sum(row["gain_pp"] < 0 for row in data["held_out_conditions"])
    if data["negative_held_out_cells"] != negative or negative != 14:
        raise ValueError("Negative-transfer count differs")
    blur_without_glass = sum(row["gain_pp"] for row in data["per_corruption"]
                             if row["family"] == "blur" and row["corruption"] != "glass_blur") / 3
    close(data["blur_gain_without_glass_pp"], blur_without_glass)

    if len(data['operator_comparison']) != 15:
        raise ValueError('Expected all fifteen operator rows')
    for row in data['operator_comparison']:
        baseline = accuracy('controlled_v2', 'clean_baseline', [row['corruption']])
        close(row['baseline_pct'], baseline)
        for family in FAMILIES:
            model = 'lofo_' + family
            close(row[model+'_gain_pp'], accuracy('historical', model, [row['corruption']])-baseline)
        close(row['all_four_representatives_gain_pp'],
              accuracy('controlled_v2', 'all_four_representatives', [row['corruption']])-baseline)
    representatives = dict(noise='gaussian_noise', blur='defocus_blur',
                           weather='fog', digital='jpeg_compression')
    if len(data['represented_family_transfer']) != 4:
        raise ValueError('Expected four represented-family summaries')
    for row in data['represented_family_transfer']:
        held_out = row['model'][5:]
        types = [c for f, cs in FAMILIES.items() if f != held_out for c in cs
                 if c != representatives[f]]
        if row['corruption_types'].split(';') != types or row['n_types'] != len(types):
            raise ValueError('Represented-family evaluation subset differs')
        baseline = accuracy('controlled_v2', 'clean_baseline', types)
        lofo = accuracy('historical', row['model'], types)
        close(row['baseline_pct'], baseline)
        close(row['lofo_pct'], lofo)
        close(row['gain_pp'], lofo-baseline)

    for filename, group in (("held_out_transfer.csv", "held_out_families"),
                            ("per_corruption_transfer.csv", "per_corruption"),
                            ("severity_transfer.csv", "severity"),
                            ("held_out_conditions.csv", "held_out_conditions"),
                            ("operator_comparison.csv", "operator_comparison"),
                            ("represented_family_transfer.csv", "represented_family_transfer")):
        with (paper / "tables" / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(data[group]):
            raise ValueError(f"Row count differs: {filename}")
        for actual, expected in zip(rows, data[group]):
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    close(actual[key], value)
                elif actual[key] != value:
                    raise ValueError(f"Row identity differs: {filename}")

    provenance = read(paper / "tables/manuscript_provenance.json")
    if provenance["script_sha256"] != sha(paper / "generate_assets.py"):
        raise ValueError("Asset-generator hash differs")
    for mapping in (provenance["inputs"], provenance["outputs"]):
        for relative, digest in mapping.items():
            path = root / relative
            if not path.resolve().is_relative_to(root) or sha(path) != digest:
                raise ValueError(f"Asset provenance differs: {relative}")
    template = read(paper / "template_provenance.json")
    for relative, digest in template["files"].items():
        if sha(paper / relative) != digest:
            raise ValueError(f"Template identity differs: {relative}")

    tex = (paper / "main.tex").read_text()
    bib = (paper / "references.bib").read_text()
    cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    entry_keys = re.findall(r"@\w+\s*\{([^,]+),", bib)
    entries = set(entry_keys)
    if not entries or len(entry_keys) != len(entries) or cited != entries:
        raise ValueError("Manuscript citations must match unique bibliography entries")
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    tables = re.findall(r"\\input\{([^}]+)\}", tex)
    if len(figures) != 4 or len(tables) != 5:
        raise ValueError("Expected four figures and five tables")
    for relative in figures + tables:
        if not (paper / relative).is_file():
            raise ValueError(f"Missing manuscript asset: {relative}")
    if "IEEEtran" in tex or "\\documentclass[twoside,a4paper,11pt]{article}" not in tex:
        raise ValueError("The official ELCVIA article preamble is required")
    if re.search(r"\[AUTHORS:|TODO|TBD", tex):
        raise ValueError("Unresolved manuscript placeholder")
    for family, gain in gains.items():
        if f"{family.title()} +{gain:.2f} pp" not in tex:
            raise ValueError(f"Missing held-out conclusion value: {family}")
    bootstrap = read(paper / "paired_bootstrap_provenance.json")
    if bootstrap["script_sha256"] != sha(paper / "paired_bootstrap.py"):
        raise ValueError("Bootstrap script identity differs")
    bootstrap_inputs_checked = 0
    unavailable_private_inputs = 0
    for relative, digest in bootstrap["inputs"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Bootstrap input escapes repository: {relative}")
        private_input = (relative.startswith("data/") or relative.startswith(
            ("results/evaluation/seed42/cache/", "results/evaluation_controlled_v2/seed42/cache/")))
        if not path.is_file() and private_input and not require_bootstrap_inputs:
            unavailable_private_inputs += 1
            continue
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Bootstrap input identity differs: {relative}")
        bootstrap_inputs_checked += 1
    for relative, digest in bootstrap["outputs"].items():
        path = paper / relative
        if not path.resolve().is_relative_to(paper) or sha(path) != digest:
            raise ValueError(f"Bootstrap output identity differs: {relative}")
    with (paper / "paired_bootstrap.csv").open() as stream:
        bootstrap_rows = list(csv.DictReader(stream))
    if len(bootstrap_rows) != 16:
        raise ValueError("Expected twelve family and four type bootstrap rows")
    for row in bootstrap_rows:
        if (int(row["n_images"]), int(row["n_resamples"]), int(row["seed"])) != (10000, 2000, 42):
            raise ValueError("Unexpected bootstrap design")
        if not float(row["lower_95"]) <= float(row["upper_95"]):
            raise ValueError("Invalid bootstrap interval")
    if "The intervals reflect test-image sampling uncertainty only; they do not capture training-seed variability and are not significance tests." not in tex:
        raise ValueError("Missing bootstrap uncertainty qualification")
    return {"models": 7, "saved_conditions": len(counts), "held_out_gain_pp": gains,
            "negative_held_out_cells": negative, "asset_hashes_verified": len(provenance["outputs"]),
            "operator_table_numeric_cells": 90, "histories_checked": 7,
            "transfer_ratios": {r['family']: r['transfer_ratio'] for r in data['held_out_families']},
            "references": len(entries), "figures": len(figures), "tables": len(tables),
            "bootstrap_rows": len(bootstrap_rows),
            "bootstrap_input_hashes_verified": bootstrap_inputs_checked,
            "bootstrap_unavailable_private_inputs": unavailable_private_inputs,
            "bootstrap_full_input_check": unavailable_private_inputs == 0,
            "training_or_inference": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-bootstrap-inputs", action="store_true",
                        help="Require private official labels and prediction caches as well as public inputs")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.require_bootstrap_inputs), indent=2))
