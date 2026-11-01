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


def verify(root):
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
                gains[family] = float(lofo - baseline)
    negative = sum(row["gain_pp"] < 0 for row in data["held_out_conditions"])
    if data["negative_held_out_cells"] != negative or negative != 14:
        raise ValueError("Negative-transfer count differs")
    blur_without_glass = sum(row["gain_pp"] for row in data["per_corruption"]
                             if row["family"] == "blur" and row["corruption"] != "glass_blur") / 3
    close(data["blur_gain_without_glass_pp"], blur_without_glass)

    for filename, group in (("held_out_transfer.csv", "held_out_families"),
                            ("per_corruption_transfer.csv", "per_corruption"),
                            ("severity_transfer.csv", "severity"),
                            ("held_out_conditions.csv", "held_out_conditions")):
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
    entries = set(re.findall(r"@\w+\s*\{([^,]+),", bib))
    if cited != entries or len(entries) != 12:
        raise ValueError("Expected twelve complete manuscript references")
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    tables = re.findall(r"\\input\{([^}]+)\}", tex)
    if len(figures) != 5 or len(tables) != 3:
        raise ValueError("Expected five figures and three tables")
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
    return {"models": 7, "saved_conditions": len(counts), "held_out_gain_pp": gains,
            "negative_held_out_cells": negative, "asset_hashes_verified": len(provenance["outputs"]),
            "references": len(entries), "figures": len(figures), "tables": len(tables),
            "training_or_inference": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(parser.parse_args().root), indent=2))
