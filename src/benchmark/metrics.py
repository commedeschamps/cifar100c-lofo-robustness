"""Macro metrics and image-cluster paired bootstrap (never repeated-image IID)."""
import numpy as np

from ..lofo import FAMILIES, REPRESENTATIVES
from .checkpoints import MODELS
from .data import CORRUPTIONS, FAMILY_OF

DEFINITIONS = {
    "per_corruption": "arithmetic mean of five severity accuracies",
    "per_family": "arithmetic mean over all member corruptions and five severities",
    "overall": "macro mean over 75 corruption/severity conditions (15 types; no clean)",
    "equal_family": "separate macro mean over four family means; noise has 3 types, others 4",
    "drop_pp": "clean accuracy minus corrupted accuracy, percentage points",
    "transfer_pp": "held-out LOFO accuracy minus clean-baseline accuracy on identical conditions",
    "mean_corruption_error_pct": "100 minus overall corrupted accuracy; no reference normalization",
    "bootstrap": "paired percentile interval; resample 10,000 original test-image identities with replacement, "
                 "sharing each draw across models, corruptions and severities; conditional on seed 42",
}


def exposure(model, corruption):
    if model == "clean_baseline":
        return "clean_only_training"
    family = FAMILY_OF[corruption]
    if model == f"lofo_{family}":
        return "entirely_held_out_family"
    return ("represented_reference_operator" if corruption == REPRESENTATIVES[family]
            else "unseen_type_within_exposed_family")


def condition_metrics(model, corruption, severity, predictions, targets, seconds, loss):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != targets.shape or len(targets) == 0:
        raise ValueError("Predictions must match all requested image identities.")
    n, correct = len(targets), int(np.count_nonzero(predictions == targets))
    return {"model": model, "seed": 42, "family": "clean" if corruption == "clean" else FAMILY_OF[corruption],
            "corruption": corruption, "severity": severity, "n_samples": n, "n_correct": correct,
            "accuracy_pct": 100*correct/n, "error_pct": 100*(n-correct)/n,
            "exposure_category": "clean" if corruption == "clean" else exposure(model, corruption),
            "seconds": seconds, "loss": loss}


def expected_keys():
    return {(m, c, s) for m in MODELS for c in CORRUPTIONS for s in range(1, 6)} | {
        (m, "clean", 0) for m in MODELS}


def validate_complete(rows):
    keys = [(r["model"], r["corruption"], r["severity"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys():
        raise ValueError("Incomplete/duplicate evaluation: require 375 corruption conditions and five clean checks.")
    for r in rows:
        if (r["n_samples"] != 10000 or type(r["n_correct"]) is not int
                or not 0 <= r["n_correct"] <= 10000 or r["seed"] != 42
                or not np.isclose(r["accuracy_pct"], r["n_correct"]/100, atol=1e-10, rtol=0)
                or not np.isclose(r["error_pct"], 100-r["accuracy_pct"], atol=1e-10, rtol=0)
                or r["family"] != ("clean" if r["corruption"] == "clean" else FAMILY_OF[r["corruption"]])):
            raise ValueError("Invalid scientific condition counts, family, or percentages.")


def aggregate(rows):
    validate_complete(rows)
    clean = {r["model"]: r["accuracy_pct"] for r in rows if r["corruption"] == "clean"}
    corrupted = [r for r in rows if r["corruption"] != "clean"]
    def mean(model, names, severity=None):
        return float(np.mean([r["accuracy_pct"] for r in corrupted if r["model"] == model
                              and r["corruption"] in names and (severity is None or r["severity"] == severity)]))
    family_rows, corruption_rows, overall, severity_rows, exposure_rows = [], [], [], [], []
    for model in MODELS:
        for family, names in FAMILIES.items():
            acc = mean(model, names)
            family_rows.append({"model": model, "family": family, "n_conditions": len(names)*5,
                                "accuracy_pct": acc, "clean_drop_pp": clean[model]-acc,
                                "baseline_difference_pp": acc-mean("clean_baseline", names),
                                "held_out": model == f"lofo_{family}"})
            for s in range(1, 6):
                severity_rows.append({"model": model, "family": family, "severity": s,
                                      "accuracy_pct": mean(model, names, s)})
        for c in CORRUPTIONS:
            acc = mean(model, (c,))
            corruption_rows.append({"model": model, "family": FAMILY_OF[c], "corruption": c,
                                    "accuracy_pct": acc, "clean_drop_pp": clean[model]-acc,
                                    "exposure_category": exposure(model, c)})
        acc = mean(model, CORRUPTIONS)
        overall.append({"model": model, "clean_accuracy_pct": clean[model], "mean_corrupted_accuracy_pct": acc,
                        "mean_corruption_error_pct": 100-acc, "clean_drop_pp": clean[model]-acc,
                        "equal_family_accuracy_pct": float(np.mean([mean(model, n) for n in FAMILIES.values()]))})
        categories = sorted({exposure(model, c) for c in CORRUPTIONS})
        for category in categories:
            names = [c for c in CORRUPTIONS if exposure(model, c) == category]
            exposure_rows.append({"model": model, "exposure_category": category, "n_types": len(names),
                                  "accuracy_pct": mean(model, names)})
    transfer, direct = [], []
    for family, names in FAMILIES.items():
        model = f"lofo_{family}"
        groups = [("family", "all", None, names)]
        groups += [("corruption", c, None, (c,)) for c in names]
        groups += [("severity", "all", s, names) for s in range(1, 6)]
        groups += [("condition", c, s, (c,)) for c in names for s in range(1, 6)]
        for scope, corruption, s, selected in groups:
            acc, base = mean(model, selected, s), mean("clean_baseline", selected, s)
            transfer.append({"model": model, "family": family, "scope": scope, "corruption": corruption,
                             "severity": s, "held_out_accuracy_pct": acc, "baseline_accuracy_pct": base,
                             "difference_pp": acc-base})
        exposed = [m for m in MODELS if m.startswith("lofo_") and m != model]
        for other in (*exposed, "exposed_lofo_mean"):
            acc = (float(np.mean([mean(m, names) for m in exposed])) if other == "exposed_lofo_mean"
                   else mean(other, names))
            direct.append({"family": family, "held_out_model": model, "exposed_comparator": other,
                           "held_out_accuracy_pct": mean(model, names), "exposed_accuracy_pct": acc,
                           "held_out_minus_exposed_pp": mean(model, names)-acc})
    return {"overall": overall, "families": family_rows, "corruptions": corruption_rows,
            "severity": severity_rows, "held_out_transfer": transfer, "direct_exposure": direct,
            "exposure_groups": exposure_rows, "definitions": DEFINITIONS}


def paired_bootstrap(a, b, repeats=2000, seed=2026):
    """Inputs [conditions, image_id]. Collapse repeats BEFORE sampling image clusters."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 2 or repeats < 100:
        raise ValueError("Paired bootstrap needs matched [conditions, image_id] arrays and >=100 replicates.")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Bootstrap inputs must encode per-image correctness.")
    delta = (a.astype(np.float64)-b.astype(np.float64)).mean(axis=0)*100
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for i in range(repeats):
        draws[i] = delta[rng.integers(0, len(delta), len(delta))].mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"difference_pp": float(delta.mean()), "ci95_low_pp": float(low), "ci95_high_pp": float(high),
            "replicates": repeats, "bootstrap_seed": seed, "n_image_clusters": len(delta),
            "method": "paired image-cluster percentile bootstrap, conditional on fixed training seed 42",
            "multiplicity": "four marginal intervals; no multiplicity adjustment or significance claim"}
