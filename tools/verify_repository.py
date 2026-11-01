"""Verify saved research numbers and available input hashes without model inference."""
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected):
    require(math.isclose(float(actual), float(expected), abs_tol=1e-9, rel_tol=0),
            f"Metric mismatch: {actual} != {expected}")


def verify(root=ROOT):
    root = Path(root).resolve()
    cohorts = {
        "historical": (root / "results/evaluation/seed42", {"clean_baseline", *(f"lofo_{f}" for f in FAMILIES)}),
        "controlled_v2": (root / "results/evaluation_controlled_v2/seed42", {"clean_baseline", "all_four_representatives"}),
    }
    conditions = {}
    for cohort, (folder, models) in cohorts.items():
        result = read(folder / "metrics/evaluation.json")
        rows = result["conditions"]
        expected = {(m, c, s) for m in models for types in FAMILIES.values() for c in types for s in range(1, 6)}
        expected |= {(m, "clean", 0) for m in models}
        keys = [(r["model"], r["corruption"], r["severity"]) for r in rows]
        require(set(keys) == expected and len(keys) == len(expected), f"Incomplete grid: {cohort}")
        for row in rows:
            require(row["seed"] == 42 and row["n_samples"] == 10000, "Unexpected seed or test size")
            require(type(row["n_correct"]) is int and 0 <= row["n_correct"] <= 10000, "Invalid correct count")
            same(row["accuracy_pct"], row["n_correct"] / 100)
            same(row["error_pct"], 100 - row["accuracy_pct"])
        if cohort == "historical":
            require(result["summary"]["benchmark_complete"], "Historical evaluation incomplete")
        else:
            require(read(folder / "status.json")["status"] == "complete", "V2 evaluation incomplete")
        conditions[cohort] = rows
    with (root / "paper/tables/seed42_results.csv").open() as stream:
        table = list(csv.DictReader(stream))
    expected_runs = {(c, m) for c, (_, models) in cohorts.items() for m in models}
    require(len(table) == 7 and {(r["cohort"], r["model"]) for r in table} == expected_runs,
            "Expected seven unique summary rows")
    metadata = root / "docs/reproducibility"
    model_index = {(r["cohort"], r["model"]): r for r in read(metadata / "models.json")["runs"]}
    parameters = {(r["cohort"], r["model"]): r for r in read(metadata / "recorded_parameters.json")["runs"]}
    require(set(model_index) == set(parameters) == expected_runs, "Recorded model/parameter inventory differs")
    checkpoint_count = 0
    for row in table:
        key = row["cohort"], row["model"]
        info = model_index[key]
        history = read(root / info["run_path"] / "metrics.json")
        require([r["epoch"] for r in history] == list(range(1, 101)), "Expected 100 complete epochs")
        best = max(history, key=lambda r: r["val_accuracy_pct"])
        same(row["best_epoch"], best["epoch"])
        same(row["best_validation_pct"], best["val_accuracy_pct"])
        same(info["selected_epoch"], best["epoch"])
        same(info["best_clean_validation_accuracy_pct"], best["val_accuracy_pct"])
        record = parameters[key]
        require(record["parameters"]["seed"] == 42 and record["parameters"]["full_epochs"] == 100,
                "Recorded training parameters differ")
        original_config = root / record["source_path"]
        if original_config.is_file():
            require(sha(original_config) == record["source_sha256"], "Saved configuration identity differs")
        rows = [r for r in conditions[key[0]] if r["model"] == key[1]]
        same(row["clean_pct"], next(r["accuracy_pct"] for r in rows if r["corruption"] == "clean"))
        values = [r["accuracy_pct"] for r in rows if r["corruption"] != "clean"]
        same(row["overall_pct"], sum(values) / len(values))
        for family, types in FAMILIES.items():
            values = [r["accuracy_pct"] for r in rows if r["corruption"] in types]
            same(row[family + "_pct"], sum(values) / len(values))
        for filename, field in (("best_model.pt", "best_checkpoint_sha256"), ("last_checkpoint.pt", "last_checkpoint_sha256")):
            path = root / info["run_path"] / "checkpoints" / filename
            if path.is_file():
                require(sha(path) == info[field], f"Checkpoint hash differs: {path}")
                checkpoint_count += 1
    provenance = read(root / "paper/tables/provenance.json")
    for mapping in (provenance["inputs"], provenance["outputs"]):
        for relative, expected in mapping.items():
            require(sha(root / relative) == expected, f"Table provenance differs: {relative}")
    split_path = root / "splits/cifar100_seed42.json"
    split = read(split_path)
    train, val = split["train_indices"], split["val_indices"]
    require(len(train) == 45000 and len(val) == 5000 and not set(train) & set(val)
            and set(train) | set(val) == set(range(50000)), "Invalid fitted/validation indices")
    require(sha(split_path) == read(metadata / "data_identities.json")["split_file_sha256"], "Split bytes differ")
    freeze = read(root / "results/controlled_v2/freeze.json")
    available, omitted = [], []
    for relative, expected in freeze["files"].items():
        path = root / relative
        if path.is_file():
            require(sha(path) == expected, f"Frozen input differs: {relative}")
            available.append(relative)
        else:
            omitted.append(relative)
    return {"models_verified": 7, "evaluation_conditions": sum(map(len, conditions.values())),
            "available_frozen_inputs_verified": len(available), "unavailable_frozen_inputs": omitted,
            "available_checkpoint_hashes_verified": checkpoint_count,
            "model_forward_passes": 0, "full_archive_verification": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
