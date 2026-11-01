"""Protocol registry and auditable exposure bins; no benchmark data access."""

FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}
REPRESENTATIVES = {"noise": "gaussian_noise", "blur": "defocus_blur",
                   "weather": "fog", "digital": "jpeg_compression"}
PLANNED_SEEDS = (42, 123, 2026)
# Canonical JSON fingerprint of the actual Part 1 split, not a newly generated split.
ORIGINAL_SPLIT_SHA256 = "ab04334a218db1266d508e8c9adaaf91d45d81e0f4410739243ef4991fdc3ee9"
CORRUPTION_NAMES = ("clean", *REPRESENTATIVES.values())


def allowed_for(held_out_family):
    if held_out_family not in FAMILIES:
        raise ValueError(f"Unknown held-out family: {held_out_family!r}.")
    return tuple(name for family, name in REPRESENTATIVES.items() if family != held_out_family)


def exposure_bins():
    return {"clean": 0, **{f"{name}:severity{level}": 0
                            for name in CORRUPTION_NAMES[1:] for level in range(1, 6)}}


def validate_exposures(config, counts, expected_count):
    if set(counts) != set(exposure_bins()) or sum(counts.values()) != expected_count:
        raise ValueError("Corruption audit does not account for every consumed training image.")
    if any(not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError("Corruption counts must be nonnegative integers.")
    for name in CORRUPTION_NAMES[1:]:
        if name not in config.allowed_corruptions and any(
                counts[f"{name}:severity{s}"] for s in range(1, 6)):
            raise ValueError(f"Forbidden corruption exposure detected: {name}.")


def protocol_metadata(config):
    classification = {}
    for family, names in FAMILIES.items():
        for name in names:
            category = ("entirely_held_out_family" if family == config.held_out_family
                        else "exact_seen" if name in config.allowed_corruptions
                        else "unseen_within_represented_family")
            classification[name] = {"family": family, "exposure_category": category}
    return {
        "experiment_name": config.experiment_name, "held_out_family": config.held_out_family,
        "allowed_corruptions": list(config.allowed_corruptions), "families": FAMILIES,
        "representatives": REPRESENTATIVES, "corruption_probability": config.corruption_probability,
        "corruption_selection": config.corruption_selection,
        "severity_selection": config.severity_selection,
        "allowed_severities": list(config.allowed_severities),
        "sampling_unit": "one independent clean/corruption draw per consumed training image per epoch",
        "audit_unit": "main-process counts of consumed batches; excludes unconsumed worker prefetch",
        "corruption_implementation": config.corruption_implementation,
        "split_sha256": config.split_sha256, "initialization": "from_scratch_weights_None",
        "evaluation_categories": classification,
        "benchmark_data_used": False, "validation": "clean_only",
    }
