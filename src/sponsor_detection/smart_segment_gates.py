from __future__ import annotations

import json
import math
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from sponsor_detection.data.profile import sha256_file, write_json_atomic


BASELINE_CONTRACT_VERSION = "sponsor-baseline-contract/1"
GATE_CONTRACT_VERSION = "smart-segment-gate-contract/1"
WILSON_Z_95 = 1.959963984540054

REQUIRED_SECTIONS = {
    "contract": ("name", "output_path", "release_manifest"),
    "sponsor_non_inferiority": (
        "confidence_interval_method",
        "precision_delta",
        "recall_delta",
        "span_f1_iou_0_5_delta",
        "false_positive_seconds_per_hour_delta",
        "missed_sponsor_seconds_per_hour_delta",
        "require_lower_ci_bound_non_inferior",
    ),
    "category_gates": ("minimum_reviewed_hours", "minimum_positive_spans"),
    "performance": (
        "device_tiers",
        "percentile",
        "maximum_latency_increase_ratio",
        "maximum_memory_increase_ratio",
        "maximum_model_size_increase_ratio",
    ),
}
REQUIRED_AUTOMATIC_ACTION_KEYS = (
    "enabled",
    "minimum_reviewed_hours",
    "minimum_positive_spans",
    "minimum_precision_lower_ci",
    "maximum_false_positive_seconds_per_hour",
)


def wilson_interval(
    successes: int, total: int, *, z: float = WILSON_Z_95
) -> dict[str, float | int]:
    """Wilson score interval for a binomial rate, the frozen CI method."""

    if total <= 0:
        return {"successes": successes, "total": total, "lower": 0.0, "upper": 0.0}
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return {
        "successes": successes,
        "total": total,
        "lower": round(max(0.0, (centre - margin) / denominator), 6),
        "upper": round(min(1.0, (centre + margin) / denominator), 6),
    }


def sponsor_baseline_contract(
    release_manifest_path: Path, *, reviewed_seconds: float | None = None
) -> dict[str, object]:
    """Freeze the current sponsor model against its frozen benchmark.

    This is the reference the Smart Segments SPONSOR head must not regress
    against; it is produced from the already-frozen release manifest so it
    cannot change with a candidate model.
    """

    manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    checkpoint = manifest["checkpoint"]["artifacts"]
    window = manifest["metrics"]["calibrated_window_test"]
    full_video = manifest["metrics"]["full_video_canary"]
    coverage = full_video["temporal_coverage"]
    span = window["span_iou_0.5"]
    presence = window["window_presence"]
    boundary = full_video["boundary_error_ms_at_iou_0.5"]

    false_positive_seconds = (coverage["predicted"] - coverage["overlap"]) / 1000.0
    missed_seconds = (coverage["expected"] - coverage["overlap"]) / 1000.0
    hours = reviewed_seconds / 3600.0 if reviewed_seconds else None

    return {
        "contract": "sponsor_baseline",
        "version": BASELINE_CONTRACT_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "release_name": manifest["name"],
        "release_manifest_sha256": sha256_file(release_manifest_path),
        "model_sha256": checkpoint["model"]["sha256"],
        "tokenizer_sha256": checkpoint["tokenizer"]["sha256"],
        "decoder_config_sha256": manifest["decoder"]["config"]["sha256"],
        "decoder_confidence_threshold": manifest["decoder"]["confidence_threshold"],
        "metrics": {
            "window_precision": presence["precision"],
            "window_recall": presence["recall"],
            "span_precision_iou_0.5": span["precision"],
            "span_recall_iou_0.5": span["recall"],
            "span_f1_iou_0.5": span["f1"],
            "false_positive_seconds": round(false_positive_seconds, 3),
            "missed_sponsor_seconds": round(missed_seconds, 3),
            "false_positive_seconds_per_hour": (
                round(false_positive_seconds / hours, 3) if hours else None
            ),
            "missed_sponsor_seconds_per_hour": (
                round(missed_seconds / hours, 3) if hours else None
            ),
            "boundary_error_start_median_ms": boundary["start_median"],
            "boundary_error_end_median_ms": boundary["end_median"],
            "window_true_positive": presence["true_positive"],
            "window_false_positive": presence["false_positive"],
            "span_true_positive": span["true_positive"],
            "span_false_positive": span["false_positive"],
            "span_false_negative": span["false_negative"],
            "reviewed_videos": full_video["videos"],
            "reviewed_seconds": reviewed_seconds,
            "confidence_intervals": {
                "method": "wilson_95",
                "window_precision": wilson_interval(
                    presence["true_positive"],
                    presence["true_positive"] + presence["false_positive"],
                ),
                "window_recall": wilson_interval(
                    presence["true_positive"],
                    presence["true_positive"] + presence["false_negative"],
                ),
                "span_precision_iou_0.5": wilson_interval(
                    span["true_positive"], span["true_positive"] + span["false_positive"]
                ),
                "span_recall_iou_0.5": wilson_interval(
                    span["true_positive"], span["true_positive"] + span["false_negative"]
                ),
            },
        },
    }


def _require_keys(section: str, configuration: dict[str, object], keys) -> None:
    missing = [key for key in keys if key not in configuration]
    if missing:
        raise ValueError(f"{section} is missing required keys: {', '.join(missing)}")


def freeze_gate_contract(
    config_path: Path, *, output_path: Path | None = None
) -> dict[str, object]:
    """Commit the release gates before any candidate test result is inspected.

    The numbers live in the TOML; this step validates that every blocking
    metric has an explicit margin and that automatic actions carry a full
    eligibility contract, then writes them together with the sponsor baseline.
    The report exists so a gate cannot be renegotiated after seeing results.
    """

    with config_path.open("rb") as source:
        configuration = tomllib.load(source)
    for section, keys in REQUIRED_SECTIONS.items():
        if section not in configuration:
            raise ValueError(f"gate contract is missing the [{section}] section")
        _require_keys(section, configuration[section], keys)

    automatic_action = configuration.get("automatic_action")
    if not isinstance(automatic_action, dict) or not automatic_action:
        raise ValueError("gate contract must define at least one automatic action")
    for category, entry in automatic_action.items():
        if not isinstance(entry, dict):
            raise ValueError(f"automatic action for {category} must be a table")
        _require_keys(f"automatic_action.{category}", entry, REQUIRED_AUTOMATIC_ACTION_KEYS)

    contract = configuration["contract"]
    reviewed_seconds = contract.get("reviewed_seconds")
    baseline = sponsor_baseline_contract(
        Path(contract["release_manifest"]),
        reviewed_seconds=float(reviewed_seconds) if reviewed_seconds else None,
    )

    report = {
        "contract": "smart_segment_gate_contract",
        "version": GATE_CONTRACT_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "name": str(contract["name"]),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "sponsor_baseline": baseline,
        "sponsor_non_inferiority": configuration["sponsor_non_inferiority"],
        "category_gates": configuration["category_gates"],
        "automatic_action": automatic_action,
        "performance": configuration["performance"],
    }
    write_json_atomic(output_path or Path(contract["output_path"]), report)
    return report
