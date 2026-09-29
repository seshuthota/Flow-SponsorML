from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sponsor_detection.smart_segment_gates import (
    freeze_gate_contract,
    sponsor_baseline_contract,
    wilson_interval,
)


RELEASE_MANIFEST = {
    "name": "fixture-rc",
    "checkpoint": {
        "artifacts": {
            "model": {"sha256": "model-sha"},
            "tokenizer": {"sha256": "tokenizer-sha"},
        }
    },
    "decoder": {"config": {"sha256": "decoder-sha"}, "confidence_threshold": 0.7},
    "metrics": {
        "calibrated_window_test": {
            "span_iou_0.5": {
                "precision": 0.9,
                "recall": 0.67,
                "f1": 0.77,
                "true_positive": 100,
                "false_positive": 11,
                "false_negative": 49,
            },
            "window_presence": {
                "precision": 0.94,
                "recall": 0.72,
                "true_positive": 100,
                "false_positive": 6,
                "false_negative": 39,
            },
        },
        "full_video_canary": {
            "temporal_coverage": {
                "expected": 100_000,
                "predicted": 90_000,
                "overlap": 80_000,
            },
            "boundary_error_ms_at_iou_0.5": {"start_median": 1000, "end_median": 2000},
            "videos": 10,
        },
    },
}


VALID_GATE_TOML = """
[contract]
name = "test-contract"
output_path = "{output}"
release_manifest = "{release}"

[sponsor_non_inferiority]
confidence_interval_method = "wilson_95"
precision_delta = -0.02
recall_delta = -0.02
span_f1_iou_0_5_delta = -0.02
false_positive_seconds_per_hour_delta = 5.0
missed_sponsor_seconds_per_hour_delta = 5.0
require_lower_ci_bound_non_inferior = true

[category_gates]
minimum_reviewed_hours = 5.0
minimum_positive_spans = 50

[performance]
device_tiers = ["arm64-v8a-mid"]
percentile = "p95"
maximum_latency_increase_ratio = 0.10
maximum_memory_increase_ratio = 0.10
maximum_model_size_increase_ratio = 0.10

[automatic_action.sponsor]
enabled = true
minimum_reviewed_hours = 20.0
minimum_positive_spans = 200
minimum_precision_lower_ci = 0.90
maximum_false_positive_seconds_per_hour = 10.0
"""


class WilsonIntervalTest(unittest.TestCase):
    def test_bounds_and_containment(self) -> None:
        interval = wilson_interval(50, 100)

        self.assertLess(interval["lower"], 0.5)
        self.assertGreater(interval["upper"], 0.5)
        self.assertGreaterEqual(interval["lower"], 0.0)
        self.assertLessEqual(interval["upper"], 1.0)

    def test_empty_sample_is_zero(self) -> None:
        self.assertEqual(
            wilson_interval(0, 0),
            {"successes": 0, "total": 0, "lower": 0.0, "upper": 0.0},
        )


class SponsorBaselineContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.manifest_path = Path(self.directory.name) / "release.json"
        self.manifest_path.write_text(
            json.dumps(RELEASE_MANIFEST), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_extracts_hashes_metrics_and_derived_seconds(self) -> None:
        contract = sponsor_baseline_contract(self.manifest_path)
        metrics = contract["metrics"]

        self.assertEqual(contract["model_sha256"], "model-sha")
        self.assertEqual(contract["tokenizer_sha256"], "tokenizer-sha")
        self.assertEqual(contract["decoder_config_sha256"], "decoder-sha")
        self.assertEqual(metrics["false_positive_seconds"], 10.0)
        self.assertEqual(metrics["missed_sponsor_seconds"], 20.0)
        self.assertIsNone(metrics["false_positive_seconds_per_hour"])
        self.assertEqual(metrics["confidence_intervals"]["method"], "wilson_95")

    def test_per_hour_uses_reviewed_seconds(self) -> None:
        contract = sponsor_baseline_contract(
            self.manifest_path, reviewed_seconds=3600
        )

        self.assertEqual(
            contract["metrics"]["false_positive_seconds_per_hour"], 10.0
        )
        self.assertEqual(
            contract["metrics"]["missed_sponsor_seconds_per_hour"], 20.0
        )


class FreezeGateContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.release = self.root / "release.json"
        self.release.write_text(json.dumps(RELEASE_MANIFEST), encoding="utf-8")
        self.output = self.root / "gates.json"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write_config(self, text: str) -> Path:
        path = self.root / "gates.toml"
        path.write_text(
            text.format(output=self.output, release=self.release), encoding="utf-8"
        )
        return path

    def test_freezes_contract_with_baseline_and_sections(self) -> None:
        config_path = self._write_config(VALID_GATE_TOML)

        report = freeze_gate_contract(config_path)

        self.assertTrue(self.output.is_file())
        self.assertEqual(report["version"], "smart-segment-gate-contract/1")
        self.assertEqual(
            report["sponsor_baseline"]["model_sha256"], "model-sha"
        )
        self.assertIn("sponsor", report["automatic_action"])
        self.assertEqual(report["performance"]["percentile"], "p95")
        self.assertEqual(report["config_sha256"], hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest())

    def test_requires_non_inferiority_margins(self) -> None:
        broken = VALID_GATE_TOML.replace("precision_delta = -0.02", "")
        with self.assertRaises(ValueError) as context:
            freeze_gate_contract(self._write_config(broken))

        self.assertIn("precision_delta", str(context.exception))

    def test_requires_automatic_action_eligibility(self) -> None:
        broken = VALID_GATE_TOML.replace("minimum_precision_lower_ci = 0.90", "")
        with self.assertRaises(ValueError) as context:
            freeze_gate_contract(self._write_config(broken))

        self.assertIn("minimum_precision_lower_ci", str(context.exception))

    def test_requires_at_least_one_automatic_action(self) -> None:
        broken = VALID_GATE_TOML.split("[automatic_action.sponsor]")[0]
        with self.assertRaises(ValueError):
            freeze_gate_contract(self._write_config(broken))


if __name__ == "__main__":
    unittest.main()
