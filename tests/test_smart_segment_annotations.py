from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.compute as compute
import pyarrow.parquet as parquet

from sponsor_detection.data.profile import EligibilityPolicy
from sponsor_detection.data.smart_segment_annotations import (
    SMART_SEGMENT_ANNOTATION_SCHEMA_VERSION,
    build_smart_segment_annotations,
)


CATEGORIES = ["sponsor", "selfpromo", "interaction", "preview", "filler"]

MIRROR_COLUMNS = [
    "videoID",
    "startTime",
    "endTime",
    "votes",
    "locked",
    "incorrectVotes",
    "UUID",
    "userID",
    "timeSubmitted",
    "views",
    "category",
    "actionType",
    "service",
    "videoDuration",
    "hidden",
    "reputation",
    "shadowHidden",
    "hashedVideoID",
    "userAgent",
    "description",
]


def _row(
    video_id: str,
    start: float,
    end: float,
    uuid: str,
    category: str,
    *,
    votes: int = 5,
    hidden: int = 0,
    shadow_hidden: int = 0,
    action_type: str = "skip",
    service: str = "YouTube",
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in MIRROR_COLUMNS}
    row.update(
        {
            "videoID": video_id,
            "startTime": start,
            "endTime": end,
            "votes": votes,
            "locked": 0,
            "incorrectVotes": 0,
            "UUID": uuid,
            "userID": "user",
            "timeSubmitted": 1_600_000_000,
            "views": 100,
            "category": category,
            "actionType": action_type,
            "service": service,
            "videoDuration": 600.0,
            "hidden": hidden,
            "reputation": 0.0,
            "shadowHidden": shadow_hidden,
        }
    )
    return row


class SmartSegmentAnnotationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.mirror_path = root / "sponsorTimes.csv"
        self.output_path = root / "out" / "smart_segment_annotations.parquet"
        self.manifest_path = root / "manifest.json"
        rows = [
            _row("VID1", 2.0, 5.0, "S1", "sponsor"),
            _row("VID1", 1.0, 3.0, "S2", "sponsor", hidden=1),
            _row("VID1", 4.5, 6.5, "P1", "selfpromo"),
            _row("VID2", 0.0, 2.0, "I1", "interaction"),
            _row("VID3", 0.0, 2.0, "S3", "sponsor", service="Patreon"),
            _row("VID1", 20.0, 25.0, "F1", "filler"),
            _row("VID1", 0.0, 1.0, "C1", "chapter"),
        ]
        lines = [",".join(MIRROR_COLUMNS)]
        for row in rows:
            lines.append(",".join(str(row[column]) for column in MIRROR_COLUMNS))
        malformed = dict(rows[0])
        malformed["startTime"] = "not-a-number"
        malformed["UUID"] = "M1"
        lines.append(",".join(str(malformed[column]) for column in MIRROR_COLUMNS))
        self.mirror_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _build(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "categories": CATEGORIES,
            "deduplication_policy": "preserve",
            "compute_sha256": True,
        }
        arguments.update(overrides)
        return build_smart_segment_annotations(
            self.mirror_path,
            self.output_path,
            self.manifest_path,
            EligibilityPolicy(),
            **arguments,
        )

    def _table(self):
        return parquet.read_table(self.output_path)

    def test_keeps_every_configured_category_and_drops_others(self) -> None:
        report = self._build()
        table = self._table()

        self.assertEqual(
            set(table.column("category").to_pylist()),
            {"sponsor", "selfpromo", "interaction", "filler"},
        )
        self.assertNotIn("chapter", table.column("category").to_pylist())
        self.assertEqual(report["counts"]["by_category"]["sponsor"], 4)
        self.assertEqual(report["counts"]["by_category"]["selfpromo"], 1)

    def test_preserves_cross_category_overlap(self) -> None:
        self._build()
        table = self._table()
        rows = table.to_pylist()
        sponsor = [row for row in rows if row["segment_id"] == "S1" and row["category"] == "sponsor"]
        selfpromo = [row for row in rows if row["segment_id"] == "P1"]
        self.assertEqual(len(sponsor), 1)
        self.assertEqual(len(selfpromo), 1)
        self.assertLess(sponsor[0]["start_ms"], selfpromo[0]["end_ms"])
        self.assertLess(selfpromo[0]["start_ms"], sponsor[0]["end_ms"])

    def test_retains_ineligible_rows_with_flags(self) -> None:
        report = self._build()
        rows = {row["segment_id"]: row for row in self._table().to_pylist()}

        self.assertFalse(rows["S2"]["is_eligible"])
        self.assertIn("hidden", rows["S2"]["rejection_flags"])
        self.assertFalse(rows["S3"]["is_eligible"])
        self.assertIn("wrong_service", rows["S3"]["rejection_flags"])
        self.assertTrue(rows["S1"]["is_eligible"])
        self.assertEqual(report["counts"]["eligible_rows"], 4)
        self.assertEqual(report["counts"]["ineligible_rows"], 2)

    def test_counts_malformed_rows_without_dropping_the_batch(self) -> None:
        report = self._build()

        self.assertEqual(report["counts"]["malformed_rows"], 1)
        self.assertEqual(report["counts"]["audited_rows"], 7)
        self.assertEqual(report["counts"]["written_rows"], 6)

    def test_records_provenance_in_manifest_and_parquet_metadata(self) -> None:
        report = self._build()
        metadata = self._table().schema.metadata

        self.assertIsNotNone(report["source"]["sha256"])
        self.assertIsNotNone(report["output"]["sha256"])
        self.assertEqual(
            int(metadata[b"smart_segments.schema_version"]),
            SMART_SEGMENT_ANNOTATION_SCHEMA_VERSION,
        )
        self.assertEqual(
            json.loads(metadata[b"smart_segments.categories"]),
            sorted(CATEGORIES),
        )
        self.assertEqual(
            metadata[b"smart_segments.source_sha256"].decode(),
            report["source"]["sha256"],
        )

    def test_rejects_unknown_deduplication_policy(self) -> None:
        with self.assertRaises(ValueError):
            self._build(deduplication_policy="collapse-everything")

    def test_expected_source_sha256_mismatch_fails(self) -> None:
        with self.assertRaises(ValueError):
            self._build(expected_source_sha256="0" * 64)

    def test_sponsor_only_filter_matches_legacy_label_counts(self) -> None:
        self._build()
        table = self._table()
        sponsor_rows = table.filter(
            compute.equal(table.column("category"), "sponsor")
        )
        eligible = [row for row in sponsor_rows.to_pylist() if row["is_eligible"]]

        self.assertEqual(len(eligible), 1)


if __name__ == "__main__":
    unittest.main()
