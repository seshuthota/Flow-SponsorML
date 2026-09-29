from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

from sponsor_detection.data.profile import EligibilityPolicy
from sponsor_detection.data.smart_segments_audit import (
    DEFAULT_CATEGORIES,
    _content_offsets,
    _count_span_tokens,
    _merge_intervals,
    build_raw_data_audit,
    build_trainability_audit,
)


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


class CharacterTokenizer:
    """Deterministic character-level token stand-in for the pinned tokenizer."""

    def __call__(
        self,
        text: str,
        *,
        max_length: int | None = None,
        stride: int = 0,
        truncation: bool = True,
        return_offsets_mapping: bool = False,
        return_overflowing_tokens: bool = False,
        padding: bool = False,
        **_: object,
    ) -> dict[str, list[object]]:
        if return_overflowing_tokens:
            capacity = (max_length or 1024) - 2
            input_ids: list[list[int]] = []
            attention_masks: list[list[int]] = []
            offset_mappings: list[list[tuple[int, int]]] = []
            start = 0
            while start < len(text):
                end = min(start + capacity, len(text))
                offsets = (
                    [(0, 0), *[(index, index + 1) for index in range(start, end)], (0, 0)]
                )
                input_ids.append([101, *range(start + 1, end + 1), 102])
                attention_masks.append([1] * len(offsets))
                offset_mappings.append(offsets)
                if end == len(text):
                    break
                start = end - stride
                if start <= 0 and end == len(text):
                    break
            return {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "offset_mapping": offset_mappings,
            }
        limit = len(text)
        if truncation and max_length is not None:
            limit = min(limit, max_length)
        content = [(index, index + 1) for index in range(limit)]
        return {
            "input_ids": [101, *range(1, limit + 1), 102],
            "attention_mask": [1] * (limit + 2),
            "offset_mapping": [(0, 0), *content, (0, 0)],
        }


def _write_parquet(path: Path, columns: dict[str, list[object]]) -> None:
    table = pa.table(columns)
    parquet.write_table(table, path)


class SmartSegmentsAuditFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.mirror_path = root / "sponsorTimes.csv"
        self.subtitles_directory = root / "subtitles"
        self.metadata_directory = root / "metadata"
        self.xenova_directory = root / "xenova"
        self.subtitles_directory.mkdir()
        self.metadata_directory.mkdir()
        self.xenova_directory.mkdir()

        rows = [
            self._mirror_row("VID1", 2.0, 5.0, "S1", "sponsor"),
            self._mirror_row("VID1", 5.0, 8.0, "I1", "interaction"),
            self._mirror_row("VID1", 4.5, 6.5, "P1", "selfpromo"),
            self._mirror_row("VID1", 1.0, 3.0, "S2", "sponsor", hidden=1),
            self._mirror_row("VID1", 50.0, 60.0, "O1", "outro"),
            self._mirror_row("VID1", 7.0, 7.5, "S3", "sponsor", action_type="mute"),
            self._mirror_row("VID2", 0.0, 2.0, "S4", "sponsor", service="Patreon"),
            self._mirror_row("VID1", 20.0, 25.0, "F1", "filler"),
        ]
        header = ",".join(MIRROR_COLUMNS)
        lines = [header, *(",".join(str(value) for value in row) for row in rows)]
        self.mirror_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cues = [
            *[
                {
                    "start": round(index * 0.08, 2),
                    "end": round(index * 0.08 + 0.08, 2),
                    "text": f"part{index:03d}",
                }
                for index in range(25)
            ],
            {"start": 2.0, "end": 5.0, "text": "this video is sponsored by acme products"},
            {"start": 5.0, "end": 8.0, "text": "please like and subscribe to the channel"},
            {"start": 8.0, "end": 12.0, "text": "and now back to the main content"},
            {"start": 12.0, "end": 16.0, "text": "we continue with the tutorial"},
            {"start": 16.0, "end": 20.0, "text": "thanks for watching see you next time"},
        ]
        _write_parquet(
            self.subtitles_directory / "part-00000.parquet",
            {
                "video_id": ["VID1"],
                "language": ["en"],
                "full_text": [" ".join(cue["text"] for cue in cues)],
                "segments_json": [json.dumps(cues)],
            },
        )
        _write_parquet(
            self.metadata_directory / "part-00000.parquet",
            {
                "id": ["VID1"],
                "channel_id": ["CH1"],
                "timestamp": [1_600_000_000],
                "upload_date": ["20200913"],
                "language": ["en"],
                "availability": ["public"],
                "live_status": [None],
                "duration": [600],
            },
        )
        (self.xenova_directory / "segments.json").write_text(
            json.dumps({"VID1": [], "VIDX": []}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    @staticmethod
    def _mirror_row(
        video_id: str,
        start: float,
        end: float,
        uuid: str,
        category: str,
        *,
        votes: int = 5,
        hidden: int = 0,
        action_type: str = "skip",
        service: str = "YouTube",
    ) -> list[object]:
        row = {column: "" for column in MIRROR_COLUMNS}
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
                "shadowHidden": 0,
            }
        )
        return [row[column] for column in MIRROR_COLUMNS]

    def _raw_audit(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "categories": DEFAULT_CATEGORIES,
            "language_prefixes": ["en"],
            "eligibility": EligibilityPolicy(),
            "xenova_directory": self.xenova_directory,
            "compute_sha256": True,
        }
        arguments.update(overrides)
        return build_raw_data_audit(
            self.mirror_path,
            self.subtitles_directory,
            self.metadata_directory,
            Path(self.directory.name) / "raw_audit.json",
            **arguments,
        )

    def _trainability_audit(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "encoder": "jhu-clsp/ettin-encoder-17m",
            "encoder_revision": "revision",
            "categories": DEFAULT_CATEGORIES,
            "language_prefixes": ["en"],
            "eligibility": EligibilityPolicy(),
            "max_length": 32,
            "overlap_tokens": 8,
            "minimum_cues": 2,
            "compute_sha256": True,
            "tokenizer": CharacterTokenizer(),
        }
        arguments.update(overrides)
        return build_trainability_audit(
            self.mirror_path,
            self.subtitles_directory,
            self.metadata_directory,
            Path(self.directory.name) / "trainability_audit.json",
            **arguments,
        )

    # Audit A

    def test_raw_audit_counts_eligible_segments_and_excludes_ineligible(self) -> None:
        report = self._raw_audit()
        table = report["table"]

        self.assertEqual(table["sponsor"]["segments"], 1)
        self.assertEqual(table["interaction"]["segments"], 1)
        self.assertEqual(table["selfpromo"]["segments"], 1)
        self.assertEqual(table["outro"]["segments"], 1)
        self.assertEqual(table["filler"]["segments"], 1)
        self.assertEqual(table["hook"]["segments"], 0)
        self.assertEqual(table["sponsor"]["transcript_videos"], 1)
        self.assertEqual(report["totals"]["videos_with_multiple_categories"], 1)

    def test_raw_audit_reports_cross_category_overlap(self) -> None:
        report = self._raw_audit()
        pairs = report["overlap"]["cross_category_overlapping_pairs"]

        self.assertEqual(pairs.get("selfpromo+sponsor"), 1)
        self.assertEqual(pairs.get("interaction+selfpromo"), 1)
        self.assertNotIn("interaction+sponsor", pairs)

    def test_raw_audit_reports_provenance_hashes(self) -> None:
        report = self._raw_audit()

        self.assertIsNotNone(report["sources"]["mirror"]["sha256"])
        self.assertEqual(len(report["sources"]["subtitles"]), 1)
        self.assertIsNotNone(report["sources"]["xenova"]["sha256"])
        self.assertEqual(
            report["configuration"]["eligibility_policy"]["service"], "YouTube"
        )
        self.assertEqual(
            report["configuration"]["eligibility_policy_version"],
            "eligibility-policy/1",
        )

    def test_raw_audit_is_model_independent(self) -> None:
        report = self._raw_audit()

        self.assertNotIn("windows", report)
        self.assertNotIn("tokens", report)
        self.assertNotIn("categories", report.get("totals", {}))

    def test_raw_audit_counts_transcript_sources_separately(self) -> None:
        report = self._raw_audit()

        self.assertEqual(report["totals"]["scriptsmith_transcript_videos"], 1)
        self.assertEqual(report["totals"]["xenova_transcript_videos"], 2)
        self.assertEqual(report["totals"]["transcript_videos"], 2)
        self.assertEqual(report["totals"]["transcript_videos_in_both_sources"], 1)

    # Audit B

    def test_trainability_aligns_positive_and_reports_unknown(self) -> None:
        report = self._trainability_audit()
        categories = report["categories"]

        self.assertGreater(categories["sponsor"]["positive_windows"], 0)
        self.assertGreater(categories["sponsor"]["positive_tokens"], 0)
        self.assertGreater(categories["interaction"]["positive_tokens"], 0)
        self.assertEqual(categories["sponsor"]["aligned_intervals"], 1)
        self.assertEqual(categories["sponsor"]["unaligned_intervals"], 0)
        self.assertEqual(categories["hook"]["positive_windows"], 0)
        self.assertEqual(
            categories["hook"]["unknown_windows"], report["totals"]["windows"]
        )

    def test_trainability_records_unaligned_intervals_with_reason(self) -> None:
        report = self._trainability_audit()
        categories = report["categories"]

        self.assertEqual(categories["outro"]["unaligned_intervals"], 1)
        self.assertEqual(
            categories["outro"]["alignment_rejection_reasons"]["no_cue_overlap"], 1
        )
        self.assertEqual(categories["filler"]["unaligned_intervals"], 1)
        self.assertEqual(categories["outro"]["positive_windows"], 0)

    def test_trainability_never_treats_missing_annotation_as_negative(self) -> None:
        report = self._trainability_audit()

        for category in DEFAULT_CATEGORIES:
            self.assertEqual(
                report["categories"][category]["confirmed_negative_windows"], 0
            )
        self.assertEqual(
            report["supervision_contract"]["confirmed_negative_evidence"], None
        )

    def test_trainability_uses_configured_negative_evidence(self) -> None:
        evidence = Path(self.directory.name) / "negatives.jsonl"
        evidence.write_text(
            json.dumps(
                {
                    "video_id": "VID1",
                    "category": "hook",
                    "start_ms": 0,
                    "end_ms": 2000,
                    "evidence_source": "manual_review",
                    "evidence_id": "review-1",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = self._trainability_audit(negative_evidence_path=evidence)

        self.assertGreater(
            report["categories"]["hook"]["confirmed_negative_windows"], 0
        )

    def test_trainability_records_pinned_preprocessing_configuration(self) -> None:
        report = self._trainability_audit()
        configuration = report["configuration"]

        self.assertEqual(configuration["max_length"], 32)
        self.assertEqual(configuration["overlap_tokens"], 8)
        self.assertEqual(configuration["encoder_revision"], "revision")
        self.assertEqual(configuration["normalization_version"], "normalize_cue_text/1")
        self.assertEqual(
            configuration["eligibility_policy_version"], "eligibility-policy/1"
        )
        self.assertEqual(
            configuration["window_builder_version"], "build_transcript_windows/1"
        )
        self.assertGreater(report["totals"]["windows"], 1)

    def test_trainability_links_the_raw_audit(self) -> None:
        raw_path = Path(self.directory.name) / "raw_audit.json"
        self._raw_audit()
        report = self._trainability_audit(raw_audit_path=raw_path)

        self.assertEqual(report["inputs"]["raw_data_audit"]["path"], str(raw_path))
        self.assertIsNotNone(report["inputs"]["raw_data_audit"]["sha256"])

    # Token counting

    def test_count_span_tokens_bounds_and_bisects(self) -> None:
        offsets = _content_offsets(((0, 0), (0, 3), (3, 6), (7, 9), (0, 0)))

        self.assertEqual(_count_span_tokens(*offsets, 0, 3), 1)
        self.assertEqual(_count_span_tokens(*offsets, 3, 7), 1)
        self.assertEqual(_count_span_tokens(*offsets, 3, 9), 2)
        self.assertEqual(_count_span_tokens(*offsets, 6, 7), 0)
        self.assertEqual(_count_span_tokens(*offsets, 0, 9), 3)

    def test_merge_intervals_unions_overlapping_spans(self) -> None:
        self.assertEqual(
            _merge_intervals([(0, 4), (3, 6), (9, 10)]),
            [(0, 6), (9, 10)],
        )
        self.assertEqual(_merge_intervals([(5, 6), (0, 2)]), [(0, 2), (5, 6)])


if __name__ == "__main__":
    unittest.main()
