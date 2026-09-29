from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet

from sponsor_detection.data.scriptsmith_dataset import build_scriptsmith_dataset


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


class SmartSegmentDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.subtitles = root / "subtitles"
        self.metadata = root / "metadata"
        self.output = root / "output"
        self.subtitles.mkdir()
        self.metadata.mkdir()

        cues = [
            {"start": 0.0, "end": 2.0, "text": "welcome to the channel"},
            {"start": 2.0, "end": 5.0, "text": "this video is sponsored by acme"},
            {"start": 5.0, "end": 8.0, "text": "please like and subscribe"},
            {"start": 8.0, "end": 12.0, "text": "and now the real content"},
        ]
        parquet.write_table(
            pa.table(
                {
                    "video_id": ["VID1"],
                    "language": ["en"],
                    "full_text": [" ".join(cue["text"] for cue in cues)],
                    "segments_json": [json.dumps(cues)],
                }
            ),
            self.subtitles / "part-00000.parquet",
        )
        parquet.write_table(
            pa.table(
                {
                    "id": ["VID1"],
                    "channel_id": ["CH1"],
                    "timestamp": [1_600_000_000],
                    "upload_date": ["20200913"],
                    "language": ["en"],
                    "availability": ["public"],
                    "live_status": [None],
                    "duration": [600],
                }
            ),
            self.metadata / "part-00000.parquet",
        )

        self.annotations = root / "annotations.parquet"
        parquet.write_table(
            pa.table(
                {
                    "video_id": ["VID1", "VID1", "VID1", "VID1"],
                    "segment_id": ["SP1", "PP1", "IN1", "SP2"],
                    "category": ["sponsor", "selfpromo", "interaction", "sponsor"],
                    "start_ms": [2000, 4500, 5000, 1000],
                    "end_ms": [5000, 6500, 8000, 1500],
                    "is_eligible": [True, True, True, False],
                }
            ),
            self.annotations,
        )

        # A decoy mirror interval that the canonical table does not contain.
        self.mirror = root / "mirror.csv"
        self.mirror.write_text(
            "videoID,startTime,endTime,votes,locked,incorrectVotes,UUID,userID,"
            "timeSubmitted,views,category,actionType,service,videoDuration,hidden,"
            "reputation,shadowHidden,hashedVideoID,userAgent,description\n"
            "VID1,9.0,9.5,5,0,0,DECOY,user,1000,10,sponsor,skip,YouTube,600,0,0.0,"
            "0,,,\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _build(self) -> dict[str, object]:
        return build_scriptsmith_dataset(
            self.subtitles,
            self.metadata,
            self.mirror,
            [],
            self.output,
            Path(self.directory.name) / "manifest.json",
            encoder="jhu-clsp/ettin-encoder-17m",
            encoder_revision="revision",
            max_length=32,
            overlap_tokens=8,
            seed="fixture",
            train_fraction=0.8,
            validation_fraction=0.1,
            language_prefixes=["en"],
            positive_categories=["sponsor", "selfpromo", "interaction"],
            hard_negative_categories=[],
            ordinary_negative_windows_per_video=2,
            maximum_video_duration_seconds=14400,
            minimum_cues=2,
            annotations_path=self.annotations,
            tokenizer=CharacterTokenizer(),
        )

    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for split in ("train", "validation", "test"):
            rows.extend(
                parquet.read_table(self.output / f"{split}.parquet").to_pylist()
            )
        return rows

    def test_reads_canonical_annotations_instead_of_the_mirror(self) -> None:
        self._build()
        start_times = {
            int(span["start_ms"])
            for row in self._rows()
            for span in row["category_spans"]
        }

        self.assertIn(2000, start_times)
        self.assertNotIn(9000, start_times)
        self.assertNotIn(1000, start_times)

    def test_preserves_category_identity_per_span(self) -> None:
        self._build()
        categories = {
            str(span["category"])
            for row in self._rows()
            for span in row["category_spans"]
        }

        self.assertEqual(categories, {"sponsor", "selfpromo", "interaction"})

    def test_keeps_cross_category_overlapping_spans(self) -> None:
        self._build()
        overlapping = [
            row
            for row in self._rows()
            if {"sponsor", "selfpromo"}
            <= {str(span["category"]) for span in row["category_spans"]}
        ]

        self.assertTrue(overlapping)
        row = overlapping[0]
        sponsor = next(s for s in row["category_spans"] if s["category"] == "sponsor")
        selfpromo = next(
            s for s in row["category_spans"] if s["category"] == "selfpromo"
        )
        self.assertLess(sponsor["start_char"], selfpromo["end_char"])
        self.assertLess(selfpromo["start_char"], sponsor["end_char"])

    def test_sponsor_spans_field_stays_sponsor_only(self) -> None:
        self._build()
        for row in self._rows():
            for span in row["sponsor_spans"]:
                self.assertEqual(span["category"], "sponsor")

    def test_positive_window_categories_reflect_present_categories(self) -> None:
        self._build()
        positives = [row for row in self._rows() if row["label_kind"] == "positive"]

        self.assertTrue(positives)
        for row in positives:
            self.assertTrue(set(row["legacy_categories"]) <= {"SPONSOR", "SELFPROMO", "INTERACTION"})

    def test_emits_explicit_positive_supervision_with_evidence(self) -> None:
        self._build()
        overlapping = [
            row
            for row in self._rows()
            if {"sponsor", "selfpromo"}
            <= {entry["category"] for entry in row["category_supervision"]}
        ]

        self.assertTrue(overlapping)
        entries = {
            entry["category"]: entry for entry in overlapping[0]["category_supervision"]
        }
        self.assertEqual(entries["sponsor"]["state"], "POSITIVE")
        self.assertEqual(entries["sponsor"]["evidence_id"], "SP1")
        self.assertEqual(entries["selfpromo"]["state"], "POSITIVE")

    def test_negative_windows_have_no_supervision(self) -> None:
        self._build()
        for row in self._rows():
            if row["label_kind"] == "positive":
                continue
            self.assertEqual(row["category_supervision"], [])


if __name__ == "__main__":
    unittest.main()
