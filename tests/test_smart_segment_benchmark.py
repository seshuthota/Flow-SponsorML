from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sponsor_detection.data.smart_segment_benchmark import (
    CONTENT_NEGATIVE_STRATUM,
    BenchmarkCandidate,
    _stratum_for,
    freeze_benchmark,
    select_candidates,
)


CATEGORIES = ["sponsor", "selfpromo", "interaction"]


def _candidate(video_id: str, channel_id: str, stratum: str = "sponsor") -> BenchmarkCandidate:
    return BenchmarkCandidate(
        video_id=video_id,
        channel_id=channel_id,
        stratum=stratum,
        duration_ms=600_000,
        language="en",
    )


class StratumTest(unittest.TestCase):
    def test_uses_plan_order_when_several_categories_present(self) -> None:
        self.assertEqual(_stratum_for({"interaction", "sponsor"}, CATEGORIES), "sponsor")

    def test_position_breaks_ties_when_the_first_is_absent(self) -> None:
        self.assertEqual(
            _stratum_for({"interaction", "selfpromo"}, CATEGORIES), "selfpromo"
        )

    def test_no_category_is_content_negative(self) -> None:
        self.assertEqual(_stratum_for(set(), CATEGORIES), CONTENT_NEGATIVE_STRATUM)


class SelectCandidatesTest(unittest.TestCase):
    def test_is_channel_disjoint(self) -> None:
        candidates = [
            _candidate("a", "channel-1"),
            _candidate("b", "channel-1"),
            _candidate("c", "channel-2"),
        ]

        selected = select_candidates(
            candidates, targets={"sponsor": 5}, seed="s", excluded_channels=set()
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(len({candidate.channel_id for candidate in selected}), 2)

    def test_respects_per_stratum_targets(self) -> None:
        candidates = [
            _candidate("a", "c1", "sponsor"),
            _candidate("b", "c2", "selfpromo"),
            _candidate("c", "c3", "selfpromo"),
        ]

        selected = select_candidates(
            candidates,
            targets={"sponsor": 1, "selfpromo": 1},
            seed="s",
            excluded_channels=set(),
        )

        self.assertEqual(
            sorted(candidate.stratum for candidate in selected),
            ["selfpromo", "sponsor"],
        )

    def test_skips_excluded_channels(self) -> None:
        candidates = [_candidate("a", "c1"), _candidate("b", "c2")]

        selected = select_candidates(
            candidates, targets={"sponsor": 5}, seed="s", excluded_channels={"c1"}
        )

        self.assertEqual([candidate.channel_id for candidate in selected], ["c2"])

    def test_is_deterministic_for_a_seed(self) -> None:
        candidates = [
            _candidate(video_id, f"channel-{video_id}")
            for video_id in ("a", "b", "c", "d", "e")
        ]

        first = select_candidates(
            candidates, targets={"sponsor": 2}, seed="seed-1", excluded_channels=set()
        )
        second = select_candidates(
            candidates, targets={"sponsor": 2}, seed="seed-1", excluded_channels=set()
        )
        other = select_candidates(
            candidates, targets={"sponsor": 2}, seed="seed-2", excluded_channels=set()
        )

        self.assertEqual([c.video_id for c in first], [c.video_id for c in second])
        self.assertNotEqual([c.video_id for c in first], [c.video_id for c in other])


class FreezeBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.review = self.root / "review.jsonl"
        self.output = self.root / "frozen.jsonl"
        self.manifest = self.root / "manifest.json"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _record(
        self,
        video_id: str,
        *,
        status: str = "reviewed",
        reviewer: str | None = "reviewer-1",
        intervals: dict[str, list[dict[str, int]]] | None = None,
        content_confirmed: bool = False,
    ) -> dict[str, object]:
        return {
            "video_id": video_id,
            "channel_id": f"channel-{video_id}",
            "stratum": "sponsor",
            "transcript": {"sha256": hashlib.sha256(video_id.encode()).hexdigest()},
            "annotation": {
                "status": status,
                "reviewer": reviewer,
                "content_confirmed": content_confirmed,
                "intervals": intervals
                or {"sponsor": [], "selfpromo": [], "interaction": []},
            },
        }

    def _write(self, records: list[dict[str, object]]) -> None:
        self.review.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_freezes_and_derives_confirmed_negatives(self) -> None:
        self._write(
            [
                self._record(
                    "pos",
                    intervals={
                        "sponsor": [{"start_ms": 1000, "end_ms": 2000}],
                        "selfpromo": [],
                        "interaction": [],
                    },
                ),
                self._record("neg", content_confirmed=True),
            ]
        )

        manifest = freeze_benchmark(
            self.review, self.output, self.manifest, categories=CATEGORIES
        )
        frozen = [
            json.loads(line) for line in self.output.read_text().splitlines() if line
        ]

        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(manifest["counts"]["videos"], 2)
        self.assertEqual(manifest["counts"]["confirmed_negative_videos"], 1)
        self.assertEqual(
            manifest["counts"]["positive_videos_by_category"], {"sponsor": 1}
        )
        negative = next(record for record in frozen if record["video_id"] == "neg")
        self.assertTrue(negative["confirmed_negative"])
        self.assertIsNotNone(manifest["output"]["sha256"])

    def test_rejects_unreviewed_records(self) -> None:
        self._write([self._record("a", status="pending")])

        with self.assertRaises(ValueError):
            freeze_benchmark(self.review, self.output, self.manifest, categories=CATEGORIES)

    def test_rejects_missing_reviewer(self) -> None:
        self._write([self._record("a", reviewer=None, content_confirmed=True)])

        with self.assertRaises(ValueError):
            freeze_benchmark(self.review, self.output, self.manifest, categories=CATEGORIES)

    def test_rejects_negative_without_confirmation(self) -> None:
        self._write([self._record("a")])

        with self.assertRaises(ValueError):
            freeze_benchmark(self.review, self.output, self.manifest, categories=CATEGORIES)

    def test_rejects_invalid_intervals(self) -> None:
        self._write(
            [
                self._record(
                    "a",
                    intervals={
                        "sponsor": [{"start_ms": 2000, "end_ms": 2000}],
                        "selfpromo": [],
                        "interaction": [],
                    },
                )
            ]
        )

        with self.assertRaises(ValueError):
            freeze_benchmark(self.review, self.output, self.manifest, categories=CATEGORIES)


if __name__ == "__main__":
    unittest.main()
