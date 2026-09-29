from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sponsor_detection.data.smart_segment_annotations import CanonicalAnnotation
from sponsor_detection.data.smart_segment_supervision import (
    NEGATIVE_CONFIRMED,
    POSITIVE,
    UNKNOWN,
    loss_mask,
    load_confirmed_negatives,
    token_states,
    window_category_states,
)


CATEGORIES = ["sponsor", "selfpromo", "interaction"]


def _annotation(category: str, start_ms: int, end_ms: int) -> CanonicalAnnotation:
    return CanonicalAnnotation(
        video_id="VID1",
        segment_id=f"{category}-{start_ms}",
        category=category,
        start_ms=start_ms,
        end_ms=end_ms,
    )


class SupervisionContractTest(unittest.TestCase):
    def test_positive_leaves_unrelated_categories_unknown(self) -> None:
        states = window_category_states(
            window_start_ms=0,
            window_end_ms=10_000,
            annotations=[_annotation("sponsor", 2000, 5000)],
            categories=CATEGORIES,
        )

        self.assertEqual(states["sponsor"], POSITIVE)
        self.assertEqual(states["selfpromo"], UNKNOWN)
        self.assertEqual(states["interaction"], UNKNOWN)

    def test_simultaneous_categories_are_independently_positive(self) -> None:
        states = window_category_states(
            window_start_ms=0,
            window_end_ms=10_000,
            annotations=[
                _annotation("sponsor", 2000, 5000),
                _annotation("selfpromo", 4500, 6500),
            ],
            categories=CATEGORIES,
        )

        self.assertEqual(states["sponsor"], POSITIVE)
        self.assertEqual(states["selfpromo"], POSITIVE)
        self.assertEqual(states["interaction"], UNKNOWN)

    def test_annotation_for_one_category_is_not_a_negative_for_another(self) -> None:
        states = window_category_states(
            window_start_ms=0,
            window_end_ms=10_000,
            annotations=[_annotation("interaction", 2000, 5000)],
            categories=CATEGORIES,
        )

        self.assertEqual(states["interaction"], POSITIVE)
        self.assertEqual(states["sponsor"], UNKNOWN)
        self.assertEqual(states["selfpromo"], UNKNOWN)

    def test_uses_confirmed_negatives_per_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negatives.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "video_id": "VID1",
                        "category": "sponsor",
                        "start_ms": 7000,
                        "end_ms": 9000,
                        "evidence_source": "manual_review",
                        "evidence_id": "review-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            negatives = load_confirmed_negatives(path)["VID1"]

        states = window_category_states(
            window_start_ms=8000,
            window_end_ms=10_000,
            annotations=[],
            categories=CATEGORIES,
            negatives=negatives,
        )

        self.assertEqual(states["sponsor"], NEGATIVE_CONFIRMED)
        self.assertEqual(states["selfpromo"], UNKNOWN)

    def test_confirmed_negative_requires_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negatives.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "video_id": "VID1",
                        "category": "sponsor",
                        "start_ms": 0,
                        "end_ms": 1000,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as context:
                load_confirmed_negatives(path)

        self.assertIn("evidence_source", str(context.exception))

    def test_confirmed_negative_rejects_invalid_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negatives.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "video_id": "VID1",
                        "category": "sponsor",
                        "start_ms": 5000,
                        "end_ms": 5000,
                        "evidence_source": "manual_review",
                        "evidence_id": "review-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_confirmed_negatives(path)

    def test_missing_negative_file_is_empty(self) -> None:
        self.assertEqual(load_confirmed_negatives(None), {})

    def test_token_states_and_loss_mask(self) -> None:
        offsets = [(0, 0), (0, 3), (3, 6), (6, 9), (9, 12), (0, 0)]

        states = token_states(
            offsets,
            positive_spans=[(3, 6)],
            negative_spans=[(9, 12)],
        )

        self.assertEqual(
            states,
            [UNKNOWN, UNKNOWN, POSITIVE, UNKNOWN, NEGATIVE_CONFIRMED, UNKNOWN],
        )
        self.assertEqual(loss_mask(states), [0, 0, 1, 0, 1, 0])

    def test_special_tokens_are_never_supervised(self) -> None:
        states = token_states([(0, 0), (0, 0)], positive_spans=[(0, 5)])

        self.assertEqual(states, [UNKNOWN, UNKNOWN])


if __name__ == "__main__":
    unittest.main()
