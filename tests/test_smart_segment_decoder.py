from __future__ import annotations

import unittest

from sponsor_detection.model.smart_segment_decoder import (
    BILOU_LABELS,
    decode_multi_head,
    softmax_probabilities,
)


def _tag(probability: float, tag: str) -> list[float]:
    """A probability vector that makes ``tag`` the argmax."""

    vector = [0.0] * len(BILOU_LABELS)
    vector[BILOU_LABELS.index(tag)] = probability
    vector[BILOU_LABELS.index("O")] = 1.0 - probability
    return vector


def _sequence(*tags: str) -> list[list[list[float]]]:
    return [[_tag(0.9, tag)] for tag in tags]


class MultiHeadDecoderTest(unittest.TestCase):
    def test_decodes_bilou_span_and_averages_confidence(self) -> None:
        probabilities = [[_tag(0.9, "B")], [_tag(0.8, "I")], [_tag(0.7, "L")]]

        spans = decode_multi_head(probabilities, categories=["sponsor"])

        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0].start_index, spans[0].end_index), (0, 3))
        self.assertAlmostEqual(spans[0].confidence, 0.8, places=6)
        self.assertEqual(spans[0].labels, ("B", "I", "L"))

    def test_single_token_and_gaps(self) -> None:
        probabilities = [
            [_tag(0.9, "O")],
            [_tag(0.9, "U")],
            [_tag(0.9, "O")],
            [_tag(0.9, "U")],
        ]

        spans = decode_multi_head(probabilities, categories=["sponsor"])

        self.assertEqual([(s.start_index, s.end_index) for s in spans], [(1, 2), (3, 4)])

    def test_categories_are_decoded_independently(self) -> None:
        probabilities = [
            [_tag(0.9, "O"), _tag(0.9, "U")],
            [_tag(0.9, "B"), _tag(0.9, "O")],
            [_tag(0.9, "L"), _tag(0.9, "O")],
        ]

        spans = decode_multi_head(
            probabilities, categories=["sponsor", "selfpromo"]
        )
        by_category = {span.category: span for span in spans}

        self.assertEqual(
            (by_category["sponsor"].start_index, by_category["sponsor"].end_index),
            (1, 3),
        )
        self.assertEqual(
            (by_category["selfpromo"].start_index, by_category["selfpromo"].end_index),
            (0, 1),
        )

    def test_same_token_can_be_positive_for_two_categories(self) -> None:
        probabilities = [[_tag(0.9, "U"), _tag(0.9, "U")]]

        spans = decode_multi_head(
            probabilities, categories=["sponsor", "interaction"]
        )

        self.assertEqual(
            {(span.category, span.start_index) for span in spans},
            {("sponsor", 0), ("interaction", 0)},
        )

    def test_per_category_thresholds_filter_independently(self) -> None:
        probabilities = [[_tag(0.8, "U"), _tag(0.8, "U")]]

        spans = decode_multi_head(
            probabilities,
            categories=["sponsor", "selfpromo"],
            thresholds={"sponsor": 0.95},
            default_threshold=0.5,
        )

        self.assertEqual([span.category for span in spans], ["selfpromo"])

    def test_unclosed_span_still_closes(self) -> None:
        probabilities = _sequence("B", "I")

        spans = decode_multi_head(probabilities, categories=["sponsor"])

        self.assertEqual((spans[0].start_index, spans[0].end_index), (0, 2))

    def test_stray_continuation_starts_a_span(self) -> None:
        probabilities = _sequence("I", "O")

        spans = decode_multi_head(probabilities, categories=["sponsor"])

        self.assertEqual((spans[0].start_index, spans[0].end_index), (0, 1))

    def test_rejects_wrong_category_axis(self) -> None:
        with self.assertRaises(ValueError):
            decode_multi_head([[[0.5] * 5]], categories=["sponsor", "selfpromo"])

    def test_rejects_wrong_label_axis(self) -> None:
        with self.assertRaises(ValueError):
            decode_multi_head([[[0.5, 0.5]]], categories=["sponsor"])

    def test_rejects_wrong_rank(self) -> None:
        with self.assertRaises(ValueError):
            decode_multi_head([[0.5] * 5], categories=["sponsor"])

    def test_softmax_normalizes_the_label_axis(self) -> None:
        probabilities = softmax_probabilities([[[2.0, 1.0, 0.0, 0.0, 0.0]]])

        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=9)
        self.assertGreater(float(probabilities[0, 0, 0]), float(probabilities[0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
