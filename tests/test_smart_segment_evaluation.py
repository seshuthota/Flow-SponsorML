from __future__ import annotations

import unittest

from sponsor_detection.smart_segment_evaluation import (
    evaluate_categories,
    match_spans,
    presence_metrics,
    span_metrics,
    temporal_coverage,
)


class MatchSpansTest(unittest.TestCase):
    def test_matches_by_descending_iou(self) -> None:
        matches, unmatched_predicted, unmatched_expected = match_spans(
            [[0, 1000], [1100, 2000]],
            [[0, 1000]],
            iou_threshold=0.5,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].predicted_index, 0)
        self.assertEqual(unmatched_predicted, [1])
        self.assertEqual(unmatched_expected, [])

    def test_leaves_unmatched_expected(self) -> None:
        matches, unmatched_predicted, unmatched_expected = match_spans(
            [], [[0, 1000], [2000, 3000]], iou_threshold=0.5
        )

        self.assertEqual(matches, [])
        self.assertEqual(unmatched_predicted, [])
        self.assertEqual(unmatched_expected, [0, 1])

    def test_one_prediction_matches_only_one_expectation(self) -> None:
        matches, unmatched_predicted, unmatched_expected = match_spans(
            [[0, 1000]], [[0, 1000], [0, 1000]], iou_threshold=0.5
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(unmatched_predicted, [])
        self.assertEqual(unmatched_expected, [1])


class SpanMetricsTest(unittest.TestCase):
    def test_perfect_match(self) -> None:
        metrics = span_metrics(
            {"v": [[0, 1000]]}, {"v": [[0, 1000]]}, iou_threshold=0.5
        )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["precision_ci_wilson_95"]["total"], 1)

    def test_counts_false_positive_and_negative(self) -> None:
        metrics = span_metrics(
            {"v": [[0, 1000], [5000, 6000]]},
            {"v": [[0, 1000], [9000, 10000]]},
            iou_threshold=0.5,
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)


class PresenceMetricsTest(unittest.TestCase):
    def test_confusion_counts(self) -> None:
        metrics = presence_metrics(
            {"a": [[0, 1]], "b": [], "c": [[0, 1]], "d": []},
            {"a": [[0, 1]], "b": [[0, 1]], "c": [], "d": []},
        )

        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["true_negative"], 1)


class TemporalCoverageTest(unittest.TestCase):
    def test_per_hour_uses_reviewed_durations(self) -> None:
        coverage = temporal_coverage(
            {"v": [[0, 10_000]]},
            {"v": [[0, 5_000]]},
            durations_ms={"v": 1_800_000},
        )

        self.assertEqual(coverage["overlap_seconds"], 5.0)
        self.assertEqual(coverage["false_positive_seconds"], 5.0)
        self.assertEqual(coverage["missed_seconds"], 0.0)
        self.assertEqual(coverage["reviewed_hours"], 0.5)
        self.assertEqual(coverage["false_positive_seconds_per_hour"], 10.0)

    def test_unavailable_durations_leave_per_hour_null(self) -> None:
        coverage = temporal_coverage(
            {"v": [[0, 10_000]]}, {"v": [[0, 5_000]]}, durations_ms=None
        )

        self.assertIsNone(coverage["false_positive_seconds_per_hour"])
        self.assertEqual(coverage["false_positive_seconds"], 5.0)


class EvaluateCategoriesTest(unittest.TestCase):
    def test_reports_each_category_independently(self) -> None:
        report = evaluate_categories(
            {"sponsor": {"v": [[0, 1000]]}},
            {"sponsor": {"v": [[0, 1000]]}, "interaction": {"v": [[0, 1000]]}},
            iou_thresholds=(0.5,),
        )

        self.assertEqual(report["sponsor"]["span_metrics"]["iou_0.5"]["f1"], 1.0)
        self.assertEqual(
            report["interaction"]["span_metrics"]["iou_0.5"]["recall"], 0.0
        )
        self.assertEqual(report["interaction"]["span_metrics"]["iou_0.5"]["true_positive"], 0)


if __name__ == "__main__":
    unittest.main()
