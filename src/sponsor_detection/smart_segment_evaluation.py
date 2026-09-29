from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from sponsor_detection.data.training_dataset import temporal_iou
from sponsor_detection.smart_segment_gates import wilson_interval


DEFAULT_IOU_THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass(frozen=True, slots=True)
class SpanMatch:
    predicted_index: int
    expected_index: int
    iou: float


def _interval(span: Sequence[int]) -> tuple[int, int]:
    return int(span[0]), int(span[1])


def match_spans(
    predicted: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
    *,
    iou_threshold: float,
) -> tuple[list[SpanMatch], list[int], list[int]]:
    """Greedily match predicted to expected spans by descending temporal IoU.

    Returns the matches plus the unmatched predicted and expected indexes.
    Greedy matching is deterministic and does not depend on input ordering.
    """

    candidates: list[SpanMatch] = []
    for predicted_index, predicted_span in enumerate(predicted):
        predicted_start, predicted_end = _interval(predicted_span)
        for expected_index, expected_span in enumerate(expected):
            expected_start, expected_end = _interval(expected_span)
            iou = temporal_iou(
                predicted_start, predicted_end, expected_start, expected_end
            )
            if iou >= iou_threshold:
                candidates.append(
                    SpanMatch(
                        predicted_index=predicted_index,
                        expected_index=expected_index,
                        iou=iou,
                    )
                )
    candidates.sort(key=lambda match: (-match.iou, match.predicted_index, match.expected_index))
    used_predicted: set[int] = set()
    used_expected: set[int] = set()
    matches: list[SpanMatch] = []
    for candidate in candidates:
        if candidate.predicted_index in used_predicted:
            continue
        if candidate.expected_index in used_expected:
            continue
        used_predicted.add(candidate.predicted_index)
        used_expected.add(candidate.expected_index)
        matches.append(candidate)
    unmatched_predicted = [
        index for index in range(len(predicted)) if index not in used_predicted
    ]
    unmatched_expected = [
        index for index in range(len(expected)) if index not in used_expected
    ]
    return matches, unmatched_predicted, unmatched_expected


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, object]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "precision_ci_wilson_95": wilson_interval(
            true_positive, true_positive + false_positive
        ),
        "recall_ci_wilson_95": wilson_interval(
            true_positive, true_positive + false_negative
        ),
    }


def _align_videos(
    predicted: dict[str, Sequence[Sequence[int]]],
    expected: dict[str, Sequence[Sequence[int]]],
) -> Iterable[tuple[str, Sequence[Sequence[int]], Sequence[Sequence[int]]]]:
    for video_id in sorted(set(predicted) | set(expected)):
        yield video_id, predicted.get(video_id, ()), expected.get(video_id, ())


def span_metrics(
    predicted: dict[str, Sequence[Sequence[int]]],
    expected: dict[str, Sequence[Sequence[int]]],
    *,
    iou_threshold: float,
) -> dict[str, object]:
    true_positive = false_positive = false_negative = 0
    for _, predicted_spans, expected_spans in _align_videos(predicted, expected):
        matches, unmatched_predicted, unmatched_expected = match_spans(
            predicted_spans, expected_spans, iou_threshold=iou_threshold
        )
        true_positive += len(matches)
        false_positive += len(unmatched_predicted)
        false_negative += len(unmatched_expected)
    return _prf(true_positive, false_positive, false_negative)


def presence_metrics(
    predicted: dict[str, Sequence[Sequence[int]]],
    expected: dict[str, Sequence[Sequence[int]]],
) -> dict[str, object]:
    true_positive = false_positive = false_negative = true_negative = 0
    for _, predicted_spans, expected_spans in _align_videos(predicted, expected):
        has_predicted = bool(predicted_spans)
        has_expected = bool(expected_spans)
        if has_predicted and has_expected:
            true_positive += 1
        elif has_predicted:
            false_positive += 1
        elif has_expected:
            false_negative += 1
        else:
            true_negative += 1
    metrics = _prf(true_positive, false_positive, false_negative)
    metrics["true_negative"] = true_negative
    return metrics


def temporal_coverage(
    predicted: dict[str, Sequence[Sequence[int]]],
    expected: dict[str, Sequence[Sequence[int]]],
    *,
    durations_ms: dict[str, int] | None = None,
) -> dict[str, object]:
    """Overlap, false-positive and missed seconds, normalised per video hour."""

    overlap = predicted_seconds = expected_seconds = 0
    for _, predicted_spans, expected_spans in _align_videos(predicted, expected):
        for start, end in predicted_spans:
            predicted_seconds += max(0, int(end) - int(start))
        for start, end in expected_spans:
            expected_seconds += max(0, int(end) - int(start))
        for predicted_span in predicted_spans:
            predicted_start, predicted_end = _interval(predicted_span)
            for expected_span in expected_spans:
                expected_start, expected_end = _interval(expected_span)
                overlap += max(
                    0,
                    min(predicted_end, expected_end) - max(predicted_start, expected_start),
                )
    false_positive_seconds = (predicted_seconds - overlap) / 1000.0
    missed_seconds = (expected_seconds - overlap) / 1000.0
    result: dict[str, object] = {
        "predicted_seconds": round(predicted_seconds / 1000.0, 3),
        "expected_seconds": round(expected_seconds / 1000.0, 3),
        "overlap_seconds": round(overlap / 1000.0, 3),
        "false_positive_seconds": round(false_positive_seconds, 3),
        "missed_seconds": round(missed_seconds, 3),
        "false_positive_seconds_per_hour": None,
        "missed_seconds_per_hour": None,
        "reviewed_hours": None,
    }
    if durations_ms:
        total_ms = sum(int(durations_ms.get(video_id, 0)) for video_id in predicted.keys() | expected.keys())
        hours = total_ms / 3_600_000.0
        if hours > 0:
            result["reviewed_hours"] = round(hours, 6)
            result["false_positive_seconds_per_hour"] = round(
                false_positive_seconds / hours, 3
            )
            result["missed_seconds_per_hour"] = round(missed_seconds / hours, 3)
    return result


def evaluate_category(
    predicted: dict[str, Sequence[Sequence[int]]],
    expected: dict[str, Sequence[Sequence[int]]],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    durations_ms: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "video_presence": presence_metrics(predicted, expected),
        "span_metrics": {
            f"iou_{threshold:.1f}": span_metrics(
                predicted, expected, iou_threshold=threshold
            )
            for threshold in iou_thresholds
        },
        "temporal_coverage": temporal_coverage(
            predicted, expected, durations_ms=durations_ms
        ),
    }


def evaluate_categories(
    predicted: dict[str, dict[str, Sequence[Sequence[int]]]],
    expected: dict[str, dict[str, Sequence[Sequence[int]]]],
    *,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    durations_ms: dict[str, int] | None = None,
) -> dict[str, object]:
    categories = sorted(set(predicted) | set(expected))
    return {
        category: evaluate_category(
            predicted.get(category, {}),
            expected.get(category, {}),
            iou_thresholds=iou_thresholds,
            durations_ms=durations_ms,
        )
        for category in categories
    }
