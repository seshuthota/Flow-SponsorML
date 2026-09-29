from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sponsor_detection.data.smart_segment_annotations import CanonicalAnnotation


POSITIVE = "POSITIVE"
NEGATIVE_CONFIRMED = "NEGATIVE_CONFIRMED"
UNKNOWN = "UNKNOWN"
SUPERVISION_STATES = (POSITIVE, NEGATIVE_CONFIRMED, UNKNOWN)
SUPERVISION_CONTRACT_VERSION = "smart-segment-supervision/1"

CONFIRMED_NEGATIVE_FIELDS = (
    "video_id",
    "category",
    "start_ms",
    "end_ms",
    "evidence_source",
    "evidence_id",
)


@dataclass(frozen=True, slots=True)
class ConfirmedNegative:
    """A reviewed interval that establishes the absence of one category.

    Provenance is mandatory: an annotation for a different category never
    proves a negative, and no negative exists without a named source and a
    review identifier.
    """

    video_id: str
    category: str
    start_ms: int
    end_ms: int
    evidence_source: str
    evidence_id: str


def _overlaps(
    left_start: int, left_end: int, right_start: int, right_end: int
) -> bool:
    return min(left_end, right_end) > max(left_start, right_start)


def load_confirmed_negatives(
    path: Path | None,
) -> dict[str, list[ConfirmedNegative]]:
    """Load reviewed confirmed negatives from JSONL, validating provenance.

    Expected shape: ``{"video_id", "category", "start_ms", "end_ms",
    "evidence_source", "evidence_id"}``.
    """

    negatives: dict[str, list[ConfirmedNegative]] = {}
    if path is None or not path.is_file():
        return negatives
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(
                    f"confirmed negative line {line_number} is not valid JSON"
                ) from error
            missing = [
                field
                for field in CONFIRMED_NEGATIVE_FIELDS
                if record.get(field) in (None, "")
            ]
            if missing:
                raise ValueError(
                    f"confirmed negative line {line_number} is missing provenance: "
                    f"{', '.join(missing)}"
                )
            start_ms = int(record["start_ms"])
            end_ms = int(record["end_ms"])
            if start_ms < 0 or end_ms <= start_ms:
                raise ValueError(
                    f"confirmed negative line {line_number} has an invalid interval"
                )
            video_id = str(record["video_id"])
            negatives.setdefault(video_id, []).append(
                ConfirmedNegative(
                    video_id=video_id,
                    category=str(record["category"]),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    evidence_source=str(record["evidence_source"]),
                    evidence_id=str(record["evidence_id"]),
                )
            )
    return negatives


def category_state(
    category: str,
    *,
    window_start_ms: int,
    window_end_ms: int,
    annotations: Sequence[CanonicalAnnotation],
    negatives: Sequence[ConfirmedNegative] = (),
) -> str:
    """Resolve one category's supervision state for one window.

    Presence of any other category has no effect: an annotation for SPONSOR
    leaves SELF_PROMO and INTERACTION UNKNOWN unless they have their own
    evidence.
    """

    for annotation in annotations:
        if annotation.category == category and _overlaps(
            annotation.start_ms, annotation.end_ms, window_start_ms, window_end_ms
        ):
            return POSITIVE
    for negative in negatives:
        if negative.category == category and _overlaps(
            negative.start_ms, negative.end_ms, window_start_ms, window_end_ms
        ):
            return NEGATIVE_CONFIRMED
    return UNKNOWN


def window_category_states(
    *,
    window_start_ms: int,
    window_end_ms: int,
    annotations: Sequence[CanonicalAnnotation],
    categories: Sequence[str],
    negatives: Sequence[ConfirmedNegative] = (),
) -> dict[str, str]:
    return {
        category: category_state(
            category,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            annotations=annotations,
            negatives=negatives,
        )
        for category in categories
    }


def token_states(
    offset_mapping: Sequence[tuple[int, int]],
    *,
    positive_spans: Sequence[tuple[int, int]] = (),
    negative_spans: Sequence[tuple[int, int]] = (),
) -> list[str]:
    """Per-token supervision for one category, in transcript character space.

    A token inside a positive span is POSITIVE, inside a confirmed negative is
    NEGATIVE_CONFIRMED, and otherwise UNKNOWN. Offsets are half-open character
    ranges; special tokens with an empty offset stay UNKNOWN.
    """

    states: list[str] = []
    for offset_start, offset_end in offset_mapping:
        if offset_end <= offset_start:
            states.append(UNKNOWN)
            continue
        if any(
            _overlaps(offset_start, offset_end, start, end)
            for start, end in positive_spans
        ):
            states.append(POSITIVE)
        elif any(
            _overlaps(offset_start, offset_end, start, end)
            for start, end in negative_spans
        ):
            states.append(NEGATIVE_CONFIRMED)
        else:
            states.append(UNKNOWN)
    return states


def loss_mask(states: Sequence[str]) -> list[int]:
    """Tokens with known supervision contribute to the loss; UNKNOWN does not."""

    return [0 if state == UNKNOWN else 1 for state in states]


def token_supervision(
    offset_mapping: Sequence[tuple[int, int]],
    *,
    positive_spans: Sequence[tuple[int, int]] = (),
    negative_spans: Sequence[tuple[int, int]] = (),
) -> tuple[list[str], list[int]]:
    states = token_states(
        offset_mapping, positive_spans=positive_spans, negative_spans=negative_spans
    )
    return states, loss_mask(states)


def positive_supervision(
    category_evidence: dict[str, str],
) -> list[dict[str, str]]:
    """Build POSITIVE supervision entries from category -> evidence id.

    Categories absent from the map stay UNKNOWN; this helper never emits a
    negative, because a missing annotation is not evidence of absence.
    """

    return [
        {"category": category, "state": POSITIVE, "evidence_id": evidence_id}
        for category, evidence_id in sorted(category_evidence.items())
    ]
