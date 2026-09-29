from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


BILOU_LABELS = ("O", "B", "I", "L", "U")


@dataclass(frozen=True, slots=True)
class DecodedCategorySpan:
    """One decoded span for a single category, in token-index space."""

    category: str
    start_index: int
    end_index: int
    confidence: float
    labels: tuple[str, ...]


def softmax_probabilities(logits):
    """Softmax over the last (BILOU) axis of a [sequence, category, label] array."""

    import numpy as np

    values = np.asarray(logits, dtype="float64")
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _validate_shape(probabilities, categories: Sequence[str]) -> None:
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [sequence, category, label]")
    if probabilities.shape[1] != len(categories):
        raise ValueError(
            "category axis does not match the ordered category list: "
            f"{probabilities.shape[1]} != {len(categories)}"
        )
    if probabilities.shape[2] != len(BILOU_LABELS):
        raise ValueError(
            f"label axis must be {len(BILOU_LABELS)} (BILOU), got {probabilities.shape[2]}"
        )


def decode_category(
    tag_ids: Sequence[int],
    tag_probabilities: Sequence[float],
    *,
    category: str,
    threshold: float,
    labels: Sequence[str] = BILOU_LABELS,
) -> list[DecodedCategorySpan]:
    """Greedy BILOU span assembly for one category.

    ``B`` opens a span, ``I`` continues it, ``L`` closes it, ``U`` is a
    single-token span and ``O`` is outside. A span left open at the sequence
    end, or a stray ``I``/``L``, still closes so a malformed tag stream cannot
    drop text silently.
    """

    spans: list[DecodedCategorySpan] = []
    start: int | None = None
    open_labels: list[str] = []
    open_probabilities: list[float] = []

    def close(end_index: int) -> None:
        nonlocal start, open_labels, open_probabilities
        if start is None:
            return
        confidence = sum(open_probabilities) / len(open_probabilities)
        if confidence >= threshold:
            spans.append(
                DecodedCategorySpan(
                    category=category,
                    start_index=start,
                    end_index=end_index,
                    confidence=round(confidence, 6),
                    labels=tuple(open_labels),
                )
            )
        start = None
        open_labels = []
        open_probabilities = []

    for index, tag_id in enumerate(tag_ids):
        tag = labels[int(tag_id)]
        probability = float(tag_probabilities[index])
        if tag == "O":
            close(index)
            continue
        if tag in ("B", "U"):
            close(index)
            start = index
            open_labels = [tag]
            open_probabilities = [probability]
            if tag == "U":
                close(index + 1)
            continue
        # I or L: continue an open span, or start one if the tag stream is broken.
        if start is None:
            start = index
            open_labels = []
            open_probabilities = []
        open_labels.append(tag)
        open_probabilities.append(probability)
        if tag == "L":
            close(index + 1)
    close(len(tag_ids))
    return spans


def decode_multi_head(
    probabilities,
    *,
    categories: Sequence[str],
    thresholds: dict[str, float] | None = None,
    default_threshold: float = 0.5,
    labels: Sequence[str] = BILOU_LABELS,
) -> list[DecodedCategorySpan]:
    """Decode every category head independently from one shared encoder pass.

    Categories are independent by design: the same token may be positive for
    more than one category, and the decoder never forces a single label per
    token.
    """

    import numpy as np

    values = np.asarray(probabilities, dtype="float64")
    _validate_shape(values, categories)
    configured = thresholds or {}
    decoded: list[DecodedCategorySpan] = []
    for category_index, category in enumerate(categories):
        category_probabilities = values[:, category_index, :]
        tag_ids = np.argmax(category_probabilities, axis=1)
        tag_probabilities = np.take_along_axis(
            category_probabilities, tag_ids[:, None], axis=1
        )[:, 0]
        decoded.extend(
            decode_category(
                tag_ids.tolist(),
                tag_probabilities.tolist(),
                category=category,
                threshold=float(configured.get(category, default_threshold)),
                labels=labels,
            )
        )
    decoded.sort(key=lambda span: (span.start_index, span.end_index, span.category))
    return decoded
