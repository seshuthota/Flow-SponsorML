from __future__ import annotations

from typing import Sequence

from sponsor_detection.data.smart_segment_supervision import (
    NEGATIVE_CONFIRMED,
    POSITIVE,
    token_states,
)


BILOU_TO_ID = {"O": 0, "B": 1, "I": 2, "L": 3, "U": 4}
IGNORED_LABEL_ID = -100


def _positive_token_runs(states: Sequence[str]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index, state in enumerate(states):
        if state == POSITIVE:
            current.append(index)
            continue
        if current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def labels_for_states(states: Sequence[str]) -> list[int]:
    """BILOU label ids for one category, driven only by its own token states.

    A contiguous POSITIVE run becomes ``U`` for a single token, otherwise
    ``B ... I ... L``. ``NEGATIVE_CONFIRMED`` tokens are ``O``. UNKNOWN tokens
    keep the ignored id so no loss is taken on them.
    """

    labels = [IGNORED_LABEL_ID] * len(states)
    for index, state in enumerate(states):
        if state == NEGATIVE_CONFIRMED:
            labels[index] = BILOU_TO_ID["O"]
    for run in _positive_token_runs(states):
        if len(run) == 1:
            labels[run[0]] = BILOU_TO_ID["U"]
            continue
        labels[run[0]] = BILOU_TO_ID["B"]
        labels[run[-1]] = BILOU_TO_ID["L"]
        for index in run[1:-1]:
            labels[index] = BILOU_TO_ID["I"]
    return labels


def loss_mask_for_labels(labels: Sequence[int]) -> list[int]:
    """Zero for ignored tokens, one for every token with known supervision."""

    return [0 if label == IGNORED_LABEL_ID else 1 for label in labels]


def category_targets(
    offsets: Sequence[tuple[int, int]],
    *,
    positive_spans: Sequence[tuple[int, int]] = (),
    negative_spans: Sequence[tuple[int, int]] = (),
) -> tuple[list[int], list[int]]:
    states = token_states(
        offsets, positive_spans=positive_spans, negative_spans=negative_spans
    )
    labels = labels_for_states(states)
    return labels, loss_mask_for_labels(labels)


def multi_category_targets(
    offsets: Sequence[tuple[int, int]],
    *,
    categories: Sequence[str],
    category_spans: dict[str, Sequence[tuple[int, int]]] | None = None,
    negative_spans: dict[str, Sequence[tuple[int, int]]] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Build independent label and mask vectors for every category.

    Categories never share state: a token may be ``U`` for sponsor and
    ``IGNORED`` for interaction in the same window, which is what makes the
    per-category loss mask necessary.
    """

    positives = category_spans or {}
    negatives = negative_spans or {}
    targets: dict[str, dict[str, list[int]]] = {}
    for category in categories:
        labels, mask = category_targets(
            offsets,
            positive_spans=positives.get(category, ()),
            negative_spans=negatives.get(category, ()),
        )
        targets[category] = {"labels": labels, "loss_mask": mask}
    return targets
