from __future__ import annotations

import unittest

from sponsor_detection.model.smart_segment_targets import (
    BILOU_TO_ID,
    IGNORED_LABEL_ID,
    category_targets,
    labels_for_states,
    loss_mask_for_labels,
    multi_category_targets,
)
from sponsor_detection.data.smart_segment_supervision import (
    NEGATIVE_CONFIRMED,
    POSITIVE,
    UNKNOWN,
)


class LabelsForStatesTest(unittest.TestCase):
    def test_single_token_run_is_u(self) -> None:
        labels = labels_for_states([UNKNOWN, POSITIVE, UNKNOWN])

        self.assertEqual(
            labels, [IGNORED_LABEL_ID, BILOU_TO_ID["U"], IGNORED_LABEL_ID]
        )

    def test_multi_token_run_is_bil(self) -> None:
        labels = labels_for_states([POSITIVE, POSITIVE, POSITIVE, UNKNOWN])

        self.assertEqual(
            labels, [BILOU_TO_ID["B"], BILOU_TO_ID["I"], BILOU_TO_ID["L"], IGNORED_LABEL_ID]
        )

    def test_confirmed_negative_is_o_and_masked_in(self) -> None:
        labels = labels_for_states([NEGATIVE_CONFIRMED, UNKNOWN])

        self.assertEqual(labels, [BILOU_TO_ID["O"], IGNORED_LABEL_ID])
        self.assertEqual(loss_mask_for_labels(labels), [1, 0])

    def test_separate_positive_runs_remain_separate(self) -> None:
        labels = labels_for_states([POSITIVE, UNKNOWN, POSITIVE])

        self.assertEqual(
            labels,
            [BILOU_TO_ID["U"], IGNORED_LABEL_ID, BILOU_TO_ID["U"]],
        )

    def test_unknown_tokens_are_never_supervised(self) -> None:
        labels = labels_for_states([UNKNOWN, UNKNOWN])

        self.assertEqual(labels, [IGNORED_LABEL_ID, IGNORED_LABEL_ID])
        self.assertEqual(loss_mask_for_labels(labels), [0, 0])


class CategoryTargetsTest(unittest.TestCase):
    def test_char_spans_map_to_bilou_and_masks(self) -> None:
        offsets = [(0, 0), (0, 3), (3, 6), (6, 9), (9, 12), (0, 0)]

        labels, mask = category_targets(offsets, positive_spans=[(3, 12)])

        self.assertEqual(
            labels,
            [
                IGNORED_LABEL_ID,
                IGNORED_LABEL_ID,
                BILOU_TO_ID["B"],
                BILOU_TO_ID["I"],
                BILOU_TO_ID["L"],
                IGNORED_LABEL_ID,
            ],
        )
        self.assertEqual(mask, [0, 0, 1, 1, 1, 0])

    def test_single_token_span_is_u(self) -> None:
        offsets = [(0, 0), (0, 3), (3, 6), (0, 0)]

        labels, _ = category_targets(offsets, positive_spans=[(3, 6)])

        self.assertEqual(
            labels,
            [IGNORED_LABEL_ID, IGNORED_LABEL_ID, BILOU_TO_ID["U"], IGNORED_LABEL_ID],
        )

    def test_special_tokens_stay_ignored(self) -> None:
        offsets = [(0, 0), (0, 5), (0, 0)]

        labels, mask = category_targets(offsets, positive_spans=[(0, 5)])

        self.assertEqual(labels[0], IGNORED_LABEL_ID)
        self.assertEqual(labels[-1], IGNORED_LABEL_ID)
        self.assertEqual(mask, [0, 1, 0])


class MultiCategoryTargetsTest(unittest.TestCase):
    def test_categories_are_independent_in_the_same_window(self) -> None:
        offsets = [(0, 3), (3, 6), (6, 9)]

        targets = multi_category_targets(
            offsets,
            categories=["sponsor", "interaction"],
            category_spans={"sponsor": [(0, 6)]},
            negative_spans={"interaction": [(6, 9)]},
        )

        self.assertEqual(
            targets["sponsor"]["labels"],
            [BILOU_TO_ID["B"], BILOU_TO_ID["L"], IGNORED_LABEL_ID],
        )
        self.assertEqual(targets["sponsor"]["loss_mask"], [1, 1, 0])
        self.assertEqual(
            targets["interaction"]["labels"],
            [IGNORED_LABEL_ID, IGNORED_LABEL_ID, BILOU_TO_ID["O"]],
        )
        self.assertEqual(targets["interaction"]["loss_mask"], [0, 0, 1])

    def test_category_without_spans_is_fully_unknown(self) -> None:
        targets = multi_category_targets(
            [(0, 3), (3, 6)],
            categories=["selfpromo"],
        )

        self.assertEqual(
            targets["selfpromo"]["labels"], [IGNORED_LABEL_ID, IGNORED_LABEL_ID]
        )
        self.assertEqual(targets["selfpromo"]["loss_mask"], [0, 0])


if __name__ == "__main__":
    unittest.main()
