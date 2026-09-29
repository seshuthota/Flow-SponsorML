from __future__ import annotations

import unittest

from sponsor_detection.model.multi_head_model import (
    LABELS,
    NUM_LABELS,
    build_token_targets,
    compute_metrics_factory,
)
from sponsor_detection.model.smart_segment_targets import BILOU_TO_ID, IGNORED_LABEL_ID


CATEGORIES = ["sponsor", "selfpromo", "interaction"]


class BuildTokenTargetsTest(unittest.TestCase):
    def test_label_and_mask_shape_is_sequence_by_category(self) -> None:
        offsets = [(0, 3), (3, 6), (6, 9)]

        labels, loss_mask = build_token_targets(
            offsets,
            [{"category": "sponsor", "start_char": 0, "end_char": 6}],
            CATEGORIES,
        )

        self.assertEqual(len(labels), 3)
        self.assertEqual([len(row) for row in labels], [3, 3, 3])
        self.assertEqual(labels[0], [BILOU_TO_ID["B"], IGNORED_LABEL_ID, IGNORED_LABEL_ID])
        self.assertEqual(labels[1], [BILOU_TO_ID["L"], IGNORED_LABEL_ID, IGNORED_LABEL_ID])
        self.assertEqual(loss_mask[0], [1, 0, 0])

    def test_categories_stay_independent(self) -> None:
        offsets = [(0, 3), (3, 6)]

        labels, loss_mask = build_token_targets(
            offsets,
            [
                {"category": "selfpromo", "start_char": 0, "end_char": 3},
                {"category": "interaction", "start_char": 3, "end_char": 6},
            ],
            CATEGORIES,
        )

        self.assertEqual(labels[0][0], IGNORED_LABEL_ID)
        self.assertEqual(labels[0][1], BILOU_TO_ID["U"])
        self.assertEqual(labels[1][2], BILOU_TO_ID["U"])
        self.assertEqual(loss_mask[0], [0, 1, 0])
        self.assertEqual(loss_mask[1], [0, 0, 1])

    def test_label_axis_matches_the_decoder(self) -> None:
        self.assertEqual(LABELS, ("O", "B", "I", "L", "U"))
        self.assertEqual(NUM_LABELS, 5)


class ComputeMetricsTest(unittest.TestCase):
    def test_reports_positive_token_recall_only(self) -> None:
        import numpy as np

        # [batch, sequence, category, label]
        predictions = np.zeros((1, 2, 3, 5), dtype=int)
        labels = np.full((1, 2, 3), IGNORED_LABEL_ID, dtype=int)
        labels[0, 0, 0] = BILOU_TO_ID["U"]
        labels[0, 1, 0] = BILOU_TO_ID["O"]
        predictions[0, 0, 0, BILOU_TO_ID["U"]] = 1
        predictions[0, 1, 0, BILOU_TO_ID["O"]] = 1

        metrics = compute_metrics_factory(CATEGORIES)(
            type("Prediction", (), {"predictions": predictions, "label_ids": labels})
        )

        self.assertEqual(metrics["sponsor_positive_token_recall"], 1.0)
        self.assertEqual(metrics["sponsor_positive_token_count"], 1)
        self.assertEqual(metrics["selfpromo_positive_token_count"], 0)
        self.assertEqual(metrics["selfpromo_positive_token_recall"], 0.0)
        self.assertNotIn("sponsor_token_precision", metrics)

    def test_recall_drops_when_a_positive_token_is_missed(self) -> None:
        import numpy as np

        predictions = np.zeros((1, 1, 3, 5), dtype=int)
        predictions[0, 0, 0, BILOU_TO_ID["O"]] = 1
        labels = np.full((1, 1, 3), IGNORED_LABEL_ID, dtype=int)
        labels[0, 0, 0] = BILOU_TO_ID["U"]

        metrics = compute_metrics_factory(CATEGORIES)(
            type("Prediction", (), {"predictions": predictions, "label_ids": labels})
        )

        self.assertEqual(metrics["sponsor_positive_token_recall"], 0.0)


class CollatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch is not installed in this environment")

    def test_pads_labels_with_ignored_and_mask_with_zero(self) -> None:
        from sponsor_detection.model.multi_head_model import MultiHeadCollator

        class Tokenizer:
            def pad(self, batch, padding, return_tensors):
                import torch

                width = max(len(ids) for ids in batch["input_ids"])
                padded_ids = [ids + [0] * (width - len(ids)) for ids in batch["input_ids"]]
                padded_mask = [
                    mask + [0] * (width - len(mask)) for mask in batch["attention_mask"]
                ]
                return {
                    "input_ids": torch.tensor(padded_ids),
                    "attention_mask": torch.tensor(padded_mask),
                }

        collator = MultiHeadCollator(Tokenizer())
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [[0, 1], [1, 1], [3, 0]],
                "loss_mask": [[1, 1], [1, 0], [1, 0]],
            },
            {
                "input_ids": [1],
                "attention_mask": [1],
                "labels": [[0, 2]],
                "loss_mask": [[1, 1]],
            },
        ]

        batch = collator(features)

        self.assertEqual(tuple(batch["labels"].shape), (2, 3, 2))
        self.assertEqual(tuple(batch["loss_mask"].shape), (2, 3, 2))
        self.assertEqual(int(batch["labels"][1, 2, 0]), IGNORED_LABEL_ID)
        self.assertEqual(int(batch["loss_mask"][1, 2, 0]), 0)


if __name__ == "__main__":
    unittest.main()
