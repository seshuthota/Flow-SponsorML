from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from sponsor_detection.model.smart_segment_targets import (
    BILOU_TO_ID,
    IGNORED_LABEL_ID,
    multi_category_targets,
)


LABELS = tuple(BILOU_TO_ID)
NUM_LABELS = len(LABELS)


def build_token_targets(
    offsets: Sequence[tuple[int, int]],
    category_spans: Sequence[dict[str, object]],
    categories: Sequence[str],
) -> tuple[list[list[int]], list[list[int]]]:
    """Per-token, per-category label ids and loss masks for one window.

    Returns ``labels`` and ``loss_mask`` shaped ``[sequence][category]`` so the
    tensor axes match the decoder's ``[batch, sequence, category, bilou]``.
    """

    spans_by_category: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in category_spans:
        spans_by_category[str(span["category"])].append(
            (int(span["start_char"]), int(span["end_char"]))
        )
    targets = multi_category_targets(
        offsets, categories=categories, category_spans=spans_by_category
    )
    labels = [
        [targets[category]["labels"][index] for category in categories]
        for index in range(len(offsets))
    ]
    loss_mask = [
        [targets[category]["loss_mask"][index] for category in categories]
        for index in range(len(offsets))
    ]
    return labels, loss_mask


def tokenize_multi_head(tokenizer, max_length: int, categories: Sequence[str]):
    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        batch_labels: list[list[list[int]]] = []
        batch_masks: list[list[list[int]]] = []
        for offsets, spans in zip(
            encoded["offset_mapping"], batch["category_spans"], strict=True
        ):
            labels, loss_mask = build_token_targets(offsets, spans, categories)
            batch_labels.append(labels)
            batch_masks.append(loss_mask)
        del encoded["offset_mapping"]
        encoded["labels"] = batch_labels
        encoded["loss_mask"] = batch_masks
        return encoded

    return tokenize


def masked_cross_entropy(logits, labels, loss_mask):
    """Cross entropy over exactly the tokens and categories with known supervision."""

    import torch.nn.functional as functional

    active = loss_mask > 0
    if not bool(active.any()):
        return logits.sum() * 0.0
    return functional.cross_entropy(logits[active], labels[active])


def load_multi_head_classifier(
    *,
    categories: Sequence[str],
    encoder_name: str,
    revision: str,
    dropout: float,
    sponsor_checkpoint: Path | None = None,
    sponsor_category: str = "sponsor",
):
    """Build the shared encoder plus one independent BILOU head per category.

    When a sponsor checkpoint is supplied its classifier weights initialise the
    sponsor head, so sponsor quality starts from the trained model instead of
    random weights.
    """

    import torch
    from torch import nn
    from transformers import AutoModel, AutoModelForTokenClassification

    categories = list(categories)

    class _Classifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.config = encoder.config
            self.main_input_name = "input_ids"
            self.categories = categories
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(
                int(encoder.config.hidden_size), len(categories) * NUM_LABELS
            )

        def forward(self, input_ids, attention_mask=None, labels=None, loss_mask=None, **kwargs):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = self.dropout(outputs.last_hidden_state)
            logits = self.classifier(hidden)
            batch, sequence, _ = logits.shape
            logits = logits.view(batch, sequence, len(self.categories), NUM_LABELS)
            loss = None
            if labels is not None:
                loss = masked_cross_entropy(logits, labels, loss_mask)
            return {"loss": loss, "logits": logits}

        def save_pretrained(self, output_directory):
            from safetensors.torch import save_file

            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            save_file(
                {key: value.contiguous() for key, value in self.state_dict().items()},
                str(output_directory / "model.safetensors"),
            )
            (output_directory / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "smart_segment_multi_head",
                        "encoder": encoder_name,
                        "hidden_size": int(self.encoder.config.hidden_size),
                        "categories": self.categories,
                        "labels": list(LABELS),
                        "dropout": float(self.dropout.p),
                        "num_labels": len(self.categories) * NUM_LABELS,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    if sponsor_checkpoint is not None:
        base = AutoModelForTokenClassification.from_pretrained(str(sponsor_checkpoint))
        encoder = base.model
        sponsor_weight = base.classifier.weight.detach().clone()
        sponsor_bias = base.classifier.bias.detach().clone()
    else:
        encoder = AutoModel.from_pretrained(encoder_name, revision=revision)
        sponsor_weight = sponsor_bias = None

    model = _Classifier(encoder)
    if sponsor_weight is not None and sponsor_category in categories:
        offset = categories.index(sponsor_category) * NUM_LABELS
        with torch.no_grad():
            model.classifier.weight[offset : offset + NUM_LABELS].copy_(sponsor_weight)
            if sponsor_bias is not None:
                model.classifier.bias[offset : offset + NUM_LABELS].copy_(sponsor_bias)
    return model


class MultiHeadCollator:
    """Pad variable-length windows and their per-category label grids."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        batch = self.tokenizer.pad(
            {
                "input_ids": [feature["input_ids"] for feature in features],
                "attention_mask": [feature["attention_mask"] for feature in features],
            },
            padding=True,
            return_tensors="pt",
        )
        sequence_length = int(batch["input_ids"].shape[1])
        category_count = len(features[0]["labels"][0])
        labels = torch.full(
            (len(features), sequence_length, category_count),
            IGNORED_LABEL_ID,
            dtype=torch.long,
        )
        loss_mask = torch.zeros(
            (len(features), sequence_length, category_count), dtype=torch.long
        )
        for index, feature in enumerate(features):
            length = len(feature["labels"])
            labels[index, :length] = torch.tensor(feature["labels"], dtype=torch.long)
            loss_mask[index, :length] = torch.tensor(
                feature["loss_mask"], dtype=torch.long
            )
        batch["labels"] = labels
        batch["loss_mask"] = loss_mask
        return batch


def build_optimizer(model, *, encoder_learning_rate: float, head_learning_rate: float, weight_decay: float):
    """Separate learning rates: the pretrained encoder moves slowly, new heads faster."""

    import torch

    groups = [
        {
            "params": [p for p in model.encoder.parameters() if p.requires_grad],
            "lr": encoder_learning_rate,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in model.classifier.parameters() if p.requires_grad],
            "lr": head_learning_rate,
            "weight_decay": weight_decay,
        },
    ]
    return torch.optim.AdamW(groups)


def compute_metrics_factory(categories: Sequence[str]):
    """Per-category token P/R/F1, then a macro average across categories."""

    import numpy as np

    categories = list(categories)

    def compute_metrics(prediction) -> dict[str, float]:
        predictions = np.argmax(prediction.predictions, axis=-1)
        labels = prediction.label_ids
        metrics: dict[str, float] = {}
        f1_scores: list[float] = []
        for index, category in enumerate(categories):
            predicted_positive = predictions[:, :, index] != BILOU_TO_ID["O"]
            actual_positive = labels[:, :, index] != BILOU_TO_ID["O"]
            valid = labels[:, :, index] != IGNORED_LABEL_ID
            true_positive = int(np.sum(predicted_positive & actual_positive & valid))
            false_positive = int(np.sum(predicted_positive & ~actual_positive & valid))
            false_negative = int(np.sum(~predicted_positive & actual_positive & valid))
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
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            metrics[f"{category}_token_precision"] = precision
            metrics[f"{category}_token_recall"] = recall
            metrics[f"{category}_token_f1"] = f1
            f1_scores.append(f1)
        metrics["macro_token_f1"] = (
            sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        )
        return metrics

    return compute_metrics
