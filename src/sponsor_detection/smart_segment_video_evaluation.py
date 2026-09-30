from __future__ import annotations

import json
import time
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sponsor_detection.data.profile import sha256_file, write_json_atomic
from sponsor_detection.data.scriptsmith_dataset import (
    _is_english,
    _iter_subtitle_rows,
    _load_metadata,
    _parse_cues,
    reconstruct_caption_lines,
)
from sponsor_detection.data.smart_segment_annotations import (
    load_canonical_annotations,
)
from sponsor_detection.inference.windowing import (
    TranscriptCue,
    assemble_transcript,
    build_transcript_windows,
)
from sponsor_detection.model.smart_segment_decoder import decode_multi_head
from sponsor_detection.smart_segment_evaluation import evaluate_categories


def _softmax(values):
    import numpy as np

    shifted = values - values.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _stitch(spans: list[tuple[int, int, float]], merge_gap_ms: int) -> list[list[int]]:
    if not spans:
        return []
    spans.sort(key=lambda span: (span[0], span[1]))
    merged: list[list[int]] = []
    for start_ms, end_ms, _confidence in spans:
        if merged and start_ms <= merged[-1][1] + merge_gap_ms:
            merged[-1][1] = max(merged[-1][1], end_ms)
            continue
        merged.append([start_ms, end_ms])
    return merged


def _window_char_span(offset_mapping, start_index: int, end_index: int):
    starts: list[int] = []
    ends: list[int] = []
    for index in range(start_index, end_index):
        if not 0 <= index < len(offset_mapping):
            continue
        offset_start, offset_end = offset_mapping[index]
        if offset_end > offset_start:
            starts.append(int(offset_start))
            ends.append(int(offset_end))
    if not starts:
        return None
    return min(starts), max(ends)


def predict_video(
    transcript,
    *,
    tokenizer,
    session,
    categories: list[str],
    thresholds: dict[str, float],
    max_length: int,
    overlap_tokens: int,
    batch_size: int,
    merge_gap_ms: int,
) -> dict[str, list[list[int]]]:
    import numpy as np

    windows = build_transcript_windows(
        transcript, tokenizer, max_length=max_length, overlap_tokens=overlap_tokens
    )
    predicted: dict[str, list[tuple[int, int, float]]] = {
        category: [] for category in categories
    }
    for offset in range(0, len(windows), batch_size):
        batch = windows[offset : offset + batch_size]
        width = max(len(window.input_ids) for window in batch)
        input_ids = np.zeros((len(batch), width), dtype=np.int64)
        attention = np.zeros((len(batch), width), dtype=np.int64)
        for row, window in enumerate(batch):
            length = len(window.input_ids)
            input_ids[row, :length] = window.input_ids
            attention[row, :length] = window.attention_mask
        logits = session.run(
            ["segment_logits"],
            {"input_ids": input_ids, "attention_mask": attention},
        )[0]
        probabilities = _softmax(logits)
        for row, window in enumerate(batch):
            spans = decode_multi_head(
                probabilities[row][: len(window.input_ids)],
                categories=categories,
                thresholds=thresholds,
                default_threshold=0.0,
            )
            for span in spans:
                char_span = _window_char_span(
                    window.offset_mapping, span.start_index, span.end_index
                )
                if char_span is None:
                    continue
                start_ms, end_ms = transcript.timestamps_for_span(*char_span)
                if end_ms <= start_ms:
                    end_ms = start_ms + 1
                predicted[span.category].append((start_ms, end_ms, span.confidence))
    return {
        category: _stitch(values, merge_gap_ms)
        for category, values in predicted.items()
    }


def evaluate_videos_from_config(config_path: Path) -> dict[str, object]:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    with config_path.open("rb") as source:
        configuration = tomllib.load(source)
    settings = configuration["smart_segment_video_evaluation"]
    bundle = Path(settings["bundle_directory"])
    categories = [str(category) for category in settings["categories"]]
    thresholds = {
        str(name): float(value)
        for name, value in settings.get("thresholds", {}).items()
    }
    default_threshold = float(settings.get("default_threshold", 0.5))
    for category in categories:
        thresholds.setdefault(category, default_threshold)
    max_length = int(settings.get("max_length", 1024))
    overlap_tokens = int(settings.get("overlap_tokens", 128))
    batch_size = int(settings.get("batch_size", 8))
    merge_gap_ms = int(settings.get("merge_gap_ms", 1500))

    review_path = Path(settings["review_path"])
    video_ids = {
        json.loads(line)["video_id"]
        for line in review_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    metadata = _load_metadata(Path(settings["metadata_directory"]))
    annotations = load_canonical_annotations(
        Path(settings["annotations_path"]),
        video_ids=video_ids,
        categories=categories,
        eligible_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(bundle), use_fast=True)
    session = ort.InferenceSession(
        str(bundle / str(settings["model_file"])), providers=["CPUExecutionProvider"]
    )

    predicted_ms: dict[str, dict[str, list]] = defaultdict(dict)
    expected_ms: dict[str, dict[str, list]] = defaultdict(dict)
    durations_ms: dict[str, int] = {}
    started = time.monotonic()
    processed = 0
    for row in _iter_subtitle_rows(Path(settings["subtitles_directory"])):
        video_id = str(row["video_id"])
        if video_id not in video_ids:
            continue
        if not _is_english(row["language"], settings.get("language_prefixes", ["en"])):
            continue
        meta = metadata.get(video_id, {})
        cues = reconstruct_caption_lines(_parse_cues(str(row["segments_json"])))
        if not cues:
            continue
        transcript = assemble_transcript(
            [
                TranscriptCue(index=i, start_ms=s, end_ms=e, text=t)
                for i, (s, e, t) in enumerate(cues)
            ]
        )
        if not transcript.text.strip():
            continue
        result = predict_video(
            transcript,
            tokenizer=tokenizer,
            session=session,
            categories=categories,
            thresholds=thresholds,
            max_length=max_length,
            overlap_tokens=overlap_tokens,
            batch_size=batch_size,
            merge_gap_ms=merge_gap_ms,
        )
        for category in categories:
            predicted_ms[category][video_id] = result.get(category, [])
            expected_ms[category][video_id] = [
                [record.start_ms, record.end_ms]
                for record in annotations.get(video_id, [])
                if record.category == category
            ]
        duration = meta.get("duration")
        if isinstance(duration, int):
            durations_ms[video_id] = duration * 1000
        processed += 1

    elapsed = time.monotonic() - started
    evaluation = evaluate_categories(
        {category: dict(predicted_ms[category]) for category in categories},
        {category: dict(expected_ms[category]) for category in categories},
        iou_thresholds=(0.3, 0.5, 0.7),
        durations_ms=durations_ms,
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "bundle_directory": str(bundle),
        "model_file": str(settings["model_file"]),
        "model_sha256": sha256_file(bundle / str(settings["model_file"])),
        "categories": categories,
        "thresholds": thresholds,
        "evaluation": evaluation,
        "dataset": {
            "videos_requested": len(video_ids),
            "videos_processed": processed,
            "review_sha256": sha256_file(review_path),
            "annotations_sha256": sha256_file(Path(settings["annotations_path"])),
            "inference_seconds": round(elapsed, 3),
            "videos_per_second": round(processed / elapsed, 4) if elapsed else None,
        },
        "limitations": [
            "Reference labels are SponsorBlock weak labels, so this is not the reviewed benchmark and cannot establish precision for release gates.",
            "Thresholds are uncalibrated defaults because no reviewed benchmark exists yet.",
            "The videos are channel-disjoint from training, so the result is a genuine held-out measurement against weak labels.",
        ],
    }
    write_json_atomic(Path(settings["output_path"]), report)
    return report
