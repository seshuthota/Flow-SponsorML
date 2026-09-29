from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

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
from sponsor_detection.inference.windowing import TranscriptCue, assemble_transcript


BENCHMARK_VERSION = "smart-segment-benchmark/1"
CONTENT_NEGATIVE_STRATUM = "content_negative"
REVIEW_STATUS_PENDING = "pending"
REVIEW_STATUS_REVIEWED = "reviewed"


@dataclass(frozen=True, slots=True)
class BenchmarkCandidate:
    video_id: str
    channel_id: str
    stratum: str
    duration_ms: int
    language: str


def _seed_order(candidate: BenchmarkCandidate, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{candidate.video_id}".encode("utf-8")).hexdigest()


def select_candidates(
    candidates: Sequence[BenchmarkCandidate],
    *,
    targets: dict[str, int],
    seed: str,
    excluded_channels: set[str],
) -> list[BenchmarkCandidate]:
    """Pick channel-disjoint candidates, deterministically, towards per-stratum targets.

    At most one video per channel is selected, and channels already used by the
    frozen training splits are skipped, so the benchmark cannot leak into the
    data a candidate model was trained on.
    """

    selected: list[BenchmarkCandidate] = []
    counts: Counter[str] = Counter()
    used_channels: set[str] = set()
    for candidate in sorted(
        candidates, key=lambda item: _seed_order(item, seed)
    ):
        if candidate.channel_id in excluded_channels:
            continue
        if candidate.channel_id in used_channels:
            continue
        target = targets.get(candidate.stratum, 0)
        if counts[candidate.stratum] >= target:
            continue
        selected.append(candidate)
        counts[candidate.stratum] += 1
        used_channels.add(candidate.channel_id)
    return selected


def _stratum_for(present: set[str], categories: Sequence[str]) -> str:
    if present:
        return min(present, key=lambda category: categories.index(category))
    return CONTENT_NEGATIVE_STRATUM


def _load_excluded_channels(split_directories: Sequence[Path]) -> set[str]:
    import pyarrow.parquet as parquet

    channels: set[str] = set()
    for directory in split_directories:
        if not directory.is_dir():
            continue
        for split in ("train", "validation", "test"):
            path = directory / f"{split}.parquet"
            if not path.is_file():
                continue
            file = parquet.ParquetFile(path)
            for batch in file.iter_batches(columns=["channel_id"], batch_size=131_072):
                channels.update(
                    str(value)
                    for value in batch.column("channel_id").to_pylist()
                    if value
                )
    return channels


def _transcript_payload(transcript) -> dict[str, object]:
    cues = [
        {
            "start_ms": cue_range.start_ms,
            "end_ms": cue_range.end_ms,
            "text": transcript.text[cue_range.start_char : cue_range.end_char],
        }
        for cue_range in transcript.cue_ranges
    ]
    return {
        "text": transcript.text,
        "cues": cues,
        "sha256": hashlib.sha256(transcript.text.encode("utf-8")).hexdigest(),
    }


def build_benchmark_candidates(
    subtitles_directory: Path,
    metadata_directory: Path,
    annotations_path: Path,
    exclude_split_directories: Sequence[Path],
    output_path: Path,
    manifest_path: Path,
    *,
    categories: Sequence[str],
    targets: dict[str, int],
    seed: str,
    language_prefixes: Sequence[str] = ("en",),
    maximum_video_duration_seconds: int = 14400,
    minimum_cues: int = 25,
    reserved_directory: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Build a full-transcript review queue for the V1 categories.

    A reviewer sees the complete transcript rather than only the intervals the
    mirror already knows about, so content-heavy negatives and missed spans are
    discoverable.

    When ``reserved_directory`` is set, the selected videos and channels are
    written as a split-shaped exclusion the training builder consumes. The
    benchmark must be reserved **before** the training splits are built;
    otherwise the two draw from the same finite channel pool and no disjoint
    benchmark can exist.
    """

    started = time.monotonic()
    if progress_callback:
        progress_callback("loading metadata and annotations")
    metadata = _load_metadata(metadata_directory)
    annotations = load_canonical_annotations(
        annotations_path, categories=list(categories), eligible_only=True
    )
    excluded_channels = _load_excluded_channels(exclude_split_directories)

    if progress_callback:
        progress_callback("scanning transcripts")
    candidates: list[BenchmarkCandidate] = []
    for row in _iter_subtitle_rows(subtitles_directory):
        video_id = str(row["video_id"])
        if video_id not in metadata:
            continue
        if not _is_english(row["language"], language_prefixes):
            continue
        meta = metadata[video_id]
        if meta.get("availability") not in (None, "public"):
            continue
        if meta.get("live_status") == "was_live":
            continue
        duration = meta.get("duration")
        if not isinstance(duration, int) or duration > maximum_video_duration_seconds:
            continue
        channel_id = meta.get("channel_id")
        if not channel_id:
            continue
        if len(_parse_cues(str(row["segments_json"]))) < minimum_cues:
            continue
        present = {record.category for record in annotations.get(video_id, [])}
        candidates.append(
            BenchmarkCandidate(
                video_id=video_id,
                channel_id=str(channel_id),
                stratum=_stratum_for(present, categories),
                duration_ms=duration * 1000,
                language=str(row["language"]),
            )
        )

    selected = select_candidates(
        candidates, targets=targets, seed=seed, excluded_channels=excluded_channels
    )
    selected_ids = {candidate.video_id for candidate in selected}

    # Materialize transcripts for the selected videos only, so the review queue
    # does not hold the whole corpus in memory.
    if progress_callback:
        progress_callback("materializing selected transcripts")
    transcripts: dict[str, dict[str, object]] = {}
    for row in _iter_subtitle_rows(subtitles_directory):
        video_id = str(row["video_id"])
        if video_id not in selected_ids:
            continue
        reconstructed = reconstruct_caption_lines(
            _parse_cues(str(row["segments_json"]))
        )
        transcript = assemble_transcript(
            [
                TranscriptCue(index=index, start_ms=start, end_ms=end, text=text)
                for index, (start, end, text) in enumerate(reconstructed)
            ]
        )
        if not transcript.text.strip():
            continue
        transcripts[video_id] = _transcript_payload(transcript)
    selected = [
        candidate for candidate in selected if candidate.video_id in transcripts
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for candidate in selected:
            sink.write(
                json.dumps(
                    {
                        "video_id": candidate.video_id,
                        "channel_id": candidate.channel_id,
                        "stratum": candidate.stratum,
                        "duration_ms": candidate.duration_ms,
                        "language": candidate.language,
                        "categories": list(categories),
                        "transcript": transcripts[candidate.video_id],
                        "annotation": {
                            "status": REVIEW_STATUS_PENDING,
                            "reviewer": None,
                            "content_confirmed": False,
                            "intervals": {category: [] for category in categories},
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    counts = Counter(candidate.stratum for candidate in selected)
    reserved: dict[str, object] = {"path": None, "sha256": None, "videos": 0}
    if reserved_directory is not None:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        reserved_directory.mkdir(parents=True, exist_ok=True)
        reserved_path = reserved_directory / "train.parquet"
        parquet.write_table(
            pa.table(
                {
                    "video_id": [candidate.video_id for candidate in selected],
                    "channel_id": [candidate.channel_id for candidate in selected],
                }
            ),
            reserved_path,
            compression="zstd",
        )
        reserved = {
            "path": str(reserved_path),
            "sha256": sha256_file(reserved_path),
            "videos": len(selected),
        }

    manifest = {
        "version": BENCHMARK_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": "annotation_pending",
        "configuration": {
            "categories": list(categories),
            "targets": targets,
            "seed": seed,
            "language_prefixes": list(language_prefixes),
            "maximum_video_duration_seconds": maximum_video_duration_seconds,
            "minimum_cues": minimum_cues,
        },
        "sources": {
            "subtitles_directory": str(subtitles_directory),
            "metadata_directory": str(metadata_directory),
            "annotations_path": str(annotations_path),
            "annotations_sha256": sha256_file(annotations_path),
        },
        "counts": {
            "candidates": len(candidates),
            "selected": len(selected),
            "excluded_channels": len(excluded_channels),
            "by_stratum": dict(sorted(counts.items())),
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "reserved": reserved,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _validated_intervals(
    record: dict[str, object], category: str
) -> list[dict[str, int]]:
    annotation = record.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError(f"video {record.get('video_id')} has no annotation block")
    intervals = annotation.get("intervals")
    if not isinstance(intervals, dict):
        raise ValueError(f"video {record.get('video_id')} has no interval map")
    category_intervals = intervals.get(category, [])
    if not isinstance(category_intervals, list):
        raise ValueError(
            f"video {record.get('video_id')} category {category} intervals must be a list"
        )
    validated: list[dict[str, int]] = []
    for interval in category_intervals:
        start = int(interval["start_ms"])
        end = int(interval["end_ms"])
        if start < 0 or end <= start:
            raise ValueError(
                f"video {record.get('video_id')} has an invalid {category} interval"
            )
        validated.append({"start_ms": start, "end_ms": end})
    return validated


def freeze_benchmark(
    review_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    categories: Sequence[str],
) -> dict[str, object]:
    """Freeze a reviewed benchmark and derive confirmed negatives from it.

    A fully reviewed video that contains no interval for any configured
    category becomes a confirmed negative for every one of them, with the
    reviewer recorded as the evidence source.
    """

    records: list[dict[str, object]] = []
    with review_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            annotation = record.get("annotation")
            if not isinstance(annotation, dict) or annotation.get("status") != REVIEW_STATUS_REVIEWED:
                raise ValueError(
                    f"review line {line_number} is not marked {REVIEW_STATUS_REVIEWED}"
                )
            reviewer = annotation.get("reviewer")
            if not reviewer:
                raise ValueError(f"review line {line_number} has no reviewer")
            records.append(record)

    frozen: list[dict[str, object]] = []
    confirmed_negatives = 0
    positive_by_category: Counter[str] = Counter()
    for record in records:
        categories_intervals = {
            category: _validated_intervals(record, category)
            for category in categories
        }
        has_any = any(categories_intervals.values())
        content_confirmed = bool(record["annotation"].get("content_confirmed"))
        if not has_any and not content_confirmed:
            raise ValueError(
                f"video {record['video_id']} has no intervals and is not confirmed content"
            )
        for category, intervals in categories_intervals.items():
            if intervals:
                positive_by_category[category] += 1
        if not has_any:
            confirmed_negatives += 1
        frozen.append(
            {
                "video_id": record["video_id"],
                "channel_id": record["channel_id"],
                "stratum": record["stratum"],
                "reviewer": str(record["annotation"]["reviewer"]),
                "categories": categories_intervals,
                "confirmed_negative": not has_any,
                "transcript_sha256": record["transcript"]["sha256"],
            }
        )

    transcript_digest = hashlib.sha256()
    for record in sorted(frozen, key=lambda item: str(item["video_id"])):
        transcript_digest.update(
            f"{record['video_id']}\0{record['transcript_sha256']}\n".encode("utf-8")
        )

    with output_path.open("w", encoding="utf-8") as sink:
        for record in sorted(frozen, key=lambda item: str(item["video_id"])):
            sink.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "version": BENCHMARK_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "status": "frozen",
        "categories": list(categories),
        "source_review_sha256": sha256_file(review_path),
        "transcript_set_sha256": transcript_digest.hexdigest(),
        "counts": {
            "videos": len(frozen),
            "confirmed_negative_videos": confirmed_negatives,
            "positive_videos_by_category": dict(sorted(positive_by_category.items())),
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
