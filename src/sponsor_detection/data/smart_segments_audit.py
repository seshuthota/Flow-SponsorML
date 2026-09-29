from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from sponsor_detection.data.profile import (
    REQUIRED_COLUMNS,
    EligibilityPolicy,
    eligibility_rejection_flags,
    sha256_file,
    write_json_atomic,
)
from sponsor_detection.data.scriptsmith_dataset import (
    LabelSegment,
    _char_span_for_interval,
    _fit_window_to_max_tokens,
    _is_english,
    _iter_subtitle_rows,
    _load_metadata,
    _parse_cues,
    reconstruct_caption_lines,
)
from sponsor_detection.data.smart_segment_supervision import (
    ConfirmedNegative,
    load_confirmed_negatives,
)
from sponsor_detection.inference.windowing import (
    TranscriptCue,
    assemble_transcript,
    build_transcript_windows,
)


# Version identifiers are part of the audit contract: a count is only
# reproducible together with the implementation version that produced it.
SMART_SEGMENTS_AUDIT_VERSION = "smart-segments-audit/1"
ELIGIBILITY_POLICY_VERSION = "eligibility-policy/1"
NORMALIZATION_VERSION = "normalize_cue_text/1"
WINDOW_BUILDER_VERSION = "build_transcript_windows/1"
ALIGNMENT_ALGORITHM_VERSION = "cue-overlap-char-span/1"
TRANSCRIPT_RECONSTRUCTION_VERSION = "reconstruct_caption_lines/1"

# The plan names categories in upper case; the SponsorBlock mirror uses the
# lower-case identifiers below. The audit reports mirror identifiers so its
# counts can be reproduced directly against the snapshot.
PLAN_CATEGORY_TO_MIRROR: dict[str, str] = {
    "SPONSOR": "sponsor",
    "SELF_PROMO": "selfpromo",
    "INTERACTION": "interaction",
    "PREVIEW_RECAP": "preview",
    "HOOK": "hook",
    "INTRO": "intro",
    "OUTRO": "outro",
    "MUSIC_OFFTOPIC": "music_offtopic",
    "FILLER": "filler",
}
DEFAULT_CATEGORIES: tuple[str, ...] = tuple(PLAN_CATEGORY_TO_MIRROR.values())


@dataclass(frozen=True)
class _CategoryStats:
    videos: set[str]
    channels: set[str]
    segment_ids: set[str]
    segment_seconds: list[float]


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def _summarize(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "median": round(statistics.median(ordered), 4),
        "p95": round(_percentile(ordered, 0.95), 4),
        "max": round(ordered[-1], 4),
    }


def _hash_directory(directory: Path, pattern: str) -> dict[str, dict[str, object]]:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob(pattern)):
        files[path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return files


def _load_subtitle_languages(directory: Path) -> dict[str, str | None]:
    import pyarrow.parquet as parquet

    languages: dict[str, str | None] = {}
    for path in sorted(directory.glob("*.parquet")):
        file = parquet.ParquetFile(path)
        for batch in file.iter_batches(
            columns=["video_id", "language"], batch_size=65_536
        ):
            values = batch.to_pydict()
            for video_id, language in zip(
                values["video_id"], values["language"], strict=True
            ):
                if video_id is None:
                    continue
                key = str(video_id)
                if key not in languages:
                    languages[key] = str(language) if language else None
    return languages


def _load_xenova_video_ids(directory: Path) -> set[str]:
    """Read the video identifiers the Xenova snapshot carries text for.

    The snapshot's ``segments.json`` is the compact index; the per-split JSONL
    files are the fallback when the index is absent.
    """

    index_path = directory / "segments.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(key) for key in payload}
    video_ids: set[str] = set()
    for name in ("train.json", "valid.json", "test.json"):
        path = directory / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                video_id = record.get("video_id")
                if video_id:
                    video_ids.add(str(video_id))
    return video_ids


def _load_eligible_labels(
    mirror_path: Path,
    relevant_video_ids: set[str],
    *,
    categories: set[str],
    policy: EligibilityPolicy,
    progress_callback: Callable[[str], None] | None = None,
    progress_every_rows: int = 0,
) -> tuple[dict[str, list[LabelSegment]], dict[str, object]]:
    """Join the mirror against transcript-bearing videos under one policy.

    The shared eligibility rules are reused verbatim from
    :func:`eligibility_rejection_flags`; only the category filter is
    generalized from the sponsor-only pipeline.
    """

    labels: dict[str, list[LabelSegment]] = defaultdict(list)
    counters: Counter[str] = Counter()
    rejections: dict[str, Counter[str]] = defaultdict(Counter)
    category_totals: Counter[str] = Counter()
    with mirror_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"missing required columns: {', '.join(missing_columns)}")
        for row in reader:
            counters["rows"] += 1
            category_totals[row.get("category") or ""] += 1
            video_id = row.get("videoID")
            if not video_id or video_id not in relevant_video_ids:
                counters["outside_transcript_set"] += 1
                continue
            category = row.get("category") or ""
            if category not in categories:
                counters["category_not_audited"] += 1
                continue
            counters["audited_rows"] += 1
            try:
                start_time = float(row["startTime"])
                end_time = float(row["endTime"])
                video_duration = float(row["videoDuration"] or 0.0)
                votes = int(row["votes"])
                hidden = int(row["hidden"])
                shadow_hidden = int(row["shadowHidden"])
            except (TypeError, ValueError):
                counters["malformed_rows"] += 1
                continue
            flags = eligibility_rejection_flags(
                policy,
                service=row.get("service") or "",
                action_type=row.get("actionType") or "",
                start_time=start_time,
                end_time=end_time,
                votes=votes,
                hidden=hidden,
                shadow_hidden=shadow_hidden,
                video_duration=video_duration,
            )
            if flags:
                counters["ineligible_rows"] += 1
                rejections[category].update(flags)
                continue
            counters["eligible_rows"] += 1
            labels[video_id].append(
                LabelSegment(
                    segment_id=str(row.get("UUID") or ""),
                    start_ms=round(start_time * 1000),
                    end_ms=round(end_time * 1000),
                    category=category,
                )
            )
            if progress_every_rows and counters["rows"] % progress_every_rows == 0:
                if progress_callback:
                    progress_callback(
                        f"scanned {counters['rows']:,} mirror rows; "
                        f"{counters['eligible_rows']:,} eligible"
                    )
    for segments in labels.values():
        segments.sort(key=lambda segment: (segment.start_ms, segment.end_ms))
    report = {
        "counters": dict(counters),
        "category_totals": dict(category_totals.most_common()),
        "ineligible_by_category": {
            category: dict(counter) for category, counter in sorted(rejections.items())
        },
    }
    return labels, report


def _channel_of(metadata: dict[str, dict[str, object]], video_id: str) -> str | None:
    channel_id = metadata.get(video_id, {}).get("channel_id")
    return str(channel_id) if channel_id else None


def build_raw_data_audit(
    mirror_path: Path,
    subtitles_directory: Path,
    metadata_directory: Path,
    output_path: Path,
    *,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    language_prefixes: Sequence[str] = ("en",),
    eligibility: EligibilityPolicy | None = None,
    xenova_directory: Path | None = None,
    scriptsmith_revision: str = "scriptsmith-sponsorblock-2024",
    maximum_videos: int = 0,
    compute_sha256: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Audit A: raw multicategory data availability, independent of any model.

    Counts the intersection between transcript-bearing dataset snapshots and
    eligible SponsorBlock annotations. Window or token counts deliberately do
    not appear here; they belong to :func:`build_trainability_audit` because
    they depend on a pinned preprocessing configuration.
    """

    started = time.monotonic()
    policy = eligibility or EligibilityPolicy()
    audited_categories = tuple(categories)

    if progress_callback:
        progress_callback("hashing source snapshots")
    source_hashes: dict[str, object] = {
        "mirror": {
            "path": str(mirror_path),
            "bytes": mirror_path.stat().st_size,
            "sha256": sha256_file(mirror_path) if compute_sha256 else None,
            "mirror_last_update": (
                (mirror_path.parent / "lastUpdate.txt").read_text(encoding="utf-8").strip()
                if (mirror_path.parent / "lastUpdate.txt").is_file()
                else None
            ),
        },
        "subtitles": _hash_directory(subtitles_directory, "*.parquet")
        if compute_sha256
        else {},
        "metadata": _hash_directory(metadata_directory, "*.parquet")
        if compute_sha256
        else {},
    }

    if progress_callback:
        progress_callback("loading transcript inventory")
    subtitle_languages = _load_subtitle_languages(subtitles_directory)
    scriptsmith_videos = set(subtitle_languages)
    english_scriptsmith = {
        video_id
        for video_id, language in subtitle_languages.items()
        if _is_english(language, language_prefixes)
    }
    xenova_videos: set[str] = set()
    if xenova_directory is not None and xenova_directory.is_dir():
        source_hashes["xenova"] = {
            "path": str(xenova_directory),
            "sha256": (
                sha256_file(xenova_directory / "segments.json")
                if compute_sha256 and (xenova_directory / "segments.json").is_file()
                else None
            ),
        }
        xenova_videos = _load_xenova_video_ids(xenova_directory)

    transcript_videos = scriptsmith_videos | xenova_videos
    if maximum_videos:
        transcript_videos = set(sorted(transcript_videos)[:maximum_videos])

    if progress_callback:
        progress_callback("loading video metadata")
    metadata = _load_metadata(metadata_directory)

    if progress_callback:
        progress_callback(
            f"joining {len(transcript_videos):,} transcript videos against the mirror"
        )
    labels, mirror_report = _load_eligible_labels(
        mirror_path,
        transcript_videos,
        categories=set(audited_categories),
        policy=policy,
        progress_callback=progress_callback,
        progress_every_rows=2_000_000,
    )

    stats = {
        category: _CategoryStats(set(), set(), set(), [])
        for category in audited_categories
    }
    for video_id, segments in labels.items():
        channel_id = _channel_of(metadata, video_id)
        for segment in segments:
            entry = stats[segment.category]
            entry.videos.add(video_id)
            if channel_id:
                entry.channels.add(channel_id)
            entry.segment_ids.add(segment.segment_id)
            entry.segment_seconds.append(
                max(0.0, (segment.end_ms - segment.start_ms) / 1000.0)
            )

    table: dict[str, dict[str, object]] = {}
    for category in audited_categories:
        entry = stats.get(category, _CategoryStats(set(), set(), set(), []))
        per_video_counts = [
            sum(1 for segment in segments if segment.category == category)
            for segments in labels.values()
            if any(segment.category == category for segment in segments)
        ]
        table[category] = {
            "transcript_videos": len(entry.videos),
            "segments": len(entry.segment_ids),
            "positive_duration_seconds": round(sum(entry.segment_seconds), 3),
            "positive_duration_hours": round(sum(entry.segment_seconds) / 3600.0, 6),
            "unique_channels": len(entry.channels),
            "segment_seconds": _summarize(entry.segment_seconds),
            "segments_per_video": _summarize(
                [float(count) for count in per_video_counts]
            ),
        }

    overlapping_videos = 0
    overlap_pairs: Counter[str] = Counter()
    for video_id, segments in labels.items():
        by_category: dict[str, list[LabelSegment]] = defaultdict(list)
        for segment in segments:
            by_category[segment.category].append(segment)
        categories_in_video = sorted(by_category)
        if len(categories_in_video) > 1:
            overlapping_videos += 1
        for left_index, left in enumerate(categories_in_video):
            for right in categories_in_video[left_index + 1 :]:
                for left_segment in by_category[left]:
                    for right_segment in by_category[right]:
                        if min(left_segment.end_ms, right_segment.end_ms) > max(
                            left_segment.start_ms, right_segment.start_ms
                        ):
                            overlap_pairs[f"{left}+{right}"] += 1

    videos_with_segments = set(labels)
    channels_with_segments = {
        channel
        for video_id in videos_with_segments
        if (channel := _channel_of(metadata, video_id))
    }
    english_videos_with_segments = sum(
        1 for video_id in videos_with_segments if video_id in english_scriptsmith
    )
    metadata_videos = transcript_videos & set(metadata)
    channel_videos = {
        video_id for video_id in transcript_videos if _channel_of(metadata, video_id)
    }

    report = {
        "audit": "smart_segments_raw_data_availability",
        "schema_version": 1,
        "audit_code_version": SMART_SEGMENTS_AUDIT_VERSION,
        "transcript_reconstruction_version": TRANSCRIPT_RECONSTRUCTION_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "configuration": {
            "categories": list(audited_categories),
            "category_name_map": dict(PLAN_CATEGORY_TO_MIRROR),
            "language_prefixes": list(language_prefixes),
            "eligibility_policy": asdict(policy),
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "maximum_videos": maximum_videos,
            "scriptsmith_revision": scriptsmith_revision,
            "compute_sha256": compute_sha256,
        },
        "sources": source_hashes,
        "totals": {
            "scriptsmith_transcript_videos": len(scriptsmith_videos),
            "xenova_transcript_videos": len(xenova_videos),
            "transcript_videos": len(transcript_videos),
            "transcript_videos_in_both_sources": len(scriptsmith_videos & xenova_videos),
            "english_scriptsmith_videos": len(english_scriptsmith),
            "transcript_videos_with_metadata": len(metadata_videos),
            "transcript_videos_with_channel": len(channel_videos),
            "videos_with_any_audited_segment": len(videos_with_segments),
            "unique_channels_with_segments": len(channels_with_segments),
            "english_videos_with_segments": english_videos_with_segments,
            "videos_with_multiple_categories": overlapping_videos,
        },
        "table": table,
        "overlap": {
            "overlapping_videos": overlapping_videos,
            "cross_category_overlapping_pairs": dict(overlap_pairs.most_common()),
        },
        "caption_coverage": {
            "auto_caption_available": False,
            "auto_caption_note": (
                "The ScriptSmith snapshot records a language per video but no "
                "manual/auto caption provenance, so the split cannot be reported."
            ),
            "english_fraction_of_scriptsmith": round(
                len(english_scriptsmith) / len(scriptsmith_videos), 6
            )
            if scriptsmith_videos
            else 0.0,
            "metadata_fraction": round(len(metadata_videos) / len(transcript_videos), 6)
            if transcript_videos
            else 0.0,
            "channel_fraction": round(len(channel_videos) / len(transcript_videos), 6)
            if transcript_videos
            else 0.0,
        },
        "mirror_join": mirror_report,
        "next_audit": "run the trainability audit on this report's eligible records",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json_atomic(output_path, report)
    return report


def _tokenizer_fingerprint(encoder: str, revision: str) -> dict[str, object]:
    fingerprint: dict[str, object] = {
        "encoder": encoder,
        "revision": revision,
        "tokenizer_sha256": None,
        "tokenizer_path": None,
    }
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return fingerprint
    try:
        path = hf_hub_download(
            repo_id=encoder,
            filename="tokenizer.json",
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        try:
            path = hf_hub_download(
                repo_id=encoder, filename="tokenizer.json", revision=revision
            )
        except Exception:
            return fingerprint
    fingerprint["tokenizer_path"] = str(path)
    fingerprint["tokenizer_sha256"] = sha256_file(Path(path))
    return fingerprint


def _empty_category_stats() -> dict[str, object]:
    return {
        "eligible_intervals": 0,
        "aligned_intervals": 0,
        "unaligned_intervals": 0,
        "alignment_rejection_reasons": Counter(),
        "positive_windows": 0,
        "unknown_windows": 0,
        "confirmed_negative_windows": 0,
        "positive_window_spans": 0,
        "boundary_crossing_spans": 0,
        "truncated_spans": 0,
        "positive_duration_seconds": 0.0,
        "positive_tokens": 0,
        "videos_with_category": set(),
        "aligned_segment_ids": set(),
        "unaligned_segment_ids": set(),
    }


def _finalize_category_stats(
    stats: dict[str, object], windows_total: int, tokens_total: int
) -> dict[str, object]:
    positive_windows = int(stats["positive_windows"])
    positive_tokens = int(stats["positive_tokens"])
    return {
        "eligible_intervals": stats["eligible_intervals"],
        "aligned_intervals": stats["aligned_intervals"],
        "unaligned_intervals": stats["unaligned_intervals"],
        "alignment_rejection_reasons": dict(
            sorted(stats["alignment_rejection_reasons"].items())
        ),
        "videos_with_category": len(stats["videos_with_category"]),
        "positive_windows": positive_windows,
        "unknown_windows": stats["unknown_windows"],
        "confirmed_negative_windows": stats["confirmed_negative_windows"],
        "positive_window_spans": stats["positive_window_spans"],
        "boundary_crossing_spans": stats["boundary_crossing_spans"],
        "truncated_spans": stats["truncated_spans"],
        "positive_duration_seconds": round(float(stats["positive_duration_seconds"]), 3),
        "positive_tokens": positive_tokens,
        "positive_window_fraction": round(positive_windows / windows_total, 8)
        if windows_total
        else 0.0,
        "positive_token_fraction": round(positive_tokens / tokens_total, 8)
        if tokens_total
        else 0.0,
    }


def build_trainability_audit(
    mirror_path: Path,
    subtitles_directory: Path,
    metadata_directory: Path,
    output_path: Path,
    *,
    encoder: str,
    encoder_revision: str,
    categories: Sequence[str] = DEFAULT_CATEGORIES,
    language_prefixes: Sequence[str] = ("en",),
    eligibility: EligibilityPolicy | None = None,
    max_length: int = 1024,
    overlap_tokens: int = 128,
    maximum_video_duration_seconds: int = 14400,
    minimum_cues: int = 10,
    maximum_videos: int = 0,
    compute_sha256: bool = True,
    raw_audit_path: Path | None = None,
    negative_evidence_path: Path | None = None,
    tokenizer: object | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Audit B: apply the pinned preprocessing pipeline to Audit A's records.

    Every count here depends on the tokenizer revision, normalization version,
    window size and overlap, so the report embeds all of them. Confirmed
    negatives require separate category-specific evidence; absent that source,
    a category with no annotation stays UNKNOWN rather than negative.
    """

    if maximum_video_duration_seconds <= 0:
        raise ValueError("maximum video duration must be positive")
    if minimum_cues <= 0:
        raise ValueError("minimum cue count must be positive")

    started = time.monotonic()
    policy = eligibility or EligibilityPolicy()
    audited_categories = tuple(categories)
    negative_evidence = load_confirmed_negatives(negative_evidence_path)

    if progress_callback:
        progress_callback("loading video metadata")
    metadata = _load_metadata(metadata_directory)
    if maximum_videos:
        metadata = dict(list(metadata.items())[:maximum_videos])
    relevant_ids = set(metadata)

    if progress_callback:
        progress_callback(f"joining labels for {len(relevant_ids):,} videos")
    labels, mirror_report = _load_eligible_labels(
        mirror_path,
        relevant_ids,
        categories=set(audited_categories),
        policy=policy,
        progress_callback=progress_callback,
        progress_every_rows=2_000_000,
    )

    if progress_callback:
        progress_callback("loading tokenizer")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(encoder, revision=encoder_revision)
    tokenizer_fingerprint = (
        _tokenizer_fingerprint(encoder, encoder_revision) if compute_sha256 else {}
    )

    funnel: Counter[str] = Counter()
    per_category = {category: _empty_category_stats() for category in audited_categories}
    windows_total = 0
    tokens_total = 0
    windows_trimmed = 0
    videos_with_windows = 0
    video_index = 0

    if progress_callback:
        progress_callback("tokenizing transcripts and aligning spans")
    for row in _iter_subtitle_rows(subtitles_directory):
        video_id = str(row["video_id"])
        video_index += 1
        if video_id not in relevant_ids:
            funnel["not_in_metadata_or_excluded"] += 1
            continue
        if not _is_english(row["language"], language_prefixes):
            funnel["non_english"] += 1
            continue
        meta = metadata.get(video_id, {})
        availability = meta.get("availability")
        if availability is not None and availability != "public":
            funnel["not_public"] += 1
            continue
        if meta.get("live_status") == "was_live":
            funnel["was_live"] += 1
            continue
        duration = meta.get("duration")
        if isinstance(duration, int) and duration > maximum_video_duration_seconds:
            funnel["too_long"] += 1
            continue
        if not meta.get("channel_id"):
            funnel["no_channel"] += 1
            continue

        raw_cues = _parse_cues(str(row["segments_json"]))
        if len(raw_cues) < minimum_cues:
            funnel["too_few_cues"] += 1
            continue
        reconstructed = reconstruct_caption_lines(raw_cues)
        if not reconstructed:
            funnel["empty_reconstruction"] += 1
            continue
        cues = [
            TranscriptCue(index=index, start_ms=start, end_ms=end, text=text)
            for index, (start, end, text) in enumerate(reconstructed)
        ]
        transcript = assemble_transcript(cues)
        if not transcript.text.strip():
            funnel["empty_transcript"] += 1
            continue

        windows = []
        for raw_window in build_transcript_windows(
            transcript,
            tokenizer,
            max_length=max_length,
            overlap_tokens=overlap_tokens,
        ):
            fitted = _fit_window_to_max_tokens(tokenizer, raw_window, max_length)
            if fitted is None:
                continue
            if fitted.text != raw_window.text:
                windows_trimmed += 1
            windows.append(fitted)
        if not windows:
            funnel["no_windows"] += 1
            continue
        videos_with_windows += 1

        video_segments = labels.get(video_id, [])
        segments_by_category: dict[str, list[LabelSegment]] = defaultdict(list)
        for segment in video_segments:
            segments_by_category[segment.category].append(segment)

        # Resolve each segment's character span once per video, then reuse it
        # for every window. A segment with no caption overlap can never align.
        resolved_spans: dict[str, list[tuple[int, int, LabelSegment]]] = {}
        for category in audited_categories:
            resolved: list[tuple[int, int, LabelSegment]] = []
            for segment in segments_by_category.get(category, []):
                stats = per_category[category]
                stats["eligible_intervals"] += 1
                span = _char_span_for_interval(transcript, segment.start_ms, segment.end_ms)
                if span is None:
                    stats["unaligned_intervals"] += 1
                    stats["alignment_rejection_reasons"]["no_cue_overlap"] += 1
                    stats["unaligned_segment_ids"].add(segment.segment_id)
                    continue
                stats["aligned_intervals"] += 1
                stats["aligned_segment_ids"].add(segment.segment_id)
                resolved.append((span[0], span[1], segment))
            resolved_spans[category] = resolved

        for window in windows:
            windows_total += 1
            tokens_total += len(window.input_ids)
            offsets_cache: tuple[list[tuple[int, int]], list[int]] | None = None
            for category in audited_categories:
                candidates = resolved_spans.get(category, [])
                intervals: list[tuple[int, int]] = []
                if candidates:
                    stats = per_category[category]
                    for char_start, char_end, segment in candidates:
                        if min(char_end, window.end_char) <= max(
                            char_start, window.start_char
                        ):
                            continue
                        global_start = max(char_start, window.start_char)
                        global_end = min(char_end, window.end_char)
                        if global_end <= global_start:
                            continue
                        intervals.append((global_start, global_end))
                        stats["videos_with_category"].add(video_id)
                        stats["positive_window_spans"] += 1
                        if char_start < window.start_char or char_end > window.end_char:
                            stats["boundary_crossing_spans"] += 1
                        if char_end > window.end_char:
                            stats["truncated_spans"] += 1
                        stats["positive_duration_seconds"] += (
                            min(segment.end_ms, window.end_ms)
                            - max(segment.start_ms, window.start_ms)
                        ) / 1000.0
                if not intervals:
                    per_category[category]["unknown_windows"] += 1
                    continue
                per_category[category]["positive_windows"] += 1
                if offsets_cache is None:
                    offsets_cache = _content_offsets(window.offset_mapping)
                per_category[category]["positive_tokens"] += sum(
                    _count_span_tokens(*offsets_cache, start, end)
                    for start, end in _merge_intervals(intervals)
                )

        if negative_evidence:
            _apply_negative_evidence(video_id, windows, negative_evidence, per_category)
        if progress_callback and video_index % 2000 == 0:
            progress_callback(
                f"processed {video_index:,} transcripts, {windows_total:,} windows"
            )

    source_hashes: dict[str, object] = {
        "mirror": {
            "path": str(mirror_path),
            "bytes": mirror_path.stat().st_size,
            "sha256": sha256_file(mirror_path) if compute_sha256 else None,
        },
        "subtitles": _hash_directory(subtitles_directory, "*.parquet")
        if compute_sha256
        else {},
        "metadata": _hash_directory(metadata_directory, "*.parquet")
        if compute_sha256
        else {},
    }
    raw_audit: dict[str, object] = {"path": None, "sha256": None}
    if raw_audit_path is not None:
        raw_audit = {
            "path": str(raw_audit_path),
            "sha256": sha256_file(raw_audit_path)
            if compute_sha256 and raw_audit_path.is_file()
            else None,
        }

    report = {
        "audit": "smart_segments_trainability",
        "schema_version": 1,
        "audit_code_version": SMART_SEGMENTS_AUDIT_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "configuration": {
            "categories": list(audited_categories),
            "language_prefixes": list(language_prefixes),
            "eligibility_policy": asdict(policy),
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "encoder": encoder,
            "encoder_revision": encoder_revision,
            "max_length": max_length,
            "overlap_tokens": overlap_tokens,
            "maximum_video_duration_seconds": maximum_video_duration_seconds,
            "minimum_cues": minimum_cues,
            "maximum_videos": maximum_videos,
            "normalization_version": NORMALIZATION_VERSION,
            "window_builder_version": WINDOW_BUILDER_VERSION,
            "alignment_algorithm_version": ALIGNMENT_ALGORITHM_VERSION,
            "transcript_reconstruction_version": TRANSCRIPT_RECONSTRUCTION_VERSION,
            "tokenizer": tokenizer_fingerprint,
        },
        "sources": source_hashes,
        "inputs": {"raw_data_audit": raw_audit},
        "supervision_contract": {
            "states": ["POSITIVE", "NEGATIVE_CONFIRMED", "UNKNOWN"],
            "confirmed_negative_evidence": (
                str(negative_evidence_path) if negative_evidence_path else None
            ),
            "note": (
                "A category with no annotation remains UNKNOWN. An annotation "
                "for another category never establishes a negative."
            ),
        },
        "funnel": dict(funnel),
        "totals": {
            "windows": windows_total,
            "tokens": tokens_total,
            "videos_with_windows": videos_with_windows,
            "windows_trimmed_by_token_fit": windows_trimmed,
        },
        "categories": {
            category: _finalize_category_stats(
                per_category[category], windows_total, tokens_total
            )
            for category in audited_categories
        },
        "mirror_join": mirror_report,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json_atomic(output_path, report)
    return report


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union half-open character intervals so overlapping spans count once."""

    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
            continue
        merged.append((start, end))
    return merged


def _content_offsets(
    offset_mapping: Sequence[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return the non-empty offsets plus a parallel end array for bisect."""

    content = [
        (int(start), int(end))
        for start, end in offset_mapping
        if int(end) > int(start)
    ]
    return content, [end for _, end in content]


def _count_span_tokens(
    content: list[tuple[int, int]],
    ends: list[int],
    start_char: int,
    end_char: int,
) -> int:
    """Count window tokens whose offsets intersect a character span.

    ``ends`` is non-decreasing, so a bisect skips every token that ends at or
    before the span, leaving only the handful that can overlap.
    """

    import bisect

    count = 0
    for index in range(bisect.bisect_right(ends, start_char), len(content)):
        offset_start, _offset_end = content[index]
        if offset_start >= end_char:
            break
        count += 1
    return count


def _apply_negative_evidence(
    video_id: str,
    windows: Iterable[object],
    evidence: dict[str, list[ConfirmedNegative]],
    per_category: dict[str, dict[str, object]],
) -> None:
    records = evidence.get(video_id)
    if not records:
        return
    for window in windows:
        window_start = int(getattr(window, "start_ms"))
        window_end = int(getattr(window, "end_ms"))
        for negative in records:
            stats = per_category.get(negative.category)
            if stats is None:
                continue
            if min(window_end, negative.end_ms) > max(window_start, negative.start_ms):
                stats["confirmed_negative_windows"] += 1
