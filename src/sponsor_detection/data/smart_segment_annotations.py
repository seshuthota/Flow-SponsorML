from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from sponsor_detection.data.profile import (
    REQUIRED_COLUMNS,
    EligibilityPolicy,
    eligibility_rejection_flags,
    sha256_file,
    write_json_atomic,
)


SMART_SEGMENT_ANNOTATION_SCHEMA_VERSION = 1
NORMALIZATION_POLICY_VERSION = "smart-segment-annotations/1"
DEDUPLICATION_POLICY_VERSION = "preserve/1"
SOURCE_NAME = "sponsorblock_mirror"

# Within-category normalization is opt-in. ``preserve`` keeps one canonical row
# per mirror annotation so raw counts stay reproducible; cross-category overlaps
# are always preserved regardless of policy.
DEDUPLICATION_POLICIES = ("preserve",)


def _parquet_schema(source: dict[str, object], policy: EligibilityPolicy, categories):
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError(
            "Parquet extraction requires the data extra: pip install -e '.[data]'"
        ) from error

    metadata = {
        b"smart_segments.schema_version": str(
            SMART_SEGMENT_ANNOTATION_SCHEMA_VERSION
        ).encode(),
        b"smart_segments.normalization_policy": NORMALIZATION_POLICY_VERSION.encode(),
        b"smart_segments.deduplication_policy": DEDUPLICATION_POLICY_VERSION.encode(),
        b"smart_segments.categories": json.dumps(
            sorted(categories), separators=(",", ":")
        ).encode(),
        b"smart_segments.source_sha256": str(source.get("sha256") or "").encode(),
        b"smart_segments.eligibility_policy": json.dumps(
            asdict(policy), sort_keys=True, separators=(",", ":")
        ).encode(),
    }
    return pa.schema(
        [
            ("video_id", pa.string()),
            ("segment_id", pa.string()),
            ("category", pa.string()),
            ("start_ms", pa.int64()),
            ("end_ms", pa.int64()),
            ("votes", pa.int32()),
            ("locked", pa.bool_()),
            ("incorrect_votes", pa.int32()),
            ("views", pa.int64()),
            ("time_submitted_ms", pa.int64()),
            ("video_duration_ms", pa.int64()),
            ("hidden", pa.bool_()),
            ("shadow_hidden", pa.bool_()),
            ("action_type", pa.string()),
            ("service", pa.string()),
            ("reputation", pa.float32()),
            ("is_eligible", pa.bool_()),
            ("rejection_flags", pa.list_(pa.string())),
            ("source", pa.string()),
        ],
        metadata=metadata,
    )


def _new_batch() -> dict[str, list[object]]:
    return {
        "video_id": [],
        "segment_id": [],
        "category": [],
        "start_ms": [],
        "end_ms": [],
        "votes": [],
        "locked": [],
        "incorrect_votes": [],
        "views": [],
        "time_submitted_ms": [],
        "video_duration_ms": [],
        "hidden": [],
        "shadow_hidden": [],
        "action_type": [],
        "service": [],
        "reputation": [],
        "is_eligible": [],
        "rejection_flags": [],
        "source": [],
    }


def _append_annotation(
    batch: dict[str, list[object]],
    row: dict[str, str],
    policy: EligibilityPolicy,
) -> bool:
    start_time = float(row["startTime"])
    end_time = float(row["endTime"])
    video_duration = float(row["videoDuration"])
    votes = int(row["votes"])
    locked = int(row["locked"])
    hidden = int(row["hidden"])
    shadow_hidden = int(row["shadowHidden"])
    flags = eligibility_rejection_flags(
        policy,
        service=row["service"],
        action_type=row["actionType"],
        start_time=start_time,
        end_time=end_time,
        votes=votes,
        hidden=hidden,
        shadow_hidden=shadow_hidden,
        video_duration=video_duration,
    )
    batch["video_id"].append(row["videoID"])
    batch["segment_id"].append(row["UUID"])
    batch["category"].append(row["category"])
    batch["start_ms"].append(round(start_time * 1000))
    batch["end_ms"].append(round(end_time * 1000))
    batch["votes"].append(votes)
    batch["locked"].append(locked != 0)
    batch["incorrect_votes"].append(int(row["incorrectVotes"]))
    batch["views"].append(int(row["views"]))
    batch["time_submitted_ms"].append(int(row["timeSubmitted"]))
    batch["video_duration_ms"].append(
        round(video_duration * 1000) if video_duration > 0 else None
    )
    batch["hidden"].append(hidden != 0)
    batch["shadow_hidden"].append(shadow_hidden != 0)
    batch["action_type"].append(row["actionType"])
    batch["service"].append(row["service"])
    batch["reputation"].append(float(row["reputation"]))
    batch["is_eligible"].append(not flags)
    batch["rejection_flags"].append(sorted(flags))
    batch["source"].append(SOURCE_NAME)
    return not flags


def build_smart_segment_annotations(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    policy: EligibilityPolicy,
    *,
    categories: list[str],
    deduplication_policy: str = "preserve",
    batch_size: int = 100_000,
    compression: str = "zstd",
    compute_sha256: bool = True,
    expected_source_sha256: str | None = None,
    progress_every_rows: int = 0,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict[str, object]:
    """Normalize the raw SponsorBlock mirror into the canonical annotation table.

    Every audited row is retained with its category, timestamps, quality
    metadata, eligibility verdict and provenance; ineligible rows stay in the
    artifact with ``is_eligible = false``. Overlaps between different categories
    are never collapsed.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Parquet extraction requires the data extra: pip install -e '.[data]'"
        ) from error

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if deduplication_policy not in DEDUPLICATION_POLICIES:
        raise ValueError(
            f"unsupported deduplication policy: {deduplication_policy}; "
            f"supported: {', '.join(DEDUPLICATION_POLICIES)}"
        )
    audited_categories = list(dict.fromkeys(categories))
    if not audited_categories:
        raise ValueError("at least one category must be audited")
    category_set = set(audited_categories)

    started = time.monotonic()
    source_sha256 = sha256_file(input_path) if compute_sha256 else None
    if expected_source_sha256 and source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError("input CSV checksum differs from the expected snapshot")
    source = {
        "path": str(input_path),
        "bytes": input_path.stat().st_size,
        "sha256": source_sha256,
    }
    schema = _parquet_schema(source, policy, audited_categories)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    total_rows = 0
    audited_rows = 0
    eligible_rows = 0
    malformed_rows = 0
    by_category: Counter[str] = Counter()
    eligible_by_category: Counter[str] = Counter()
    eligibility_rejections: dict[str, Counter[str]] = {}
    batch = _new_batch()

    try:
        with input_path.open("r", encoding="utf-8", newline="") as source_file:
            reader = csv.DictReader(source_file)
            missing_columns = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames or []))
            if missing_columns:
                raise ValueError(f"missing required columns: {', '.join(missing_columns)}")
            with pq.ParquetWriter(
                temporary_path, schema, compression=compression, use_dictionary=True
            ) as writer:
                for row in reader:
                    total_rows += 1
                    category = row.get("category") or ""
                    if category not in category_set:
                        continue
                    audited_rows += 1
                    by_category[category] += 1
                    try:
                        eligible = _append_annotation(batch, row, policy)
                    except (KeyError, TypeError, ValueError):
                        malformed_rows += 1
                        continue
                    if eligible:
                        eligible_rows += 1
                        eligible_by_category[category] += 1
                    else:
                        rejection_counter = eligibility_rejections.setdefault(
                            category, Counter()
                        )
                        for flag in batch["rejection_flags"][-1]:
                            rejection_counter[flag] += 1
                    if len(batch["video_id"]) >= batch_size:
                        writer.write_table(pa.Table.from_pydict(batch, schema=schema))
                        batch = _new_batch()
                    if progress_every_rows and total_rows % progress_every_rows == 0:
                        if progress_callback:
                            progress_callback(
                                total_rows, audited_rows, time.monotonic() - started
                            )
                if batch["video_id"]:
                    writer.write_table(pa.Table.from_pydict(batch, schema=schema))
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": SMART_SEGMENT_ANNOTATION_SCHEMA_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "normalization_policy": NORMALIZATION_POLICY_VERSION,
        "deduplication_policy": deduplication_policy,
        "source": source,
        "eligibility_policy": asdict(policy),
        "categories": audited_categories,
        "counts": {
            "source_rows": total_rows,
            "audited_rows": audited_rows,
            "written_rows": audited_rows - malformed_rows,
            "eligible_rows": eligible_rows,
            "ineligible_rows": audited_rows - malformed_rows - eligible_rows,
            "malformed_rows": malformed_rows,
            "by_category": dict(sorted(by_category.items())),
            "eligible_by_category": dict(sorted(eligible_by_category.items())),
        },
        "rejections_by_category": {
            category: dict(sorted(counter.items()))
            for category, counter in sorted(eligibility_rejections.items())
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path) if compute_sha256 else None,
            "compression": compression,
        },
        "build_seconds": round(time.monotonic() - started, 3),
    }
    write_json_atomic(manifest_path, report)
    return report
