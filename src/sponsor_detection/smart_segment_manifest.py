from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sponsor_detection.data.profile import sha256_file


MODEL_MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (MODEL_MANIFEST_SCHEMA_VERSION,)
SUPPORTED_MINIMUM_RUNTIME_VERSIONS = (1,)
EXPECTED_LABELS = ("O", "B", "I", "L", "U")
EXPECTED_OUTPUT_AXES = ("batch", "sequence", "category", "bilou")
OUTPUT_NAME = "segment_logits"


class ManifestError(ValueError):
    """Raised when a model bundle cannot be trusted or is incompatible."""


@dataclass(frozen=True, slots=True)
class ModelManifest:
    schema_version: int
    model_version: str
    model_file: str
    model_sha256: str
    tokenizer_file: str
    tokenizer_sha256: str
    decoder_config_sha256: str
    normalization_version: int
    decoder_version: int
    minimum_android_runtime_version: int
    inputs: dict[str, dict[str, object]]
    outputs: dict[str, dict[str, object]]
    categories: dict[int, str]
    labels: dict[int, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _file_entry(section: object, name: str, field: str) -> tuple[str, str]:
    _require(isinstance(section, dict), f"{field} must be an object")
    filename = section.get("file")
    digest = section.get("sha256")
    _require(
        isinstance(filename, str) and filename,
        f"{field}.file must be a non-empty string",
    )
    _require(
        isinstance(digest, str) and len(digest) == 64,
        f"{field}.sha256 must be a 64-character hex digest",
    )
    return filename, digest


def _indexed_mapping(value: object, field: str) -> dict[int, str]:
    _require(isinstance(value, dict) and value, f"{field} must be a non-empty object")
    mapping: dict[int, str] = {}
    for key, item in value.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as error:
            raise ManifestError(f"{field} keys must be integers") from error
        _require(isinstance(item, str) and item, f"{field}[{key}] must be a string")
        mapping[index] = item
    expected = list(range(len(mapping)))
    _require(
        sorted(mapping) == expected,
        f"{field} must be contiguous from 0",
    )
    return mapping


def parse_model_manifest(payload: object) -> ModelManifest:
    """Validate and parse the versioned model manifest contract.

    Rejects an unknown schema, an unreadable tensor layout, a non-contiguous
    category axis or a BILOU label mapping that is not ``O, B, I, L, U`` —
    anything that would otherwise have to be guessed at inference time.
    """

    _require(isinstance(payload, dict), "manifest must be an object")
    schema_version = payload.get("schema_version")
    _require(
        isinstance(schema_version, int) and schema_version in SUPPORTED_SCHEMA_VERSIONS,
        f"unsupported schema_version: {schema_version!r}",
    )
    model_version = payload.get("model_version")
    _require(
        isinstance(model_version, str) and model_version,
        "model_version must be a non-empty string",
    )

    model_file, model_sha256 = _file_entry(payload.get("model"), "model", "model")
    tokenizer_file, tokenizer_sha256 = _file_entry(
        payload.get("tokenizer"), "tokenizer", "tokenizer"
    )

    inputs = payload.get("inputs")
    _require(isinstance(inputs, dict) and inputs, "inputs must be a non-empty object")
    for name, entry in inputs.items():
        _require(isinstance(entry, dict), f"inputs.{name} must be an object")
        _require(
            isinstance(entry.get("dtype"), str) and entry["dtype"],
            f"inputs.{name}.dtype must be a string",
        )
        _require(
            isinstance(entry.get("shape"), list),
            f"inputs.{name}.shape must be a list",
        )

    outputs = payload.get("outputs")
    _require(isinstance(outputs, dict), "outputs must be an object")
    _require(OUTPUT_NAME in outputs, f"outputs must include {OUTPUT_NAME}")
    output_shape = outputs[OUTPUT_NAME].get("shape")
    _require(
        output_shape == list(EXPECTED_OUTPUT_AXES),
        f"{OUTPUT_NAME}.shape must be {list(EXPECTED_OUTPUT_AXES)}, got {output_shape}",
    )

    categories = _indexed_mapping(payload.get("categories"), "categories")
    labels = _indexed_mapping(payload.get("labels"), "labels")
    _require(
        tuple(labels[index] for index in sorted(labels)) == EXPECTED_LABELS,
        f"labels must be exactly {list(EXPECTED_LABELS)} from index 0",
    )

    decoder_config_sha256 = payload.get("decoder_config_sha256")
    _require(
        isinstance(decoder_config_sha256, str) and len(decoder_config_sha256) == 64,
        "decoder_config_sha256 must be a 64-character hex digest",
    )
    normalization_version = payload.get("normalization_version")
    decoder_version = payload.get("decoder_version")
    minimum_runtime = payload.get("minimum_android_runtime_version")
    for field, value in (
        ("normalization_version", normalization_version),
        ("decoder_version", decoder_version),
        ("minimum_android_runtime_version", minimum_runtime),
    ):
        _require(
            isinstance(value, int) and value > 0, f"{field} must be a positive integer"
        )
    _require(
        minimum_runtime in SUPPORTED_MINIMUM_RUNTIME_VERSIONS,
        f"unsupported minimum_android_runtime_version: {minimum_runtime!r}",
    )

    return ModelManifest(
        schema_version=schema_version,
        model_version=model_version,
        model_file=model_file,
        model_sha256=model_sha256,
        tokenizer_file=tokenizer_file,
        tokenizer_sha256=tokenizer_sha256,
        decoder_config_sha256=decoder_config_sha256,
        normalization_version=normalization_version,
        decoder_version=decoder_version,
        minimum_android_runtime_version=minimum_runtime,
        inputs={str(name): dict(entry) for name, entry in inputs.items()},
        outputs={str(name): dict(entry) for name, entry in outputs.items()},
        categories=categories,
        labels=labels,
    )


def load_model_manifest(
    path: Path, *, pinned_manifest_sha256: str | None = None
) -> ModelManifest:
    """Load a manifest, verifying the application-pinned hash first.

    The pinned hash must be checked before any manifest-provided hash is
    trusted; otherwise a tampered manifest could vouch for tampered artifacts.
    """

    if pinned_manifest_sha256 is not None:
        actual = sha256_file(path)
        if actual != pinned_manifest_sha256:
            raise ManifestError(
                "manifest hash does not match the pinned revision: "
                f"{actual} != {pinned_manifest_sha256}"
            )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ManifestError(f"manifest is not valid JSON: {path}") from error
    return parse_model_manifest(payload)


def artifact_paths(manifest: ModelManifest, bundle_directory: Path) -> dict[str, Path]:
    return {
        "model": bundle_directory / manifest.model_file,
        "tokenizer": bundle_directory / manifest.tokenizer_file,
    }


def verify_artifacts(manifest: ModelManifest, bundle_directory: Path) -> dict[str, str]:
    """Verify every artifact against the hash the pinned manifest declares."""

    expected = {
        "model": manifest.model_sha256,
        "tokenizer": manifest.tokenizer_sha256,
    }
    verified: dict[str, str] = {}
    for name, path in artifact_paths(manifest, bundle_directory).items():
        if not path.is_file():
            raise ManifestError(f"{name} artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected[name]:
            raise ManifestError(
                f"{name} artifact hash mismatch: {actual} != {expected[name]}"
            )
        verified[name] = actual
    return verified


def validate_runtime(
    manifest: ModelManifest,
    *,
    runtime_version: int,
    output_rank: int,
    output_axes: tuple[str, ...] | None = None,
) -> None:
    """Reject a bundle the runtime cannot execute or interpret safely."""

    if runtime_version < manifest.minimum_android_runtime_version:
        raise ManifestError(
            "runtime is older than the bundle requires: "
            f"{runtime_version} < {manifest.minimum_android_runtime_version}"
        )
    if output_rank != len(EXPECTED_OUTPUT_AXES):
        raise ManifestError(
            f"output rank must be {len(EXPECTED_OUTPUT_AXES)}, got {output_rank}"
        )
    if output_axes is not None and tuple(output_axes) != EXPECTED_OUTPUT_AXES:
        raise ManifestError(
            f"output axes must be {list(EXPECTED_OUTPUT_AXES)}, got {list(output_axes)}"
        )
