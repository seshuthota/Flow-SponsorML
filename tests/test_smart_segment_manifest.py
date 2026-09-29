from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sponsor_detection.data.profile import sha256_file
from sponsor_detection.smart_segment_manifest import (
    ManifestError,
    load_model_manifest,
    parse_model_manifest,
    validate_runtime,
    verify_artifacts,
)


def _manifest(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "model_version": "smart-segments-v1",
        "model": {"file": "smart_segments.int8.ort", "sha256": "a" * 64},
        "tokenizer": {"file": "tokenizer.json", "sha256": "b" * 64},
        "inputs": {
            "input_ids": {"dtype": "int64", "shape": ["batch", "sequence"]},
            "attention_mask": {"dtype": "int64", "shape": ["batch", "sequence"]},
        },
        "outputs": {
            "segment_logits": {
                "dtype": "float32",
                "shape": ["batch", "sequence", "category", "bilou"],
            }
        },
        "categories": {"0": "sponsor", "1": "selfpromo", "2": "interaction"},
        "labels": {"0": "O", "1": "B", "2": "I", "3": "L", "4": "U"},
        "normalization_version": 1,
        "decoder_version": 1,
        "decoder_config_sha256": "c" * 64,
        "minimum_android_runtime_version": 1,
    }
    payload.update(overrides)
    return payload


class ParseModelManifestTest(unittest.TestCase):
    def test_parses_a_valid_manifest(self) -> None:
        manifest = parse_model_manifest(_manifest())

        self.assertEqual(manifest.model_version, "smart-segments-v1")
        self.assertEqual(manifest.categories[2], "interaction")
        self.assertEqual(manifest.labels[4], "U")

    def test_rejects_unknown_schema_version(self) -> None:
        with self.assertRaises(ManifestError):
            parse_model_manifest(_manifest(schema_version=2))

    def test_rejects_non_contiguous_categories(self) -> None:
        with self.assertRaises(ManifestError):
            parse_model_manifest(
                _manifest(categories={"0": "sponsor", "2": "interaction"})
            )

    def test_rejects_wrong_label_mapping(self) -> None:
        with self.assertRaises(ManifestError):
            parse_model_manifest(
                _manifest(labels={"0": "O", "1": "B", "2": "I", "3": "U", "4": "L"})
            )

    def test_rejects_unknown_output_layout(self) -> None:
        with self.assertRaises(ManifestError):
            parse_model_manifest(
                _manifest(
                    outputs={
                        "segment_logits": {
                            "dtype": "float32",
                            "shape": ["batch", "bilou", "sequence", "category"],
                        }
                    }
                )
            )

    def test_rejects_unsupported_minimum_runtime(self) -> None:
        with self.assertRaises(ManifestError):
            parse_model_manifest(_manifest(minimum_android_runtime_version=99))

    def test_rejects_missing_input_dtype(self) -> None:
        with self.assertRaises(ManifestError):
            parse_model_manifest(
                _manifest(inputs={"input_ids": {"shape": ["batch", "sequence"]}})
            )


class LoadModelManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "manifest.json"
        self.path.write_text(json.dumps(_manifest()), encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_verifies_the_pinned_manifest_hash(self) -> None:
        manifest = load_model_manifest(
            self.path, pinned_manifest_sha256=sha256_file(self.path)
        )

        self.assertEqual(manifest.schema_version, 1)

    def test_rejects_a_pinned_hash_mismatch(self) -> None:
        with self.assertRaises(ManifestError):
            load_model_manifest(self.path, pinned_manifest_sha256="0" * 64)


class VerifyArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "smart_segments.int8.ort").write_bytes(b"model-bytes")
        (self.root / "tokenizer.json").write_bytes(b"tokenizer-bytes")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _real_manifest(self) -> dict[str, object]:
        return _manifest(
            model={
                "file": "smart_segments.int8.ort",
                "sha256": sha256_file(self.root / "smart_segments.int8.ort"),
            },
            tokenizer={
                "file": "tokenizer.json",
                "sha256": sha256_file(self.root / "tokenizer.json"),
            },
        )

    def test_verifies_artifact_hashes(self) -> None:
        manifest = parse_model_manifest(self._real_manifest())

        verified = verify_artifacts(manifest, self.root)

        self.assertEqual(set(verified), {"model", "tokenizer"})

    def test_rejects_a_tampered_artifact(self) -> None:
        manifest = parse_model_manifest(self._real_manifest())
        (self.root / "tokenizer.json").write_bytes(b"tampered")

        with self.assertRaises(ManifestError):
            verify_artifacts(manifest, self.root)

    def test_rejects_a_missing_artifact(self) -> None:
        manifest = parse_model_manifest(self._real_manifest())
        (self.root / "tokenizer.json").unlink()

        with self.assertRaises(ManifestError):
            verify_artifacts(manifest, self.root)


class ValidateRuntimeTest(unittest.TestCase):
    def test_accepts_matching_runtime_and_rank(self) -> None:
        manifest = parse_model_manifest(_manifest())

        validate_runtime(
            manifest,
            runtime_version=1,
            output_rank=4,
            output_axes=("batch", "sequence", "category", "bilou"),
        )

    def test_rejects_old_runtime(self) -> None:
        manifest = parse_model_manifest(_manifest(minimum_android_runtime_version=1))

        with self.assertRaises(ManifestError):
            validate_runtime(manifest, runtime_version=0, output_rank=4)

    def test_rejects_wrong_rank(self) -> None:
        manifest = parse_model_manifest(_manifest())

        with self.assertRaises(ManifestError):
            validate_runtime(manifest, runtime_version=1, output_rank=3)

    def test_rejects_wrong_axes(self) -> None:
        manifest = parse_model_manifest(_manifest())

        with self.assertRaises(ManifestError):
            validate_runtime(
                manifest,
                runtime_version=1,
                output_rank=4,
                output_axes=("batch", "sequence", "bilou", "category"),
            )


if __name__ == "__main__":
    unittest.main()
