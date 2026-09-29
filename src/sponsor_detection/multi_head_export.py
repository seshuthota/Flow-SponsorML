from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from sponsor_detection.data.profile import sha256_file, write_json_atomic
from sponsor_detection.model.multi_head_model import load_saved_multi_head
from sponsor_detection.smart_segment_manifest import (
    MODEL_MANIFEST_SCHEMA_VERSION,
    parse_model_manifest,
    verify_artifacts,
)


def _onnx_wrapper(model):
    from torch import nn

    class LogitsOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["logits"]

    return LogitsOnly().eval()


def _export_fp32(model, path: Path, *, sequence_length: int, opset_version: int) -> None:
    import torch

    wrapper = _onnx_wrapper(model)
    input_ids = torch.ones((1, sequence_length), dtype=torch.long)
    attention_mask = torch.ones((1, sequence_length), dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["segment_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "segment_logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )


def _quantize(fp32_path: Path, int8_path: Path, op_types: list[str]) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=list(op_types),
    )


def _parity(model, onnx_path: Path, tokenizer, text: str, *, max_length: int) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        reference = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )["logits"].numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(
        ["segment_logits"],
        {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
        },
    )[0]
    difference = np.abs(reference - outputs)
    return {
        "max_abs_logit_difference": round(float(difference.max()), 6),
        "mean_abs_logit_difference": round(float(difference.mean()), 6),
        "argmax_agreement": round(
            float(np.mean(np.argmax(reference, axis=-1) == np.argmax(outputs, axis=-1))),
            6,
        ),
    }


def export_multi_head_from_config(config_path: Path) -> dict[str, object]:
    """Export the trained multi-head model to a versioned, hash-pinned bundle."""

    import onnx

    with config_path.open("rb") as source:
        configuration = tomllib.load(source)
    export = configuration["export"]
    release = configuration["release"]
    checkpoint_directory = Path(export["checkpoint_directory"])
    output_directory = Path(export["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    revision = str(export["encoder_revision"])
    sequence_length = int(export.get("sequence_length", 256))
    opset_version = int(export.get("opset_version", 17))

    model = load_saved_multi_head(checkpoint_directory, revision=revision)
    categories = list(model.categories)
    fp32_path = output_directory / "smart_segments.fp32.onnx"
    int8_path = output_directory / "smart_segments.int8.onnx"
    _export_fp32(
        model, fp32_path, sequence_length=sequence_length, opset_version=opset_version
    )
    onnx.checker.check_model(str(fp32_path))
    _quantize(fp32_path, int8_path, export.get("quantize_op_types", ["Gather"]))
    onnx.checker.check_model(str(int8_path))

    tokenizer_path = output_directory / "tokenizer.json"
    import shutil

    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = checkpoint_directory / name
        if source.is_file():
            shutil.copy2(source, output_directory / name)
    parity_inputs = _parity_reference_text(checkpoint_directory)
    fp32_parity = _parity(
        model, fp32_path, _load_tokenizer(checkpoint_directory), parity_inputs, max_length=sequence_length
    )
    int8_parity = _parity(
        model, int8_path, _load_tokenizer(checkpoint_directory), parity_inputs, max_length=sequence_length
    )

    decoder_path = output_directory / "decoder_config.json"
    decoder_path.write_text(
        json.dumps(
            {
                "version": 1,
                "calibrated": False,
                "note": "Thresholds require the reviewed benchmark and are not yet set.",
                "categories": {category: {} for category in categories},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "model_version": str(release["model_version"]),
        "model": {"file": int8_path.name, "sha256": sha256_file(int8_path)},
        "tokenizer": {"file": tokenizer_path.name, "sha256": sha256_file(tokenizer_path)},
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
        "categories": {str(index): category for index, category in enumerate(categories)},
        "labels": {"0": "O", "1": "B", "2": "I", "3": "L", "4": "U"},
        "normalization_version": 1,
        "decoder_version": 1,
        "decoder_config_sha256": sha256_file(decoder_path),
        "minimum_android_runtime_version": 1,
    }
    manifest_path = output_directory / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    parsed = parse_model_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    verified = verify_artifacts(parsed, output_directory)

    report = {
        "model_version": manifest["model_version"],
        "categories": categories,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "artifacts": {
            "fp32": {
                "path": str(fp32_path),
                "bytes": fp32_path.stat().st_size,
                "sha256": sha256_file(fp32_path),
            },
            "int8": {
                "path": str(int8_path),
                "bytes": int8_path.stat().st_size,
                "sha256": sha256_file(int8_path),
            },
            "manifest": str(manifest_path),
        },
        "verified_artifacts": verified,
        "parity": {"fp32": fp32_parity, "int8": int8_parity},
        "input_characters": len(parity_inputs),
    }
    write_json_atomic(output_directory / "export_report.json", report)
    return report


def _load_tokenizer(checkpoint_directory: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint_directory))


def _parity_reference_text(checkpoint_directory: Path) -> str:
    """Use a real training window so parity is measured on production text."""

    import pyarrow.parquet as parquet

    dataset = Path("data/training/smart_segments_windows/test.parquet")
    if not dataset.is_file():
        return "this video is sponsored by our sponsor"
    table = parquet.read_table(dataset, columns=["text"]).slice(0, 1)
    return str(table.column("text").to_pylist()[0])
