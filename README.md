# Flow-SponsorML

On-device sponsor detection model for [Flow](https://github.com/seshuthota/Flow), the Android
music/video client. This repository holds the full ML pipeline: dataset collection and
annotation, training, calibration, evaluation, ONNX export, and the custom ONNX Runtime build
the app links against.

It was split out of the Flow repository so that ML iteration no longer lands in the app's git
history. The two repositories develop in parallel and have no build-time coupling: Flow consumes
only two published artifacts from here, both pinned by version.

## What Flow consumes from this repo

| Artifact | Delivered to Flow as | Pinned by |
| --- | --- | --- |
| ONNX model + tokenizer | Downloaded from Hugging Face at runtime, not bundled in the APK | `SPONSOR_MODEL_NAME` and the two SHA256 constants in `SponsorFeedbackModels.kt` |
| Custom ONNX Runtime AAR | Vendored at `app/libs/onnxruntime-android-1.29.0.aar` in the Flow repo | Filename in `app/build.gradle.kts` |

Everything else here — checkpoints, datasets, evaluation reports, the ONNX Runtime source
checkout — stays local to this repository and is gitignored.

### The custom ONNX Runtime build

Flow vendors a minimal ONNX Runtime AAR instead of the upstream Maven Central artifact, which
saves roughly 10-15 MB in the APK. The build is driven by
[`config/onnxruntime_android_build.json`](config/onnxruntime_android_build.json) plus the
operator set emitted by the Android export stage.

```bash
# 1. Export the model and its required-operator config
.venv/bin/sponsor-detection export-android --config config/export_android_combined.toml

# 2. Build the minimal AAR (clones onnxruntime v1.29.0 and runs a Docker build)
./scripts/build_android_ort_runtime.sh
```

Step 2 writes the AAR to
`artifacts/android/onnxruntime_custom/build/output/aar_out/MinSizeRel/`. Copy it into the Flow
repo at `app/libs/onnxruntime-android-1.29.0.aar` when bumping the runtime version.

## Layout

All paths in this repository are relative to the repository root, and every command is run from
it.

| Path | Contents |
| --- | --- |
| `config/` | Per-stage TOML configs. Each names the manifest, data and output paths for one pipeline stage. |
| `src/sponsor_detection/` | The pipeline package. `cli.py` holds the argparse defaults for every stage. |
| `scripts/` | Build and training entry points, including the ONNX Runtime build and the Hugging Face upload. |
| `tests/` | Python unit tests. |
| `reports/` | Evaluation outputs and manifests. Manifests are read back as inputs by later stages, so their path fields must stay valid. |
| `artifacts/`, `data/`, `checkpoints/` | Generated. Gitignored. |

## Workflow

See [PLAN.md](PLAN.md) for the pipeline design and [STATUS_AND_NEXT_STEPS.md](STATUS_AND_NEXT_STEPS.md)
for where the work currently stands.

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Device feedback loop

Flow writes user feedback to `sponsor_training_v1.jsonl` on device. Those journals are the input
for the next round of annotation and retraining; the wiring for exporting them out of the app is
not built yet.
